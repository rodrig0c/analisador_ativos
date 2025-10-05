# app.py
# Versão atualizada com:
# - Backtest profissional com custos de transação, spread e slippage
# - Validação temporal (walk-forward / walk-forward rolling)
# - Position sizing (fracção fixa e sizing baseado em volatilidade)
# - Stop-loss e take-profit por ordem
# - Controle de dados (checagens, normalização de índice, ajustes)
# - Hyperparameter tuning com TimeSeriesSplit para modelos de árvore e XGBoost (se instalado)
# - Logging estruturado em arquivo e na interface
# - Exportação de resultados e metadados com versão e hash simples
# - Código entregue completo e legível

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime
from pandas.tseries.offsets import BDay
import plotly.express as px
import plotly.graph_objects as go
import io
import zipfile
import json
import logging
import os
import hashlib
import warnings

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Analisador de Ativos - Profissional", layout="wide")

# -----------------------
# Configurações principais
# -----------------------
APP_VERSION = "2025-10-05_1"
LOG_FOLDER = "logs"
os.makedirs(LOG_FOLDER, exist_ok=True)
LOG_FILE = os.path.join(LOG_FOLDER, f"app_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Backtest / mercado
COMMISSION_PER_TRADE = 0.0035        # 0.35% por execução (ex: corretora + taxas)
SLIPPAGE_PCT = 0.001                 # 0.1% slippage por execução
SPREAD_PCT = 0.0005                  # 0.05% spread estimado
MIN_DAYS_CHARTS = 60
MIN_DAYS_ADVANCED = 180
FORECAST_DAYS = 5

# Risk management
POSITION_SIZE_FRACTION = 0.05        # fração do capital por posição (5%)
VOL_SIZING_LOOKBACK = 30             # volatilidade lookback para sizing
MAX_POSITION_LEVERAGE = 1.0

# Walk-forward
WALKFWD_TRAIN_WINDOW = 252 * 2      # 2 anos de treino por janela
WALKFWD_TEST_WINDOW = 63            # 3 meses de teste por janela
WALKFWD_EXPAND = True               # se True, treino expande; se False, janela móvel

# Hyperparameter tuning settings
TS_CV_SPLITS = 4

# Optional libraries
HAS_XGB = False
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

HAS_TF = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    HAS_TF = True
except Exception:
    HAS_TF = False

# -----------------------
# Logging
# -----------------------
logger = logging.getLogger("assetz_prof")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(LOG_FILE)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(fh)

# Console handler for Streamlit only at INFO
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# -----------------------
# Utilitários
# -----------------------
def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

@st.cache_data
def get_tickers_from_csv():
    try:
        df = pd.read_csv('acoes-listadas-b3.csv')
        # Normaliza colunas
        df = df.rename(columns={c: c.strip() for c in df.columns})
        # tenta mapear colunas esperadas
        possible_ticker_cols = [c for c in df.columns if c.lower() in ('ticker', 'codigo', 'sigla')]
        possible_name_cols = [c for c in df.columns if c.lower() in ('nome', 'empresa', 'name')]
        ticker_col = possible_ticker_cols[0] if possible_ticker_cols else df.columns[0]
        name_col = possible_name_cols[0] if possible_name_cols else df.columns[1] if df.shape[1] > 1 else ticker_col
        df = df.rename(columns={ticker_col: 'ticker', name_col: 'nome'})
        df['ticker'] = df['ticker'].astype(str).str.replace('.SA', '', case=False).str.strip()
        df['display'] = df['nome'].astype(str) + ' (' + df['ticker'] + ')'
        logger.info("Lista de tickers carregada do CSV local.")
        return df[['ticker', 'nome', 'display']]
    except Exception as e:
        logger.warning(f"Falha ao carregar CSV de tickers: {e}. Usando fallback.")
        fallback = {'ticker': ['PETR4', 'VALE3', 'ITUB4', 'MGLU3'], 'nome': ['Petrobras', 'Vale', 'Itaú', 'Magazine Luiza']}
        df = pd.DataFrame(fallback)
        df['display'] = df['nome'] + ' (' + df['ticker'] + ')'
        return df[['ticker', 'nome', 'display']]

def safe_yf_download(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Faz download seguro dos dados do Yahoo e aplica checks básicos:
    - Normaliza colunas
    - Converte índice para datetime
    - Remove duplicados
    - Ajusta colunas se MultiIndex
    """
    logger.info(f"Iniciando download Yahoo Finance para {ticker} de {start} até {end}")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        logger.warning(f"Download vazio para {ticker}.")
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # garantir colunas esperadas
    for c in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']:
        if c not in df.columns:
            df[c] = np.nan
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()
    logger.info(f"Download completo. {len(df)} linhas para {ticker}.")
    return df

@st.cache_data
def load_data(ticker, start, end):
    s = pd.to_datetime(start)
    e = pd.to_datetime(end) + pd.Timedelta(days=1)
    df = safe_yf_download(ticker, s.strftime('%Y-%m-%d'), e.strftime('%Y-%m-%d'))
    return df

def calculate_indicators(df):
    df = df.copy()
    df['Close'] = df['Close'].astype(float)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MM_Curta'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['MM_Longa'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['BB_Media'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['BB_Superior'] = df['BB_Media'] + 2 * df['Close'].rolling(window=20, min_periods=1).std()
    df['BB_Inferior'] = df['BB_Media'] - 2 * df['Close'].rolling(window=20, min_periods=1).std()
    df['Daily Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily Return'].rolling(window=30, min_periods=1).std() * (252 ** 0.5)
    return df

def prepare_advanced_features(df, forecast_days=FORECAST_DAYS):
    d = df[['Close', 'Volume', 'RSI', 'MM_Curta', 'MM_Longa', 'Volatility']].copy()
    periods = [1, 3, 5, 10, 20]
    for p in periods:
        d[f'return_{p}d'] = d['Close'].pct_change(p)
        d[f'volatility_{p}d'] = d['Close'].pct_change().rolling(window=p, min_periods=1).std()
    if 'Volume' in d.columns and d['Volume'].sum() > 0:
        for p in periods:
            d[f'volume_ma_{p}d'] = d['Volume'].rolling(window=p, min_periods=1).mean()
    d['price_vs_ma20'] = d['Close'] / d['MM_Curta'].replace(0, np.nan)
    d['price_vs_ma50'] = d['Close'] / d['MM_Longa'].replace(0, np.nan)
    d['target_future_return'] = d['Close'].shift(-forecast_days) / d['Close'] - 1
    d['target_future_price'] = d['Close'].shift(-forecast_days)
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    potential = [c for c in d.columns if c.startswith(('return_', 'volume_ma_', 'volatility_', 'price_vs_'))]
    potential.extend(['RSI', 'Volatility'])
    features = [c for c in potential if c in d.columns and not d[c].isnull().all()]
    required = features + ['target_future_return', 'target_future_price']
    d.dropna(subset=required, inplace=True)
    return d, features

def create_base_models():
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'Neural Net': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    }
    if HAS_XGB:
        models['XGBoost'] = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    return models

def compute_price_metrics(y_true_price, y_pred_price):
    y_t = np.array(y_true_price, dtype=float)
    y_p = np.array(y_pred_price, dtype=float)
    mask = np.isfinite(y_p) & np.isfinite(y_t)
    if mask.sum() == 0:
        return {'MAE': None, 'RMSE': None, 'MAPE': None}
    y_t_m, y_p_m = y_t[mask], y_p[mask]
    mae = float(mean_absolute_error(y_t_m, y_p_m))
    rmse = float(np.sqrt(mean_squared_error(y_t_m, y_p_m)))
    denom = np.where(np.abs(y_t_m) < 1e-6, 1e-6, np.abs(y_t_m))
    mape = float(np.mean(np.abs((y_t_m - y_p_m) / denom)))
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def compute_return_hitrate(y_true_ret, y_pred_ret):
    y_t, y_p = np.array(y_true_ret, dtype=float), np.array(y_pred_ret, dtype=float)
    mask = np.isfinite(y_p) & np.isfinite(y_t)
    if mask.sum() == 0:
        return None
    return float(np.mean((np.sign(y_t[mask]) == np.sign(y_p[mask])).astype(float)))

# -----------------------
# Backtest profissional
# -----------------------
def simulate_trade_sequence(dates, base_prices, pred_returns, real_returns,
                            initial_capital=10000.0,
                            position_fraction=POSITION_SIZE_FRACTION,
                            commission_pct=COMMISSION_PER_TRADE,
                            slippage_pct=SLIPPAGE_PCT,
                            spread_pct=SPREAD_PCT,
                            stop_loss_pct=0.03,
                            take_profit_pct=0.06,
                            vol_sizing=False,
                            vol_lookback=VOL_SIZING_LOOKBACK):
    """
    Simula trades baseados em sinais de previsão de retorno.
    Estratégia simples:
    - Entrar comprado no dia t se pred_returns[t] > 0
    - Sair quando stop-loss ou take-profit for atingido ou no final do período
    - Position sizing considera posição como fraction do capital; se vol_sizing True usa volatilidade
    - Aplica comissão, slippage e spread nos preços de entrada/saída
    """
    n = len(dates)
    capital = initial_capital
    positions = []
    logs = []
    equity_curve = np.full(n, np.nan)
    cash = capital
    position = None  # dict com keys: size_shares, entry_price, entry_index, notional
    for i in range(n):
        price = float(base_prices[i])
        pred_ret = float(pred_returns[i]) if np.isfinite(pred_returns[i]) else 0.0
        real_ret = float(real_returns[i]) if np.isfinite(real_returns[i]) else 0.0
        date_i = dates[i]
        # Update equity
        if position is None:
            equity_curve[i] = cash
        else:
            # mark-to-market
            mtm = position['size_shares'] * price
            equity_curve[i] = cash + mtm

        # Entry rule: signal positive
        enter_signal = pred_ret > 0.0
        exit_signal = False

        # If we have a position, check stop-loss / take-profit
        if position is not None:
            # current return since entry
            entry_price = position['entry_price']
            # use mid-price adjustments: assume we cross at market with slippage and spread
            effective_price = price * (1 - spread_pct)  # selling receives slightly less
            current_ret = (effective_price / entry_price) - 1.0
            if current_ret <= -abs(stop_loss_pct):
                exit_signal = True
                exit_reason = 'stop_loss'
            elif current_ret >= abs(take_profit_pct):
                exit_signal = True
                exit_reason = 'take_profit'
            else:
                exit_signal = False
                exit_reason = None

        # Execute exit if signaled
        if position is not None and exit_signal:
            # Simulate exit price with slippage and spread
            exit_price = price * (1 - spread_pct) * (1 - slippage_pct)
            proceeds = position['size_shares'] * exit_price
            commission = proceeds * commission_pct
            cash += proceeds - commission
            pnl = proceeds - commission - position['notional']
            logs.append({'date': date_i, 'action': 'SELL', 'price': exit_price, 'shares': position['size_shares'], 'notional_in': position['notional'], 'proceeds': proceeds, 'commission': commission, 'pnl': pnl, 'reason': exit_reason})
            position = None

        # Execute entry if no position and entry signal
        if position is None and enter_signal:
            # Determine position size
            if vol_sizing and i >= vol_lookback:
                recent_vol = np.nanstd(real_returns[max(0, i - vol_lookback):i]) * np.sqrt(252)
                # avoid division by zero
                target_risk_dollar = capital * position_fraction
                # scale by volatility: larger volatility -> smaller notional (inverse)
                vol_factor = 1.0 / (recent_vol + 1e-6)
                notional = max(0.0, min(capital * position_fraction * vol_factor, capital * MAX_POSITION_LEVERAGE))
            else:
                notional = capital * position_fraction
            # Shares to buy considering entry execution costs
            entry_price = price * (1 + spread_pct) * (1 + slippage_pct)
            if entry_price <= 0 or notional <= 0:
                continue
            size_shares = np.floor(notional / entry_price)
            if size_shares <= 0:
                continue
            notional_used = size_shares * entry_price
            commission = notional_used * commission_pct
            cash -= notional_used + commission
            position = {'size_shares': size_shares, 'entry_price': entry_price, 'entry_index': i, 'notional': notional_used}
            logs.append({'date': date_i, 'action': 'BUY', 'price': entry_price, 'shares': size_shares, 'notional_in': notional_used, 'commission': commission})

        # If end and we still have position, close it
        if i == n - 1 and position is not None:
            exit_price = price * (1 - spread_pct) * (1 - slippage_pct)
            proceeds = position['size_shares'] * exit_price
            commission = proceeds * commission_pct
            cash += proceeds - commission
            pnl = proceeds - commission - position['notional']
            logs.append({'date': date_i, 'action': 'SELL_END', 'price': exit_price, 'shares': position['size_shares'], 'notional_in': position['notional'], 'proceeds': proceeds, 'commission': commission, 'pnl': pnl, 'reason': 'end_period'})
            position = None
            equity_curve[i] = cash

        # If no trades this day set equity to previous or cash
        if np.isnan(equity_curve[i]):
            equity_curve[i] = cash if position is None else cash + position['size_shares'] * price

    # result
    trades_df = pd.DataFrame(logs)
    eq = pd.DataFrame({'Date': dates, 'Equity': equity_curve})
    return {'equity_curve': eq, 'trades': trades_df, 'final_capital': cash, 'initial_capital': initial_capital}

# -----------------------
# Walk-forward training + backtest
# -----------------------
def walk_forward_backtest(df_adv, features, initial_capital=10000.0,
                          train_window=WALKFWD_TRAIN_WINDOW, test_window=WALKFWD_TEST_WINDOW,
                          expand_train=WALKFWD_EXPAND, models_to_train=None,
                          tune_models=True, cv_splits=TS_CV_SPLITS,
                          progress_callback=None):
    """
    Executa walk-forward. Para cada janela:
    - Treina modelos no período de treino
    - (Opcional) Tuning com TimeSeriesSplit
    - Testa no período de teste
    - Coleta previsões e resultados para backtest de estratégia com gerenciamento de risco
    Retorna dicionário com métricas agregadas e detalhes por janela.
    """
    if models_to_train is None:
        models_to_train = create_base_models()

    n = len(df_adv)
    results = []
    trained_models_history = []
    i_start = 0
    windows = []
    # calculo de janelas baseado no índice
    while True:
        train_start = i_start
        train_end = train_start + train_window
        test_end = train_end + test_window
        if train_end >= n:
            break
        if test_end > n:
            test_end = n
        windows.append((train_start, train_end, train_end, test_end))
        if not expand_train:
            i_start = i_start + test_window
        else:
            i_start = i_start + test_window

        if test_end == n:
            break

    total_windows = len(windows)
    logger.info(f"Walk-forward janelas calculadas: {total_windows}")

    aggregated_preds = []
    aggregated_reals = []
    aggregated_dates = []
    aggregated_base_prices = []
    ensemble_details = []

    for idx, (train_s, train_e, test_s, test_e) in enumerate(windows):
        if progress_callback:
            progress_callback(idx / total_windows)
        train_slice = df_adv.iloc[train_s:train_e]
        test_slice = df_adv.iloc[test_s:test_e]
        if len(train_slice) < 10 or len(test_slice) < 1:
            logger.warning(f"Janela {idx} ignorada por tamanho insuficiente.")
            continue

        X_train = train_slice[features].values
        y_train = train_slice['target_future_return'].values
        X_test = test_slice[features].values
        y_test = test_slice['target_future_return'].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        trained_models = {}
        preds_for_window = {}
        metrics_for_window = {}

        # hyperparameter tuning com TimeSeriesSplit para modelos baseados em árvore
        tscv = TimeSeriesSplit(n_splits=min(cv_splits, max(2, len(train_slice)//10)))

        for name, model in models_to_train.items():
            try:
                model_to_fit = model
                if tune_models and name in ('Random Forest', 'Gradient Boosting', 'XGBoost') and tscv.get_n_splits() > 1:
                    # grid simples para economizar tempo
                    if name == 'Random Forest':
                        param_grid = {'n_estimators': [100, 200], 'max_depth': [6, 12]}
                        gs = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                        gs.fit(X_train_s, y_train)
                        model_to_fit = gs.best_estimator_
                        logger.info(f"GridSearch best for {name}: {gs.best_params_}")
                    elif name == 'Gradient Boosting':
                        param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 6]}
                        gs = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                        gs.fit(X_train_s, y_train)
                        model_to_fit = gs.best_estimator_
                        logger.info(f"GridSearch best for {name}: {gs.best_params_}")
                    elif name == 'XGBoost' and HAS_XGB:
                        param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 6]}
                        gs = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                        gs.fit(X_train_s, y_train)
                        model_to_fit = gs.best_estimator_
                        logger.info(f"GridSearch best for {name}: {gs.best_params_}")

                model_to_fit.fit(X_train_s, y_train)
                preds = model_to_fit.predict(X_test_s)
                trained_models[name] = {'model': model_to_fit, 'scaler': scaler}
                preds_for_window[name] = preds
                price_base = test_slice['Close'].values
                preds_price = (1 + preds) * price_base
                price_metrics = compute_price_metrics(test_slice['target_future_price'].values[:len(preds_price)], preds_price)
                metrics_for_window[name] = {'price': price_metrics, 'hitrate': compute_return_hitrate(y_test, preds)}
            except Exception as e:
                logger.error(f"Erro treinando {name} na janela {idx}: {e}")
                trained_models[name] = {'model': None, 'scaler': scaler}
                preds_for_window[name] = np.full(len(X_test_s), np.nan)
                metrics_for_window[name] = {'error': str(e)}

        # LSTM opcional
        if HAS_TF:
            try:
                X_train_seq = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
                X_test_seq = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))
                lstm = Sequential([LSTM(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])), Dropout(0.2), Dense(32, activation='relu'), Dense(1)])
                lstm.compile(optimizer='adam', loss='mse')
                lstm.fit(X_train_seq, y_train, epochs=10, batch_size=16, verbose=0)
                preds_lstm = lstm.predict(X_test_seq).flatten()
                trained_models['LSTM'] = {'model': lstm, 'scaler': scaler}
                preds_for_window['LSTM'] = preds_lstm
                metrics_for_window['LSTM'] = {'price': compute_price_metrics(test_slice['target_future_price'].values[:len(preds_lstm)], (1 + preds_lstm) * test_slice['Close'].values[:len(preds_lstm)]), 'hitrate': compute_return_hitrate(y_test, preds_lstm)}
            except Exception as e:
                logger.warning(f"LSTM failure: {e}")
                trained_models['LSTM'] = {'model': None, 'scaler': scaler}
                preds_for_window['LSTM'] = np.full(len(X_test_s), np.nan)
                metrics_for_window['LSTM'] = {'error': str(e)}

        # Ensemble: média dos modelos válidos
        pred_matrix = np.vstack([v for v in preds_for_window.values() if np.array(v).shape[0] == len(X_test_s)])
        with np.errstate(all='ignore'):
            ensemble_preds = np.nanmean(np.where(np.isfinite(pred_matrix), pred_matrix, np.nan), axis=0) if pred_matrix.size else np.full(len(X_test_s), np.nan)

        # Guardar resultados agregados
        dates_window = test_slice.index.to_numpy()
        base_prices_window = test_slice['Close'].values
        aggregated_preds.extend(list(ensemble_preds))
        aggregated_reals.extend(list(test_slice['target_future_return'].values))
        aggregated_dates.extend(list(dates_window))
        aggregated_base_prices.extend(list(base_prices_window))
        ensemble_details.append({'window': idx, 'train_range': (df_adv.index[train_s], df_adv.index[train_e - 1]), 'test_range': (df_adv.index[test_s], df_adv.index[test_e - 1]), 'models_metrics': metrics_for_window})

        trained_models_history.append(trained_models)

    # Converter para arrays
    if len(aggregated_preds) == 0:
        logger.error("Nenhuma previsão gerada pelo walk-forward.")
        return None

    agg_preds = np.array(aggregated_preds, dtype=float)
    agg_reals = np.array(aggregated_reals, dtype=float)
    agg_dates = np.array(aggregated_dates)
    agg_base_prices = np.array(aggregated_base_prices, dtype=float)

    # Simular estratégia com funções profissionais
    sim = simulate_trade_sequence(agg_dates, agg_base_prices, agg_preds, agg_reals,
                                  initial_capital=10000.0,
                                  position_fraction=POSITION_SIZE_FRACTION,
                                  commission_pct=COMMISSION_PER_TRADE,
                                  slippage_pct=SLIPPAGE_PCT,
                                  spread_pct=SPREAD_PCT,
                                  stop_loss_pct=0.03,
                                  take_profit_pct=0.06,
                                  vol_sizing=True,
                                  vol_lookback=VOL_SIZING_LOOKBACK)

    # métricas do ensemble
    price_metrics = compute_price_metrics((1 + agg_reals) * agg_base_prices, (1 + agg_preds) * agg_base_prices)
    hitrate = compute_return_hitrate(agg_reals, agg_preds)

    return {'ensemble_preds': agg_preds, 'ensemble_reals': agg_reals, 'dates': agg_dates, 'base_prices': agg_base_prices,
            'backtest': sim, 'price_metrics': price_metrics, 'hitrate': hitrate, 'windows': ensemble_details, 'trained_history': trained_models_history}

# -----------------------
# Interface Streamlit
# -----------------------
st.title('📊 Assetz - Analisador de Ativos Profissional')
st.write('Protótipo com backtest profissional, walk-forward, controle de risco e tuning temporal.')

st.sidebar.header('⚙️ Parâmetros de Análise')
start_default = date(2018, 1, 1)
start_date = st.sidebar.date_input("Data de Início", start_default, format="DD/MM/YYYY")
end_date = st.sidebar.date_input("Data de Fim", date.today(), format="DD/MM/YYYY")
view_period = st.sidebar.selectbox("Período de visualização", ["Últimos 3 meses", "Últimos 6 meses", "Último 1 ano", "Todo período"], index=2)
view_map = {"Últimos 3 meses": 63, "Últimos 6 meses": 126, "Último 1 ano": 252, "Todo período": None}
viz_days = view_map[view_period]

tickers_df = get_tickers_from_csv()
selected_display = st.sidebar.selectbox('Escolha a Ação', tickers_df['display'])
ticker_symbol = tickers_df.loc[tickers_df['display'] == selected_display, 'ticker'].iloc[0]
company_name = tickers_df.loc[tickers_df['display'] == selected_display, 'nome'].iloc[0]
ticker = f"{ticker_symbol}.SA"

# Opções avançadas
st.sidebar.header("⚙️ Opções Avançadas de Backtest")
commission_input = st.sidebar.number_input("Comissão por trade (pct)", min_value=0.0, max_value=0.05, value=COMMISSION_PER_TRADE, step=0.0001, format="%.4f")
slippage_input = st.sidebar.number_input("Slippage média (pct)", min_value=0.0, max_value=0.05, value=SLIPPAGE_PCT, step=0.0001, format="%.4f")
spread_input = st.sidebar.number_input("Spread estimado (pct)", min_value=0.0, max_value=0.05, value=SPREAD_PCT, step=0.0001, format="%.4f")
pos_frac_input = st.sidebar.number_input("Fração do capital por posição (pct)", min_value=0.001, max_value=1.0, value=POSITION_SIZE_FRACTION, step=0.001, format="%.3f")
stop_loss_input = st.sidebar.number_input("Stop-loss por trade (pct)", min_value=0.0, max_value=0.5, value=0.03, step=0.005, format="%.3f")
take_profit_input = st.sidebar.number_input("Take-profit por trade (pct)", min_value=0.0, max_value=1.0, value=0.06, step=0.005, format="%.3f")

# Atualizar variáveis globais com inputs do usuário
COMMISSION_PER_TRADE = commission_input
SLIPPAGE_PCT = slippage_input
SPREAD_PCT = spread_input
POSITION_SIZE_FRACTION = pos_frac_input

# Carregar dados
data = load_data(ticker, start_date, end_date)
ibov = load_data('^BVSP', start_date, end_date)

if data.empty or len(data) < 1:
    st.error("❌ Não foi possível baixar dados para este ticker no período solicitado.")
    st.stop()

data = calculate_indicators(data)

# Cabeçalho com métricas rápidas
ticker_obj = yf.Ticker(ticker)
try:
    last_price_info = ticker_obj.fast_info.get('lastPrice')
except Exception:
    last_price_info = None

try:
    live_data = ticker_obj.history(period='1d', interval='1m')
    if live_data.empty:
        live_data = ticker_obj.history(period='1d', interval='5m')
except Exception:
    live_data = pd.DataFrame()

if last_price_info and np.isfinite(last_price_info) and last_price_info > 0:
    last_price = last_price_info
elif not live_data.empty:
    last_price = live_data['Close'].iloc[-1]
else:
    last_price = data['Close'].iloc[-1]

prev_price = data['Close'].iloc[-2] if len(data) >= 2 else last_price
price_change = last_price - prev_price
percent_change = (price_change / prev_price * 100) if prev_price != 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("🏢 Empresa", company_name)
c2.metric("💹 Ticker", ticker_symbol)
c3.metric("💰 Último Preço", f"R$ {last_price:.2f}")
c4.metric("📊 Variação (vs. Fech. Anterior)", f"{price_change:+.2f} R$", f"{percent_change:+.2f}%")

# Gráficos
st.subheader('📈 Visão Geral do Ativo')
view_slice = slice(-viz_days, None) if viz_days is not None else slice(None)

with st.expander("Gráficos de Preço e Indicadores"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index[view_slice], y=data['BB_Superior'][view_slice], line=dict(width=0), showlegend=False, name='Banda Superior'))
    fig.add_trace(go.Scatter(x=data.index[view_slice], y=data['BB_Inferior'][view_slice], fill='tonexty', fillcolor='rgba(0, 176, 246, 0.2)', line=dict(width=0), name='Bandas de Bollinger', showlegend=True))
    fig.add_trace(go.Scatter(x=data.index[view_slice], y=data['MM_Longa'][view_slice], name='Média Móvel 50 Dias', line=dict(color='purple', dash='dash')))
    fig.add_trace(go.Scatter(x=data.index[view_slice], y=data['MM_Curta'][view_slice], name='Média Móvel 20 Dias', line=dict(color='yellow', width=1.5)))
    fig.add_trace(go.Scatter(x=data.index[view_slice], y=data['Close'][view_slice], name='Preço de Fechamento', line=dict(color='cyan', width=2)))
    fig.update_layout(height=450, xaxis=dict(title="Data"), yaxis=dict(title="Preço (R$)"))
    st.plotly_chart(fig, use_container_width=True)

# Seção avançada
st.subheader('🔮 Previsão Avançada, Walk-Forward e Backtest Profissional')
st.write(f"Requer no mínimo {MIN_DAYS_ADVANCED} dias de histórico para análises mais robustas. Modelos: Random Forest, Gradient Boosting, SVR, Neural Net, XGBoost (se instalado) e LSTM (se instalado).")

if st.button('Executar Análise Walk-Forward e Backtest'):
    with st.spinner("Preparando features..."):
        adv_df, used_features = prepare_advanced_features(data, forecast_days=FORECAST_DAYS)
    if len(adv_df) < MIN_DAYS_ADVANCED:
        st.warning(f"Dados insuficientes para análise. Linhas válidas: {len(adv_df)}. Mínimo: {MIN_DAYS_ADVANCED}.")
    else:
        st.info(f"Utilizando {len(adv_df)} dias de dados para a análise walk-forward.")
        progress_bar = st.progress(0.0)
        def prog(p): progress_bar.progress(min(1.0, p))
        models_base = create_base_models()
        try:
            bt_result = walk_forward_backtest(adv_df, used_features, initial_capital=10000.0,
                                              train_window=WALKFWD_TRAIN_WINDOW, test_window=WALKFWD_TEST_WINDOW,
                                              expand_train=WALKFWD_EXPAND, models_to_train=models_base,
                                              tune_models=True, cv_splits=TS_CV_SPLITS, progress_callback=prog)
            progress_bar.progress(1.0)
        except Exception as e:
            logger.error(f"Erro no walk-forward completo: {e}")
            st.error(f"Erro ao executar walk-forward: {e}")
            bt_result = None

        if bt_result:
            # Exibir métricas
            price_metrics = bt_result['price_metrics']
            hitrate = bt_result['hitrate']
            st.subheader("Métricas Agregadas do Ensemble")
            st.write(f"MAE (R$): {price_metrics.get('MAE'):.4f} | RMSE (R$): {price_metrics.get('RMSE'):.4f} | MAPE: {price_metrics.get('MAPE'):.4%}")
            st.write(f"Hit Rate (sinal de retorno): {hitrate:.4%}")

            # Equity curve
            eq = bt_result['backtest']['equity_curve']
            fig_eq = px.line(eq, x='Date', y='Equity', title='Equity Curve da Estratégia')
            st.plotly_chart(fig_eq, use_container_width=True)

            # Trades summary
            trades = bt_result['backtest']['trades']
            if not trades.empty:
                st.subheader("Resumo de Trades")
                st.dataframe(trades.sort_values('date', ascending=False).reset_index(drop=True), use_container_width=True)
                total_pnl = trades['pnl'].sum() if 'pnl' in trades.columns else None
                st.write(f"Resultado total de P&L (soma das trades): R$ {total_pnl:,.2f}" if total_pnl is not None else "Sem P&L calculado nas trades.")

            # Simulação comparativa com Buy & Hold
            try:
                plot_df = pd.DataFrame({'Date': bt_result['dates'],
                                        'Estratégia': bt_result['backtest']['equity_curve']['Equity'].values,
                                        'BasePrice': bt_result['base_prices']})
                # construir buy & hold capital
                returns_real = bt_result['ensemble_reals']
                capital_initial = 10000.0
                bh_curve = (1 + returns_real).cumprod() * capital_initial
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(x=bt_result['dates'], y=bt_result['backtest']['equity_curve']['Equity'].values, name='Estratégia'))
                fig_comp.add_trace(go.Scatter(x=bt_result['dates'], y=bh_curve, name='Buy & Hold', line=dict(dash='dash')))
                fig_comp.update_layout(title_text='Comparação: Estratégia vs Buy & Hold', xaxis_title='Data', yaxis_title='Capital (R$)')
                st.plotly_chart(fig_comp, use_container_width=True)
            except Exception as e:
                logger.warning(f"Erro ao plotar comparação: {e}")

            # Exportar resultados com metadata
            metadata = {
                'app_version': APP_VERSION,
                'timestamp': pd.Timestamp.now(tz='America/Sao_Paulo').isoformat(),
                'ticker': ticker_symbol,
                'parameters': {
                    'commission_pct': COMMISSION_PER_TRADE,
                    'slippage_pct': SLIPPAGE_PCT,
                    'spread_pct': SPREAD_PCT,
                    'position_fraction': POSITION_SIZE_FRACTION,
                    'stop_loss_pct': stop_loss_input,
                    'take_profit_pct': take_profit_input,
                    'walkfwd_train_window': WALKFWD_TRAIN_WINDOW,
                    'walkfwd_test_window': WALKFWD_TEST_WINDOW
                },
                'price_metrics': price_metrics,
                'hitrate': hitrate,
                'hash_run': sha256_of_text(json.dumps({'timestamp': pd.Timestamp.now().isoformat(), 'ticker': ticker_symbol, 'version': APP_VERSION}))
            }
            # preparar arquivos para download
            mem = io.BytesIO()
            with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                # equity
                buf = io.BytesIO()
                bt_result['backtest']['equity_curve'].to_csv(buf, index=False)
                zf.writestr('equity_curve.csv', buf.getvalue())
                # trades
                buf2 = io.BytesIO()
                bt_result['backtest']['trades'].to_csv(buf2, index=False)
                zf.writestr('trades.csv', buf2.getvalue())
                # preds
                preds_df = pd.DataFrame({'Date': pd.to_datetime(bt_result['dates']).strftime('%Y-%m-%d'), 'PredReturn': bt_result['ensemble_preds'], 'RealReturn': bt_result['ensemble_reals'], 'BasePrice': bt_result['base_prices']})
                zf.writestr('predictions.csv', preds_df.to_csv(index=False))
                zf.writestr('metadata.json', json.dumps(metadata, indent=4, default=str))
            mem.seek(0)
            st.download_button(label="Baixar Resultados da Análise (ZIP)", data=mem,
                               file_name=f"analise_{ticker_symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.zip", mime="application/zip")
            logger.info("Análise finalizada e disponível para download.")
        else:
            st.error("Não foi possível gerar resultados do backtest.")

# Importar análises exportadas
st.markdown("---")
st.subheader("📂 Importar e Comparar Previsões Exportadas")
uploaded = st.file_uploader("Carregar ZIP de análise exportada por esta ferramenta", type=["zip"])
if uploaded is not None:
    try:
        z = zipfile.ZipFile(io.BytesIO(uploaded.read()))
        if 'metadata.json' in z.namelist() and 'predictions.csv' in z.namelist():
            meta = json.loads(z.read('metadata.json'))
            preds = pd.read_csv(io.BytesIO(z.read('predictions.csv')))
            st.write(f"Análise importada para o ticker **{meta.get('ticker')}** de **{meta.get('timestamp')}**.")
            # Exibir previsões importadas
            st.dataframe(preds, use_container_width=True)
            if st.button("Comparar Previsão Importada com Preços Reais"):
                ticker_to_check = f"{meta.get('ticker')}.SA"
                dates_to_check = pd.to_datetime(preds['Date'], dayfirst=False)
                start_check, end_check = dates_to_check.min() - BDay(5), dates_to_check.max() + BDay(5)
                actual_data = safe_yf_download(ticker_to_check, start_check.strftime('%Y-%m-%d'), end_check.strftime('%Y-%m-%d'))
                if not actual_data.empty:
                    actual_data.index = pd.to_datetime(actual_data.index).normalize()
                    preds['Date'] = pd.to_datetime(preds['Date']).dt.normalize()
                    merged_df = pd.merge(preds, actual_data[['Close']], left_on='Date', right_index=True, how='left').rename(columns={'Close': 'Preço Real'})
                    merged_df['Erro (R$)'] = merged_df['Preço Real'] - merged_df['Preço Previsto']
                    merged_df['Erro (%)'] = (merged_df['Erro (R$)'] / merged_df['Preço Previsto'])
                    st.dataframe(merged_df, use_container_width=True)
                else:
                    st.warning("Não foi possível baixar os dados reais para comparação.")
        else:
            st.error("ZIP inválido. Arquivos 'metadata.json' ou 'predictions.csv' não encontrados.")
    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo ZIP: {e}")

# Rodapé e informações do log
st.markdown("---")
horario_consulta = pd.Timestamp.now(tz='America/Sao_Paulo').strftime('%d/%m/%Y %H:%M:%S')
st.caption(f"Última consulta dos dados: **{horario_consulta}** — Dados: Yahoo Finance.")
st.markdown(f"<p style='text-align:center;color:#888'>Desenvolvido por Rodrigo Costa de Araujo | Versão do App: {APP_VERSION}</p>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;color:#888'>Logs gravados em: {LOG_FILE}</p>", unsafe_allow_html=True)
logger.info("Execução da interface concluída.")
