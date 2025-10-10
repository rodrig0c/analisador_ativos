# app.py
# Versão final: backtest profissional, custos 0.1%, stop-loss -2%, take-profit +4%
# Validação temporal: estática 80/20, rolling window, expanding window
# Inclui: Plotly dark theme, gráficos restaurados, logs de predição, resumo por janela,
# sensibilidade do sinal (threshold) e exportação de relatórios.
# Autor: atualização automática solicitada por Rodrigo

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, datetime, timedelta
from pandas.tseries.offsets import BDay
import plotly.express as px
import plotly.graph_objects as go
import io, zipfile, json, logging, os, hashlib, warnings
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
APP_VERSION = "2025-10-05_final_all_methods"
LOG_FOLDER = "logs"
os.makedirs(LOG_FOLDER, exist_ok=True)
LOG_FILE = os.path.join(LOG_FOLDER, f"app_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Defaults de mercado / backtest
DEFAULT_COMMISSION_PCT = 0.001    # 0.1%
DEFAULT_SLIPPAGE_PCT = 0.001     # 0.1%
DEFAULT_SPREAD_PCT = 0.0005      # 0.05%
DEFAULT_STOP_LOSS = 0.02         # 2%
DEFAULT_TAKE_PROFIT = 0.04       # 4%
DEFAULT_POSITION_FRACTION = 0.05 # 5% do capital
VOL_SIZING_LOOKBACK = 30
MAX_POSITION_LEVERAGE = 1.0

MIN_DAYS_CHARTS = 60
MIN_DAYS_ADVANCED = 180
FORECAST_DAYS = 5

# Walk-forward params
ROLLING_TRAIN_WINDOW = 252 * 2
ROLLING_TEST_WINDOW = 63

# Optional libs
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
logger = logging.getLogger("assetz_prof_full")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    fh = logging.FileHandler(LOG_FILE)
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

def sha256_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# -----------------------
# Data loaders & indicadores
# -----------------------
@st.cache_data
def get_tickers_from_csv():
    try:
        df = pd.read_csv('acoes-listadas-b3.csv')
        df = df.rename(columns={c: c.strip() for c in df.columns})
        poss_t = [c for c in df.columns if c.lower() in ('ticker','codigo','sigla')]
        poss_n = [c for c in df.columns if c.lower() in ('nome','empresa','name')]
        tcol = poss_t[0] if poss_t else df.columns[0]
        ncol = poss_n[0] if poss_n else (df.columns[1] if df.shape[1]>1 else tcol)
        df = df.rename(columns={tcol:'ticker', ncol:'nome'})
        df['ticker'] = df['ticker'].astype(str).str.replace('.SA','',case=False).str.strip()
        df['display'] = df['nome'].astype(str) + ' (' + df['ticker'] + ')'
        return df[['ticker','nome','display']]
    except Exception as e:
        logger.warning(f"Erro ao carregar CSV: {e}. Usando fallback.")
        fallback = {'ticker':['PETR4','VALE3','ITUB4','MGLU3'],'nome':['Petrobras','Vale','Itaú','Magazine Luiza']}
        df = pd.DataFrame(fallback); df['display'] = df['nome']+' ('+df['ticker']+')'
        return df[['ticker','nome','display']]

def safe_yf_download(ticker: str, start: str, end: str) -> pd.DataFrame:
    logger.info(f"Baixando {ticker} de {start} até {end}")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        logger.warning(f"Dados vazios para {ticker}")
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for c in ['Open','High','Low','Close','Volume','Adj Close']:
        if c not in df.columns:
            df[c] = np.nan
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep='first')].sort_index()
    return df

@st.cache_data
def load_data(ticker, start, end):
    s = pd.to_datetime(start); e = pd.to_datetime(end) + pd.Timedelta(days=1)
    return safe_yf_download(ticker, s.strftime('%Y-%m-%d'), e.strftime('%Y-%m-%d'))

def calculate_indicators(df):
    df = df.copy()
    df['Close'] = df['Close'].astype(float)
    delta = df['Close'].diff()
    gain = delta.where(delta>0,0.0)
    loss = -delta.where(delta<0,0.0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MM_Curta'] = df['Close'].rolling(20, min_periods=1).mean()
    df['MM_Longa'] = df['Close'].rolling(50, min_periods=1).mean()
    df['BB_Media'] = df['Close'].rolling(20, min_periods=1).mean()
    df['BB_Superior'] = df['BB_Media'] + 2 * df['Close'].rolling(20, min_periods=1).std()
    df['BB_Inferior'] = df['BB_Media'] - 2 * df['Close'].rolling(20, min_periods=1).std()
    df['Daily Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily Return'].rolling(30, min_periods=1).std() * np.sqrt(252)
    # ATR (alternative volatility) for informational use
    df['TR1'] = (df['High'] - df['Low']).abs()
    df['TR2'] = (df['High'] - df['Close'].shift(1)).abs()
    df['TR3'] = (df['Low'] - df['Close'].shift(1)).abs()
    df['TR'] = df[['TR1','TR2','TR3']].max(axis=1)
    df['ATR_14'] = df['TR'].rolling(14, min_periods=1).mean()
    return df

def prepare_advanced_features(df, forecast_days=FORECAST_DAYS):
    d = df[['Close','Volume','RSI','MM_Curta','MM_Longa','Volatility','ATR_14']].copy()
    periods = [1,3,5,10,20]
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
    potential = [c for c in d.columns if c.startswith(('return_','volume_ma_','volatility_','price_vs_'))]
    potential.extend(['RSI','Volatility','ATR_14'])
    features = [c for c in potential if c in d.columns and not d[c].isnull().all()]
    required = features + ['target_future_return','target_future_price']
    d.dropna(subset=required, inplace=True)
    return d, features

def create_base_models():
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'Neural Net': MLPRegressor(hidden_layer_sizes=(64,32), max_iter=1000, random_state=42)
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
    if mask.sum() == 0: return None
    return float(np.mean((np.sign(y_t[mask]) == np.sign(y_p[mask])).astype(float)))

# -----------------------
# Simulação de trades (backtest) - função reutilizável
# -----------------------
def simulate_trade_sequence(dates, base_prices, pred_returns, real_returns,
                            initial_capital=10000.0,
                            position_fraction=DEFAULT_POSITION_FRACTION,
                            commission_pct=DEFAULT_COMMISSION_PCT,
                            slippage_pct=DEFAULT_SLIPPAGE_PCT,
                            spread_pct=DEFAULT_SPREAD_PCT,
                            stop_loss_pct=DEFAULT_STOP_LOSS,
                            take_profit_pct=DEFAULT_TAKE_PROFIT,
                            vol_sizing=False,
                            vol_lookback=VOL_SIZING_LOOKBACK,
                            signal_threshold=0.0):
    n = len(dates)
    capital = initial_capital
    cash = capital
    position = None
    logs = []
    equity_curve = np.full(n, np.nan)

    for i in range(n):
        price = float(base_prices[i])
        pred_ret = float(pred_returns[i]) if np.isfinite(pred_returns[i]) else 0.0
        real_ret = float(real_returns[i]) if np.isfinite(real_returns[i]) else 0.0
        date_i = pd.to_datetime(dates[i])

        # atualizar mtm
        if position is None:
            equity_curve[i] = cash
        else:
            mtm = position['size_shares'] * price
            equity_curve[i] = cash + mtm

        # verificar stop / take se posição aberta
        if position is not None:
            entry_price = position['entry_price']
            effective_price = price * (1 - spread_pct)
            current_ret = (effective_price / entry_price) - 1.0
            if current_ret <= -abs(stop_loss_pct):
                exit_price = price * (1 - spread_pct) * (1 - slippage_pct)
                proceeds = position['size_shares'] * exit_price
                commission = proceeds * commission_pct
                cash += proceeds - commission
                pnl = proceeds - commission - position['notional']
                logs.append({'date': date_i, 'action': 'SELL', 'price': exit_price, 'shares': position['size_shares'], 'notional_in': position['notional'], 'proceeds': proceeds, 'commission': commission, 'pnl': pnl, 'reason': 'stop_loss'})
                position = None
            elif current_ret >= abs(take_profit_pct):
                exit_price = price * (1 - spread_pct) * (1 - slippage_pct)
                proceeds = position['size_shares'] * exit_price
                commission = proceeds * commission_pct
                cash += proceeds - commission
                pnl = proceeds - commission - position['notional']
                logs.append({'date': date_i, 'action': 'SELL', 'price': exit_price, 'shares': position['size_shares'], 'notional_in': position['notional'], 'proceeds': proceeds, 'commission': commission, 'pnl': pnl, 'reason': 'take_profit'})
                position = None

        # regra de entrada: sem posição e sinal acima do threshold
        if position is None and pred_ret > signal_threshold:
            if vol_sizing and i >= vol_lookback:
                recent_vol = np.nanstd(real_returns[max(0, i - vol_lookback):i]) * np.sqrt(252)
                vol_factor = 1.0 / (recent_vol + 1e-6)
                notional = max(0.0, min(capital * position_fraction * vol_factor, capital * MAX_POSITION_LEVERAGE))
            else:
                notional = capital * position_fraction
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

        # fim do período: fechar posições abertas
        if i == n - 1 and position is not None:
            exit_price = price * (1 - spread_pct) * (1 - slippage_pct)
            proceeds = position['size_shares'] * exit_price
            commission = proceeds * commission_pct
            cash += proceeds - commission
            pnl = proceeds - commission - position['notional']
            logs.append({'date': date_i, 'action': 'SELL_END', 'price': exit_price, 'shares': position['size_shares'], 'notional_in': position['notional'], 'proceeds': proceeds, 'commission': commission, 'pnl': pnl, 'reason': 'end_period'})
            position = None
            equity_curve[i] = cash

        if np.isnan(equity_curve[i]):
            equity_curve[i] = cash if position is None else cash + position['size_shares'] * price

    trades_df = pd.DataFrame(logs)
    eq = pd.DataFrame({'Date': pd.to_datetime(dates), 'Equity': equity_curve})
    return {'equity_curve': eq, 'trades': trades_df, 'final_capital': cash, 'initial_capital': initial_capital}

# -----------------------
# Validação temporal: 3 métodos
# - static split 80/20
# - rolling window
# - expanding (growing) window
# Todos retornam previsões alinhadas com datas
# -----------------------
def validate_and_backtest(df_adv, features, method='rolling', static_split_ratio=0.8,
                          rolling_train=ROLLING_TRAIN_WINDOW, rolling_test=ROLLING_TEST_WINDOW,
                          initial_capital=10000.0, tune_models=True, signal_threshold=0.0):
    models = create_base_models()
    n = len(df_adv)
    results = {}

    if method == 'static':
        split = int(n * static_split_ratio)
        train = df_adv.iloc[:split]; test = df_adv.iloc[split:]
        if len(test) < 1:
            raise ValueError("Período de teste insuficiente para método estático.")
        X_train = train[features].values; y_train = train['target_future_return'].values
        X_test = test[features].values; y_test = test['target_future_return'].values
        scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
        trained = {}
        preds_for_models = {}
        for name, model in models.items():
            try:
                model.fit(X_train_s, y_train)
                preds = model.predict(X_test_s)
                preds_for_models[name] = preds
                trained[name] = model
            except Exception as e:
                logger.error(f"Erro treinando {name} em estático: {e}")
                preds_for_models[name] = np.full(len(X_test_s), np.nan)
                trained[name] = None
        # ensemble média
        pred_matrix = np.vstack([v for v in preds_for_models.values() if np.array(v).shape[0] == len(X_test_s)]) if len(preds_for_models)>0 else np.array([])
        ensemble_preds = np.nanmean(np.where(np.isfinite(pred_matrix), pred_matrix, np.nan), axis=0) if pred_matrix.size else np.full(len(X_test_s), np.nan)
        dates = test.index.to_numpy(); base_prices = test['Close'].values; reals = test['target_future_return'].values
        # clip returns to avoid outliers distortion
        reals_clipped = np.clip(reals, -0.5, 0.5)
        sim = simulate_trade_sequence(dates, base_prices, ensemble_preds, reals_clipped, initial_capital=initial_capital,
                                      position_fraction=DEFAULT_POSITION_FRACTION, commission_pct=DEFAULT_COMMISSION_PCT,
                                      slippage_pct=DEFAULT_SLIPPAGE_PCT, spread_pct=DEFAULT_SPREAD_PCT,
                                      stop_loss_pct=DEFAULT_STOP_LOSS, take_profit_pct=DEFAULT_TAKE_PROFIT, vol_sizing=True,
                                      signal_threshold=signal_threshold)
        results = {'method': 'static', 'dates': dates, 'preds': ensemble_preds, 'reals': reals_clipped, 'base_prices': base_prices, 'sim': sim, 'trained': trained}

    elif method == 'rolling':
        windows = []
        start = 0
        while True:
            train_s = start
            train_e = train_s + rolling_train
            test_s = train_e
            test_e = test_s + rolling_test
            if test_s >= n:
                break
            if test_e > n:
                test_e = n
            windows.append((train_s, train_e, test_s, test_e))
            start = start + rolling_test
            if test_e == n:
                break
        logger.info(f"Rolling windows: {len(windows)}")
        agg_preds = []; agg_reals = []; agg_dates = []; agg_base_prices = []
        model_summary = []
        for idx, (ts, te, ss, se) in enumerate(windows):
            train = df_adv.iloc[ts:te]; test = df_adv.iloc[ss:se]
            if len(train) < 10 or len(test) < 1:
                continue
            X_train = train[features].values; y_train = train['target_future_return'].values
            X_test = test[features].values; y_test = test['target_future_return'].values
            scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
            preds_for_window = {}
            metrics_window = {}
            tscv = TimeSeriesSplit(n_splits=min(4, max(2, len(train)//10)))
            for name, model in models.items():
                try:
                    model_to_fit = model
                    # simple tuning for tree models
                    if name in ('Random Forest','Gradient Boosting') and tscv.get_n_splits()>1 and tune_models:
                        if name == 'Random Forest':
                            param_grid = {'n_estimators':[100,200],'max_depth':[6,12]}
                        else:
                            param_grid = {'n_estimators':[100,200],'max_depth':[3,6]}
                        gs = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                        gs.fit(X_train_s, y_train)
                        model_to_fit = gs.best_estimator_
                        logger.info(f"Window {idx}: best {name} params {gs.best_params_}")
                    model_to_fit.fit(X_train_s, y_train)
                    preds = model_to_fit.predict(X_test_s)
                    preds_for_window[name] = preds
                    metrics_window[name] = {'hitrate': compute_return_hitrate(y_test, preds)}
                except Exception as e:
                    logger.error(f"Window {idx}: erro treinando {name}: {e}")
                    preds_for_window[name] = np.full(len(X_test_s), np.nan)
                    metrics_window[name] = {'error': str(e)}
            # optionally LSTM omitted for speed
            pred_matrix = np.vstack([v for v in preds_for_window.values() if np.array(v).shape[0] == len(X_test_s)]) if len(preds_for_window)>0 else np.array([])
            ensemble_preds = np.nanmean(np.where(np.isfinite(pred_matrix), pred_matrix, np.nan), axis=0) if pred_matrix.size else np.full(len(X_test_s), np.nan)
            agg_preds.extend(list(ensemble_preds))
            agg_reals.extend(list(y_test))
            agg_dates.extend(list(test.index.to_numpy()))
            agg_base_prices.extend(list(test['Close'].values))
            model_summary.append({'window': idx, 'train_range': (df_adv.index[ts], df_adv.index[te-1]), 'test_range': (df_adv.index[ss], df_adv.index[se-1]), 'metrics': metrics_window, 'n_test': len(test)})
        if len(agg_preds) == 0:
            raise ValueError("Nenhuma previsão no método rolling.")
        agg_preds = np.array(agg_preds); agg_reals = np.array(agg_reals); agg_dates = np.array(agg_dates); agg_base_prices = np.array(agg_base_prices)
        agg_reals_clipped = np.clip(agg_reals, -0.5, 0.5)
        sim = simulate_trade_sequence(agg_dates, agg_base_prices, agg_preds, agg_reals_clipped, initial_capital=initial_capital,
                                      position_fraction=DEFAULT_POSITION_FRACTION, commission_pct=DEFAULT_COMMISSION_PCT,
                                      slippage_pct=DEFAULT_SLIPPAGE_PCT, spread_pct=DEFAULT_SPREAD_PCT,
                                      stop_loss_pct=DEFAULT_STOP_LOSS, take_profit_pct=DEFAULT_TAKE_PROFIT, vol_sizing=True,
                                      signal_threshold=signal_threshold)
        results = {'method': 'rolling', 'dates': agg_dates, 'preds': agg_preds, 'reals': agg_reals_clipped, 'base_prices': agg_base_prices, 'sim': sim, 'summary': model_summary}

    elif method == 'expanding':
        # expanding window: start with minimal size then expand adding next chunk as test
        min_train = max(50, int(n*0.2))
        step = ROLLING_TEST_WINDOW
        start_train = min_train
        agg_preds = []; agg_reals = []; agg_dates = []; agg_base_prices = []; model_summary = []
        train_end = start_train
        while train_end < n - 1:
            test_start = train_end
            test_end = min(n, test_start + step)
            train = df_adv.iloc[:train_end]; test = df_adv.iloc[test_start:test_end]
            if len(test) < 1:
                break
            X_train = train[features].values; y_train = train['target_future_return'].values
            X_test = test[features].values; y_test = test['target_future_return'].values
            scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
            preds_for_window = {}; metrics_window = {}
            tscv = TimeSeriesSplit(n_splits=min(4, max(2, len(train)//10)))
            for name, model in models.items():
                try:
                    model_to_fit = model
                    if name in ('Random Forest','Gradient Boosting') and tscv.get_n_splits()>1:
                        if name == 'Random Forest':
                            param_grid = {'n_estimators':[100,200],'max_depth':[6,12]}
                        else:
                            param_grid = {'n_estimators':[100,200],'max_depth':[3,6]}
                        gs = GridSearchCV(model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1)
                        gs.fit(X_train_s, y_train)
                        model_to_fit = gs.best_estimator_
                    model_to_fit.fit(X_train_s, y_train)
                    preds = model_to_fit.predict(X_test_s)
                    preds_for_window[name] = preds
                    metrics_window[name] = {'hitrate': compute_return_hitrate(y_test, preds)}
                except Exception as e:
                    logger.error(f"Expanding: erro treinando {name}: {e}")
                    preds_for_window[name] = np.full(len(X_test_s), np.nan)
                    metrics_window[name] = {'error': str(e)}
            pred_matrix = np.vstack([v for v in preds_for_window.values() if np.array(v).shape[0] == len(X_test_s)]) if len(preds_for_window)>0 else np.array([])
            ensemble_preds = np.nanmean(np.where(np.isfinite(pred_matrix), pred_matrix, np.nan), axis=0) if pred_matrix.size else np.full(len(X_test_s), np.nan)
            agg_preds.extend(list(ensemble_preds)); agg_reals.extend(list(y_test)); agg_dates.extend(list(test.index.to_numpy())); agg_base_prices.extend(list(test['Close'].values))
            model_summary.append({'train_end': df_adv.index[train_end-1], 'test_range':(df_adv.index[test_start], df_adv.index[test_end-1]), 'metrics': metrics_window, 'n_test': len(test)})
            train_end = train_end + step
        if len(agg_preds) == 0:
            raise ValueError("Nenhuma previsão no método expanding.")
        agg_preds = np.array(agg_preds); agg_reals = np.array(agg_reals); agg_dates = np.array(agg_dates); agg_base_prices = np.array(agg_base_prices)
        agg_reals_clipped = np.clip(agg_reals, -0.5, 0.5)
        sim = simulate_trade_sequence(agg_dates, agg_base_prices, agg_preds, agg_reals_clipped, initial_capital=initial_capital,
                                      position_fraction=DEFAULT_POSITION_FRACTION, commission_pct=DEFAULT_COMMISSION_PCT,
                                      slippage_pct=DEFAULT_SLIPPAGE_PCT, spread_pct=DEFAULT_SPREAD_PCT,
                                      stop_loss_pct=DEFAULT_STOP_LOSS, take_profit_pct=DEFAULT_TAKE_PROFIT, vol_sizing=True,
                                      signal_threshold=signal_threshold)
        results = {'method': 'expanding', 'dates': agg_dates, 'preds': agg_preds, 'reals': agg_reals_clipped, 'base_prices': agg_base_prices, 'sim': sim, 'summary': model_summary}

    else:
        raise ValueError("Método inválido. Escolha 'static', 'rolling' ou 'expanding'.")

    return results

# -----------------------
# Interface Streamlit
# -----------------------
st.title('📊 Assetz - Analisador de Ativos Profissional')
st.write('Versão com validações estática/rolling/expanding, logs de predição e gráficos.')

# Sidebar inputs
st.sidebar.header('Parâmetros de Dados e Backtest')
start_default = date(2019,1,1)
start_date = st.sidebar.date_input("Data de Início", start_default, format="DD/MM/YYYY")
end_date = st.sidebar.date_input("Data de Fim", date.today(), format="DD/MM/YYYY")
view_period = st.sidebar.selectbox("Período de visualização", ["Últimos 3 meses","Últimos 6 meses","Último 1 ano","Todo período"], index=2)
view_map = {"Últimos 3 meses":63,"Últimos 6 meses":126,"Último 1 ano":252,"Todo período":None}
viz_days = view_map[view_period]

tickers_df = get_tickers_from_csv()
selected_display = st.sidebar.selectbox('Escolha a Ação', tickers_df['display'])
ticker_symbol = tickers_df.loc[tickers_df['display']==selected_display,'ticker'].iloc[0]
company_name = tickers_df.loc[tickers_df['display']==selected_display,'nome'].iloc[0]
ticker = f"{ticker_symbol}.SA"

st.sidebar.header('Parâmetros do Backtest e Validação')
method = st.sidebar.selectbox("Método de validação temporal", ['rolling','expanding','static'], index=0)
commission_input = st.sidebar.number_input("Comissão por trade (pct)", min_value=0.0, max_value=0.05, value=DEFAULT_COMMISSION_PCT, step=0.0001, format="%.4f")
slippage_input = st.sidebar.number_input("Slippage média (pct)", min_value=0.0, max_value=0.05, value=DEFAULT_SLIPPAGE_PCT, step=0.0001, format="%.4f")
spread_input = st.sidebar.number_input("Spread estimado (pct)", min_value=0.0, max_value=0.05, value=DEFAULT_SPREAD_PCT, step=0.0001, format="%.4f")
pos_frac_input = st.sidebar.number_input("Fração do capital por posição (pct)", min_value=0.001, max_value=1.0, value=DEFAULT_POSITION_FRACTION, step=0.001, format="%.3f")
signal_threshold = st.sidebar.slider("Sensibilidade do sinal (threshold) — entrada se pred > threshold", min_value=-0.05, max_value=0.2, value=0.0, step=0.001)
stop_loss_input = st.sidebar.number_input("Stop-loss por trade (pct)", min_value=0.0, max_value=0.5, value=DEFAULT_STOP_LOSS, step=0.001, format="%.3f")
take_profit_input = st.sidebar.number_input("Take-profit por trade (pct)", min_value=0.0, max_value=1.0, value=DEFAULT_TAKE_PROFIT, step=0.001, format="%.3f")
run_tuning = st.sidebar.checkbox("Ativar hyperparameter tuning (mais lento)", value=True)

# Aplicar inputs nas variáveis
DEFAULT_COMMISSION_PCT = commission_input
DEFAULT_SLIPPAGE_PCT = slippage_input
DEFAULT_SPREAD_PCT = spread_input
DEFAULT_POSITION_FRACTION = pos_frac_input

# Load data
data = load_data(ticker, start_date, end_date)
ibov = load_data('^BVSP', start_date, end_date)
if data.empty or len(data) < 2:
    st.error("Não foi possível baixar dados para este ticker no período solicitado.")
    st.stop()
data = calculate_indicators(data)

# Header metrics
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
if last_price_info and np.isfinite(last_price_info) and last_price_info>0:
    last_price = last_price_info
elif not live_data.empty:
    last_price = live_data['Close'].iloc[-1]
else:
    last_price = data['Close'].iloc[-1]
prev_price = data['Close'].iloc[-2] if len(data) >= 2 else last_price
price_change = last_price - prev_price
percent_change = (price_change / prev_price * 100) if prev_price != 0 else 0.0
c1,c2,c3,c4 = st.columns(4)
c1.metric("🏢 Empresa", company_name)
c2.metric("💹 Ticker", ticker_symbol)
c3.metric("💰 Último Preço", f"R$ {last_price:.2f}")
c4.metric("📊 Variação (vs. Fech. Anterior)", f"{price_change:+.2f} R$", f"{percent_change:+.2f}%")

st.markdown("---")

# Restore tabs
tab1, tab2, tab3, tab4 = st.tabs(["Preço e Indicadores","Volatilidade","Comparativo com IBOVESPA","Oscilação Intradiária Último Dia"])
view_slice = slice(-viz_days, None) if viz_days is not None else slice(None)

with tab1:
    st.subheader('Preço, Médias Móveis e Bandas de Bollinger')
    if len(data) < MIN_DAYS_CHARTS:
        st.warning(f"Dados insuficientes para gráficos históricos (mínimo {MIN_DAYS_CHARTS} dias).")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index[view_slice], y=data['BB_Superior'][view_slice], line=dict(width=0), showlegend=False, name='Banda Superior'))
        fig.add_trace(go.Scatter(x=data.index[view_slice], y=data['BB_Inferior'][view_slice], fill='tonexty', fillcolor='rgba(0,176,246,0.12)', line=dict(width=0), name='Bandas de Bollinger'))
        fig.add_trace(go.Scatter(x=data.index[view_slice], y=data['MM_Longa'][view_slice], name='MM 50', line=dict(color='purple', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index[view_slice], y=data['MM_Curta'][view_slice], name='MM 20', line=dict(color='yellow', width=1.5)))
        fig.add_trace(go.Scatter(x=data.index[view_slice], y=data['Close'][view_slice], name='Preço', line=dict(color='cyan', width=2)))
        fig.update_layout(height=450, xaxis=dict(title="Data"), yaxis=dict(title="Preço (R$)"), template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader('Volatilidade e ATR')
    if len(data) < MIN_DAYS_CHARTS:
        st.info("Volatilidade não disponível por dados insuficientes.")
    else:
        fig_vol = px.line(data[view_slice], x=data.index[view_slice], y=['Volatility','ATR_14'], title='Volatilidade Anualizada (30d) e ATR(14)')
        fig_vol.update_layout(height=420, xaxis=dict(title="Data"), yaxis=dict(title="Volatilidade / ATR"), template='plotly_dark')
        st.plotly_chart(fig_vol, use_container_width=True)
        current_vol = float(data['Volatility'].iloc[-1]) if pd.notna(data['Volatility'].iloc[-1]) else 0.0
        if current_vol >= 0.5: vol_label = "ALTA VOLATILIDADE"
        elif current_vol >= 0.25: vol_label = "VOLATILIDADE MÉDIA"
        else: vol_label = "BAIXA VOLATILIDADE"
        st.markdown(f"<div style='background:#0b1220;padding:8px;border-radius:6px'><span style='color:#ddd;font-size:16px'>{vol_label}: {current_vol:.4f}</span></div>", unsafe_allow_html=True)

with tab3:
    st.subheader('Comparativo com IBOVESPA')
    if len(data) < MIN_DAYS_CHARTS or ibov.empty:
        st.info("Comparador IBOVESPA indisponível por dados insuficientes.")
    else:
        common_idx = data.index.intersection(ibov.index)
        if len(common_idx) > 1:
            comp_df = pd.DataFrame({
                'IBOVESPA': ibov.loc[common_idx,'Close'] / ibov.loc[common_idx[0],'Close'],
                ticker_symbol: data.loc[common_idx,'Close'] / data.loc[common_idx[0],'Close']
            })
            fig_comp = px.line(comp_df, x=comp_df.index, y=comp_df.columns, title='Performance Normalizada (base 1)')
            fig_comp.update_layout(height=400, xaxis=dict(title="Data"), yaxis=dict(title="Performance Normalizada"), template='plotly_dark')
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("Períodos não coincidem o suficiente para comparação.")

with tab4:
    st.subheader('Oscilação Intradiária do Último Dia Disponível')
    last_trade_day = data.index[-1].date()
    try:
        intraday = ticker_obj.history(start=last_trade_day, end=last_trade_day + timedelta(days=1), interval='5m')
        if intraday is None or intraday.empty:
            st.info("Dados intradiários não disponíveis para o último dia.")
        else:
            intraday.index = pd.to_datetime(intraday.index)
            fig_intra = go.Figure()
            fig_intra.add_trace(go.Scatter(x=intraday.index, y=intraday['Close'], mode='lines', name='Preço Intraday'))
            fig_intra.update_layout(height=350, xaxis_title='Horário', yaxis_title='Preço (R$)', template='plotly_dark')
            st.plotly_chart(fig_intra, use_container_width=True)
    except Exception as e:
        st.info("Não foi possível obter dados intradiários.")

st.markdown("---")

st.subheader('🔮 Previsão Avançada e Backtest (métodos múltiplos)')
st.write(f"Requer pelo menos {MIN_DAYS_ADVANCED} dias para análises mais robustas.")

if st.button('Executar Análise e Backtest'):
    with st.spinner("Preparando features..."):
        adv_df, used_features = prepare_advanced_features(data, forecast_days=FORECAST_DAYS)
    if len(adv_df) < MIN_DAYS_ADVANCED:
        st.warning(f"Dados insuficientes para análise. Linhas válidas: {len(adv_df)}. Mínimo: {MIN_DAYS_ADVANCED}.")
    else:
        st.info(f"Executando validação '{method}' com sensibilidade {signal_threshold:.3f}.")
        progress_bar = st.progress(0.0)
        def prog(p): progress_bar.progress(min(1.0,p))
        try:
            res = validate_and_backtest(adv_df, used_features, method=method, static_split_ratio=0.8,
                                        rolling_train=ROLLING_TRAIN_WINDOW, rolling_test=ROLLING_TEST_WINDOW,
                                        initial_capital=10000.0, tune_models=run_tuning, signal_threshold=signal_threshold)
            progress_bar.progress(1.0)
        except Exception as e:
            logger.error(f"Erro na validação/backtest: {e}")
            st.error(f"Erro: {e}")
            res = None

        if res:
            # Métricas agregadas
            price_metrics = compute_price_metrics((1 + res['reals']) * res['base_prices'], (1 + res['preds']) * res['base_prices'])
            hitrate = compute_return_hitrate(res['reals'], res['preds'])
            st.subheader("Métricas Agregadas do Ensemble")
            pm_mae = price_metrics.get('MAE'); pm_rmse = price_metrics.get('RMSE'); pm_mape = price_metrics.get('MAPE')
            st.write(f"MAE (R$): {pm_mae:.4f} | RMSE (R$): {pm_rmse:.4f} | MAPE: {pm_mape:.4%}" if pm_mape is not None else f"MAE: {pm_mae}, RMSE: {pm_rmse}")
            st.write(f"Hit Rate (sinal): {hitrate:.4%}" if hitrate is not None else "Hit Rate: N/A")

            # Equity curve
            eq = res['sim']['equity_curve']
            fig_eq = px.line(eq, x='Date', y='Equity', title='Equity Curve da Estratégia')
            fig_eq.update_layout(template='plotly_dark')
            st.plotly_chart(fig_eq, use_container_width=True)

            # Trades
            trades = res['sim']['trades']
            if not trades.empty:
                st.subheader("Resumo de Trades (ordens executadas)")
                trades_display = trades.copy()
                if 'date' in trades_display.columns:
                    trades_display['date'] = pd.to_datetime(trades_display['date'])
                st.dataframe(trades_display.sort_values('date', ascending=False).reset_index(drop=True), use_container_width=True)
                total_pnl = trades['pnl'].sum() if 'pnl' in trades.columns else None
                if total_pnl is not None:
                    st.write(f"Resultado total de P&L (soma das trades): R$ {total_pnl:,.2f}")

            # Comparação com Buy & Hold
            try:
                returns_real = res['reals']
                capital_initial = 10000.0
                bh_curve = (1 + returns_real).cumprod() * capital_initial
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(x=res['dates'], y=res['sim']['equity_curve']['Equity'].values, name='Estratégia'))
                fig_comp.add_trace(go.Scatter(x=res['dates'], y=bh_curve, name='Buy & Hold', line=dict(dash='dash')))
                fig_comp.update_layout(title_text='Comparação: Estratégia vs Buy & Hold', xaxis_title='Data', yaxis_title='Capital (R$)', template='plotly_dark')
                st.plotly_chart(fig_comp, use_container_width=True)
            except Exception as e:
                logger.warning(f"Erro ao plotar comparação: {e}")

            # Logs de predições: resumo por janela (se disponível)
            st.subheader("Log de Predições e Resumo por Janela")
            if 'summary' in res and res['summary']:
                # rolling/expanding summary
                summary = res['summary']
                rows = []
                for w in summary:
                    if 'window' in w:
                        tr = f"{pd.to_datetime(w['train_range'][0]).strftime('%Y-%m-%d')} -> {pd.to_datetime(w['train_range'][1]).strftime('%Y-%m-%d')}"
                        te = f"{pd.to_datetime(w['test_range'][0]).strftime('%Y-%m-%d')} -> {pd.to_datetime(w['test_range'][1]).strftime('%Y-%m-%d')}"
                        ntest = w.get('n_test', None)
                        metrics = {k: v.get('hitrate') if isinstance(v, dict) else None for k,v in w.get('metrics',{}).items()} if 'metrics' in w else {}
                        rows.append({'window': w.get('window', 'n/a'), 'train_range': tr, 'test_range': te, 'n_test': ntest, 'metrics_example': json.dumps(metrics)})
                df_summary = pd.DataFrame(rows)
                st.dataframe(df_summary, use_container_width=True)
            else:
                st.info("Resumo por janela não disponível para método selecionado (static).")

            # Exportar resultados
            metadata = {'app_version': APP_VERSION, 'timestamp': pd.Timestamp.now(tz='America/Sao_Paulo').isoformat(),
                        'ticker': ticker_symbol, 'method': method, 'params': {'commission_pct': DEFAULT_COMMISSION_PCT, 'slippage_pct': DEFAULT_SLIPPAGE_PCT, 'spread_pct': DEFAULT_SPREAD_PCT, 'position_fraction': DEFAULT_POSITION_FRACTION, 'stop_loss_pct': stop_loss_input, 'take_profit_pct': take_profit_input, 'signal_threshold': signal_threshold},
                        'price_metrics': price_metrics, 'hitrate': hitrate, 'hash_run': sha256_of_text(json.dumps({'timestamp': pd.Timestamp.now().isoformat(), 'ticker': ticker_symbol, 'version': APP_VERSION}))}
            mem = io.BytesIO()
            with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                buf = io.BytesIO(); res['sim']['equity_curve'].to_csv(buf, index=False); zf.writestr('equity_curve.csv', buf.getvalue())
                buf2 = io.BytesIO(); res['sim']['trades'].to_csv(buf2, index=False); zf.writestr('trades.csv', buf2.getvalue())
                preds_df = pd.DataFrame({'Date': pd.to_datetime(res['dates']).strftime('%Y-%m-%d'), 'PredReturn': res['preds'], 'RealReturn': res['reals'], 'BasePrice': res['base_prices']})
                zf.writestr('predictions.csv', preds_df.to_csv(index=False))
                zf.writestr('metadata.json', json.dumps(metadata, indent=4, default=str))
            mem.seek(0)
            st.download_button(label="Baixar Resultados (ZIP)", data=mem, file_name=f"analise_{ticker_symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.zip", mime="application/zip")

            logger.info("Análise concluída. Export disponível.")
        else:
            st.error("Não foi possível executar a análise.")

# Importar análises exportadas
st.markdown("---")
st.subheader("Importar e Comparar Previsões Exportadas")
uploaded = st.file_uploader("Carregar ZIP de análise exportada por esta ferramenta", type=["zip"])
if uploaded is not None:
    try:
        z = zipfile.ZipFile(io.BytesIO(uploaded.read()))
        if 'metadata.json' in z.namelist() and 'predictions.csv' in z.namelist():
            meta = json.loads(z.read('metadata.json')); preds = pd.read_csv(io.BytesIO(z.read('predictions.csv')))
            st.write(f"Análise importada para o ticker **{meta.get('ticker')}** de **{meta.get('timestamp')}**.")
            st.dataframe(preds, use_container_width=True)
            if st.button("Comparar Previsão Importada com Preços Reais"):
                ticker_to_check = f"{meta.get('ticker')}.SA"
                dates_to_check = pd.to_datetime(preds['Date'])
                start_check, end_check = dates_to_check.min() - BDay(5), dates_to_check.max() + BDay(5)
                actual_data = safe_yf_download(ticker_to_check, start_check.strftime('%Y-%m-%d'), end_check.strftime('%Y-%m-%d'))
                if not actual_data.empty:
                    actual_data.index = pd.to_datetime(actual_data.index).normalize()
                    preds['Date'] = pd.to_datetime(preds['Date']).dt.normalize()
                    # compute predicted price if not present
                    if 'Preço Previsto' not in preds.columns and 'BasePrice' in preds.columns and 'PredReturn' in preds.columns:
                        preds['Preço Previsto'] = preds['BasePrice'] * (1 + preds['PredReturn'])
                    merged_df = pd.merge(preds, actual_data[['Close']], left_on='Date', right_index=True, how='left').rename(columns={'Close':'Preço Real'})
                    merged_df['Erro (R$)'] = merged_df['Preço Real'] - merged_df.get('Preço Previsto', np.nan)
                    merged_df['Erro (%)'] = merged_df['Erro (R$)'] / merged_df.get('Preço Previsto', np.nan)
                    st.dataframe(merged_df, use_container_width=True)
                else:
                    st.warning("Não foi possível baixar dados reais.")
        else:
            st.error("ZIP inválido. Arquivos 'metadata.json' ou 'predictions.csv' não encontrados.")
    except Exception as e:
        st.error(f"Erro ao processar ZIP: {e}")

st.markdown("---")
horario_consulta = pd.Timestamp.now(tz='America/Sao_Paulo').strftime('%d/%m/%Y %H:%M:%S')
st.caption(f"Última consulta dos dados: **{horario_consulta}** — Dados: Yahoo Finance.")
st.markdown(f"<p style='text-align:center;color:#888'>Desenvolvido por Rodrigo Costa de Araujo</p>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center;color:#888'>Logs gravados em: {LOG_FILE}</p>", unsafe_allow_html=True)
logger.info("Interface carregada com parâmetros finais.")
