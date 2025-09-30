# --- Analisador com Confiança + Backtesting Automático ---
# Substitua seu arquivo por este.
# Requisitos opcionais extras: pip install prophet tensorflow
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date
from pandas.tseries.offsets import BDay
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io
import zipfile
import json
import warnings

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Analisador de Ativos (Confiança+Backtest)", layout="wide")

# --- Opção: importar Prophet e TensorFlow (LSTM) se disponíveis ---
HAS_PROPHET = False
HAS_TF = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    HAS_TF = True
except Exception:
    HAS_TF = False

# --- UI titulo / sidebar ---
st.title('📊 Analisador Interativo de Ativos - Confiança & Backtest')
st.write('Previsões com indicador de confiança baseado em backtest. Todas as datas em dd/mm/YYYY.')

st.sidebar.header('⚙️ Parâmetros')
start_default = date(2019, 1, 1)
start_date = st.sidebar.date_input("Data de Início", start_default, format="DD/MM/YYYY")
end_date = st.sidebar.date_input("Data de Fim", date.today(), format="DD/MM/YYYY")
view_period = st.sidebar.selectbox("Período de visualização", ["Últimos 3 meses", "Últimos 6 meses", "Último 1 ano", "Todo período"], index=2)
view_map = {"Últimos 3 meses": 63, "Últimos 6 meses": 126, "Último 1 ano": 252, "Todo período": None}
viz_days = view_map[view_period]

# --- Helpers: dados e indicadores ---
@st.cache_data
def get_tickers_from_csv():
    try:
        df = pd.read_csv('acoes-listadas-b3.csv')
        df = df.rename(columns={'Ticker': 'ticker', 'Nome': 'nome'})
        df['display'] = df['nome'] + ' (' + df['ticker'] + ')'
        return df
    except FileNotFoundError:
        fallback = {'ticker': ['PETR4','VALE3','ITUB4','MGLU3'], 'nome': ['Petrobras','Vale','Itaú','Magazine Luiza']}
        df = pd.DataFrame(fallback)
        df['display'] = df['nome'] + ' (' + df['ticker'] + ')'
        return df

@st.cache_data
def load_data(ticker, start, end):
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) + pd.Timedelta(days=1)
    df = yf.download(ticker, start=start_dt.strftime('%Y-%m-%d'), end=end_dt.strftime('%Y-%m-%d'), progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    df.index = pd.to_datetime(df.index)
    return df

def calculate_indicators(df):
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MM_Curta'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['MM_Longa'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['BB_Media'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['BB_Superior'] = df['BB_Media'] + 2 * df['Close'].rolling(window=20, min_periods=1).std()
    df['BB_Inferior'] = df['BB_Media'] - 2 * df['Close'].rolling(window=20, min_periods=1).std()
    df['Daily Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily Return'].rolling(window=30, min_periods=1).std() * (252**0.5)
    return df

def prepare_advanced_features(df, forecast_days=5):
    dff = df[['Close','Volume','RSI','MM_Curta','MM_Longa','Volatility']].copy()
    periods = [1,3,5,10,20]
    for p in periods:
        dff[f'return_{p}d'] = dff['Close'].pct_change(p)
        dff[f'high_{p}d'] = dff['Close'].rolling(window=p, min_periods=1).max()
        dff[f'low_{p}d'] = dff['Close'].rolling(window=p, min_periods=1).min()
        dff[f'volatility_{p}d'] = dff['Close'].pct_change().rolling(window=p, min_periods=1).std()
    if 'Volume' in dff.columns and dff['Volume'].sum() > 0:
        for p in periods:
            dff[f'volume_ma_{p}d'] = dff['Volume'].rolling(window=p, min_periods=1).mean()
    dff['price_vs_ma20'] = dff['Close'] / dff['MM_Curta'].replace(0,np.nan)
    dff['price_vs_ma50'] = dff['Close'] / dff['MM_Longa'].replace(0,np.nan)
    dff['ma_cross'] = (dff['MM_Curta'] > dff['MM_Longa']).astype(int)
    dff['target_future_return'] = dff['Close'].shift(-forecast_days) / dff['Close'] - 1
    dff['target_direction'] = (dff['target_future_return'] > 0).astype(int)
    dff.replace([np.inf,-np.inf], np.nan, inplace=True)
    potential = [c for c in dff.columns if c.startswith(('return_','volume_ma_','high_','low_','volatility_','price_vs_','ma_cross'))]
    potential.extend(['RSI','Volatility'])
    features = [c for c in potential if c in dff.columns and not dff[c].isnull().all()]
    required = features + ['target_future_return','target_direction']
    dff.dropna(subset=required, inplace=True)
    return dff, features

# --- Modeling helpers ---
def create_classic_models():
    return {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'Neural Net': MLPRegressor(hidden_layer_sizes=(64,32), max_iter=1000, random_state=42)
    }

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Confidence & thresholds ---
MAX_REASONABLE_STD = 0.20
CONF_HIGH = 0.7
CONF_MED = 0.4

def confidence_label_from_mape(mape):
    # thresholds provided: <5% high, 5-10% medium, >10% low
    if mape < 0.05: return "ALTA CONFIANÇA", "#2ECC71"
    if mape < 0.10: return "MÉDIA CONFIANÇA", "#F1C40F"
    return "BAIXA CONFIANÇA", "#E74C3C"

# --- Backtest function (train on first 80%, test last 20%) ---
def backtest_models(models_dict, adv_df, features, forecast_days=5, include_prophet=False, include_lstm=False, lstm_epochs=20):
    """
    models_dict: classic sklearn models (dict)
    adv_df: prepared dataframe with target_future_return
    returns: dict with metrics and predictions for each model and ensemble
    """
    results = {}
    X = adv_df[features]
    y = adv_df['target_future_return']
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    # scale for classic models and NN/LSTM
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    trained = {}

    # classic sklearn models
    for name, model in models_dict.items():
        try:
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            trained[name] = {'model': model, 'y_pred': y_pred}
            results[name] = compute_metrics(y_test.values, y_pred)
        except Exception as e:
            results[name] = {'error': str(e)}
            trained[name] = {'model': None, 'y_pred': np.full(len(y_test), np.nan)}

    # Prophet: train on Close price, forecast and convert to return for the same test index set
    if include_prophet and HAS_PROPHET:
        try:
            # train prophet on training portion (using dates aligned to adv_df index)
            train_close = adv_df['Close'].iloc[:split].reset_index()
            train_close.columns = ['ds','y']
            m = Prophet()
            m.fit(train_close)
            # forecast for dates covering test set end (we need predictions at test dates + forecast_days ahead)
            # We'll forecast daily up to last test date + forecast_days
            last_needed_date = adv_df.index[split + len(X_test) - 1] + pd.Timedelta(days=forecast_days + 5)
            future = m.make_future_dataframe(periods=(len(pd.date_range(start=adv_df.index[split], end=last_needed_date, freq='B'))), freq='D')
            fcst = m.predict(future)
            fcst = fcst.set_index('ds')
            # Build predicted returns for test rows: return = price(t+forecast_days)/price(t)-1
            preds = []
            for idx in adv_df.index[split:split+len(X_test)]:
                t = idx
                t_plus = (pd.to_datetime(t) + pd.Timedelta(days=forecast_days))
                # find nearest business day in forecast
                try:
                    pred_price = fcst.loc[pd.to_datetime(t_plus).normalize(), 'yhat']
                except Exception:
                    # fallback: take last available forecasted yhat
                    pred_price = fcst['yhat'].iloc[-1]
                base_price = adv_df.loc[t, 'Close']
                preds.append((pred_price / base_price) - 1)
            preds = np.array(preds)
            results['Prophet'] = compute_metrics(y_test.values, preds)
            trained['Prophet'] = {'model': m, 'y_pred': preds}
        except Exception as e:
            results['Prophet'] = {'error': str(e)}
            trained['Prophet'] = {'model': None, 'y_pred': np.full(len(y_test), np.nan)}
    elif include_prophet and not HAS_PROPHET:
        results['Prophet'] = {'error': 'prophet not installed'}

    # LSTM: simple implementation on scaled sequences (if available)
    if include_lstm and HAS_TF:
        try:
            # prepare simple sequences: use last row only (non-seq) converted to shape (n_samples, 1, n_features)
            X_train_seq = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
            X_test_seq = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))
            lstm = build_lstm((X_train_seq.shape[1], X_train_seq.shape[2]))
            lstm.fit(X_train_seq, y_train.values, epochs=lstm_epochs, batch_size=16, verbose=0)
            y_pred = lstm.predict(X_test_seq).flatten()
            results['LSTM'] = compute_metrics(y_test.values, y_pred)
            trained['LSTM'] = {'model': lstm, 'y_pred': y_pred}
        except Exception as e:
            results['LSTM'] = {'error': str(e)}
            trained['LSTM'] = {'model': None, 'y_pred': np.full(len(y_test), np.nan)}
    elif include_lstm and not HAS_TF:
        results['LSTM'] = {'error': 'tensorflow not installed'}

    # Ensemble: mean of valid model predictions (exclude models with nan)
    all_preds = []
    for v in trained.values():
        if 'y_pred' in v:
            all_preds.append(v['y_pred'])
    if len(all_preds) == 0:
        ensemble_pred = np.full(len(y_test), np.nan)
    else:
        stacked = np.vstack(all_preds)
        # mask nan rows
        stacked = np.where(np.isfinite(stacked), stacked, np.nan)
        ensemble_pred = np.nanmean(stacked, axis=0)
    results['Ensemble'] = compute_metrics(y_test.values, ensemble_pred)
    # collect trained dict and test index
    return {'results': results, 'trained': trained, 'y_test_index': adv_df.index[split:split+len(X_test)], 'y_test': y_test.values, 'ensemble_pred': ensemble_pred}

def compute_metrics(y_true, y_pred):
    mask = np.isfinite(y_pred)
    if mask.sum() == 0:
        return {'MAE': None, 'RMSE': None, 'MAPE': None, 'HitRate': None}
    y_true_m = np.array(y_true)[mask]
    y_pred_m = np.array(y_pred)[mask]
    mae = float(mean_absolute_error(y_true_m, y_pred_m))
    rmse = float(np.sqrt(mean_squared_error(y_true_m, y_pred_m)))
    # MAPE: handle zero true values
    denom = np.where(np.abs(y_true_m) < 1e-8, 1e-8, np.abs(y_true_m))
    mape = float(np.mean(np.abs((y_true_m - y_pred_m) / denom)))
    # hit rate for direction
    hits = np.mean((np.sign(y_true_m) == np.sign(y_pred_m)).astype(float))
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'HitRate': hits}

# --- Main UI flow ---
tickers_df = get_tickers_from_csv()
selected_display = st.sidebar.selectbox('Escolha a Ação', tickers_df['display'])
ticker_symbol = tickers_df[tickers_df['display'] == selected_display]['ticker'].iloc[0]
company_name = tickers_df[tickers_df['display'] == selected_display]['nome'].iloc[0]
ticker = f"{ticker_symbol}.SA"

# Clear stored analyses when ticker changes
if 'last_ticker' not in st.session_state:
    st.session_state['last_ticker'] = ticker_symbol
else:
    if st.session_state['last_ticker'] != ticker_symbol:
        st.session_state['last_ticker'] = ticker_symbol
        st.session_state['advanced_result'] = None
        st.session_state['vol_result'] = None

data = load_data(ticker, start_date, end_date)
ibov = load_data('^BVSP', start_date, end_date)
if data.empty or len(data) < 60:
    st.error("❌ Dados insuficientes ou ausentes (mínimo 60 dias). Ajuste as datas ou ticker.")
    st.stop()

data = calculate_indicators(data)

# header metrics
st.subheader('📈 Visão Geral do Ativo')
last_price = float(data['Close'].iloc[-1])
prev_price = float(data['Close'].iloc[-2]) if len(data) >= 2 else last_price
price_change = last_price - prev_price
percent_change = ((price_change / prev_price) * 100) if prev_price != 0 else 0.0
c1, c2, c3, c4 = st.columns(4)
c1.metric("🏢 Empresa", company_name)
c2.metric("💹 Ticker", ticker_symbol)
c3.metric("💰 Último Preço", f"R$ {last_price:.2f}")
c4.metric("📊 Variação (Dia)", f"{price_change:+.2f} R$", f"{percent_change:+.2f}%")
st.markdown("---")

# initial charts (as before)
tab1, tab2, tab3 = st.tabs(["Preço e Indicadores", "Volatilidade", "Comparativo com IBOVESPA"])
if viz_days is None:
    view_slice = slice(None)
else:
    view_slice = slice(-viz_days, None)

with tab1:
    st.subheader('Preço, Médias Móveis e Bandas de Bollinger')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index[view_slice], y=data['Close'][view_slice], name='Close'))
    fig.add_trace(go.Scatter(x=data.index[view_slice], y=data['MM_Curta'][view_slice], name='MM20'))
    fig.add_trace(go.Scatter(x=data.index[view_slice], y=data['MM_Longa'][view_slice], name='MM50'))
    fig.add_trace(go.Scatter(x=data.index[view_slice], y=data['BB_Superior'][view_slice], name='BB Sup', fill=None))
    fig.add_trace(go.Scatter(x=data.index[view_slice], y=data['BB_Inferior'][view_slice], name='BB Inf', fill='tonexty', opacity=0.1))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader('RSI')
    fig_rsi = px.line(data[view_slice], x=data[view_slice].index, y='RSI', title='RSI')
    fig_rsi.add_hline(y=70, line_dash="dash", annotation_text="Sobrecompra")
    fig_rsi.add_hline(y=30, line_dash="dash", annotation_text="Sobrevenda")
    st.plotly_chart(fig_rsi, use_container_width=True)

with tab2:
    st.subheader('Volatilidade (janela de 30 dias)')
    fig_vol = px.line(data[view_slice], x=data[view_slice].index, y='Volatility', title='Volatilidade Anualizada')
    st.plotly_chart(fig_vol, use_container_width=True)
    current_vol = float(data['Volatility'].iloc[-1]) if not pd.isna(data['Volatility'].iloc[-1]) else 0.0
    # color-coded label
    if current_vol >= 0.5:
        vol_label, vol_color = "ALTA VOLATILIDADE", "#E74C3C"
    elif current_vol >= 0.25:
        vol_label, vol_color = "VOLATILIDADE MÉDIA", "#F1C40F"
    else:
        vol_label, vol_color = "BAIXA VOLATILIDADE", "#2ECC71"
    st.markdown(f"<div style='background:#0b1220;padding:8px;border-radius:6px'><span style='color:{vol_color};font-size:18px;font-weight:700'>{vol_label}</span>  <span style='color:#ddd;margin-left:12px;font-size:16px'>Volatilidade atual: <strong>{current_vol:.4f}</strong></span></div>", unsafe_allow_html=True)

with tab3:
    st.subheader('Comparativo com IBOVESPA')
    if not ibov.empty:
        common = data.index.intersection(ibov.index)
        comp_df = pd.DataFrame({
            'IBOVESPA': ibov.loc[common,'Close'] / ibov.loc[common,'Close'].iloc[0],
            ticker_symbol: data.loc[common,'Close'] / data.loc[common,'Close'].iloc[0]
        }, index=common)
        if comp_df.empty:
            st.warning("Períodos não coincidem entre ação e IBOVESPA.")
        else:
            fig_comp = px.line(comp_df, x=comp_df.index, y=comp_df.columns, title='Performance Normalizada')
            st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.warning("Não foi possível carregar IBOVESPA.")

st.markdown("---")

# --- Simple volatility model (keeps previous behavior) ---
st.subheader('🧠 Volatilidade — Modelo Simples (RandomForest)')
if st.button('Executar Previsão de Volatilidade (Simples)'):
    df_vol = data[['Volatility']].copy().dropna()
    if len(df_vol) < 30:
        st.warning("Dados insuficientes para treinar o modelo de volatilidade.")
    else:
        for lag in range(1,6):
            df_vol[f'vol_lag_{lag}'] = df_vol['Volatility'].shift(lag)
        df_vol.dropna(inplace=True)
        Xv = df_vol.drop('Volatility',axis=1)
        yv = df_vol['Volatility']
        model_vol = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model_vol.fit(Xv, yv)
        pred_vol = float(model_vol.predict(Xv.iloc[-1:].values)[0])
        next_day = (pd.to_datetime(data.index[-1]) + BDay(1)).strftime('%d/%m/%Y')
        st.session_state['vol_result'] = {'timestamp': pd.Timestamp.now().isoformat(), 'ticker': ticker_symbol, 'pred_vol': pred_vol, 'date': next_day}
if st.session_state.get('vol_result') is not None:
    vol = st.session_state['vol_result']
    v = vol['pred_vol']
    if v >= 0.5: label_color = ("ALTA VOLATILIDADE","#E74C3C")
    elif v >= 0.25: label_color = ("VOLATILIDADE MÉDIA","#F1C40F")
    else: label_color = ("BAIXA VOLATILIDADE","#2ECC71")
    st.markdown(f"<div style='background:#0b1220;padding:10px;border-radius:8px;display:flex;gap:16px;align-items:center'><div style='font-size:20px;color:{label_color[1]};font-weight:800'>{label_color[0]}</div><div style='color:#ddd;font-size:18px'>Data prevista: <strong>{vol['date']}</strong></div><div style='color:#ddd;font-size:18px'>Valor previsto: <strong>{v:.4f}</strong></div></div>", unsafe_allow_html=True)
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('volatility.json', json.dumps(vol))
        zf.writestr('meta.txt', f"Ticker:{vol['ticker']}\nExport:{vol['timestamp']}\n")
    mem.seek(0)
    st.download_button("Exportar Volatilidade (ZIP)", mem.getvalue(), file_name=f"volatility_{ticker_symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.zip", mime="application/zip")

st.markdown("---")

# --- Advanced prediction with backtest and confidence ---
st.subheader('🔮 Previsão de Preço Avançada (Machine Learning + Backtest)')
st.write("Executar treina modelos (RandomForest, GB, SVR, MLP). Prophet e LSTM serão usados se instalados. Backtest automático (80/20).")

if st.button('Executar Previsão Avançada'):
    adv_df, used_features = prepare_advanced_features(data, forecast_days=5)
    dias_utilizados = len(adv_df)
    # single-line header requested
    st.markdown(f"<div style='background:#0b1220;padding:10px;border-radius:8px'><span style='color:#fff;font-weight:700'>Dias solicitados:</span> <span style='color:#ddd;margin-left:8px'>{pd.to_datetime(start_date).strftime('%d/%m/%Y')} — {pd.to_datetime(end_date).strftime('%d/%m/%Y')} (<strong style='color:#fff'>{dias_utilizados} dias utilizados</strong>)</span></div>", unsafe_allow_html=True)
    if dias_utilizados < 60:
        st.warning(f"Dados insuficientes para análise avançada. Linhas válidas: {dias_utilizados}. Aumente o intervalo.")
    else:
        # build models
        classic = create_classic_models()
        include_prophet = HAS_PROPHET
        include_lstm = HAS_TF
        # run backtest (this shows progress visually via st.spinner)
        with st.spinner('Executando backtest e treinando modelos...'):
            bt = backtest_models(classic, adv_df, used_features, forecast_days=5, include_prophet=include_prophet, include_lstm=include_lstm, lstm_epochs=25)
        metrics = bt['results']
        # Show backtest metrics table
        metrics_table = []
        for k, v in metrics.items():
            if 'error' in v:
                metrics_table.append({'Modelo': k, 'MAE': None, 'RMSE': None, 'MAPE': None, 'HitRate': None, 'Erro': v.get('error')})
            else:
                metrics_table.append({'Modelo': k, 'MAE': v['MAE'], 'RMSE': v['RMSE'], 'MAPE': v['MAPE'], 'HitRate': v['HitRate']})
        metrics_df = pd.DataFrame(metrics_table)
        st.subheader("Backtest (80% treino / 20% teste) — métricas")
        def fmt(v):
            return f"{v:.4f}" if pd.notna(v) else "N/A"
        st.dataframe(metrics_df.fillna("N/A"), use_container_width=True)

        # Determine overall MAPE from Ensemble (preferred) -> compute confidence label
        ensemble_metrics = metrics.get('Ensemble', {})
        ensemble_mape = ensemble_metrics.get('MAPE', None)
        if ensemble_mape is None:
            ensemble_mape = 1.0  # very bad if missing
        conf_label, conf_color = confidence_label_from_mape(ensemble_mape)
        # also incorporate ensemble agreement (std of per-model preds on test)
        std_preds = np.nanstd(bt['ensemble_pred']) if bt.get('ensemble_pred') is not None else 0.0
        # Normalize std into [0,1] agreement
        agreement = max(0.0, 1.0 - min(std_preds, MAX_REASONABLE_STD) / MAX_REASONABLE_STD)
        # final confidence percent = (1 - normalized mape) * agreement, but we map using thresholds already given.
        # For display produce a combined score (1 - MAPE) * agreement
        combined_score = (1.0 - min(ensemble_mape, 1.0)) * agreement
        combined_pct = combined_score * 100.0

        # Show confidence in header (colored text foreground)
        st.markdown(f"<div style='background:#0b1220;padding:8px;border-radius:8px'><span style='color:#ddd;font-size:16px;font-weight:700'>Confiança combinada:</span> <span style='color:{conf_color};font-size:20px;font-weight:900;margin-left:12px'>{conf_label} ({combined_pct:.1f}%)</span></div>", unsafe_allow_html=True)

        # Build final predictions (using ensemble of trained models on last available features)
        trained = bt['trained']
        # use last real date from data as base
        current_date = pd.to_datetime(data.index[-1])
        current_price = float(data['Close'].iloc[-1])
        # per-model predictions (for horizon=5 days return)
        per_model_future = {}
        for name, info in trained.items():
            ypred = info.get('y_pred', None)
            if ypred is None or len(ypred) == 0:
                per_model_future[name] = float('nan')
            else:
                # For consistency we take the model's average predicted return on the test set last element as proxy for the future return.
                # Prefer last element of that y_pred if available.
                try:
                    per_model_future[name] = float(ypred[-1])
                except Exception:
                    per_model_future[name] = float('nan')
        # ensemble future return
        valid_vals = np.array([v for v in per_model_future.values() if np.isfinite(v)], dtype=float)
        if valid_vals.size == 0:
            ensemble_future = 0.0
        else:
            ensemble_future = float(np.mean(valid_vals))
        capped = float(np.clip(ensemble_future, -0.5, 0.5))
        daily_rate = (1 + capped)**(1/5) - 1
        # future price series
        temp = current_price
        preds_display = []
        for d in range(1,6):
            temp *= (1 + daily_rate)
            fut_date = (current_date + BDay(d)).normalize()
            preds_display.append({'Dias': d, 'Data': fut_date.strftime('%d/%m/%Y'), 'Preço Previsto': temp, 'Variação': temp/current_price - 1})
        preds_df = pd.DataFrame(preds_display)

        # show predictions emphasized (big font)
        st.subheader("Projeção de Preço para os Próximos 5 Dias (baseado na última data disponível)")
        st.markdown("<div style='background:#071626;padding:12px;border-radius:10px'>", unsafe_allow_html=True)
        for row in preds_df.to_dict(orient='records'):
            st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center;padding:8px 6px;border-radius:6px;margin-bottom:6px'><div style='color:#ddd;font-size:16px'>{row['Data']}</div><div style='color:#00BFFF;font-size:26px;font-weight:900'>R$ {row['Preço Previsto']:,.2f}</div><div style='color:#ddd;font-size:16px'>{row['Variação']:+.2%}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Backtest plot: predicted vs actual returns (ensemble)
        y_test_index = bt['y_test_index']
        y_test = bt['y_test']
        ensemble_pred = bt['ensemble_pred']
        # convert to DataFrame for plotting
        df_plot = pd.DataFrame({'Data': y_test_index, 'Real Return': y_test, 'Ensemble Pred': ensemble_pred})
        df_plot.set_index('Data', inplace=True)
        st.subheader("Backtest: Retorno Real vs Retorno Previsto (Ensemble) — Período de teste")
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Real Return'], name='Retorno Real'))
        fig_bt.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Ensemble Pred'], name='Retorno Previsto (Ensemble)', line=dict(dash='dash')))
        fig_bt.update_layout(xaxis_title='Data', yaxis_title='Retorno (decimal)')
        st.plotly_chart(fig_bt, use_container_width=True)

        # Export advanced result (single download button)
        adv_result = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'ticker': ticker_symbol,
            'data_used_period': f"{pd.to_datetime(start_date).strftime('%d/%m/%Y')} — {pd.to_datetime(end_date).strftime('%d/%m/%Y')}",
            'rows_used': int(dias_utilizados),
            'features_used': used_features,
            'backtest_metrics': metrics,
            'per_model_return_predictions': per_model_future,
            'ensemble_future_return': ensemble_future,
            'predictions_df': preds_df.to_dict(orient='records'),
            'confidence_combined_pct': combined_pct
        }
        st.session_state['advanced_result'] = adv_result
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('predictions.csv', pd.DataFrame(preds_df).to_csv(index=False))
            zf.writestr('metadata.json', json.dumps(adv_result))
        mem.seek(0)
        st.download_button("Exportar Previsão Avançada (ZIP)", mem.getvalue(), file_name=f"analise_avancada_{ticker_symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.zip", mime="application/zip")

st.markdown("---")

# --- Import & compare final section ---
st.subheader("📂 Importar e Comparar Previsões Exportadas")
uploaded = st.file_uploader("Carregar ZIP de análise exportada por esta ferramenta", type=["zip"])
if uploaded is not None:
    try:
        z = zipfile.ZipFile(io.BytesIO(uploaded.read()))
        if 'predictions.csv' in z.namelist():
            preds = pd.read_csv(io.BytesIO(z.read('predictions.csv')), dtype=str)
            preds['Data'] = pd.to_datetime(preds['Data'], dayfirst=True, errors='coerce')
            preds_display = preds.copy()
            preds_display['Data'] = preds_display['Data'].dt.strftime('%d/%m/%Y')
            st.write("Predições importadas:")
            st.dataframe(preds_display, use_container_width=True)
            if st.button("Comparar com preços reais (Yahoo Finance)"):
                min_date = preds['Data'].min()
                max_date = preds['Data'].max()
                df_actual = yf.download(f"{preds['Ticker'].iloc[0]}.SA", start=(min_date - pd.Timedelta(days=3)).strftime('%Y-%m-%d'), end=(max_date + pd.Timedelta(days=3)).strftime('%Y-%m-%d'), progress=False)
                if df_actual.empty:
                    st.error("Não foi possível baixar preços reais para as datas requeridas.")
                else:
                    df_actual.index = pd.to_datetime(df_actual.index).normalize()
                    rows = []
                    for _, row in preds.iterrows():
                        pred_date = row['Data'].normalize()
                        actual_price = None
                        if pred_date in df_actual.index:
                            actual_price = float(df_actual.loc[pred_date,'Close'])
                        else:
                            for k in range(1,6):
                                try_date = (pred_date + pd.Timedelta(days=k))
                                if try_date in df_actual.index:
                                    actual_price = float(df_actual.loc[try_date,'Close'])
                                    break
                        rows.append({'Data Prevista':pred_date.strftime('%d/%m/%Y'),'Preço Previsto':float(row['Preço Previsto']),'Preço Real (fechamento próximo disponível)':actual_price})
                    comp = pd.DataFrame(rows)
                    comp['Erro Abs'] = comp.apply(lambda r: None if pd.isna(r['Preço Real (fechamento próximo disponível)']) else abs(r['Preço Real (fechamento próximo disponível)'] - r['Preço Previsto']), axis=1)
                    comp['Erro %'] = comp.apply(lambda r: None if pd.isna(r['Preço Real (fechamento próximo disponível)']) else (r['Preço Real (fechamento próximo disponível)'] / r['Preço Previsto'] - 1), axis=1)
                    st.subheader("Comparação Previsto x Real")
                    st.dataframe(comp.style.format({'Preço Previsto':'R$ {:.2f}','Preço Real (fechamento próximo disponível)':'R$ {:.2f}','Erro Abs':'R$ {:.2f}','Erro %':'{:+.2%}'}), use_container_width=True)
        elif 'volatility.json' in z.namelist():
            vol = json.loads(z.read('volatility.json'))
            st.write("Arquivo de Volatilidade importado:")
            st.json(vol)
            if st.button("Comparar Volatilidade importada com valor real (Yahoo)"):
                try:
                    date_to_check = pd.to_datetime(vol['date'], dayfirst=True)
                    hist = yf.download(f"{vol['ticker']}.SA", start=(date_to_check - pd.Timedelta(days=60)).strftime('%Y-%m-%d'), end=(date_to_check + pd.Timedelta(days=3)).strftime('%Y-%m-%d'), progress=False)
                    if hist.empty:
                        st.error("Não foi possível baixar histórico para cálculo da volatilidade real.")
                    else:
                        hist['Daily Return'] = hist['Close'].pct_change()
                        real_vol = hist['Daily Return'].rolling(window=30, min_periods=1).std().iloc[-1] * (252**0.5)
                        st.write(f"Volatilidade real aproximada (janela 30d) na data mais próxima: {real_vol:.4f}")
                        st.write(f"Previsto: {vol['pred_vol']:.4f}")
                        st.write(f"Erro absoluto: {abs(real_vol - vol['pred_vol']):.4f}")
                except Exception as e:
                    st.error(f"Erro ao comparar volatilidade: {e}")
        else:
            st.error("ZIP inválido. Arquivo 'predictions.csv' ou 'volatility.json' não encontrado.")
    except Exception as e:
        st.error(f"Erro ao processar ZIP: {e}")

# footer
last_update = pd.to_datetime(data.index[-1]).strftime('%d/%m/%Y')
st.markdown("---")
st.caption(f"Última atualização dos preços: **{last_update}** — Dados: Yahoo Finance.")
st.markdown("<p style='text-align:center;color:#888'>Desenvolvido por Rodrigo Costa de Araujo | rodrigocosta@usp.br</p>", unsafe_allow_html=True)


