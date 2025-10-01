# app.py
# Analisador completo com ensemble (RF, GB, SVR, MLP, Prophet, LSTM quando disponíveis),
# Prophet fix (ds,y), tratamento robusto de erros por modelo, .na_rep() para Styler,
# backtest com linhas (preço real x previsto), confiança baseada em MAPE de preço.
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date
from pandas.tseries.offsets import BDay
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io, zipfile, json, warnings
from math import isfinite

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Analisador (Corrigido)", layout="wide")

# Optional libraries
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
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    HAS_TF = True
except Exception:
    HAS_TF = False

# UI
st.title('📊 Analisador de Ativos — Ensemble robusto (Prophet corrigido)')
st.write('Todas as datas em dd/mm/YYYY. Confiança = (1 − MAPE_preço) × 100.')

st.sidebar.header('⚙️ Parâmetros')
start_default = date(2019, 1, 1)
start_date = st.sidebar.date_input("Data de Início", start_default, format="DD/MM/YYYY")
end_date = st.sidebar.date_input("Data de Fim", date.today(), format="DD/MM/YYYY")
view_period = st.sidebar.selectbox("Período de visualização", ["Últimos 3 meses", "Últimos 6 meses", "Último 1 ano", "Todo período"], index=2)
view_map = {"Últimos 3 meses": 63, "Últimos 6 meses": 126, "Último 1 ano": 252, "Todo período": None}
viz_days = view_map[view_period]

MIN_DAYS_CHARTS = 60
MIN_DAYS_ADVANCED = 180
FORECAST_DAYS = 5

# --- Helpers
@st.cache_data
def get_tickers_from_csv():
    try:
        df = pd.read_csv('acoes-listadas-b3.csv')
        df = df.rename(columns={'Ticker':'ticker','Nome':'nome'})
        df['display'] = df['nome'] + ' (' + df['ticker'] + ')'
        return df
    except Exception:
        fallback = {'ticker':['PETR4','VALE3','ITUB4','MGLU3'],'nome':['Petrobras','Vale','Itaú','Magazine Luiza']}
        df = pd.DataFrame(fallback)
        df['display'] = df['nome'] + ' (' + df['ticker'] + ')'
        return df

@st.cache_data
def load_data(ticker, start, end):
    s = pd.to_datetime(start)
    e = pd.to_datetime(end) + pd.Timedelta(days=1)
    df = yf.download(ticker, start=s.strftime('%Y-%m-%d'), end=e.strftime('%Y-%m-%d'), progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if 'Volume' not in df.columns:
        df['Volume'] = 0
    df.index = pd.to_datetime(df.index)
    return df

def calculate_indicators(df):
    df = df.copy()
    delta = df['Close'].diff()
    gain = delta.where(delta>0, 0)
    loss = -delta.where(delta<0, 0)
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

def prepare_advanced_features(df, forecast_days=FORECAST_DAYS):
    d = df[['Close','Volume','RSI','MM_Curta','MM_Longa','Volatility']].copy()
    periods = [1,3,5,10,20]
    for p in periods:
        d[f'return_{p}d'] = d['Close'].pct_change(p)
        d[f'high_{p}d'] = d['Close'].rolling(window=p, min_periods=1).max()
        d[f'low_{p}d'] = d['Close'].rolling(window=p, min_periods=1).min()
        d[f'volatility_{p}d'] = d['Close'].pct_change().rolling(window=p, min_periods=1).std()
    if 'Volume' in d.columns and d['Volume'].sum() > 0:
        for p in periods:
            d[f'volume_ma_{p}d'] = d['Volume'].rolling(window=p, min_periods=1).mean()
    d['price_vs_ma20'] = d['Close'] / d['MM_Curta'].replace(0, np.nan)
    d['price_vs_ma50'] = d['Close'] / d['MM_Longa'].replace(0, np.nan)
    d['ma_cross'] = (d['MM_Curta'] > d['MM_Longa']).astype(int)
    d['target_future_return'] = d['Close'].shift(-forecast_days) / d['Close'] - 1
    d['target_future_price'] = d['Close'].shift(-forecast_days)
    d['target_direction'] = (d['target_future_return'] > 0).astype(int)
    d.replace([np.inf,-np.inf], np.nan, inplace=True)
    potential = [c for c in d.columns if c.startswith(('return_','volume_ma_','high_','low_','volatility_','price_vs_','ma_cross'))]
    potential.extend(['RSI','Volatility'])
    features = [c for c in potential if c in d.columns and not d[c].isnull().all()]
    required = features + ['target_future_return','target_future_price','target_direction']
    d.dropna(subset=required, inplace=True)
    return d, features

def create_classic_models():
    return {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'Neural Net': MLPRegressor(hidden_layer_sizes=(64,32), max_iter=1000, random_state=42)
    }

def compute_price_metrics(y_true_price, y_pred_price):
    y_t = np.array(y_true_price, dtype=float)
    y_p = np.array(y_pred_price, dtype=float)
    mask = np.isfinite(y_p) & np.isfinite(y_t)
    if mask.sum() == 0:
        return {'MAE': None, 'RMSE': None, 'MAPE': None}
    y_t_m = y_t[mask]
    y_p_m = y_p[mask]
    mae = float(mean_absolute_error(y_t_m, y_p_m))
    rmse = float(np.sqrt(mean_squared_error(y_t_m, y_p_m)))
    denom = np.where(np.abs(y_t_m) < 1e-6, 1e-6, np.abs(y_t_m))
    mape = float(np.mean(np.abs((y_t_m - y_p_m) / denom)))
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def compute_return_hitrate(y_true_ret, y_pred_ret):
    y_t = np.array(y_true_ret, dtype=float)
    y_p = np.array(y_pred_ret, dtype=float)
    mask = np.isfinite(y_p) & np.isfinite(y_t)
    if mask.sum() == 0:
        return None
    return float(np.mean((np.sign(y_t[mask]) == np.sign(y_p[mask])).astype(float)))

def backtest_ensemble(adv_df, features, include_prophet=HAS_PROPHET, include_lstm=HAS_TF, progress_callback=None):
    X = adv_df[features]
    y_ret = adv_df['target_future_return']
    y_price = adv_df['target_future_price']
    split = int(len(X) * 0.8)
    if split < 2:
        raise ValueError("Período de treino insuficiente para backtest.")
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train_ret, y_test_ret = y_ret.iloc[:split], y_ret.iloc[split:]
    y_test_price = y_price.iloc[split:split+len(X_test)]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = create_classic_models()
    trained = {}
    results = {}
    total = len(models) + (1 if include_prophet else 0) + (1 if include_lstm else 0)
    done = 0

    for name, model in models.items():
        try:
            model.fit(X_train_s, y_train_ret)
            preds_ret = model.predict(X_test_s)
            base_prices = adv_df['Close'].iloc[split:split+len(X_test)].values
            preds_price = (1 + preds_ret) * base_prices
            price_metrics = compute_price_metrics(y_test_price.values, preds_price)
            hit = compute_return_hitrate(y_test_ret.values, preds_ret)
            results[name] = {'price': price_metrics, 'hitrate': hit}
            trained[name] = {'pred_ret': preds_ret, 'pred_price': preds_price}
        except Exception as e:
            results[name] = {'error': str(e)}
            trained[name] = {'pred_ret': np.full(len(X_test), np.nan), 'pred_price': np.full(len(X_test), np.nan)}
        done += 1
        if progress_callback: progress_callback(done / total)

    # Prophet as price predictor with strict ds/y creation per-target
    if include_prophet:
        if HAS_PROPHET:
            try:
                train_close = adv_df[['Close']].iloc[:split].reset_index().rename(columns={'index':'ds','Close':'y'})
                train_close['ds'] = pd.to_datetime(train_close['ds'])
                train_close['y'] = pd.to_numeric(train_close['y'], errors='coerce')
                train_close.dropna(inplace=True)
                if train_close.empty:
                    raise ValueError("Dados Prophet vazios após limpeza.")
                m = Prophet()
                m.fit(train_close)
                preds_price = []
                test_base_dates = adv_df.index[split:split+len(X_test)]
                for t in test_base_dates:
                    target_date = (pd.to_datetime(t) + pd.Timedelta(days=FORECAST_DAYS)).normalize()
                    df_pred = pd.DataFrame({'ds': [target_date]})
                    df_pred['ds'] = pd.to_datetime(df_pred['ds'])
                    fc = m.predict(df_pred)
                    # safe extraction
                    pred_price = float(fc['yhat'].iloc[0]) if 'yhat' in fc.columns and not fc['yhat'].isna().all() else float(train_close['y'].iloc[-1])
                    preds_price.append(pred_price)
                preds_price = np.array(preds_price)
                base_prices = adv_df['Close'].iloc[split:split+len(X_test)].values
                preds_ret = preds_price / base_prices - 1
                price_metrics = compute_price_metrics(y_test_price.values, preds_price)
                hit = compute_return_hitrate(y_test_ret.values, preds_ret)
                results['Prophet'] = {'price': price_metrics, 'hitrate': hit}
                trained['Prophet'] = {'pred_ret': preds_ret, 'pred_price': preds_price}
            except Exception as e:
                results['Prophet'] = {'error': f"Prophet error: {e}"}
                trained['Prophet'] = {'pred_ret': np.full(len(X_test), np.nan), 'pred_price': np.full(len(X_test), np.nan)}
        else:
            results['Prophet'] = {'error': 'Prophet não instalado'}
            trained['Prophet'] = {'pred_ret': np.full(len(X_test), np.nan), 'pred_price': np.full(len(X_test), np.nan)}
        done += 1
        if progress_callback: progress_callback(done / total)

    # LSTM optional
    if include_lstm:
        if HAS_TF:
            try:
                X_train_seq = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
                X_test_seq = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))
                model = Sequential()
                model.add(LSTM(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
                model.add(Dropout(0.2))
                model.add(Dense(32, activation='relu'))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_train_seq, y_train_ret.values, epochs=20, batch_size=16, verbose=0)
                preds_ret = model.predict(X_test_seq).flatten()
                base_prices = adv_df['Close'].iloc[split:split+len(X_test)].values
                preds_price = (1 + preds_ret) * base_prices
                price_metrics = compute_price_metrics(y_test_price.values, preds_price)
                hit = compute_return_hitrate(y_test_ret.values, preds_ret)
                results['LSTM'] = {'price': price_metrics, 'hitrate': hit}
                trained['LSTM'] = {'pred_ret': preds_ret, 'pred_price': preds_price}
            except Exception as e:
                results['LSTM'] = {'error': f"LSTM error: {e}"}
                trained['LSTM'] = {'pred_ret': np.full(len(X_test), np.nan), 'pred_price': np.full(len(X_test), np.nan)}
        else:
            results['LSTM'] = {'error': 'TensorFlow não instalado'}
            trained['LSTM'] = {'pred_ret': np.full(len(X_test), np.nan), 'pred_price': np.full(len(X_test), np.nan)}
        done += 1
        if progress_callback: progress_callback(done / total)

    # Ensemble predicted price: mean across models' predicted prices
    pred_price_matrix = np.vstack([v['pred_price'] for v in trained.values()])
    pred_price_matrix = np.where(np.isfinite(pred_price_matrix), pred_price_matrix, np.nan)
    with np.errstate(all='ignore'):
        ensemble_price = np.nanmean(pred_price_matrix, axis=0)
    price_metrics_ens = compute_price_metrics(y_test_price.values, ensemble_price)
    base_prices = adv_df['Close'].iloc[split:split+len(X_test)].values
    ensemble_ret = ensemble_price / base_prices - 1
    hit_ens = compute_return_hitrate(y_test_ret.values, ensemble_ret)
    results['Ensemble'] = {'price': price_metrics_ens, 'hitrate': hit_ens}

    df_plot = pd.DataFrame({
        'Data': adv_df.index[split:split+len(X_test)],
        'RealPrice': np.array(y_test_price.values, dtype=float),
        'PredPrice': ensemble_price
    })
    df_plot['RealPrice'] = df_plot['RealPrice'].round(2)
    df_plot['PredPrice'] = np.round(df_plot['PredPrice'].astype(float), 2)
    return {'results': results, 'trained': trained, 'df_plot': df_plot, 'ensemble_ret': ensemble_ret}

def confidence_from_price_mape(mape):
    if mape is None: return 0.0, "BAIXA CONFIANÇA", "#E74C3C"
    conf_pct = max(0.0, min(1.0, 1.0 - mape)) * 100.0
    if mape < 0.05: return conf_pct, "ALTA CONFIANÇA", "#2ECC71"
    if mape < 0.10: return conf_pct, "MÉDIA CONFIANÇA", "#F1C40F"
    return conf_pct, "BAIXA CONFIANÇA", "#E74C3C"

# --- Main flow
tickers_df = get_tickers_from_csv()
selected_display = st.sidebar.selectbox('Escolha a Ação', tickers_df['display'])
ticker_symbol = tickers_df[tickers_df['display'] == selected_display]['ticker'].iloc[0]
company_name = tickers_df[tickers_df['display'] == selected_display]['nome'].iloc[0]
ticker = f"{ticker_symbol}.SA"

# clear analyses when ticker changes
if 'last_ticker' not in st.session_state:
    st.session_state['last_ticker'] = ticker_symbol
else:
    if st.session_state['last_ticker'] != ticker_symbol:
        st.session_state['last_ticker'] = ticker_symbol
        st.session_state['advanced_result'] = None
        st.session_state['vol_result'] = None

data = load_data(ticker, start_date, end_date)
ibov = load_data('^BVSP', start_date, end_date)
if data.empty or len(data) < 1:
    st.error("❌ Não foi possível baixar dados para este ticker no período solicitado.")
    st.stop()

data = calculate_indicators(data)

# Header
st.subheader('📈 Visão Geral do Ativo')
last_price = float(data['Close'].iloc[-1])
prev_price = float(data['Close'].iloc[-2]) if len(data) >= 2 else last_price
price_change = last_price - prev_price
percent_change = ((price_change / prev_price) * 100) if prev_price != 0 else 0.0
c1,c2,c3,c4 = st.columns(4)
c1.metric("🏢 Empresa", company_name)
c2.metric("💹 Ticker", ticker_symbol)
c3.metric("💰 Último Preço", f"R$ {last_price:.2f}")
c4.metric("📊 Variação (Dia)", f"{price_change:+.2f} R$", f"{percent_change:+.2f}%")
st.markdown("---")

# Initial charts
tab1, tab2, tab3 = st.tabs(["Preço e Indicadores", "Volatilidade", "Comparativo com IBOVESPA"])
view_slice = slice(-viz_days, None) if viz_days is not None else slice(None)

with tab1:
    st.subheader('Preço, Médias Móveis e Bandas de Bollinger')
    if len(data) < MIN_DAYS_CHARTS:
        st.warning(f"Dados insuficientes para gráficos históricos (mínimo {MIN_DAYS_CHARTS} dias). Histórico: {len(data)} dias.")
    else:
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
    if len(data) < MIN_DAYS_CHARTS:
        st.info("Volatilidade não disponível por dados históricos insuficientes.")
    else:
        fig_vol = px.line(data[view_slice], x=data[view_slice].index, y='Volatility', title='Volatilidade Anualizada')
        st.plotly_chart(fig_vol, use_container_width=True)
        current_vol = float(data['Volatility'].iloc[-1]) if not pd.isna(data['Volatility'].iloc[-1]) else 0.0
        if current_vol >= 0.5: vol_label, vol_color = "ALTA VOLATILIDADE", "#E74C3C"
        elif current_vol >= 0.25: vol_label, vol_color = "VOLATILIDADE MÉDIA", "#F1C40F"
        else: vol_label, vol_color = "BAIXA VOLATILIDADE", "#2ECC71"
        st.markdown(f"<div style='background:#0b1220;padding:8px;border-radius:6px'><span style='color:{vol_color};font-size:18px;font-weight:700'>{vol_label}</span>  <span style='color:#ddd;margin-left:12px;font-size:16px'>Volatilidade atual: <strong>{current_vol:.4f}</strong></span></div>", unsafe_allow_html=True)

with tab3:
    st.subheader('Comparativo com IBOVESPA')
    if len(data) < MIN_DAYS_CHARTS or ibov.empty:
        st.info("Comparador IBOVESPA indisponível por dados insuficientes.")
    else:
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

st.markdown("---")

# Simple volatility
st.subheader('🧠 Volatilidade — Modelo Simples (RandomForest)')
if st.button('Executar Previsão de Volatilidade (Simples)', key='vol_simple'):
    df_vol = data[['Volatility']].copy().dropna()
    if len(df_vol) < 30:
        st.warning("Dados insuficientes (mínimo 30 dias).")
    else:
        for lag in range(1,6): df_vol[f'vol_lag_{lag}'] = df_vol['Volatility'].shift(lag)
        df_vol.dropna(inplace=True)
        Xv = df_vol.drop('Volatility', axis=1); yv = df_vol['Volatility']
        model_vol = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model_vol.fit(Xv, yv)
        pred_vol = float(model_vol.predict(Xv.iloc[-1:].values)[0])
        next_day = (pd.to_datetime(data.index[-1]) + BDay(1)).strftime('%d/%m/%Y')
        st.session_state['vol_result'] = {'timestamp': pd.Timestamp.now().isoformat(), 'ticker': ticker_symbol, 'pred_vol': pred_vol, 'date': next_day}
if st.session_state.get('vol_result') is not None:
    vol = st.session_state['vol_result']; v = vol['pred_vol']
    if v >= 0.5: label, color = "ALTA VOLATILIDADE", "#E74C3C"
    elif v >= 0.25: label, color = "VOLATILIDADE MÉDIA", "#F1C40F"
    else: label, color = "BAIXA VOLATILIDADE", "#2ECC71"
    st.markdown(f"<div style='background:#0b1220;padding:10px;border-radius:8px;display:flex;gap:16px;align-items:center'><div style='font-size:20px;color:{color};font-weight:800'>{label}</div><div style='color:#ddd;font-size:18px'>Data prevista: <strong>{vol['date']}</strong></div><div style='color:#ddd;font-size:18px'>Valor previsto: <strong>{v:.4f}</strong></div></div>", unsafe_allow_html=True)
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('volatility.json', json.dumps(vol)); zf.writestr('meta.txt', f"Ticker:{vol['ticker']}\nExport:{vol['timestamp']}\n")
    mem.seek(0)
    st.download_button("Exportar Volatilidade (ZIP)", mem.getvalue(), file_name=f"volatility_{ticker_symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.zip", mime="application/zip")

st.markdown("---")

# Advanced
st.subheader('🔮 Previsão de Preço Avançada (Ensemble + Backtest)')
st.write(f"Requer mínimo {MIN_DAYS_ADVANCED} dias de histórico para rodar. Usa Prophet/LSTM se disponíveis.")

if st.button('Executar Previsão Avançada', key='run_advanced'):
    adv_df, used_features = prepare_advanced_features(data, forecast_days=FORECAST_DAYS)
    dias_utilizados = len(adv_df)
    st.markdown(f"<div style='background:#0b1220;padding:10px;border-radius:8px'><span style='color:#fff;font-weight:700'>Dias solicitados:</span> <span style='color:#ddd;margin-left:8px'>{pd.to_datetime(start_date).strftime('%d/%m/%Y')} — {pd.to_datetime(end_date).strftime('%d/%m/%Y')} (<strong style='color:#fff'>{dias_utilizados} dias usados</strong>)</span></div>", unsafe_allow_html=True)

    if dias_utilizados < MIN_DAYS_ADVANCED:
        st.warning(f"Dados insuficientes para análise avançada. Linhas válidas: {dias_utilizados}. Mínimo: {MIN_DAYS_ADVANCED}.")
    else:
        progress_bar = st.progress(0)
        def prog(p): progress_bar.progress(min(100, int(p*100)))
        with st.spinner("Executando backtest (80/20) e treinando modelos..."):
            bt = backtest_ensemble(adv_df, used_features, include_prophet=True, include_lstm=True, progress_callback=prog)

        metrics = bt['results']
        rows = []
        for k, v in metrics.items():
            if 'error' in v:
                rows.append({'Modelo': k, 'MAE (R$)': None, 'RMSE (R$)': None, 'MAPE (%)': None, 'HitRate': None, 'Erro': v.get('error')})
            else:
                price = v.get('price', {})
                mae = price.get('MAE'); rmse = price.get('RMSE'); mape = price.get('MAPE')
                hit = v.get('hitrate')
                rows.append({'Modelo': k, 'MAE (R$)': mae, 'RMSE (R$)': rmse, 'MAPE (%)': (mape * 100 if mape is not None else None), 'HitRate': (hit if hit is not None else None), 'Erro': None})
        metrics_df = pd.DataFrame(rows)

        # Render styled dataframe safely: do NOT fillna before formatting; use na_rep
        sty = metrics_df.style.format({
            'MAE (R$)': lambda v: f"R$ {v:,.2f}" if pd.notna(v) else "N/A",
            'RMSE (R$)': lambda v: f"R$ {v:,.2f}" if pd.notna(v) else "N/A",
            'MAPE (%)': lambda v: f"{v:.2f}%" if pd.notna(v) else "N/A",
            'HitRate': lambda v: f"{v:.2%}" if pd.notna(v) else "N/A",
            'Erro': lambda v: v if pd.notna(v) else ""
        }).na_rep("N/A")
        st.subheader("Backtest (80% treino / 20% teste) — métricas (preço)")
        st.dataframe(sty, use_container_width=True)

        # ensemble confidence from price MAPE
        ensemble_price_metrics = metrics.get('Ensemble', {}).get('price', {})
        ensemble_mape = ensemble_price_metrics.get('MAPE', None)
        conf_pct, conf_label, conf_color = confidence_from_price_mape(ensemble_mape)
        st.markdown(f"<div style='background:#0b1220;padding:8px;border-radius:8px'><span style='color:#ddd;font-size:16px;font-weight:700'>Confiança (1 − MAPE_preço):</span> <span style='color:{conf_color};font-size:20px;font-weight:900;margin-left:12px'>{conf_label} ({conf_pct:.1f}%)</span></div>", unsafe_allow_html=True)

        # Final predictions
        trained = bt['trained']
        per_model_latest_ret = {}
        for name, info in trained.items():
            preds_ret = info.get('pred_ret', None)
            if preds_ret is None or len(preds_ret) == 0:
                per_model_latest_ret[name] = float('nan')
            else:
                val = preds_ret[-1]
                per_model_latest_ret[name] = float(val) if isfinite(val) else float('nan')
        valid_vals = np.array([v for v in per_model_latest_ret.values() if np.isfinite(v)], dtype=float)
        ensemble_future_ret = float(np.mean(valid_vals)) if valid_vals.size > 0 else 0.0
        ensemble_future_ret = float(np.clip(ensemble_future_ret, -0.5, 0.5))
        daily_rate = (1 + ensemble_future_ret) ** (1/FORECAST_DAYS) - 1
        current_price = float(data['Close'].iloc[-1])
        current_date = pd.to_datetime(data.index[-1])
        preds_display = []
        tmp = current_price
        for d in range(1, FORECAST_DAYS+1):
            tmp *= (1 + daily_rate)
            preds_display.append({'Dias': d, 'Data': (current_date + BDay(d)).strftime('%d/%m/%Y'), 'Preço Previsto': round(tmp, 2), 'Variação': round(tmp/current_price - 1, 4)})
        preds_df = pd.DataFrame(preds_display)

        st.subheader("Projeção de Preço para os Próximos 5 Dias (baseado na última data disponível)")
        st.markdown("<div style='background:#071626;padding:12px;border-radius:10px'>", unsafe_allow_html=True)
        for r in preds_df.to_dict(orient='records'):
            st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center;padding:8px 6px;border-radius:6px;margin-bottom:6px'><div style='color:#ddd;font-size:16px'>{r['Data']}</div><div style='color:#00BFFF;font-size:28px;font-weight:900'>R$ {r['Preço Previsto']:,.2f}</div><div style='color:#ddd;font-size:16px'>{r['Variação']:+.2%}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Backtest plot
        df_plot = bt['df_plot'].copy()
        df_plot['Data'] = pd.to_datetime(df_plot['Data'])
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=df_plot['Data'], y=df_plot['RealPrice'], name='Preço Real (t+5)', line=dict(color='blue')))
        fig_bt.add_trace(go.Scatter(x=df_plot['Data'], y=df_plot['PredPrice'], name='Preço Previsto (Ensemble)', line=dict(color='red', dash='dash')))
        fig_bt.update_layout(title='Backtest: Preço Real vs Preço Previsto (Período de Teste)', xaxis_title='Data', yaxis_title='Preço (R$)')
        st.subheader("Backtest: Preço Real vs Preço Previsto (linhas)")
        st.plotly_chart(fig_bt, use_container_width=True)

        # Export
        adv_result = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'ticker': ticker_symbol,
            'data_used_period': f"{pd.to_datetime(start_date).strftime('%d/%m/%Y')} — {pd.to_datetime(end_date).strftime('%d/%m/%Y')}",
            'rows_used': int(dias_utilizados),
            'features_used': used_features,
            'backtest_metrics': metrics,
            'per_model_latest_ret': per_model_latest_ret,
            'ensemble_future_ret': ensemble_future_ret,
            'predictions_df': preds_df.to_dict(orient='records'),
            'confidence_pct': conf_pct
        }
        st.session_state['advanced_result'] = adv_result
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('predictions.csv', pd.DataFrame(preds_df).to_csv(index=False))
            zf.writestr('metadata.json', json.dumps(adv_result))
        mem.seek(0)
        st.download_button("Exportar Previsão Avançada (ZIP)", mem.getvalue(), file_name=f"analise_avancada_{ticker_symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.zip", mime="application/zip")

st.markdown("---")

# Import & compare
st.subheader("📂 Importar e Comparar Previsões Exportadas")
uploaded = st.file_uploader("Carregar ZIP de análise exportada por esta ferramenta", type=["zip"])
if uploaded is not None:
    try:
        z = zipfile.ZipFile(io.BytesIO(uploaded.read()))
        if 'predictions.csv' in z.namelist():
            preds = pd.read_csv(io.BytesIO(z.read('predictions.csv')), dtype=str)
            preds['Data'] = pd.to_datetime(preds['Data'], dayfirst=True, errors='coerce')
            preds_display = preds.copy(); preds_display['Data'] = preds_display['Data'].dt.strftime('%d/%m/%Y')
            st.write("Predições importadas:")
            st.dataframe(preds_display, use_container_width=True)
            if st.button("Comparar com preços reais (Yahoo Finance)"):
                min_date = preds['Data'].min(); max_date = preds['Data'].max()
                df_actual = yf.download(f"{preds['Ticker'].iloc[0]}.SA", start=(min_date - pd.Timedelta(days=3)).strftime('%Y-%m-%d'), end=(max_date + pd.Timedelta(days=3)).strftime('%Y-%m-%d'), progress=False)
                if df_actual.empty: st.error("Não foi possível baixar preços reais para as datas requeridas.")
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
                                    actual_price = float(df_actual.loc[try_date,'Close']); break
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
                    if hist.empty: st.error("Não foi possível baixar histórico para cálculo da volatilidade real.")
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


