# app.py
# Analisador com validação 180 dias, confiança = (1 - MAPE)*100 e backtest visível
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date
from pandas.tseries.offsets import BDay
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io, zipfile, json, warnings

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Analisador de Ativos (Confiança+Backtest)", layout="wide")

# Optional: Prophet and TensorFlow (LSTM) if installed
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

# UI
st.title('📊 Analisador Interativo de Ativos — Confiança & Backtest')
st.write('Todas as datas no formato dd/mm/YYYY. Use com cautela.')

st.sidebar.header('⚙️ Parâmetros')
start_default = date(2019, 1, 1)
start_date = st.sidebar.date_input("Data de Início", start_default, format="DD/MM/YYYY")
end_date = st.sidebar.date_input("Data de Fim", date.today(), format="DD/MM/YYYY")
view_period = st.sidebar.selectbox("Período de visualização", ["Últimos 3 meses", "Últimos 6 meses", "Último 1 ano", "Todo período"], index=2)
view_map = {"Últimos 3 meses": 63, "Últimos 6 meses": 126, "Último 1 ano": 252, "Todo período": None}
viz_days = view_map[view_period]

MIN_DAYS_CHARTS = 60         # mínimo para mostrar gráficos históricos
MIN_DAYS_ADVANCED = 180      # mínimo para permitir análise avançada (requisitado)

# Helpers: data and indicators
@st.cache_data
def get_tickers_from_csv():
    try:
        df = pd.read_csv('acoes-listadas-b3.csv')
        df = df.rename(columns={'Ticker':'ticker','Nome':'nome'})
        df['display'] = df['nome'] + ' (' + df['ticker'] + ')'
        return df
    except Exception:
        fallback = {'ticker':['PETR4','VALE3','ITUB4','MGLU3'],'nome':['Petrobras','Vale','Itaú','Magazine Luiza']}
        df = pd.DataFrame(fallback); df['display'] = df['nome'] + ' (' + df['ticker'] + ')'
        return df

@st.cache_data
def load_data(ticker, start, end):
    s = pd.to_datetime(start)
    e = pd.to_datetime(end) + pd.Timedelta(days=1)
    df = yf.download(ticker, start=s.strftime('%Y-%m-%d'), end=e.strftime('%Y-%m-%d'), progress=False)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if 'Volume' not in df.columns: df['Volume'] = 0
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

def prepare_advanced_features(df, forecast_days=5):
    d = df[['Close','Volume','RSI','MM_Curta','MM_Longa','Volatility']].copy()
    periods = [1,3,5,10,20]
    for p in periods:
        d[f'return_{p}d'] = d['Close'].pct_change(p)
        d[f'high_{p}d'] = d['Close'].rolling(window=p, min_periods=1).max()
        d[f'low_{p}d'] = d['Close'].rolling(window=p, min_periods=1).min()
        d[f'volatility_{p}d'] = d['Close'].pct_change().rolling(window=p, min_periods=1).std()
    if 'Volume' in d.columns and d['Volume'].sum() > 0:
        for p in periods: d[f'volume_ma_{p}d'] = d['Volume'].rolling(window=p, min_periods=1).mean()
    d['price_vs_ma20'] = d['Close'] / d['MM_Curta'].replace(0, np.nan)
    d['price_vs_ma50'] = d['Close'] / d['MM_Longa'].replace(0, np.nan)
    d['ma_cross'] = (d['MM_Curta'] > d['MM_Longa']).astype(int)
    d['target_future_return'] = d['Close'].shift(-forecast_days) / d['Close'] - 1
    d['target_direction'] = (d['target_future_return'] > 0).astype(int)
    d.replace([np.inf,-np.inf], np.nan, inplace=True)
    potential = [c for c in d.columns if c.startswith(('return_','volume_ma_','high_','low_','volatility_','price_vs_','ma_cross'))]
    potential.extend(['RSI','Volatility'])
    features = [c for c in potential if c in d.columns and not d[c].isnull().all()]
    required = features + ['target_future_return','target_direction']
    d.dropna(subset=required, inplace=True)
    return d, features

# Modeling helpers
def create_classic_models():
    return {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'Neural Net': MLPRegressor(hidden_layer_sizes=(64,32), max_iter=1000, random_state=42)
    }

def compute_metrics(y_true, y_pred):
    mask = np.isfinite(y_pred)
    if mask.sum() == 0:
        return {'MAE': None, 'RMSE': None, 'MAPE': None, 'HitRate': None}
    y_t = np.array(y_true)[mask]
    y_p = np.array(y_pred)[mask]
    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    denom = np.where(np.abs(y_t) < 1e-8, 1e-8, np.abs(y_t))
    mape = float(np.mean(np.abs((y_t - y_p) / denom)))
    hit = float(np.mean((np.sign(y_t) == np.sign(y_p)).astype(float)))
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'HitRate': hit}

def backtest_models(models_dict, adv_df, features, forecast_days=5, include_prophet=False, include_lstm=False, lstm_epochs=20, progress_callback=None):
    # 80/20 split
    X = adv_df[features]
    y = adv_df['target_future_return']
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    trained = {}
    results = {}

    total_models = len(models_dict) + (1 if include_prophet else 0) + (1 if include_lstm else 0)
    done = 0
    for name, model in models_dict.items():
        try:
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            results[name] = compute_metrics(y_test.values, y_pred)
            trained[name] = {'model': model, 'y_pred': y_pred}
        except Exception as e:
            results[name] = {'error': str(e)}
            trained[name] = {'model': None, 'y_pred': np.full(len(y_test), np.nan)}
        done += 1
        if progress_callback: progress_callback(done / total_models)

    # Prophet (price-based) if available
    if include_prophet and HAS_PROPHET:
        try:
            train_close = adv_df['Close'].iloc[:split].reset_index().rename(columns={'index':'ds','Close':'y'})
            m = Prophet()
            m.fit(train_close)
            # forecast range sufficient to cover test end + forecast_days
            last_needed = adv_df.index[split + len(X_test) - 1] + pd.Timedelta(days=forecast_days+5)
            future = m.make_future_dataframe(periods=(len(pd.date_range(start=adv_df.index[split], end=last_needed, freq='B'))), freq='D')
            fcst = m.predict(future).set_index('ds')
            preds = []
            for t in adv_df.index[split:split+len(X_test)]:
                t_plus = (pd.to_datetime(t) + pd.Timedelta(days=forecast_days)).normalize()
                if t_plus in fcst.index:
                    pred_price = fcst.loc[t_plus, 'yhat']
                else:
                    pred_price = fcst['yhat'].iloc[-1]
                base = adv_df.loc[t, 'Close']
                preds.append((pred_price / base) - 1)
            preds = np.array(preds)
            results['Prophet'] = compute_metrics(y_test.values, preds)
            trained['Prophet'] = {'model': m, 'y_pred': preds}
        except Exception as e:
            results['Prophet'] = {'error': str(e)}
            trained['Prophet'] = {'model': None, 'y_pred': np.full(len(y_test), np.nan)}
        done += 1
        if progress_callback: progress_callback(done / total_models)

    # LSTM optional skipped here for brevity if TF not installed
    if include_lstm and HAS_TF:
        try:
            X_train_seq = X_train_s.reshape((X_train_s.shape[0], 1, X_train_s.shape[1]))
            X_test_seq = X_test_s.reshape((X_test_s.shape[0], 1, X_test_s.shape[1]))
            model = Sequential()
            model.add(LSTM(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
            model.add(Dropout(0.2))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train_seq, y_train.values, epochs=lstm_epochs, batch_size=16, verbose=0)
            y_pred = model.predict(X_test_seq).flatten()
            results['LSTM'] = compute_metrics(y_test.values, y_pred)
            trained['LSTM'] = {'model': model, 'y_pred': y_pred}
        except Exception as e:
            results['LSTM'] = {'error': str(e)}
            trained['LSTM'] = {'model': None, 'y_pred': np.full(len(y_test), np.nan)}
        done += 1
        if progress_callback: progress_callback(done / total_models)

    # Ensemble mean across available predictions
    all_preds = [v['y_pred'] for v in trained.values() if 'y_pred' in v]
    if len(all_preds) == 0:
        ensemble_pred = np.full(len(y_test), np.nan)
    else:
        stacked = np.vstack([np.where(np.isfinite(a), a, np.nan) for a in all_preds])
        ensemble_pred = np.nanmean(stacked, axis=0)
    results['Ensemble'] = compute_metrics(y_test.values, ensemble_pred)
    return {'results': results, 'trained': trained, 'y_test_index': adv_df.index[split:split+len(X_test)], 'y_test': y_test.values, 'ensemble_pred': ensemble_pred}

# Confidence helper using MAPE thresholds
def confidence_from_mape(mape):
    # mape in decimal (0.05 = 5%)
    if mape is None: return 0.0, "BAIXA CONFIANÇA", "#E74C3C"
    conf_pct = max(0.0, min(1.0, 1.0 - mape)) * 100.0
    if mape < 0.05: label, color = "ALTA CONFIANÇA", "#2ECC71"
    elif mape < 0.10: label, color = "MÉDIA CONFIANÇA", "#F1C40F"
    else: label, color = "BAIXA CONFIANÇA", "#E74C3C"
    return conf_pct, label, color

# Load tickers
tickers_df = get_tickers_from_csv()
selected_display = st.sidebar.selectbox('Escolha a Ação', tickers_df['display'])
ticker_symbol = tickers_df[tickers_df['display'] == selected_display]['ticker'].iloc[0]
company_name = tickers_df[tickers_df['display'] == selected_display]['nome'].iloc[0]
ticker = f"{ticker_symbol}.SA"

# Clear analyses when ticker changes
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

# Header metrics
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

# Initial charts: require MIN_DAYS_CHARTS to show, otherwise friendly warning (no error)
tab1, tab2, tab3 = st.tabs(["Preço e Indicadores", "Volatilidade", "Comparativo com IBOVESPA"])
if viz_days is None:
    view_slice = slice(None)
else:
    view_slice = slice(-viz_days, None)

with tab1:
    st.subheader('Preço, Médias Móveis e Bandas de Bollinger')
    if len(data) < MIN_DAYS_CHARTS:
        st.warning(f"Dados insuficientes para gráficos históricos (mínimo {MIN_DAYS_CHARTS} dias). Histórico disponível: {len(data)} dias.")
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

# Simple volatility model (independent)
st.subheader('🧠 Volatilidade — Modelo Simples (RandomForest)')
st.write("Modelo simples para prever volatilidade do próximo dia útil. Não sobrescreve análise avançada.")
if st.button('Executar Previsão de Volatilidade (Simples)', key='vol_simple'):
    df_vol = data[['Volatility']].copy().dropna()
    if len(df_vol) < 30:
        st.warning("Dados insuficientes para treinar o modelo de volatilidade (mínimo 30 dias).")
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

# Advanced ML + backtest
st.subheader('🔮 Previsão de Preço Avançada (ML + Backtest)')
st.write(f"Requer mínimo {MIN_DAYS_ADVANCED} dias de histórico para rodar. Prophet/TensorFlow usados se instalados.")

if st.button('Executar Previsão Avançada', key='run_advanced'):
    adv_df, used_features = prepare_advanced_features(data, forecast_days=5)
    dias_utilizados = len(adv_df)
    st.markdown(f"<div style='background:#0b1220;padding:10px;border-radius:8px'><span style='color:#fff;font-weight:700'>Dias solicitados:</span> <span style='color:#ddd;margin-left:8px'>{pd.to_datetime(start_date).strftime('%d/%m/%Y')} — {pd.to_datetime(end_date).strftime('%d/%m/%Y')} (<strong style='color:#fff'>{dias_utilizados} dias usados</strong>)</span></div>", unsafe_allow_html=True)

    if dias_utilizados < MIN_DAYS_ADVANCED:
        st.warning(f"Dados insuficientes para análise avançada. Linhas válidas após limpeza: {dias_utilizados}. Mínimo requerido: {MIN_DAYS_ADVANCED}.")
    else:
        models = create_classic_models()
        include_prophet = HAS_PROPHET
        include_lstm = HAS_TF
        progress_bar = st.progress(0)
        def prog_callback(fraction):
            try:
                progress_bar.progress(min(100, int(fraction*100)))
            except Exception:
                pass

        with st.spinner("Treinando modelos e executando backtest (80/20)..."):
            bt = backtest_models(models, adv_df, used_features, forecast_days=5, include_prophet=include_prophet, include_lstm=include_lstm, lstm_epochs=25, progress_callback=prog_callback)

        metrics = bt['results']
        # Prepare table
        rows = []
        for name, m in metrics.items():
            if 'error' in m:
                rows.append({'Modelo': name, 'MAE': None, 'RMSE': None, 'MAPE': None, 'HitRate': None, 'Erro': m['error']})
            else:
                rows.append({'Modelo': name, 'MAE': m['MAE'], 'RMSE': m['RMSE'], 'MAPE': m['MAPE'], 'HitRate': m['HitRate']})
        metrics_df = pd.DataFrame(rows)
        st.subheader("Backtest (80% treino / 20% teste) — métricas")
        st.dataframe(metrics_df.fillna("N/A"), use_container_width=True)

        # Ensemble metrics -> compute confidence = (1 - MAPE) * 100
        ensemble_m = metrics.get('Ensemble', {})
        ensemble_mape = ensemble_m.get('MAPE', None)
        conf_pct, conf_label, conf_color = (0.0, "BAIXA CONFIANÇA", "#E74C3C")
        if ensemble_mape is not None:
            conf_pct = max(0.0, min(1.0, 1.0 - ensemble_mape)) * 100.0
            if ensemble_mape < 0.05: conf_label, conf_color = "ALTA CONFIANÇA", "#2ECC71"
            elif ensemble_mape < 0.10: conf_label, conf_color = "MÉDIA CONFIANÇA", "#F1C40F"
            else: conf_label, conf_color = "BAIXA CONFIANÇA", "#E74C3C"

        st.markdown(f"<div style='background:#0b1220;padding:8px;border-radius:8px'><span style='color:#ddd;font-size:16px;font-weight:700'>Confiança (baseado em MAPE do backtest):</span> <span style='color:{conf_color};font-size:20px;font-weight:900;margin-left:12px'>{conf_label} ({conf_pct:.1f}%)</span></div>", unsafe_allow_html=True)

        # Build final predictions using ensemble mean of trained models' last test predictions as proxy
        trained = bt['trained']
        per_model_future = {}
        for name, info in trained.items():
            ypred = info.get('y_pred', None)
            if ypred is None or len(ypred) == 0:
                per_model_future[name] = float('nan')
            else:
                try:
                    per_model_future[name] = float(ypred[-1])
                except Exception:
                    per_model_future[name] = float('nan')
        valid_vals = np.array([v for v in per_model_future.values() if np.isfinite(v)], dtype=float)
        if valid_vals.size == 0:
            ensemble_future = 0.0
        else:
            ensemble_future = float(np.mean(valid_vals))
        capped = float(np.clip(ensemble_future, -0.5, 0.5))
        daily_rate = (1 + capped)**(1/5) - 1
        current_price = float(data['Close'].iloc[-1])
        current_date = pd.to_datetime(data.index[-1])
        preds_display = []
        temp = current_price
        for d in range(1,6):
            temp *= (1 + daily_rate)
            fut_date = (current_date + BDay(d)).normalize()
            preds_display.append({'Dias': d, 'Data': fut_date.strftime('%d/%m/%Y'), 'Preço Previsto': temp, 'Variação': (temp/current_price - 1)})
        preds_df = pd.DataFrame(preds_display)

        # Emphasized forecast block
        st.subheader("Projeção de Preço para os Próximos 5 Dias (baseado na última data disponível)")
        st.markdown("<div style='background:#071626;padding:12px;border-radius:10px'>", unsafe_allow_html=True)
        for row in preds_df.to_dict(orient='records'):
            st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center;padding:8px 6px;border-radius:6px;margin-bottom:6px'><div style='color:#ddd;font-size:16px'>{row['Data']}</div><div style='color:#00BFFF;font-size:28px;font-weight:900'>R$ {row['Preço Previsto']:,.2f}</div><div style='color:#ddd;font-size:16px'>{row['Variação']:+.2%}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Backtest plot: real vs ensemble predicted returns on test set
        y_idx = bt['y_test_index']; y_test = bt['y_test']; ensemble_pred = bt['ensemble_pred']
        df_bt = pd.DataFrame({'Data': y_idx, 'Real': y_test, 'Previsto': ensemble_pred}).set_index('Data')
        st.subheader("Backtest: Retorno Real vs Retorno Previsto (Ensemble) — Período de Teste")
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Real'], name='Retorno Real'))
        fig_bt.add_trace(go.Scatter(x=df_bt.index, y=df_bt['Previsto'], name='Retorno Previsto (Ensemble)', line=dict(dash='dash')))
        fig_bt.update_layout(xaxis_title='Data', yaxis_title='Retorno (decimal)')
        st.plotly_chart(fig_bt, use_container_width=True)

        # Export results
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

# Import & compare final section (always at end)
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

# Footer
last_update = pd.to_datetime(data.index[-1]).strftime('%d/%m/%Y')
st.markdown("---")
st.caption(f"Última atualização dos preços: **{last_update}** — Dados: Yahoo Finance.")
st.markdown("<p style='text-align:center;color:#888'>Desenvolvido por Rodrigo Costa de Araujo | rodrigocosta@usp.br</p>", unsafe_allow_html=True)



