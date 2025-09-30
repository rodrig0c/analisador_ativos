# --- Importações e configuração ---
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date
from pandas.tseries.offsets import BDay
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import numpy as np
import warnings
import io
import zipfile
import json

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Analisador de Ativos", layout="wide")

# --- Parâmetros de UI ---
st.title('📊 Analisador Interativo de Ativos Financeiros')
st.write('Análise de preços e previsões. Use como apoio. Todas as datas no formato dd/mm/YYYY.')

st.sidebar.header('⚙️ Parâmetros')
start_default = date(2019, 1, 1)
start_date = st.sidebar.date_input("Data de Início", start_default, format="DD/MM/YYYY")
end_date = st.sidebar.date_input("Data de Fim", date.today(), format="DD/MM/YYYY")
view_period = st.sidebar.selectbox("Período de visualização", ["Últimos 3 meses", "Últimos 6 meses", "Último 1 ano", "Todo período"], index=2)
view_map = {"Últimos 3 meses": 63, "Últimos 6 meses": 126, "Último 1 ano": 252, "Todo período": None}
viz_days = view_map[view_period]

# --- Helpers / modelos / thresholds ---
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
    data = yf.download(ticker, start=start_dt.strftime('%Y-%m-%d'), end=end_dt.strftime('%Y-%m-%d'), progress=False)
    if data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if 'Volume' not in data.columns:
        data['Volume'] = 0
    data.index = pd.to_datetime(data.index)
    return data

def calculate_indicators(data):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['MM_Curta'] = data['Close'].rolling(window=20, min_periods=1).mean()
    data['MM_Longa'] = data['Close'].rolling(window=50, min_periods=1).mean()
    data['BB_Media'] = data['Close'].rolling(window=20, min_periods=1).mean()
    data['BB_Superior'] = data['BB_Media'] + 2 * data['Close'].rolling(window=20, min_periods=1).std()
    data['BB_Inferior'] = data['BB_Media'] - 2 * data['Close'].rolling(window=20, min_periods=1).std()
    data['Daily Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Daily Return'].rolling(window=30, min_periods=1).std() * (252**0.5)
    return data

def prepare_advanced_features(data, forecast_days=5):
    df = data[['Close','Volume','RSI','MM_Curta','MM_Longa','Volatility']].copy()
    periods = [1,3,5,10,20]
    for d in periods:
        df[f'return_{d}d'] = df['Close'].pct_change(d)
        df[f'high_{d}d'] = df['Close'].rolling(window=d, min_periods=1).max()
        df[f'low_{d}d'] = df['Close'].rolling(window=d, min_periods=1).min()
        df[f'volatility_{d}d'] = df['Close'].pct_change().rolling(window=d, min_periods=1).std()
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        for d in periods:
            df[f'volume_ma_{d}d'] = df['Volume'].rolling(window=d, min_periods=1).mean()
    df['price_vs_ma20'] = df['Close'] / df['MM_Curta'].replace(0,np.nan)
    df['price_vs_ma50'] = df['Close'] / df['MM_Longa'].replace(0,np.nan)
    df['ma_cross'] = (df['MM_Curta'] > df['MM_Longa']).astype(int)
    df['target_future_return'] = df['Close'].shift(-forecast_days) / df['Close'] - 1
    df['target_direction'] = (df['target_future_return'] > 0).astype(int)
    df.replace([np.inf,-np.inf], np.nan, inplace=True)
    potential = [c for c in df.columns if c.startswith(('return_','volume_ma_','high_','low_','volatility_','price_vs_','ma_cross'))]
    potential.extend(['RSI','Volatility'])
    features = [c for c in potential if c in df.columns and not df[c].isnull().all()]
    required = features + ['target_future_return','target_direction']
    df.dropna(subset=required, inplace=True)
    return df, features

def create_advanced_model():
    return {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(64,32), max_iter=1000, random_state=42)
    }

def ensemble_predict(models, X):
    preds = [m.predict(X) for m in models.values()]
    return np.mean(preds, axis=0)

# Thresholds and label helpers
MAX_REASONABLE_STD = 0.20
CONF_HIGH = 0.7
CONF_MED = 0.4
VOL_HIGH = 0.5
VOL_MED = 0.25

def confidence_label_and_color(score):
    if score >= CONF_HIGH:
        return "ALTA CONFIANÇA", "#2ECC71"
    if score >= CONF_MED:
        return "MÉDIA CONFIANÇA", "#F1C40F"
    return "BAIXA CONFIANÇA", "#E74C3C"

def volatility_label_and_color(v):
    if v >= VOL_HIGH:
        return "ALTA VOLATILIDADE", "#E74C3C"
    if v >= VOL_MED:
        return "VOLATILIDADE MÉDIA", "#F1C40F"
    return "BAIXA VOLATILIDADE", "#2ECC71"

# --- Load tickers and data ---
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
if 'advanced_result' not in st.session_state: st.session_state['advanced_result'] = None
if 'vol_result' not in st.session_state: st.session_state['vol_result'] = None

if data.empty or len(data) < 60:
    st.error("❌ Dados insuficientes ou ausentes (mínimo 60 dias). Ajuste as datas ou o ticker.")
    st.stop()

data = calculate_indicators(data)

# --- Header metrics ---
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

# --- Charts as before ---
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
    st.markdown("Nota: a janela de cálculo é 30 dias. Use o seletor de visualização para reduzir ruído.")
    fig_vol = px.line(data[view_slice], x=data[view_slice].index, y='Volatility', title='Volatilidade Anualizada')
    st.plotly_chart(fig_vol, use_container_width=True)
    current_vol = float(data['Volatility'].iloc[-1]) if not pd.isna(data['Volatility'].iloc[-1]) else 0.0
    vol_label, vol_color = volatility_label_and_color(current_vol)
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

# --- Volatilidade (simples) section ---
st.subheader('🧠 Volatilidade — Modelo Simples (RandomForest)')
st.write("Modelo simples para prever volatilidade do próximo dia útil.")
if st.button('Executar Previsão de Volatilidade (Simples)', key='run_vol_simple'):
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
        last_date = pd.to_datetime(data.index[-1])
        next_day = (last_date + BDay(1)).strftime('%d/%m/%Y')
        st.session_state['vol_result'] = {'timestamp': pd.Timestamp.now().isoformat(), 'ticker': ticker_symbol, 'pred_vol': pred_vol, 'date': next_day}

if st.session_state.get('vol_result') is not None:
    vol = st.session_state['vol_result']
    v = vol['pred_vol']
    vol_label, vol_color = volatility_label_and_color(v)
    st.markdown(
        f"<div style='background:#0b1220;padding:10px;border-radius:8px;display:flex;gap:16px;align-items:center'>"
        f"<div style='font-size:20px;color:{vol_color};font-weight:800'>{vol_label}</div>"
        f"<div style='color:#ddd;font-size:18px'>Data prevista: <strong>{vol['date']}</strong></div>"
        f"<div style='color:#ddd;font-size:18px'>Valor previsto: <strong>{v:.4f}</strong></div>"
        f"</div>", unsafe_allow_html=True)
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('volatility.json', json.dumps(vol))
        zf.writestr('meta.txt', f"Ticker:{vol['ticker']}\nExport:{vol['timestamp']}\n")
    mem.seek(0)
    st.download_button("Exportar Volatilidade (ZIP)", mem.getvalue(), file_name=f"volatility_{ticker_symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.zip", mime="application/zip")

st.markdown("---")

# --- Advanced prediction section (simplified header + progress bar only) ---
st.subheader('🔮 Previsão de Preço Avançada (Machine Learning)')
st.write("Clique em Executar. Usa apenas dados reais do Yahoo Finance no período selecionado. Resultado não sobrescreve a seção simples.")

if st.button('Executar Previsão de Preço Avançada', key='run_advanced'):
    adv_df, used_features = prepare_advanced_features(data, forecast_days=5)
    dias_utilizados = len(adv_df)
    # show only the single requested header line
    st.markdown(
        f"<div style='background:#0b1220;padding:10px;border-radius:8px'>"
        f"<span style='color:#fff;font-weight:700'>Dias solicitados:</span> "
        f"<span style='color:#ddd;margin-left:8px'>{pd.to_datetime(start_date).strftime('%d/%m/%Y')} — {pd.to_datetime(end_date).strftime('%d/%m/%Y')} "
        f"(<strong style=\"color:#fff\">{dias_utilizados} dias utilizados</strong>)</span>"
        f"</div>", unsafe_allow_html=True
    )

    if dias_utilizados < 60:
        st.warning(f"Dados insuficientes para análise avançada. Linhas válidas: {dias_utilizados}. Aumente o intervalo.")
    else:
        # prepare train/test and scale
        X = adv_df[used_features].copy()
        y_return = adv_df['target_future_return'].copy()
        y_dir = adv_df['target_direction'].copy()
        split = int(len(X)*0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y_return.iloc[:split], y_return.iloc[split:]
        ydir_train, ydir_test = y_dir.iloc[:split], y_dir.iloc[split:]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        models = create_advanced_model()
        trained = {}
        preds_test = {}
        n_models = len(models)
        progress = st.progress(0)
        status = st.empty()
        for i,(name,model) in enumerate(models.items()):
            status.text(f"Treinando {name} ({i+1}/{n_models})...")
            try:
                model.fit(X_train_s, y_train)
                trained[name] = model
                preds_test[name] = model.predict(X_test_s)
            except Exception as e:
                status.text(f"Erro ao treinar {name}: {e}")
                trained[name] = None
                preds_test[name] = np.full(shape=(len(X_test_s),), fill_value=np.nan)
            progress.progress(int(((i+1)/n_models)*100))
        status.text("Treinamento concluído!")

        # per-model future predictions using last real features
        current_price = float(data['Close'].iloc[-1])
        current_date = pd.to_datetime(data.index[-1])  # last real date
        last_feat = X.iloc[-1:].copy()
        try:
            last_scaled = scaler.transform(last_feat)
        except Exception:
            last_scaled = scaler.transform(X.tail(1).fillna(0))

        per_model_future = {}
        for name, model in trained.items():
            try:
                per_model_future[name] = float(model.predict(last_scaled)[0]) if model is not None else float('nan')
            except Exception:
                per_model_future[name] = float('nan')

        valid_vals = np.array([v for v in per_model_future.values() if not (pd.isna(v) or np.isinf(v))], dtype=float)
        if valid_vals.size == 0:
            ensemble_future = 0.0
            std_preds = 0.0
        else:
            ensemble_future = float(np.mean(valid_vals))
            std_preds = float(np.std(valid_vals))

        capped = float(np.clip(ensemble_future, -0.5, 0.5))
        daily_rate = (1 + capped)**(1/5) - 1

        # Build future dates starting from current_date + BDay(1..5) => ensures we predict FUTURE
        temp_price = current_price
        predicted_prices_data = []
        for i_day in range(1,6):
            temp_price *= (1 + daily_rate)
            fut_date = (current_date + BDay(i_day)).normalize()
            predicted_prices_data.append({
                'Dias': i_day,
                'Data': fut_date.strftime('%d/%m/%Y'),
                'Preço Previsto': temp_price,
                'Variação': (temp_price / current_price - 1),
                'Ticker': ticker_symbol
            })
        predictions_df = pd.DataFrame(predicted_prices_data)

        # Confidence computation
        agreement = max(0.0, 1.0 - min(std_preds, MAX_REASONABLE_STD) / MAX_REASONABLE_STD)
        magnitude_penalty = max(0.0, 1.0 - (abs(ensemble_future) / 0.5))
        confidence = agreement * magnitude_penalty
        confidence = float(np.clip(confidence, 0.0, 1.0))
        concordance_pct = agreement * 100.0
        confidence_pct = confidence * 100.0
        rec_label, rec_color = confidence_label_and_color(confidence)

        # Store results
        st.session_state['advanced_result'] = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'ticker': ticker_symbol,
            'data_used_period': f"{pd.to_datetime(start_date).strftime('%d/%m/%Y')} — {pd.to_datetime(end_date).strftime('%d/%m/%Y')}",
            'rows_historico': int(len(data)),
            'rows_validas': int(dias_utilizados),
            'features_used': used_features,
            'per_model_return_predictions': {k: (None if pd.isna(v) else float(v)) for k,v in per_model_future.items()},
            'ensemble_future_return': float(ensemble_future),
            'predictions_df': predictions_df.to_dict(orient='records'),
            'confidence': float(confidence),
            'concordance_pct': float(concordance_pct),
            'confidence_pct': float(confidence_pct),
            'rec_label': rec_label,
            'rec_color': rec_color
        }

# --- Display advanced results (single header line only) ---
if st.session_state.get('advanced_result') is not None:
    adv = st.session_state['advanced_result']
    st.subheader("Resultados - Previsão Avançada")
    # Only the single requested header line (Dias solicitados + (dias utilizados))
    st.markdown(
        f"<div style='background:#0b1220;padding:10px;border-radius:8px'>"
        f"<span style='color:#fff;font-weight:700'>Dias solicitados:</span> "
        f"<span style='color:#ddd;margin-left:8px'>{adv['data_used_period']} "
        f"(<strong style=\"color:#fff\">{adv['rows_validas']} dias utilizados</strong>)</span>"
        f"</div>", unsafe_allow_html=True
    )

    # Concordância, Score e Recomendação as colored text
    concord_pct = adv.get('concordance_pct', 0.0)
    confidence_pct = adv.get('confidence_pct', 0.0)
    rec_label = adv.get('rec_label', 'BAIXA CONFIANÇA')
    rec_color = adv.get('rec_color', '#E74C3C')
    st.markdown(
        f"<div style='display:flex;gap:20px;align-items:center;margin-top:8px'>"
        f"<div style='color:#ddd;font-size:16px'><strong>Concordância:</strong> <span style='color:{rec_color};font-size:20px;font-weight:800'>{concord_pct:.1f}%</span></div>"
        f"<div style='color:#ddd;font-size:16px'><strong>Score de Confiança:</strong> <span style='color:{rec_color};font-size:20px;font-weight:800'>{confidence_pct:.1f}%</span></div>"
        f"<div style='color:#ddd;font-size:16px'><strong>Recomendação:</strong> <span style='color:{rec_color};font-size:20px;font-weight:800'>{rec_label}</span></div>"
        f"</div>", unsafe_allow_html=True)

    # Highlighted big card for the 5-day predictions (larger font for price)
    preds_display = pd.DataFrame(adv['predictions_df'])
    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='background:#071626;padding:12px;border-radius:10px'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#fff;font-size:18px;font-weight:700;margin-bottom:8px'>Projeção de Preço para os Próximos 5 Dias (baseado na data mais recente disponível)</div>", unsafe_allow_html=True)

    for row in preds_display.to_dict(orient='records'):
        date_str = row['Data']
        price = float(row['Preço Previsto'])
        var = float(row['Variação'])
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;padding:8px 6px;border-radius:6px;margin-bottom:6px'>"
            f"<div style='color:#ddd;font-size:16px'>{date_str}</div>"
            f"<div style='color:#00BFFF;font-size:26px;font-weight:900'>R$ {price:,.2f}</div>"
            f"<div style='color:#ddd;font-size:16px'>{var:+.2%}</div>"
            f"</div>", unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # per-model predictions
    per_model_df = pd.DataFrame([{'Modelo':k,'Retorno Estimado':v} for k,v in adv['per_model_return_predictions'].items()])
    if not per_model_df.empty:
        st.subheader("Previsões por modelo (retorno estimado para 5 dias)")
        st.dataframe(per_model_df.style.format({'Retorno Estimado':'{:+.4f}'}), use_container_width=True)

    # Export button
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        preds_df = pd.DataFrame(adv['predictions_df'])
        zf.writestr('predictions.csv', preds_df.to_csv(index=False))
        zf.writestr('metadata.json', json.dumps({
            'timestamp': adv['timestamp'],
            'ticker': adv['ticker'],
            'data_used_period': adv['data_used_period'],
            'rows_historico': adv['rows_historico'],
            'rows_validas': adv['rows_validas'],
            'features_used': adv['features_used'],
            'per_model_return_predictions': adv['per_model_return_predictions'],
            'ensemble_future_return': adv['ensemble_future_return'],
            'confidence': adv['confidence']
        }))
    mem.seek(0)
    st.download_button("Exportar Previsão Avançada (ZIP)", mem.getvalue(), file_name=f"analise_avancada_{ticker_symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.zip", mime="application/zip")

st.markdown("---")

# --- Import / Compare final section ---
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

# --- Footer ---
last_update = pd.to_datetime(data.index[-1]).strftime('%d/%m/%Y')
st.markdown("---")
st.caption(f"Última atualização dos preços: **{last_update}** — Dados: Yahoo Finance.")
st.markdown("<p style='text-align:center;color:#888'>Desenvolvido por Rodrigo Costa de Araujo | rodrigocosta@usp.br</p>", unsafe_allow_html=True)

