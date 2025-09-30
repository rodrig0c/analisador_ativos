# --- Importações ---
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

# --- Configuração da página ---
st.set_page_config(page_title="Analisador de Ativos", layout="wide")
st.title('📊 Analisador Interativo de Ativos Financeiros')
st.write('Análise de preços e previsões. Use com cuidado.')

# --- Sidebar ---
st.sidebar.header('⚙️ Parâmetros')
start_default = date(2019, 1, 1)
start_date = st.sidebar.date_input("Data de Início", start_default, format="DD/MM/YYYY")
end_date = st.sidebar.date_input("Data de Fim", date.today(), format="DD/MM/YYYY")
view_period = st.sidebar.selectbox("Período de visualização", ["Últimos 3 meses", "Últimos 6 meses", "Último 1 ano", "Todo período"], index=2)
view_map = {"Últimos 3 meses": 63, "Últimos 6 meses": 126, "Último 1 ano": 252, "Todo período": None}
viz_days = view_map[view_period]

# --- Helpers ---
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

def confidence_style(score):
    # score in [0,1]. thresholds: >=0.7 high (green), 0.4-0.7 medium (yellow), <0.4 low (red)
    if score >= 0.7:
        return {"color":"#ffffff","bg":"#2ECC71","label":"ALTA CONFIANÇA"}  # green
    if score >= 0.4:
        return {"color":"#000000","bg":"#F1C40F","label":"MÉDIA CONFIANÇA"}  # yellow
    return {"color":"#ffffff","bg":"#E74C3C","label":"BAIXA CONFIANÇA"}      # red

def volatility_style(v):
    # annualized volatility thresholds: <0.25 low (green), 0.25-0.5 medium (yellow), >=0.5 high (red)
    if v >= 0.5:
        return {"color":"#ffffff","bg":"#E74C3C","label":"ALTA VOLATILIDADE"}
    if v >= 0.25:
        return {"color":"#000000","bg":"#F1C40F","label":"VOLATILIDADE MÉDIA"}
    return {"color":"#ffffff","bg":"#2ECC71","label":"BAIXA VOLATILIDADE"}

# --- Carregar tickers e dados ---
tickers_df = get_tickers_from_csv()
selected_display = st.sidebar.selectbox('Escolha a Ação', tickers_df['display'])
ticker_symbol = tickers_df[tickers_df['display'] == selected_display]['ticker'].iloc[0]
company_name = tickers_df[tickers_df['display'] == selected_display]['nome'].iloc[0]
ticker = f"{ticker_symbol}.SA"

data = load_data(ticker, start_date, end_date)
ibov = load_data('^BVSP', start_date, end_date)

if 'advanced_result' not in st.session_state: st.session_state['advanced_result'] = None
if 'vol_result' not in st.session_state: st.session_state['vol_result'] = None

if data.empty or len(data) < 60:
    st.error("❌ Dados insuficientes ou ausentes (mínimo 60 dias). Ajuste as datas ou o ticker.")
    st.stop()

data = calculate_indicators(data)

# --- Cabeçalho e métricas ---
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

# --- Gráficos iniciais: mesma estrutura de antes ---
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
    vol_style = volatility_style(current_vol)
    st.markdown(f"<div style='font-size:18px;padding:10px;border-radius:6px;background:{vol_style['bg']};color:{vol_style['color']};text-align:center'><strong>{vol_style['label']}</strong> — Volatilidade atual: {current_vol:.4f}</div>", unsafe_allow_html=True)

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

# --- Seção: Volatilidade (ML simples) independente abaixo dos gráficos ---
st.subheader('🧠 Volatilidade — Modelo Simples (RandomForest)')
st.write("Modelo simples para prever volatilidade do próximo dia útil. Executar não altera a seção avançada.")
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
        next_day = (last_date + BDay(1))
        next_day_str = next_day.strftime('%d/%m/%Y')
        st.session_state['vol_result'] = {'timestamp': pd.Timestamp.now().isoformat(), 'ticker': ticker_symbol, 'pred_vol': pred_vol, 'date': next_day_str}
# Mostrar resultado simples se existir (não apagará o avançado)
if st.session_state.get('vol_result') is not None:
    vol = st.session_state['vol_result']
    v = vol['pred_vol']
    style = volatility_style(v)
    st.markdown(f"<div style='display:flex;gap:12px;align-items:center'><div style='padding:12px;border-radius:8px;background:{style['bg']};color:{style['color']};font-size:20px;font-weight:700;text-align:center;min-width:260px'>{style['label']}</div><div style='font-size:18px'>Data prevista: <strong>{vol['date']}</strong></div><div style='font-size:18px'>Valor previsto: <strong> {v:.4f}</strong></div></div>", unsafe_allow_html=True)
    # export button (only the button)
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('volatility.json', json.dumps(vol))
        zf.writestr('meta.txt', f"Ticker:{vol['ticker']}\nExport:{vol['timestamp']}\n")
    mem.seek(0)
    st.download_button("Exportar Volatilidade (ZIP)", mem.getvalue(), file_name=f"volatility_{ticker_symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.zip", mime="application/zip")

st.markdown("---")

# --- Seção: Previsão Avançada em bloco próprio ---
st.subheader('🔮 Previsão de Preço Avançada (Machine Learning)')
st.write("Clique em Executar para treinar modelos com dados reais do Yahoo Finance e gerar previsões para os próximos 5 dias úteis.")
if st.button('Executar Previsão de Preço Avançada', key='run_advanced'):
    adv_df, used_features = prepare_advanced_features(data, forecast_days=5)
    if len(adv_df) < 60:
        st.warning(f"Dados insuficientes para análise avançada. Disponíveis: {len(adv_df)} dias.")
    else:
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
        preds = {}
        for name, model in models.items():
            model.fit(X_train_s, y_train)
            trained[name] = model
            preds[name] = model.predict(X_test_s)
        ensemble_pred = ensemble_predict(trained, X_test_s)
        metrics = []
        for name in models.keys():
            mae = mean_absolute_error(y_test, preds[name])
            rmse = np.sqrt(mean_squared_error(y_test, preds[name]))
            r2 = r2_score(y_test, preds[name])
            acc_dir = accuracy_score(ydir_test, (preds[name] > 0).astype(int))
            metrics.append({'Modelo':name,'MAE':mae,'RMSE':rmse,'R²':r2,'Acerto Direção':acc_dir})
        metrics_df = pd.DataFrame(metrics)
        last_feat = X.iloc[-1:].copy()
        last_idx = X.index[-1]
        last_scaled = scaler.transform(last_feat)
        per_model_future = {k: float(m.predict(last_scaled)[0]) for k,m in trained.items()}
        ensemble_future = float(np.mean(list(per_model_future.values())))
        capped = float(np.clip(ensemble_future, -0.5, 0.5))
        daily_rate = (1 + capped)**(1/5) - 1
        cur_price = float(adv_df['Close'].loc[last_idx])
        temp = cur_price
        preds_list = []
        for d in range(1,6):
            temp *= (1+daily_rate)
            fut_date = (pd.to_datetime(last_idx) + BDay(d)).normalize()
            preds_list.append({'Dias':d,'Data':fut_date.strftime('%d/%m/%Y'),'Preço Previsto':temp,'Variação':(temp/cur_price - 1),'Ticker':ticker_symbol})
        preds_df = pd.DataFrame(preds_list)
        arr = np.array(list(per_model_future.values()))
        std_preds = float(np.std(arr))
        max_std = 0.20
        agreement = max(0.0, 1.0 - min(std_preds, max_std)/max_std)
        magnitude_penalty = max(0.0, 1.0 - (abs(ensemble_future)/0.5))
        confidence = agreement * magnitude_penalty
        style = confidence_style(confidence)
        # store
        st.session_state['advanced_result'] = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'ticker': ticker_symbol,
            'used_features': used_features,
            'metrics': metrics_df.to_dict(orient='records'),
            'per_model_return_predictions': per_model_future,
            'ensemble_future_return': ensemble_future,
            'predictions_df': preds_df.to_dict(orient='records'),
            'scaler_mean': getattr(scaler, 'mean_', None).tolist() if hasattr(scaler, 'mean_') else None,
            'scaler_scale': getattr(scaler, 'scale_', None).tolist() if hasattr(scaler, 'scale_') else None,
            'confidence': confidence
        }

# Mostrar resultado avançado se existir (não será apagado pela execução do simples)
if st.session_state.get('advanced_result') is not None:
    adv = st.session_state['advanced_result']
    st.subheader("Resultados - Previsão Avançada")
    st.markdown("**Performance (conjunto de teste)**")
    metrics_df = pd.DataFrame(adv['metrics'])
    st.dataframe(metrics_df.style.format({'MAE':'{:.4f}','RMSE':'{:.4f}','R²':'{:.4f}','Acerto Direção':'{:.2%}'}), use_container_width=True)
    # concordância / confiança / recomendação alinhados, coloridos e fonte maior
    conf = float(adv.get('confidence', 0.0))
    conf_style = confidence_style(conf)
    concordance = max(0.0, (1.0 - np.std(list(adv['per_model_return_predictions'].values())) / 0.20))  # rough normalized
    concord_pct = max(0.0, min(100.0, concordance * 100))
    rec_label = conf_style['label']
    # layout
    st.markdown(f"""
    <div style='display:flex;gap:12px;align-items:stretch'>
      <div style='flex:1;padding:14px;border-radius:8px;background:{conf_style["bg"]};color:{conf_style["color"]};font-size:20px;text-align:center'>
        <div style='font-weight:700'>Concordância</div>
        <div style='font-size:22px;margin-top:6px'>{concord_pct:.1f}%</div>
      </div>
      <div style='flex:1;padding:14px;border-radius:8px;background:{conf_style["bg"]};color:{conf_style["color"]};font-size:20px;text-align:center'>
        <div style='font-weight:700'>Score de Confiança</div>
        <div style='font-size:22px;margin-top:6px'>{conf*100:.1f}%</div>
      </div>
      <div style='flex:1;padding:14px;border-radius:8px;background:{conf_style["bg"]};color:{conf_style["color"]};font-size:20px;text-align:center'>
        <div style='font-weight:700'>Recomendação</div>
        <div style='font-size:22px;margin-top:6px'>{rec_label}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("**Previsão para próximos 5 dias**")
    st.dataframe(pd.DataFrame(adv['predictions_df']).style.format({'Preço Previsto':'R$ {:.2f}','Variação':'{:+.2%}'}), use_container_width=True)
    per_model_df = pd.DataFrame([{'Modelo':k,'Retorno Estimado':v} for k,v in adv['per_model_return_predictions'].items()])
    st.subheader("Previsões por modelo")
    st.dataframe(per_model_df.style.format({'Retorno Estimado':'{:+.4f}'}), use_container_width=True)
    # export button only (no extra title)
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        preds_df = pd.DataFrame(adv['predictions_df'])
        zf.writestr('predictions.csv', preds_df.to_csv(index=False))
        zf.writestr('metrics.json', json.dumps({'timestamp':adv['timestamp'],'ticker':adv['ticker'],'metrics':adv['metrics'],'per_model':adv['per_model_return_predictions'],'ensemble':adv['ensemble_future_return'],'used_features':adv['used_features']}))
        zf.writestr('meta.txt', f"Ticker:{adv['ticker']}\nExport:{adv['timestamp']}\n")
    mem.seek(0)
    st.download_button("Exportar Previsão Avançada (ZIP)", mem.getvalue(), file_name=f"analise_avancada_{ticker_symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.zip", mime="application/zip")

st.markdown("---")

# --- Aba final: Importar e Comparar ---
st.subheader("📂 Importar e Comparar Previsões Exportadas")
uploaded = st.file_uploader("Carregar ZIP de análise exportada por esta ferramenta", type=["zip"])
if uploaded is not None:
    try:
        z = zipfile.ZipFile(io.BytesIO(uploaded.read()))
        if 'predictions.csv' in z.namelist():
            preds = pd.read_csv(io.BytesIO(z.read('predictions.csv')), dtype=str)
            # parse dates flexible (dayfirst True to accept dd/mm/yyyy)
            preds['Data'] = pd.to_datetime(preds['Data'], dayfirst=True, errors='coerce')
            st.write("Predições importadas:")
            st.dataframe(preds, use_container_width=True)
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

# --- Rodapé ---
last_update = pd.to_datetime(data.index[-1]).strftime('%d/%m/%Y')
st.markdown("---")
st.caption(f"Última atualização dos preços: **{last_update}** — Dados: Yahoo Finance.")
st.markdown("<p style='text-align:center;color:#888'>Desenvolvido por Rodrigo Costa de Araujo | rodrigocosta@usp.br</p>", unsafe_allow_html=True)





