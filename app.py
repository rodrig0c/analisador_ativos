# --- Importações das Bibliotecas ---
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

# --- Configuração da Página ---
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

# --- Helpers: coleta e indicadores ---
@st.cache_data
def get_tickers_from_csv():
    file_path = 'acoes-listadas-b3.csv'
    try:
        df = pd.read_csv(file_path)
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

# --- UI: seleção de ticker e carregamento ---
tickers_df = get_tickers_from_csv()
selected_display = st.sidebar.selectbox('Escolha a Ação', tickers_df['display'])
ticker_symbol = tickers_df[tickers_df['display'] == selected_display]['ticker'].iloc[0]
company_name = tickers_df[tickers_df['display'] == selected_display]['nome'].iloc[0]
ticker = f"{ticker_symbol}.SA"

data = load_data(ticker, start_date, end_date)
ibov = load_data('^BVSP', start_date, end_date)

# session state
if 'advanced_result' not in st.session_state: st.session_state['advanced_result'] = None
if 'vol_result' not in st.session_state: st.session_state['vol_result'] = None

# --- Validação mínima ---
if data.empty or len(data) < 60:
    st.error("❌ Dados insuficientes ou ausentes (mínimo 60 dias). Ajuste as datas ou ticker.")
    st.stop()

data = calculate_indicators(data)

# --- Top: Gráficos iniciais (como antes) ---
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

tab1, tab2, tab3 = st.tabs(["Preço e Indicadores", "Volatilidade", "Comparativo com IBOVESPA"])

# view slice
if viz_days is None:
    view_slice = slice(None)
else:
    view_slice = slice(-viz_days, None)

with tab1:
    st.subheader('Preço, Médias e Bandas de Bollinger')
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

# --- Seção: Volatilidade (ML simples) - independente e abaixo dos gráficos ---
st.subheader('🧠 Volatilidade — Modelo Simples (RandomForest)')
st.write("Modelo simples para prever volatilidade do próximo dia útil. Executar não altera os gráficos acima.")
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
        last_date = pd.to_datetime(data.index[-1])
        next_day = (last_date + BDay(1)).strftime('%d/%m/%Y')
        st.markdown(f"**Previsão para {next_day}**: {pred_vol:.4f}")
        if pred_vol < 0.30:
            st.success("Baixa Volatilidade")
        elif pred_vol >= 0.60:
            st.error("Alta Volatilidade")
        else:
            st.info("Volatilidade Média")
        # store result for export
        st.session_state['vol_result'] = {'timestamp': pd.Timestamp.now().isoformat(), 'ticker': ticker_symbol, 'pred_vol': pred_vol, 'date': next_day}

# export button for simple volatility (always visible after run)
if st.session_state.get('vol_result') is not None:
    st.markdown("**Exportar Análise de Volatilidade Simples**")
    vol = st.session_state['vol_result']
    if st.button("Exportar Volatilidade (ZIP)"):
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('volatility.json', json.dumps(vol))
            zf.writestr('meta.txt', f"Ticker:{vol['ticker']}\nExport:{vol['timestamp']}\n")
        mem.seek(0)
        st.download_button("Baixar ZIP - Volatilidade", mem.getvalue(), file_name=f"volatility_{ticker_symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.zip", mime="application/zip")

st.markdown("---")

# --- Seção: Previsão Avançada (ML avançado) em bloco próprio abaixo ---
st.subheader('🔮 Previsão de Preço Avançada (Machine Learning)')
st.write("Clique em Executar para treinar. Usa apenas dados reais baixados do Yahoo Finance no intervalo selecionado.")
if st.button('Executar Previsão de Preço Avançada'):
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
        progress = st.progress(0)
        status = st.empty()
        for i,(name,model) in enumerate(models.items()):
            status.text(f"Treinando {name}...")
            model.fit(X_train_s, y_train)
            trained[name] = model
            preds[name] = model.predict(X_test_s)
            progress.progress(int(((i+1)/len(models))*100))
        status.text("Treinamento concluído")
        ensemble_pred = ensemble_predict(trained, X_test_s)

        # métricas
        metrics = []
        for name in models.keys():
            mae = mean_absolute_error(y_test, preds[name])
            rmse = np.sqrt(mean_squared_error(y_test, preds[name]))
            r2 = r2_score(y_test, preds[name])
            acc_dir = accuracy_score(ydir_test, (preds[name] > 0).astype(int))
            metrics.append({'Modelo':name,'MAE':mae,'RMSE':rmse,'R²':r2,'Acerto Direção':acc_dir})
        metrics_df = pd.DataFrame(metrics)
        st.subheader("Performance (conjunto de teste)")
        st.dataframe(metrics_df.style.format({'MAE':'{:.4f}','RMSE':'{:.4f}','R²':'{:.4f}','Acerto Direção':'{:.2%}'}), use_container_width=True)

        # previsão futura usando última linha válida
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
            preds_list.append({'Dias':d,'Data':fut_date.strftime('%Y-%m-%d'),'Preço Previsto':temp,'Variação':(temp/cur_price - 1),'Ticker':ticker_symbol})
        preds_df = pd.DataFrame(preds_list)
        st.subheader("Previsão para próximos 5 dias")
        st.dataframe(preds_df.style.format({'Preço Previsto':'R$ {:.2f}','Variação':'{:+.2%}'}), use_container_width=True)

        # confiança
        arr = np.array(list(per_model_future.values()))
        std_preds = float(np.std(arr))
        max_std = 0.20
        agreement = max(0.0, 1.0 - min(std_preds, max_std)/max_std)
        magnitude_penalty = max(0.0, 1.0 - (abs(ensemble_future)/0.5))
        confidence = agreement * magnitude_penalty
        concord_pct = agreement*100
        conf_pct = confidence*100
        st.columns(3)[0].metric("Concordância entre Modelos", f"{concord_pct:.1f}%")
        st.columns(3)[1].metric("Score de Confiança", f"{conf_pct:.1f}%")
        rec = "BAIXA CONFIANÇA"
        if confidence > 0.75: rec = "ALTA CONFIANÇA"
        elif confidence > 0.5: rec = "MÉDIA CONFIANÇA"
        st.columns(3)[2].metric("Recomendação", rec)

        st.subheader("Previsões por modelo (retorno estimado para 5 dias)")
        per_model_df = pd.DataFrame([{'Modelo':k,'Retorno Estimado':v} for k,v in per_model_future.items()])
        st.dataframe(per_model_df.style.format({'Retorno Estimado':'{:+.4f}'}), use_container_width=True)

        st.warning("Previsões são probabílisticas. Não constituem recomendação de investimento.")

        # armazenar resultado para export
        st.session_state['advanced_result'] = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'ticker': ticker_symbol,
            'used_features': used_features,
            'metrics': metrics_df.to_dict(orient='records'),
            'per_model_return_predictions': per_model_future,
            'ensemble_future_return': ensemble_future,
            'predictions_df': preds_df.to_dict(orient='records'),
            'scaler_mean': scaler.mean_.tolist() if hasattr(scaler,'mean_') else None,
            'scaler_scale': scaler.scale_.tolist() if hasattr(scaler,'scale_') else None
        }

# export button for advanced (always visible after run)
if st.session_state.get('advanced_result') is not None:
    st.markdown("**Exportar Análise Avançada**")
    adv = st.session_state['advanced_result']
    if st.button("Exportar Previsão Avançada (ZIP)"):
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            preds_df = pd.DataFrame(adv['predictions_df'])
            zf.writestr('predictions.csv', preds_df.to_csv(index=False))
            zf.writestr('metrics.json', json.dumps({'timestamp':adv['timestamp'],'ticker':adv['ticker'],'metrics':adv['metrics'],'per_model':adv['per_model_return_predictions'],'ensemble':adv['ensemble_future_return'],'used_features':adv['used_features']}))
            zf.writestr('meta.txt', f"Ticker:{adv['ticker']}\nExport:{adv['timestamp']}\n")
        mem.seek(0)
        st.download_button("Baixar ZIP - Análise Avançada", mem.getvalue(), file_name=f"analise_avancada_{ticker_symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.zip", mime="application/zip")

st.markdown("---")

# --- Aba final: Importar e Comparar (sempre independente e no final) ---
st.subheader("📂 Importar e Comparar Previsões Exportadas")
uploaded = st.file_uploader("Carregar ZIP de análise exportada por esta ferramenta", type=["zip"])
if uploaded is not None:
    try:
        z = zipfile.ZipFile(io.BytesIO(uploaded.read()))
        if 'predictions.csv' in z.namelist():
            preds = pd.read_csv(io.BytesIO(z.read('predictions.csv')))
            st.write("Predições importadas:")
            st.dataframe(preds, use_container_width=True)
            if st.button("Comparar com preços reais (Yahoo Finance)"):
                min_date = pd.to_datetime(preds['Data'].min())
                max_date = pd.to_datetime(preds['Data'].max())
                df_actual = yf.download(f"{preds['Ticker'].iloc[0]}.SA", start=(min_date - pd.Timedelta(days=3)).strftime('%Y-%m-%d'), end=(max_date + pd.Timedelta(days=3)).strftime('%Y-%m-%d'), progress=False)
                if df_actual.empty:
                    st.error("Não foi possível baixar preços reais para as datas requeridas.")
                else:
                    df_actual.index = pd.to_datetime(df_actual.index).normalize()
                    rows = []
                    for _, row in preds.iterrows():
                        pred_date = pd.to_datetime(row['Data']).normalize()
                        actual_price = None
                        if pred_date in df_actual.index:
                            actual_price = float(df_actual.loc[pred_date,'Close'])
                        else:
                            for k in range(1,6):
                                try_date = (pred_date + pd.Timedelta(days=k))
                                if try_date in df_actual.index:
                                    actual_price = float(df_actual.loc[try_date,'Close'])
                                    break
                        rows.append({'Data Prevista':pred_date.strftime('%Y-%m-%d'),'Preço Previsto':float(row['Preço Previsto']),'Preço Real (fechamento próximo disponível)':actual_price})
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
                # tentar buscar volatilidade real aproximada com mesma data: calcular volatilidade da série
                try:
                    date_to_check = pd.to_datetime(vol['date'])
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

# rodapé
last_update = pd.to_datetime(data.index[-1]).strftime('%d/%m/%Y')
st.markdown("---")
st.caption(f"Última atualização dos preços: **{last_update}** — Dados: Yahoo Finance.")
st.markdown("<p style='text-align:center;color:#888'>Desenvolvido por Rodrigo Costa de Araujo | rodrigocosta@usp.br</p>", unsafe_allow_html=True)




