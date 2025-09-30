# --- Importações das Bibliotecas ---
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# --- Configuração da Página ---
st.set_page_config(page_title="Analisador de Ativos", layout="wide")
st.title('📊 Analisador Interativo de Ativos Financeiros')
st.write('Analise o preço, a volatilidade e os principais indicadores técnicos de ações da B3. '
         'Compare com o IBOVESPA e obtenha previsões avançadas com Machine Learning.')

# --- Barra Lateral ---
st.sidebar.header('⚙️ Parâmetros de Análise')

# ---------- 1. FUNÇÕES AUXILIARES ----------
@st.cache_data
def get_tickers_from_csv():
    """Carrega lista de tickers de CSV local ou fallback."""
    file_path = 'acoes-listadas-b3.csv'
    try:
        df = pd.read_csv(file_path)
        df = df.rename(columns={'Ticker': 'ticker', 'Nome': 'nome'})
        df['display'] = df['nome'] + ' (' + df['ticker'] + ')'
        return df
    except FileNotFoundError:
        st.sidebar.warning("Arquivo CSV não encontrado; usando lista reduzida.")
        fb = {'ticker': ['PETR4', 'VALE3', 'ITUB4', 'MGLU3'],
              'nome': ['Petrobras', 'Vale', 'Itaú Unibanco', 'Magazine Luiza']}
        df = pd.DataFrame(fb)
        df['display'] = df['nome'] + ' (' + df['ticker'] + ')'
        return df

@st.cache_data
def load_clean_data(ticker, start, end):
    """
    Baixa do yfinance, remove duplicatas, fins de semana, linhas NaN,
    garante que sobrem apenas dias úteis com preço/volume reais.
    """
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        return raw

    # achata MultiIndex caso apareça
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # remove duplicatas e fins de semana
    raw = raw[~raw.index.duplicated(keep='last')]
    raw = raw[raw.index.dayofweek < 5]          # segunda a sexta
    raw = raw.dropna(subset=['Close', 'Volume'])

    # exige pelo menos 60 dias úteis para o pipeline ML
    if len(raw) < 60:
        st.warning(f"⚠️ Ticker {ticker} retornou apenas {len(raw)} dias úteis no período.")
        return pd.DataFrame()   # força mensagem de "dados insuficientes"

    return raw

def calculate_indicators(data):
    """Calcula indicadores técnicos."""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    data['MM_Curta'] = data['Close'].rolling(20).mean()
    data['MM_Longa']  = data['Close'].rolling(50).mean()

    data['BB_Media']    = data['Close'].rolling(20).mean()
    std20               = data['Close'].rolling(20).std()
    data['BB_Superior'] = data['BB_Media'] + 2 * std20
    data['BB_Inferior'] = data['BB_Media'] - 2 * std20

    data['Daily Return'] = data['Close'].pct_change()
    data['Volatility']   = data['Daily Return'].rolling(30).std() * np.sqrt(252)
    return data

def prepare_advanced_features(data, lookback_days=60, forecast_days=5):
    """Gera features estendidas e alvos futuros."""
    df = data[['Close', 'Volume', 'RSI', 'MM_Curta', 'MM_Longa', 'Volatility']].copy()

    # features de retorno e volume
    for d in [1, 3, 5, 10, 20]:
        df[f'return_{d}d']     = df['Close'].pct_change(d)
        df[f'volume_ma_{d}d']  = df['Volume'].rolling(d).mean()
        if d <= 20:
            df[f'high_{d}d']   = df['Close'].rolling(d).max()
            df[f'low_{d}d']    = df['Close'].rolling(d).min()
        df[f'volatility_{d}d'] = df['Close'].pct_change().rolling(d).std()

    df['price_vs_ma20'] = df['Close'] / df['MM_Curta']
    df['price_vs_ma50'] = df['Close'] / df['MM_Longa']
    df['ma_cross']      = (df['MM_Curta'] > df['MM_Longa']).astype(int)

    df['target_future_return'] = df['Close'].shift(-forecast_days) / df['Close'] - 1
    df['target_direction']     = (df['target_future_return'] > 0).astype(int)

    # lista final de features
    feat = [c for c in df.columns if c.startswith(('return_', 'volume_ma_', 'high_', 'low_', 'volatility_', 'price_vs_', 'ma_cross'))]
    feat += ['RSI', 'Volatility']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=feat + ['target_future_return', 'target_direction'], inplace=True)
    return df

def create_advanced_model():
    """Ensemble de modelos."""
    return {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
    }

def ensemble_predict(models, X):
    """Média das previsões."""
    return np.mean([m.predict(X) for m in models.values()], axis=0)

# ---------- 2. ENTRADA DO USUÁRIO ----------
tickers_df = get_tickers_from_csv()
selected_display = st.sidebar.selectbox('Escolha a Ação', tickers_df['display'])
ticker_symbol = tickers_df[tickers_df['display'] == selected_display]['ticker'].iloc[0]
company_name  = tickers_df[tickers_df['display'] == selected_display]['nome'].iloc[0]
ticker = f"{ticker_symbol}.SA"

start_date = st.sidebar.date_input("Data de Início", date(2019, 1, 1), format="DD/MM/YYYY")
end_date   = st.sidebar.date_input("Data de Fim",   date.today(),      format="DD/MM/YYYY")

# ---------- 3. CARREGA DADOS LIMPOS ----------
data = load_clean_data(ticker, start_date, end_date)
ibov = load_clean_data('^BVSP', start_date, end_date)

if data.empty:
    st.error("❌ Dados insuficientes ou ticker inválido. Escolha outro ativo ou aumente o período.")
    st.stop()

st.success(f"✅ Dias úteis reais carregados: {len(data)}")
data = calculate_indicators(data)

# ---------- 4. VISÃO GERAL ----------
st.subheader('📈 Visão Geral do Ativo')
last_price  = data['Close'].iloc[-1]
prev_price  = data['Close'].iloc[-2]
price_change = last_price - prev_price
pct_change   = (price_change / prev_price) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("🏢 Empresa", company_name)
col2.metric("💹 Ticker", ticker_symbol)
col3.metric("💰 Último Preço", f"R$ {last_price:.2f}")
col4.metric("📊 Variação (Dia)", f"{price_change:+.2f} R$", f"{pct_change:+.2f}%")
st.markdown("---")

# ---------- 5. ABAS DE GRÁFICOS ----------
tab1, tab2, tab3 = st.tabs(["Preço e Indicadores", "Volatilidade", "Comparativo com IBOVESPA"])

with tab1:
    st.subheader('📉 Preço, Médias Móveis e Bandas de Bollinger')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Preço de Fechamento', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data['MM_Curta'], name='MM 20', line=dict(color='orange', dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['MM_Longa'], name='MM 50', line=dict(color='purple', dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Superior'], name='Banda Sup', line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Inferior'], name='Banda Inf', line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader('📊 RSI')
    fig2 = px.line(data, x=data.index, y='RSI', title='RSI')
    fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrecompra")
    fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobrevenda")
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader('📈 Volatilidade Anualizada')
    current_vol = data['Volatility'].iloc[-1]
    vol_med     = data['Volatility'].median()
    col1, col2 = st.columns([3, 1])
    with col1:
        fig3 = px.line(data, x=data.index, y='Volatility', title='Volatilidade (janela 30 dias)')
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        st.metric("Vol Atual", f"{current_vol:.3f}")
        st.metric("Vol Mediana", f"{vol_med:.3f}")

with tab3:
    st.subheader('🏁 Performance vs IBOVESPA (normalizada)')
    if not ibov.empty:
        comp = pd.DataFrame({
            ticker_symbol: data['Close'] / data['Close'].iloc[0],
            'IBOVESPA': ibov['Close'] / ibov['Close'].iloc[0]
        })
        fig4 = px.line(comp, x=comp.index, y=comp.columns, title='Ação vs IBOVESPA')
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Dados do IBOVESPA não disponíveis.")

st.markdown("---")

# ---------- 6. PREVISÃO AVANÇADA ----------
with st.expander("🔮 Previsão de Preço Avançada (ML)", expanded=True):
    st.write("Previsão 5 dias à frente via ensemble (Random Forest, GBM, SVR, Rede Neural).")
    if st.button('Executar Previsão Avançada'):
        with st.spinner('Processando...'):
            advanced_data = prepare_advanced_features(data, lookback_days=30, forecast_days=5)
            if len(advanced_data) < 50:
                st.warning(f"⚠️ Apenas {len(advanced_data)} dias úteis após processamento. Escolha período maior.")
            else:
                # features e alvos
                feat_cols = [c for c in advanced_data.columns if c.startswith(('return_', 'volume_ma_', 'high_', 'low_', 'volatility_', 'price_vs_', 'ma_cross'))] + ['RSI', 'Volatility']
                X = advanced_data[feat_cols]
                y_ret = advanced_data['target_future_return']
                y_dir = advanced_data['target_direction']

                # split temporal 80/20
                split = int(len(X) * 0.8)
                X_train, X_test = X[:split], X[split:]
                y_train_r, y_test_r = y_ret[:split], y_ret[split:]
                y_train_d, y_test_d = y_dir[:split], y_dir[split:]

                # treina modelos
                models = create_advanced_model()
                trained, preds = {}, {}
                progress = st.progress(0)
                for i, (name, mdl) in enumerate(models.items()):
                    mdl.fit(X_train, y_train_r)
                    trained[name] = mdl
                    preds[name]   = mdl.predict(X_test)
                    progress.progress((i+1)/len(models))

                # ensemble
                ensemble_pred = ensemble_predict(trained, X_test)

                # métricas
                metrics = []
                for name in models:
                    mae = mean_absolute_error(y_test_r, preds[name])
                    rmse= np.sqrt(mean_squared_error(y_test_r, preds[name]))
                    r2  = r2_score(y_test_r, preds[name])
                    acc = accuracy_score(y_test_d, (preds[name] > 0).astype(int))
                    metrics.append({'Modelo': name, 'MAE': mae, 'RMSE': rmse, 'R²': r2, 'Acerto Direção': acc})
                met_df = pd.DataFrame(metrics)
                st.dataframe(met_df.style.format({'MAE': '{:.4f}', 'RMSE': '{:.4f}', 'R²': '{:.4f}', 'Acerto Direção': '{:.2%}'}), use_container_width=True)

                # comparação real vs previsto
                fig5 = go.Figure()
                fig5.add_trace(go.Scatter(x=y_test_r.index, y=y_test_r, name='Real', line=dict(color='blue', width=3)))
                fig5.add_trace(go.Scatter(x=y_test_r.index, y=ensemble_pred, name='Ensemble', line=dict(color='red', dash='dash')))
                fig5.update_layout(title="Real vs Previsão (dados teste)")
                st.plotly_chart(fig5, use_container_width=True)

                # previsão futura 5 dias
                latest = X.iloc[-1:].values
                fut_preds = {n: m.predict(latest)[0] for n, m in trained.items()}
                ensemble_fut = np.mean(list(fut_preds.values()))
                cur_price = data['Close'].iloc[-1]

                fut_table = []
                for d in range(1, 6):
                    ret = ensemble_fut * (d / 5)
                    price = cur_price * (1 + ret)
                    fut_table.append({'Dias': d, 'Data': (data.index[-1] + pd.Timedelta(days=d)).strftime('%d/%m/%Y'), 'Preço Previsto': price, 'Variação %': ret*100})
                fut_df = pd.DataFrame(fut_table)
                st.dataframe(fut_df.style.format({'Preço Previsto': 'R$ {:.2f}', 'Variação %': '{:+.2f}%'}), use_container_width=True)

                # gráfico futuro
                fig6 = go.Figure()
                hist = data['Close'].iloc[-30:]
                fig6.add_trace(go.Scatter(x=hist.index, y=hist, name='Histórico', line=dict(color='blue')))
                fut_dates = [data.index[-1] + pd.Timedelta(days=i) for i in range(1, 6)]
                fut_prices = fut_df['Preço Previsto'].values
                fig6.add_trace(go.Scatter(x=fut_dates, y=fut_prices, name='Previsão', line=dict(color='red', dash='dash'), marker=dict(size=8)))
                fig6.update_layout(title="Previsão 5 dias")
                st.plotly_chart(fig6, use_container_width=True)

                # confiança
                agreement = np.std(list(fut_preds.values()))
                score = max(0, 1 - agreement * 5)
                st.metric("Concordância", f"{(1-agreement)*100:.1f}%")
                st.metric("Score Confiança", f"{score*100:.1f}%")
                st.warning("Disclaimer: previsões são probabilísticas, não garantias.")

# ---------- 7. PREVISÃO DE VOLATILIDADE ----------
with st.expander("🧠 Previsão de Volatilidade (RF tradicional)", expanded=False):
    st.write("Modelo Random Forest para volatilidade 1 dia útil à frente.")
    if st.button('Executar Previsão de Volatilidade'):
        vol_df = data[['Volatility']].dropna()
        if len(vol_df) < 20:
            st.warning("Dados insuficientes para treino.")
        else:
            for i in range(1, 6):
                vol_df[f'vol_lag_{i}'] = vol_df['Volatility'].shift(i)
            vol_df.dropna(inplace=True)
            X_vol = vol_df.drop('Volatility', axis=1)
            y_vol = vol_df['Volatility']
            X_tr, X_te, y_tr, y_te = train_test_split(X_vol, y_vol, test_size=0.2, shuffle=False)
            rf_vol = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_vol.fit(X_tr, y_tr)
            y_pred = rf_vol.predict(X_te)
            st.metric("MAE", f"{mean_absolute_error(y_te, y_pred):.4f}")
            fig7 = go.Figure()
            fig7.add_trace(go.Scatter(x=y_te.index, y=y_te, name='Real'))
            fig7.add_trace(go.Scatter(x=y_te.index, y=y_pred, name='Previsto'))
            fig7.update_layout(title="Vol Real vs Prev (teste)")
            st.plotly_chart(fig7, use_container_width=True)

            # próximo dia
            nxt = rf_vol.predict(X_vol.iloc[-1:].values)[0]
            last_d = data.index[-1]
            nxt_d = last_d + pd.Timedelta(days=1)
            while nxt_d.weekday() >= 5:  # pula sáb/dom
                nxt_d += pd.Timedelta(days=1)
            st.metric(f"Vol prevista {nxt_d.strftime('%d/%m/%Y')}", f"{nxt:.4f}")

# ---------- 8. RODAPÉ ----------
last_upd = data.index[-1].strftime('%d/%m/%Y')
st.markdown("---")
st.caption(f"📅 Última cotação: **{last_upd}** — Fonte Yahoo Finance.")
st.markdown("<p style='text-align:center;color:#888;'>Desenvolvido por Rodrigo Costa de Araujo | rodrigocosta@usp.br</p>", unsafe_allow_html=True)


