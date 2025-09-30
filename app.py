# --- Importações das Bibliotecas ---
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

# --- Configuração da Página ---
st.set_page_config(page_title="Analisador de Ativos", layout="wide")
st.title('📊 Analisador Interativo de Ativos Financeiros')
st.write('Analise o preço, a volatilidade e os principais indicadores técnicos de ações da B3. '
         'Compare com o IBOVESPA e obtenha uma previsão de volatilidade com Machine Learning.')

# --- Barra Lateral ---
st.sidebar.header('⚙️ Parâmetros de Análise')

# --- Funções de Cálculo e Coleta de Dados ---
@st.cache_data
def get_tickers_from_csv():
    """Carrega a lista de tickers de um arquivo CSV local."""
    file_path = 'acoes-listadas-b3.csv'
    try:
        df = pd.read_csv(file_path)
        df = df.rename(columns={'Ticker': 'ticker', 'Nome': 'nome'})
        df['display'] = df['nome'] + ' (' + df['ticker'] + ')'
        return df
    except FileNotFoundError:
        st.sidebar.error(f"Arquivo '{file_path}' não encontrado. Usando lista de fallback.")
        fallback_data = {'ticker': ['PETR4', 'VALE3', 'ITUB4'], 'nome': ['Petrobras', 'Vale', 'Itaú Unibanco']}
        fallback_df = pd.DataFrame(fallback_data)
        fallback_df['display'] = fallback_df['nome'] + ' (' + fallback_df['ticker'] + ')'
        return fallback_df

@st.cache_data
def load_data(ticker, start, end):
    """Baixa os dados do yfinance e simplifica os nomes das colunas."""
    data = yf.download(ticker, start, end, progress=False)
    if not data.empty:
        data.columns = data.columns.get_level_values(0)
    return data

def calculate_indicators(data):
    """Calcula os indicadores técnicos para o DataFrame."""
    # --- CORREÇÃO: Cálculo de RSI com Média Móvel Exponencial (padrão da indústria) ---
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Médias Móveis
    data['MM_Curta'] = data['Close'].rolling(window=20).mean()
    data['MM_Longa'] = data['Close'].rolling(window=50).mean()

    # Bandas de Bollinger
    data['BB_Media'] = data['Close'].rolling(window=20).mean()
    data['BB_Superior'] = data['BB_Media'] + 2 * data['Close'].rolling(window=20).std()
    data['BB_Inferior'] = data['BB_Media'] - 2 * data['Close'].rolling(window=20).std()
    
    # Volatilidade
    data['Daily Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Daily Return'].rolling(window=30).std() * (252**0.5)
    return data

# --- Lógica Principal da Barra Lateral e Coleta de Dados ---
tickers_df = get_tickers_from_csv()

selected_display = st.sidebar.selectbox('Escolha a Ação', tickers_df['display'])
ticker_symbol = tickers_df[tickers_df['display'] == selected_display]['ticker'].iloc[0]
company_name = tickers_df[tickers_df['display'] == selected_display]['nome'].iloc[0]
ticker = f"{ticker_symbol}.SA"

start_date = st.sidebar.date_input("Data de Início", date(2020, 1, 1), format="DD/MM/YYYY")
end_date = st.sidebar.date_input("Data de Fim", date.today(), format="DD/MM/YYYY")

data = load_data(ticker, start_date, end_date)
ibov = load_data('^BVSP', start_date, end_date)

# --- Exibição da Análise ---
if data.empty:
    st.error("❌ Nenhum dado encontrado para o período selecionado. Ajuste as datas ou o código da ação.")
else:
    data = calculate_indicators(data)

    # --- Métricas principais ---
    st.subheader('📈 Visão Geral do Ativo')
    last_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    price_change = last_price - prev_price
    percent_change = (price_change / prev_price) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏢 Empresa", company_name)
    col2.metric("💹 Ticker", ticker_symbol)
    col3.metric("💰 Último Preço", f"R$ {last_price:.2f}")
    col4.metric("📊 Variação (Dia)", f"{price_change:+.2f} R$", f"{percent_change:+.2f}%")

    st.markdown("---")
    
    # --- Abas para Organização dos Gráficos ---
    tab1, tab2, tab3 = st.tabs(["Preço e Indicadores", "Volatilidade", "Comparativo com IBOVESPA"])

    with tab1:
        st.subheader('📉 Preço, Médias Móveis e Bandas de Bollinger')
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Preço de Fechamento', line=dict(color='blue')))
        fig_price.add_trace(go.Scatter(x=data.index, y=data['MM_Curta'], name='Média Móvel 20p', line=dict(color='orange', dash='dash')))
        fig_price.add_trace(go.Scatter(x=data.index, y=data['MM_Longa'], name='Média Móvel 50p', line=dict(color='purple', dash='dash')))
        fig_price.add_trace(go.Scatter(x=data.index, y=data['BB_Superior'], name='Banda Superior', line=dict(color='gray', width=1, dash='dot')))
        fig_price.add_trace(go.Scatter(x=data.index, y=data['BB_Inferior'], name='Banda Inferior', line=dict(color='gray', width=1, dash='dot')))
        st.plotly_chart(fig_price, use_container_width=True)

        st.subheader('📊 Índice de Força Relativa (RSI)')
        fig_rsi = px.line(data, x=data.index, y='RSI', title='RSI (Índice de Força Relativa)')
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrecompra")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobrevenda")
        st.plotly_chart(fig_rsi, use_container_width=True)

    with tab2:
        st.subheader('📈 Análise de Volatilidade')
        current_vol = data['Volatility'].iloc[-1]
        vol_median = data['Volatility'].median()
        
        col1, col2 = st.columns([3, 1])
        with col1:
             fig_vol = px.line(data, x=data.index, y='Volatility', title='Volatilidade Anualizada (30 dias)')
             st.plotly_chart(fig_vol, use_container_width=True)
        with col2:
            st.metric("Volatilidade Atual", f"{current_vol:.3f}")
            st.metric("Volatilidade Mediana", f"{vol_median:.3f}")

    with tab3:
        st.subheader('🏁 Comparativo com o IBOVESPA')
        comp_df = pd.DataFrame({
            ticker_symbol: data['Close'] / data['Close'].iloc[0],
            'IBOVESPA': ibov['Close'] / ibov['Close'].iloc[0]
        })
        fig_comp = px.line(comp_df, x=comp_df.index, y=comp_df.columns, title='Performance Normalizada: Ação vs IBOVESPA')
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    
    # --- MELHORIA: Seção de Machine Learning mais Profissional ---
    with st.expander("🧠 Análise Preditiva com Machine Learning", expanded=True):
        st.write("""
        Esta seção utiliza um modelo de Machine Learning (Random Forest) para prever a volatilidade do ativo no próximo dia útil. 
        O modelo é treinado com base na volatilidade dos 5 dias anteriores.
        """)

        if st.button('Executar Análise Preditiva'):
            df_model = data[['Volatility']].copy().dropna()
            if len(df_model) < 20: # Aumenta o requisito mínimo de dados
                st.warning("⚠️ Dados históricos insuficientes para treinar e avaliar o modelo de forma confiável.")
            else:
                for i in range(1, 6):
                    df_model[f'vol_lag_{i}'] = df_model['Volatility'].shift(i)
                df_model.dropna(inplace=True)

                X = df_model.drop('Volatility', axis=1)
                y = df_model['Volatility']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                with st.spinner('Treinando o modelo...'):
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    model.fit(X_train, y_train)
                
                # --- Avaliação do Modelo ---
                st.subheader("Avaliando a Performance do Modelo")
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)

                col1, col2 = st.columns(2)
                col1.metric("Erro Médio Absoluto (MAE)", f"{mae:.4f}", help="Indica o erro médio das previsões do modelo no período de teste.")
                
                fig_eval = go.Figure()
                fig_eval.add_trace(go.Scatter(x=y_test.index, y=y_test, name='Volatilidade Real', line=dict(color='blue')))
                fig_eval.add_trace(go.Scatter(x=y_test.index, y=y_pred, name='Previsão do Modelo', line=dict(color='red', dash='dash')))
                fig_eval.update_layout(title="Comparativo: Volatilidade Real vs. Previsão do Modelo (Dados de Teste)")
                st.plotly_chart(fig_eval, use_container_width=True)

                # --- Previsão Final ---
                st.subheader("Previsão para o Próximo Dia")
                prediction = model.predict(X.iloc[-1:].values)
                next_day = (data.index[-1] + pd.Timedelta(days=1)).strftime('%d/%m/%Y')
                
                st.metric(label=f"📅 Previsão de Volatilidade para {next_day}", value=f"{prediction[0]:.4f}")
                st.info('**Disclaimer:** Este modelo é apenas para fins educacionais e não constitui uma recomendação de investimento.')

    # --- Nota de atualização ---
    last_update_date = data.index[-1].strftime('%d/%m/%Y')
    st.markdown("---")
    st.caption(f"📅 Última atualização dos preços: **{last_update_date}** — Dados fornecidos pelo Yahoo Finance (podem ter atraso).")
