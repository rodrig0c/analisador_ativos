# --- Importações das Bibliotecas ---
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date, datetime
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# --- Configuração da Página ---
st.set_page_config(page_title="Analisador de Ativos", layout="wide")
st.title('📊 Analisador Interativo de Ativos Financeiros')
st.write('Analise o preço, a volatilidade e os principais indicadores técnicos de ações da B3. '
         'Compare com o IBOVESPA e obtenha uma previsão de volatilidade com Machine Learning.')

# --- Barra Lateral ---
st.sidebar.header('⚙️ Parâmetros de Análise')

# Ajuste para datas no formato brasileiro
def date_input_br(label, default):
    return st.sidebar.date_input(label, default, format="DD/MM/YYYY")

# --- Função para buscar tickers ---
@st.cache_data
def get_tickers_from_csv():
    file_path = 'acoes-listadas-b3.csv'
    try:
        df = pd.read_csv(file_path)
        df = df.rename(columns={'Ticker': 'ticker', 'Nome': 'nome'})
        df['display'] = df['nome'] + ' (' + df['ticker'] + ')'
        return df
    except FileNotFoundError:
        fallback_data = {
            'ticker': ['PETR4', 'VALE3', 'ITUB4'],
            'nome': ['Petrobras', 'Vale', 'Itaú Unibanco'],
        }
        fallback_df = pd.DataFrame(fallback_data)
        fallback_df['display'] = fallback_df['nome'] + ' (' + fallback_df['ticker'] + ')'
        return fallback_df

tickers_df = get_tickers_from_csv()

# --- Seleção do Ativo ---
selected_display = st.sidebar.selectbox('Escolha a Ação', tickers_df['display'])
ticker_symbol = tickers_df[tickers_df['display'] == selected_display]['ticker'].iloc[0]
company_name = tickers_df[tickers_df['display'] == selected_display]['nome'].iloc[0]
ticker = f"{ticker_symbol}.SA"

# Datas em formato brasileiro
start_date = date_input_br('Data de Início', date(2020, 1, 1))
end_date = date_input_br('Data de Fim', date.today())

# --- Coleta de Dados com Cache ---
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start, end, progress=False)
    if not data.empty:
        data.columns = data.columns.get_level_values(0)
    return data

data = load_data(ticker, start_date, end_date)
ibov = load_data('^BVSP', start_date, end_date)

# --- Verificação de Dados ---
if data.empty:
    st.error("❌ Nenhum dado encontrado para o período selecionado. Ajuste as datas ou o código da ação.")
else:
    # --- Converter datas ---
    data.index = pd.to_datetime(data.index)
    ibov.index = pd.to_datetime(ibov.index)

    # --- Cálculo de Indicadores Técnicos ---
    st.markdown("### 📘 Indicadores Técnicos Explicados")
    st.info(
        "- **RSI (Índice de Força Relativa):** Mede a força da tendência. RSI > 70 indica sobrecompra, RSI < 30 indica sobrevenda.\n"
        "- **Médias Móveis (Curta e Longa):** Ajudam a identificar tendências (cruzamento indica reversão de tendência).\n"
        "- **Bandas de Bollinger:** Mostram a volatilidade. Quando o preço toca as bandas, pode indicar movimentos extremos."
    )

    # RSI
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Médias Móveis
    data['MM_Curta'] = data['Close'].rolling(window=20).mean()
    data['MM_Longa'] = data['Close'].rolling(window=50).mean()

    # Bandas de Bollinger
    data['BB_Media'] = data['Close'].rolling(window=20).mean()
    data['BB_Superior'] = data['BB_Media'] + 2 * data['Close'].rolling(window=20).std()
    data['BB_Inferior'] = data['BB_Media'] - 2 * data['Close'].rolling(window=20).std()

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

    # --- Gráfico de Preço com Indicadores ---
    st.subheader('📉 Preço com Médias Móveis e Bandas de Bollinger')
    fig_price = px.line(data, x=data.index, y=['Close', 'MM_Curta', 'MM_Longa'],
                        labels={'value': 'Preço (R$)', 'variable': 'Indicador'},
                        title=f'{company_name} ({ticker_symbol}) - Preço e Médias Móveis')
    fig_price.add_scatter(x=data.index, y=data['BB_Superior'], mode='lines', name='Banda Superior', line=dict(width=1, dash='dot'))
    fig_price.add_scatter(x=data.index, y=data['BB_Inferior'], mode='lines', name='Banda Inferior', line=dict(width=1, dash='dot'))
    fig_price.update_xaxes(dtick="M2", tickformat="%d/%m/%Y")
    st.plotly_chart(fig_price, use_container_width=True)

    # --- RSI ---
    st.subheader('📊 Índice de Força Relativa (RSI)')
    fig_rsi = px.line(data, x=data.index, y='RSI', title='RSI (Índice de Força Relativa)', labels={'RSI': 'RSI', 'Date': 'Data'})
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    st.plotly_chart(fig_rsi, use_container_width=True)

    # --- Comparativo com IBOV ---
    st.subheader('🏁 Comparativo com o IBOVESPA')
    comp_df = pd.DataFrame({
        'Ação': data['Close'] / data['Close'].iloc[0] * 100,
        'IBOVESPA': ibov['Close'] / ibov['Close'].iloc[0] * 100
    })
    fig_comp = px.line(comp_df, x=comp_df.index, y=['Ação', 'IBOVESPA'],
                       labels={'value': 'Performance (%)', 'variable': 'Ativo', 'index': 'Data'},
                       title='Comparativo de Performance: Ação vs IBOVESPA')
    st.plotly_chart(fig_comp, use_container_width=True)

    # --- Volatilidade ---
    st.subheader('📈 Análise de Volatilidade')
    data['Daily Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Daily Return'].rolling(window=30).std() * (252**0.5)

    current_vol = data['Volatility'].iloc[-1]
    vol_q1 = data['Volatility'].quantile(0.25)
    vol_q3 = data['Volatility'].quantile(0.75)
    vol_median = data['Volatility'].median()

    if current_vol < vol_q1:
        status = "🟢 Baixa Volatilidade"
        color = "#00C853"
    elif current_vol > vol_q3:
        status = "🔴 Alta Volatilidade"
        color = "#D50000"
    else:
        status = "🟡 Média Volatilidade"
        color = "#FFD600"

    st.markdown(
        f"""
        <div style='padding:15px; border-radius:10px; background-color:{color}; color:black; font-weight:bold; text-align:center'>
            {status}<br>
            <small>Atual: {current_vol:.3f} | Mediana: {vol_median:.3f}</small>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Machine Learning ---
    st.subheader('🤖 Previsão de Volatilidade com Machine Learning')
    if st.button('Treinar Modelo e Fazer Previsão'):
        df_model = data[['Volatility']].copy().dropna()
        if len(df_model) < 10:
            st.warning("⚠️ Dados insuficientes para treinar o modelo.")
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

            prediction = model.predict(X.iloc[-1:].values)
            next_day = (data.index[-1] + pd.Timedelta(days=1)).strftime('%d/%m/%Y')
            st.success('✅ Modelo treinado com sucesso!')
            st.metric(label=f"📅 Previsão de Volatilidade para {next_day}", value=f"{prediction[0]:.4f}")
            st.info('**Disclaimer:** Este modelo é apenas educacional e não constitui recomendação de investimento.')

    # --- Nota de atualização ---
    last_update_date = data.index[-1].strftime('%d/%m/%Y')
    st.markdown("---")
    st.caption(f"📅 Última atualização dos preços: **{last_update_date}** — Dados do Yahoo Finance (podem ter atraso).")
