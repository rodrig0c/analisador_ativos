# --- Importações das Bibliotecas ---
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# --- Configuração da Página ---
st.set_page_config(page_title="Analisador de Ativos", layout="wide")
st.title('Analisador Interativo de Ativos Financeiros')
st.write('Uma ferramenta para analisar o preço e a volatilidade de ações da B3, usando um modelo de Machine Learning para previsão.')

# --- Barra Lateral para Entradas do Usuário ---
st.sidebar.header('Escolha seus Parâmetros')

### ALTERAÇÃO: Função para buscar os tickers do arquivo CSV local ###
@st.cache_data
def get_tickers_from_csv():
    """
    Busca a lista de tickers e nomes de um arquivo CSV local para testes.
    """
    # --- Para Teste Local ---
    # Certifique-se de que o arquivo 'acoes-listadas-b3.csv' está na mesma pasta que o app.py
    file_path = 'acoes-listadas-b3.csv'
    
    # --- Para Deploy no GitHub (comente a linha acima e descomente a abaixo com sua URL) ---
    # file_path = 'URL_RAW_DO_SEU_ARQUIVO_NO_GITHUB'
    
    try:
        df = pd.read_csv(file_path)
        # Renomeia as colunas para um padrão
        df = df.rename(columns={'Ticker': 'ticker', 'Nome': 'nome'})
        # Cria uma coluna para exibição amigável no menu
        df['display'] = df['nome'] + ' (' + df['ticker'] + ')'
        return df
    except FileNotFoundError:
        st.error(f"Arquivo '{file_path}' não encontrado. Certifique-se de que ele está na mesma pasta que o app.py.")
        # Retorna um DataFrame de fallback para o app não quebrar
        fallback_data = {
            'ticker': ['PETR4', 'VALE3', 'ITUB4'],
            'nome': ['Petrobras', 'Vale', 'Itaú Unibanco'],
        }
        fallback_df = pd.DataFrame(fallback_data)
        fallback_df['display'] = fallback_df['nome'] + ' (' + fallback_df['ticker'] + ')'
        return fallback_df
    except Exception as e:
        st.error(f"Ocorreu um erro ao ler o arquivo CSV. Erro: {e}")
        return pd.DataFrame() # Retorna DataFrame vazio em outros erros

tickers_df = get_tickers_from_csv()

# --- Menu de Seleção Amigável ---
# Usa a coluna 'display' para o usuário, mas vamos extrair o ticker para a busca
selected_display = st.sidebar.selectbox('Escolha a Ação', tickers_df['display'])

# Encontra o ticker correspondente à seleção do usuário
ticker_symbol = tickers_df[tickers_df['display'] == selected_display]['ticker'].iloc[0]
company_name = tickers_df[tickers_df['display'] == selected_display]['nome'].iloc[0]

# Adicionamos .SA para compatibilidade com o yfinance
ticker = f"{ticker_symbol}.SA"

start_date = st.sidebar.date_input('Data de Início', date(2020, 1, 1))
end_date = st.sidebar.date_input('Data de Fim', date.today())

# --- Coleta de Dados com Cache ---
@st.cache_data
def load_data(ticker, start, end):
    """Baixa os dados do yfinance e simplifica os nomes das colunas."""
    data = yf.download(ticker, start, end, progress=False)
    if not data.empty:
        data.columns = data.columns.get_level_values(0)
    return data

data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error("Não foram encontrados dados para o ativo no período selecionado. Por favor, ajuste as datas ou o código da ação.")
else:
    ### NOVO: Dashboard de Métricas Principais ###
    st.subheader('Visão Geral do Ativo')
    
    # Pega os dois últimos dias para calcular a variação
    last_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    price_change = last_price - prev_price
    percent_change = (price_change / prev_price) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Empresa", company_name)
    col2.metric("Ticker", ticker_symbol)
    col3.metric("Último Preço", f"R$ {last_price:.2f}")
    col4.metric("Variação (Dia)", f"{price_change:+.2f} R$", f"{percent_change:+.2f}%")

    st.markdown("---")
    
    # --- Exibindo os Dados ---
    st.subheader(f'Dados Históricos para {ticker}')
    st.dataframe(data.tail())

    # --- Gráfico de Preço de Fechamento ---
    st.subheader('Gráfico de Preço de Fechamento')
    fig_price = px.line(data, x=data.index, y='Close', title=f'Preço de Fechamento de {ticker}')
    st.plotly_chart(fig_price, use_container_width=True)

    # --- Cálculo e Gráfico de Volatilidade ---
    st.subheader('Análise de Volatilidade')
    data['Daily Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Daily Return'].rolling(window=30).std() * (252**0.5) # Anualizada

    fig_vol = px.line(data, x=data.index, y='Volatility', title=f'Volatilidade Anualizada (30 dias) de {ticker}')
    st.plotly_chart(fig_vol, use_container_width=True)

    # --- Resumo e Análise da Volatilidade Atual ---
    st.subheader('Resumo da Volatilidade Atual')
    current_vol = data['Volatility'].iloc[-1]
    vol_q1 = data['Volatility'].quantile(0.25)
    vol_q3 = data['Volatility'].quantile(0.75)
    vol_median = data['Volatility'].median()

    if current_vol < vol_q1:
        status = "Baixa Volatilidade"
        delta_color = "normal"
        help_text = f"A volatilidade atual está abaixo de 25% dos valores históricos (limite: {vol_q1:.3f})."
    elif current_vol > vol_q3:
        status = "Alta Volatilidade"
        delta_color = "inverse"
        help_text = f"A volatilidade atual está acima de 75% dos valores históricos (limite: {vol_q3:.3f})."
    else:
        status = "Média Volatilidade"
        delta_color = "off"
        help_text = f"A volatilidade atual está entre os quartis de 25% e 75% dos valores históricos."

    st.metric(label=f"Status para {ticker_symbol}", value=status, delta=f"Atual: {current_vol:.3f} | Mediana: {vol_median:.3f}", delta_color=delta_color, help=help_text)


    # --- Machine Learning ---
    st.subheader('Previsão de Volatilidade com Machine Learning')
    if st.button('Treinar Modelo e Fazer Previsão'):
        df_model = data[['Volatility']].copy().dropna()
        if len(df_model) < 10:
             st.warning("Não há dados históricos suficientes para treinar o modelo.")
        else:
            for i in range(1, 6):
                df_model[f'vol_lag_{i}'] = df_model['Volatility'].shift(i)
            df_model.dropna(inplace=True)

            X = df_model.drop('Volatility', axis=1)
            y = df_model['Volatility']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            with st.spinner('Treinando o modelo... Isso pode levar um momento.'):
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)

            last_features = X.iloc[-1:].values
            prediction = model.predict(last_features)

            st.success('Modelo treinado com sucesso!')
            next_day = (pd.to_datetime(data.index[-1], dayfirst=True) + pd.Timedelta(days=1)).strftime('%d/%m/%Y')
            st.metric(label=f"Previsão de Volatilidade para {next_day}", value=f"{prediction[0]:.4f}")
            st.info('**Disclaimer:** Este é um modelo educacional simplificado e não deve ser usado como conselho de investimento.')
    
    # --- Nota sobre a Atualização dos Dados ---
    st.markdown("---")
    last_update_date = data.index[-1].strftime('%d/%m/%Y')
    st.caption(f"**Nota sobre os dados:** As análises são baseadas em dados históricos de fechamento diário. "
               f"O último registro utilizado é do dia **{last_update_date}**. "
               f"Os dados são fornecidos pelo Yahoo Finance e podem ter atraso.")

