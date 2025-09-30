
# app.py
import streamlit as st

st.title('Analisador Interativo de Ativos Financeiros')
st.write('Bem-vindo à sua ferramenta de análise de ações!')

# app.py
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date

st.title('Analisador Interativo de Ativos Financeiros')

# --- Barra Lateral para Entradas do Usuário ---
st.sidebar.header('Escolha seus Parâmetros')
ticker = st.sidebar.text_input('Código da Ação (ex: PETR4.SA)', 'PETR4.SA').upper()
start_date = st.sidebar.date_input('Data de Início', date(2020, 1, 1))
end_date = st.sidebar.date_input('Data de Fim', date.today())

# --- Coleta de Dados com Cache ---
@st.cache_data
def load_data(ticker, start, end):
	data = yf.download(ticker, start, end)
	return data

data = load_data(ticker, start_date, end_date)

# --- Exibindo os Dados ---
st.subheader(f'Dados Brutos para {ticker}')
st.dataframe(data.tail())

# Adicione no final do app.py
import plotly.express as px

# --- Gráfico de Preço de Fechamento ---
st.subheader('Gráfico de Preço de Fechamento')
fig_price = px.line(data, x=data.index, y='Adj Close', title=f'Preço de Fechamento de {ticker}')
st.plotly_chart(fig_price, use_container_width=True)

# --- Cálculo e Gráfico de Volatilidade ---
st.subheader('Análise de Volatilidade')
data['Daily Return'] = data['Adj Close'].pct_change()
# Usamos uma janela móvel de 30 dias para a volatilidade
data['Volatility'] = data['Daily Return'].rolling(window=30).std() * (252**0.5) # Anualizada

fig_vol = px.line(data, x=data.index, y='Volatility', title=f'Volatilidade Anualizada (30 dias) de {ticker}')
st.plotly_chart(fig_vol, use_container_width=True)


# Adicione no final do app.py
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

st.subheader('Previsão de Volatilidade com Machine Learning')

if st.button('Treinar Modelo e Fazer Previsão'):
	# 1. Engenharia de Features Simples
	df_model = data[['Volatility']].copy().dropna()
	# Criamos features baseadas na volatilidade passada (lags)
	for i in range(1, 6):
		df_model[f'vol_lag_{i}'] = df_model['Volatility'].shift(i)
	df_model.dropna(inplace=True)
	
	# 2. Preparação dos Dados
	X = df_model.drop('Volatility', axis=1)
	y = df_model['Volatility']
	
	# Usamos os últimos 20% dos dados para teste
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
	
	# 3. Treinamento do Modelo
	with st.spinner('Treinando o modelo... Isso pode levar um momento.'):
		model = RandomForestRegressor(n_estimators=100, random_state=42)
		model.fit(X_train, y_train)
		
	# 4. Fazendo a "previsão" para o próximo dia
	last_features = X.iloc[-1:].values
	prediction = model.predict(last_features)
	
	st.success('Modelo treinado com sucesso!')
	st.write('**Previsão de Volatilidade para o próximo dia:**')
	st.metric(label="Volatilidade Prevista", value=f"{prediction[0]:.4f}")
	
	st.info('**Disclaimer:** Este é um modelo educacional simplificado e não deve ser usado como conselho de investimento.')