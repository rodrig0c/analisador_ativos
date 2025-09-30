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
        fallback_data = {'ticker': ['PETR4', 'VALE3', 'ITUB4', 'MGLU3'], 'nome': ['Petrobras', 'Vale', 'Itaú Unibanco', 'Magazine Luiza']}
        fallback_df = pd.DataFrame(fallback_data)
        fallback_df['display'] = fallback_df['nome'] + ' (' + fallback_df['ticker'] + ')'
        return fallback_df

@st.cache_data
def load_data(ticker, start, end):
    """Baixa os dados do yfinance e simplifica os nomes das colunas."""
    data = yf.download(ticker, start, end, progress=False)
    if not data.empty:
        if 'Volume' not in data.columns:
            data['Volume'] = 0
        data.columns = data.columns.get_level_values(0)
    return data

def calculate_indicators(data):
    """Calcula os indicadores técnicos para o DataFrame."""
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['MM_Curta'] = data['Close'].rolling(window=20).mean()
    data['MM_Longa'] = data['Close'].rolling(window=50).mean()
    data['BB_Media'] = data['Close'].rolling(window=20).mean()
    data['BB_Superior'] = data['BB_Media'] + 2 * data['Close'].rolling(window=20).std()
    data['BB_Inferior'] = data['BB_Media'] - 2 * data['Close'].rolling(window=20).std()
    data['Daily Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Daily Return'].rolling(window=30).std() * (252**0.5)
    return data

# ##################################################################################
# ### ALTERAÇÃO CRÍTICA 1: Lógica de Limpeza Robusta na Preparação dos Dados     ###
# ##################################################################################
def prepare_advanced_features(data, forecast_days=5):
    """
    Prepara features com lógica robusta para lidar com dados ausentes (como volume).
    """
    df = data[['Close', 'Volume', 'RSI', 'MM_Curta', 'MM_Longa', 'Volatility']].copy()
    periods = [1, 3, 5, 10, 20]

    # Criação de features baseadas em Retorno, Máximas, Mínimas e Volatilidade
    for days in periods:
        df[f'return_{days}d'] = df['Close'].pct_change(days)
        df[f'high_{days}d'] = df['Close'].rolling(window=days).max()
        df[f'low_{days}d'] = df['Close'].rolling(window=days).min()
        df[f'volatility_{days}d'] = df['Close'].pct_change().rolling(window=days).std()

    # Cria features de volume APENAS se houver dados de volume válidos
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        for days in periods:
            df[f'volume_ma_{days}d'] = df['Volume'].rolling(window=days).mean()

    # Features técnicas avançadas
    df['price_vs_ma20'] = df['Close'] / df['MM_Curta']
    df['price_vs_ma50'] = df['Close'] / df['MM_Longa']
    df['ma_cross'] = (df['MM_Curta'] > df['MM_Longa']).astype(int)
    
    # Alvo (Target)
    df['target_future_return'] = df['Close'].shift(-forecast_days) / df['Close'] - 1
    df['target_direction'] = (df['target_future_return'] > 0).astype(int)

    # Lógica de limpeza robusta
    potential_feature_columns = [col for col in df.columns if col.startswith(
        ('return_', 'volume_ma_', 'high_', 'low_', 'volatility_', 'price_vs_', 'ma_cross'))]
    potential_feature_columns.extend(['RSI', 'Volatility'])
    
    feature_columns = [col for col in potential_feature_columns if col in df.columns and not df[col].isnull().all()]
    target_columns = ['target_future_return', 'target_direction']
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=feature_columns + target_columns, inplace=True)

    return df, feature_columns
# ##################################################################################
# ### FIM DA ALTERAÇÃO 1                                                         ###
# ##################################################################################

def create_advanced_model():
    """Cria ensemble de modelos"""
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
    }
    return models

def ensemble_predict(models, X):
    """Combina previsões de múltiplos modelos"""
    predictions = [model.predict(X) for name, model in models.items()]
    return np.mean(predictions, axis=0)

# --- Lógica Principal da Barra Lateral e Coleta de Dados ---
tickers_df = get_tickers_from_csv()
selected_display = st.sidebar.selectbox('Escolha a Ação', tickers_df['display'])
ticker_symbol = tickers_df[tickers_df['display'] == selected_display]['ticker'].iloc[0]
company_name = tickers_df[tickers_df['display'] == selected_display]['nome'].iloc[0]
ticker = f"{ticker_symbol}.SA"
start_date = st.sidebar.date_input("Data de Início", date(2019, 1, 1), format="DD/MM/YYYY")
end_date = st.sidebar.date_input("Data de Fim", date.today(), format="DD/MM/YYYY")

data = load_data(ticker, start_date, end_date)
ibov = load_data('^BVSP', start_date, end_date)

# --- Exibição da Análise ---
if data.empty or len(data) < 60:
    st.error("❌ Nenhum dado encontrado ou dados insuficientes para o período selecionado (mínimo de 60 dias). Ajuste as datas ou o código da ação.")
else:
    data = calculate_indicators(data)

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

    tab1, tab2, tab3 = st.tabs(["Preço e Indicadores", "Volatilidade", "Comparativo com IBOVESPA"])
    with tab1:
        st.subheader('📉 Preço, Médias Móveis e Bandas de Bollinger')
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Preço de Fechamento', line=dict(color='blue')))
        fig_price.add_trace(go.Scatter(x=data.index, y=data['MM_Curta'], name='Média Móvel 20p', line=dict(color='orange', dash='dash')))
        fig_price.add_trace(go.Scatter(x=data.index, y=data['MM_Longa'], name='Média Móvel 50p', line=dict(color='purple', dash='dash')))
        fig_price.add_trace(go.Scatter(x=data.index, y=data['BB_Superior'], name='Banda Superior', line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))
        fig_price.add_trace(go.Scatter(x=data.index, y=data['BB_Inferior'], name='Banda Inferior', line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))
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
             fig_vol = px.line(data, x=data.index, y='Volatility', title='Volatilidade Anualizada (janela de 30 dias)')
             st.plotly_chart(fig_vol, use_container_width=True)
        with col2:
            st.metric("Volatilidade Atual", f"{current_vol:.3f}")
            st.metric("Volatilidade Mediana", f"{vol_median:.3f}")
    with tab3:
        st.subheader('🏁 Comparativo com o IBOVESPA')
        if not ibov.empty:
            comp_df = pd.DataFrame({'IBOVESPA': ibov['Close'] / ibov['Close'].iloc[0], ticker_symbol: data['Close'] / data['Close'].iloc[0]})
            fig_comp = px.line(comp_df, x=comp_df.index, y=comp_df.columns, title='Performance Normalizada: Ação vs IBOVESPA')
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.warning("Não foi possível carregar os dados do IBOVESPA para comparação.")
    
    st.markdown("---")
    
    with st.expander("🔮 Previsão de Preço Avançada (Machine Learning)", expanded=True):
        st.write("""
        **Previsão para os próximos 5 dias usando múltiplos algoritmos de ML**
        - Features: Preço, Volume, RSI, Médias Móveis, Volatilidade
        - Modelos: Random Forest, Gradient Boosting, SVR, Neural Network
        """)
        
        if st.button('Executar Previsão de Preço Avançada'):
            with st.spinner('Processando dados e treinando modelos...'):
                advanced_data, used_features = prepare_advanced_features(data, forecast_days=5)
                if len(advanced_data) < 50:
                    st.warning(f"⚠️ Dados insuficientes para análise avançada. Disponíveis: {len(advanced_data)} dias.")
                    st.info(f"💡 Dica: Selecione um período de datas mais longo.")
                else:
                    X = advanced_data[used_features]
                    y_return = advanced_data['target_future_return']
                    y_direction = advanced_data['target_direction']
                    st.info(f"📊 Dados disponíveis para treinamento: {len(X)} dias úteis. Features utilizadas: {len(used_features)}.")
                    split_idx = int(len(X) * 0.8)
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train_return, y_test_return = y_return[:split_idx], y_return[split_idx:]
                    y_train_dir, y_test_dir = y_direction[:split_idx], y_direction[split_idx:]
                    models = create_advanced_model()
                    trained_models, return_predictions = {}, {}
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    for i, (name, model) in enumerate(models.items()):
                        status_text.text(f"Treinando {name}...")
                        model.fit(X_train, y_train_return)
                        trained_models[name] = model
                        return_predictions[name] = model.predict(X_test)
                        progress_bar.progress((i + 1) / len(models))
                    status_text.text("Treinamento concluído!")
                    ensemble_pred = ensemble_predict(trained_models, X_test)
                    
                    st.subheader("📊 Performance dos Modelos (Dados de Teste)")
                    metrics_data = []
                    for name in models.keys():
                        metrics_data.append({
                            'Modelo': name,
                            'MAE': mean_absolute_error(y_test_return, return_predictions[name]),
                            'RMSE': np.sqrt(mean_squared_error(y_test_return, return_predictions[name])),
                            'R²': r2_score(y_test_return, return_predictions[name]),
                            'Acerto Direção': accuracy_score(y_test_dir, (return_predictions[name] > 0).astype(int))
                        })
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df.style.format({'MAE': '{:.4f}', 'RMSE': '{:.4f}', 'R²': '{:.4f}', 'Acerto Direção': '{:.2%}'}), use_container_width=True)
                    
                    fig_comparison = go.Figure()
                    fig_comparison.add_trace(go.Scatter(x=y_test_return.index, y=y_test_return.values, name='Retorno Real', line=dict(color='blue', width=3)))
                    fig_comparison.add_trace(go.Scatter(x=y_test_return.index, y=ensemble_pred, name='Previsão Ensemble', line=dict(color='red', width=2, dash='dash')))
                    fig_comparison.update_layout(title="Comparação: Retorno Real vs Previsão do Modelo (Dados de Teste)", xaxis_title="Data", yaxis_title="Retorno Esperado (%)")
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    st.subheader("🎯 Previsão para os Próximos Dias")
                    latest_features = X.iloc[-1:].values
                    future_predictions = {name: model.predict(latest_features)[0] for name, model in trained_models.items()}
                    ensemble_future_return = np.mean(list(future_predictions.values()))

                    # ##################################################################################
                    # ### ALTERAÇÃO CRÍTICA 2: Correção do Cálculo de Previsão com Juros Compostos ###
                    # ##################################################################################
                    
                    # Limitar o retorno previsto para evitar valores explosivos e irreais
                    # Um ganho de 50% em 5 dias já é extremamente otimista/pessimista.
                    capped_return = np.clip(ensemble_future_return, -0.5, 0.5)

                    # Calcular a taxa diária composta equivalente com base no retorno limitado
                    # A fórmula (1+retorno)^(1/dias) - 1 calcula a taxa diária que, se aplicada
                    # repetidamente, resulta no retorno total ao final do período.
                    daily_comp_rate = (1 + capped_return) ** (1/5) - 1
                    
                    current_price = data['Close'].iloc[-1]
                    predicted_prices_data = []
                    temp_price = current_price
                    
                    for i in range(1, 6):
                        # Aplica a taxa composta ao preço do dia anterior
                        temp_price *= (1 + daily_comp_rate)
                        total_return = (temp_price / current_price) - 1
                        predicted_prices_data.append({
                            'Dias': i,
                            'Data': (data.index[-1] + pd.Timedelta(days=i)).strftime('%d/%m/%Y'),
                            'Preço Previsto': temp_price,
                            'Variação %': total_return * 100
                        })
                    
                    predictions_df = pd.DataFrame(predicted_prices_data)
                    # ##################################################################################
                    # ### FIM DA ALTERAÇÃO 2                                                         ###
                    # ##################################################################################

                    st.dataframe(predictions_df.style.format({'Preço Previsto': 'R$ {:.2f}', 'Variação %': '{:+.2f}%'}), use_container_width=True)
                    
                    fig_forecast = go.Figure()
                    hist_data = data['Close'].iloc[-30:]
                    fig_forecast.add_trace(go.Scatter(x=hist_data.index, y=hist_data.values, name='Histórico', line=dict(color='blue', width=2)))
                    future_dates = [pd.to_datetime(d['Data'], format='%d/%m/%Y') for d in predicted_prices_data]
                    future_prices = predictions_df['Preço Previsto'].values
                    fig_forecast.add_trace(go.Scatter(x=future_dates, y=future_prices, name='Previsão', line=dict(color='red', width=2, dash='dash')))
                    fig_forecast.update_layout(title="Projeção de Preço para os Próximos 5 Dias", xaxis_title="Data", yaxis_title="Preço (R$)")
                    st.plotly_chart(fig_forecast, use_container_width=True)

                    st.subheader("📈 Análise de Confiança da Previsão")
                    model_agreement = np.std(list(future_predictions.values()))
                    confidence_score = max(0, 1 - model_agreement * 5)
                    col1_conf, col2_conf, col3_conf = st.columns(3)
                    col1_conf.metric("Concordância entre Modelos", f"{(1 - model_agreement) * 100:.1f}%")
                    col2_conf.metric("Score de Confiança", f"{confidence_score * 100:.1f}%")
                    if confidence_score > 0.7: recomendacao = "ALTA CONFIANÇA"
                    elif confidence_score > 0.5: recomendacao = "MÉDIA CONFIANÇA"
                    else: recomendacao = "BAIXA CONFIANÇA"
                    col3_conf.metric("Recomendação", recomendacao)
                    st.warning("""
                    **⚠️ Disclaimer Importante:** Previsões são probabilísticas, não garantias. O mercado financeiro é influenciado por fatores imprevisíveis. Use como ferramenta auxiliar e consulte sempre um profissional.
                    """)

    with st.expander("🧠 Previsão de Volatilidade (Modelo Simples)", expanded=False):
        st.write("Previsão da volatilidade para o próximo dia útil usando um modelo simples de Random Forest.")
        if st.button('Executar Análise Preditiva de Volatilidade'):
            df_model = data[['Volatility']].copy().dropna()
            if len(df_model) < 20:
                st.warning("⚠️ Dados históricos insuficientes para treinar o modelo de volatilidade.")
            else:
                for i in range(1, 6):
                    df_model[f'vol_lag_{i}'] = df_model['Volatility'].shift(i)
                df_model.dropna(inplace=True)
                X_vol = df_model.drop('Volatility', axis=1)
                y_vol = df_model['Volatility']
                with st.spinner('Treinando o modelo de volatilidade...'):
                    model_vol = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    model_vol.fit(X_vol, y_vol)
                st.subheader("Previsão de Volatilidade para o Próximo Dia Útil")
                prediction_vol = model_vol.predict(X_vol.iloc[-1:].values)[0]
                last_date = data.index[-1]
                next_day = last_date + pd.Timedelta(days=1)
                if next_day.weekday() >= 5:
                    next_day += pd.Timedelta(days=7 - next_day.weekday())
                next_day_str = next_day.strftime('%d/%m/%Y')
                if prediction_vol < 0.30: status_text, status_color = "Baixa Volatilidade", "#28a745"
                elif prediction_vol >= 0.60: status_text, status_color = "Alta Volatilidade", "#dc3545"
                else: status_text, status_color = "Média Volatilidade", "#ffc107"
                st.markdown(f"""<div style='border:1px solid #444;border-radius:10px;padding:20px;text-align:center'><p style='font-size:1.1em;margin-bottom:5px;color:#FAFAFA'>Previsão de Volatilidade para <strong>{next_day_str}</strong></p><p style='font-size:2.5em;font-weight:bold;color:{status_color};margin:0'>{prediction_vol:.4f}</p><p style='font-size:1.2em;font-weight:bold;color:{status_color};margin-top:5px'>{status_text}</p></div>""", unsafe_allow_html=True)
                st.info('**Disclaimer:** Este modelo é apenas para fins educacionais e não constitui uma recomendação de investimento.')
    
    last_update_date = data.index[-1].strftime('%d/%m/%Y')
    st.markdown("---")
    st.caption(f"📅 Última atualização dos preços: **{last_update_date}** — Dados fornecidos pelo Yahoo Finance.")
    st.markdown("<p style='text-align:center;color:#888'>Desenvolvido por Rodrigo Costa de Araujo | rodrigocosta@usp.br</p>", unsafe_allow_html=True)
