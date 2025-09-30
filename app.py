
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
        data.columns = data.columns.get_level_values(0)
    return data

def calculate_indicators(data):
    """Calcula os indicadores técnicos para o DataFrame."""
    # RSI (Índice de Força Relativa)
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
    
    # Volatilidade (Anualizada)
    data['Daily Return'] = data['Close'].pct_change()
    data['Volatility'] = data['Daily Return'].rolling(window=30).std() * (252**0.5)
    return data

# --- VERSÃO CORRIGIDA DA FUNÇÃO ---
def prepare_advanced_features(data, lookback_days=60, forecast_days=5):
    """
    Prepara features com janela temporal expandida
    lookback_days: quantos dias no passado considerar
    forecast_days: prever para quantos dias à frente
    """
    df = data[['Close', 'Volume', 'RSI', 'MM_Curta', 'MM_Longa', 'Volatility']].copy()
    
    # Features de preço com múltiplas janelas
    periods = [1, 3, 5, 10, 20]
    for days in periods:
        df[f'return_{days}d'] = df['Close'].pct_change(days)
        df[f'volume_ma_{days}d'] = df['Volume'].rolling(days).mean()
        if days <= 20:
            df[f'high_{days}d'] = df['Close'].rolling(days).max()
            df[f'low_{days}d'] = df['Close'].rolling(days).min()
        df[f'volatility_{days}d'] = df['Close'].pct_change().rolling(days).std()
    
    # Features técnicas avançadas
    df['price_vs_ma20'] = df['Close'] / df['MM_Curta']
    df['price_vs_ma50'] = df['Close'] / df['MM_Longa']
    df['ma_cross'] = (df['MM_Curta'] > df['MM_Longa']).astype(int)
    
    # Target: Retorno futuro (5 dias)
    df['target_future_return'] = df['Close'].shift(-forecast_days) / df['Close'] - 1
    
    # Target: Direção (1 = sobe, 0 = desce)
    df['target_direction'] = (df['target_future_return'] > 0).astype(int)
    
    # --- ✅ CORREÇÃO: Tratamento robusto de dados ausentes e infinitos ---
    # 1. Substituir valores infinitos (resultantes de divisão por zero) por NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 2. Remover TODAS as linhas que contenham QUALQUER valor NaN.
    #    Isso garante que o modelo receba apenas dados completos, eliminando
    #    NaNs do início (janelas móveis) e do fim (alvo futuro).
    df.dropna(inplace=True)
    
    return df

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
    predictions = []
    for name, model in models.items():
        pred = model.predict(X)
        predictions.append(pred)
    
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
            comp_df = pd.DataFrame({
                ticker_symbol: data['Close'] / data['Close'].iloc[0],
                'IBOVESPA': ibov['Close'] / ibov['Close'].iloc[0]
            })
            fig_comp = px.line(comp_df, x=comp_df.index, y=comp_df.columns, title='Performance Normalizada: Ação vs IBOVESPA')
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.warning("Não foi possível carregar os dados do IBOVESPA para comparação.")

    st.markdown("---")
    
    # --- NOVA SEÇÃO: Previsão de Preço Avançada ---
    with st.expander("🔮 Previsão de Preço Avançada (Machine Learning)", expanded=True):
        st.write("""
        **Previsão para os próximos 5 dias usando múltiplos algoritmos de ML**
        - Período de análise: 60 dias (~3 meses)
        - Features: Preço, Volume, RSI, Médias Móveis, Volatilidade
        - Modelos: Random Forest, Gradient Boosting, SVR, Neural Network
        """)
        
        if st.button('Executar Previsão de Preço Avançada'):
            with st.spinner('Processando dados e treinando modelos...'):
                # Preparar dados avançados
                advanced_data = prepare_advanced_features(data, lookback_days=60, forecast_days=5)
                
                # Verificar se temos dados suficientes após o processamento
                if len(advanced_data) < 50:
                    st.warning(f"⚠️ Dados insuficientes para análise avançada. Necessários pelo menos 50 dias úteis após processamento. Disponíveis: {len(advanced_data)} dias.")
                    st.info(f"💡 Dica: Selecione um período mais longo (a partir de 2019) para ter dados suficientes.")
                else:
                    # Separar features e target
                    feature_columns = [col for col in advanced_data.columns if not col.startswith('target_')]
                    X = advanced_data[feature_columns]
                    y_return = advanced_data['target_future_return']  # Retorno percentual
                    y_direction = advanced_data['target_direction']    # Direção
                    
                    # Mostrar informações sobre os dados
                    st.info(f"📊 Dados disponíveis para treinamento: {len(X)} dias úteis")
                    
                    # Split temporal (não shuffle para time series)
                    split_idx = int(len(X) * 0.8)
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train_return, y_test_return = y_return[:split_idx], y_return[split_idx:]
                    y_train_dir, y_test_dir = y_direction[:split_idx], y_direction[split_idx:]
                    
                    # Treinar modelos
                    models = create_advanced_model()
                    trained_models = {}
                    return_predictions = {}
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, (name, model) in enumerate(models.items()):
                        status_text.text(f"Treinando {name}...")
                        model.fit(X_train, y_train_return)
                        trained_models[name] = model
                        return_predictions[name] = model.predict(X_test)
                        progress_bar.progress((i + 1) / len(models))
                    
                    status_text.text("Treinamento concluído!")
                    
                    # Ensemble
                    ensemble_pred = ensemble_predict(trained_models, X_test)
                    
                    # --- Avaliação dos Modelos ---
                    st.subheader("📊 Performance dos Modelos")
                    
                    metrics_data = []
                    for name in models.keys():
                        mae = mean_absolute_error(y_test_return, return_predictions[name])
                        rmse = np.sqrt(mean_squared_error(y_test_return, return_predictions[name]))
                        r2 = r2_score(y_test_return, return_predictions[name])
                        accuracy_dir = accuracy_score(y_test_dir, (return_predictions[name] > 0).astype(int))
                        
                        metrics_data.append({
                            'Modelo': name,
                            'MAE': mae,
                            'RMSE': rmse,
                            'R²': r2,
                            'Acerto Direção': accuracy_dir
                        })
                    
                    metrics_df = pd.DataFrame(metrics_data)
                    st.dataframe(metrics_df.style.format({
                        'MAE': '{:.4f}',
                        'RMSE': '{:.4f}', 
                        'R²': '{:.4f}',
                        'Acerto Direção': '{:.2%}'
                    }), use_container_width=True)
                    
                    # --- Gráfico de Comparação ---
                    fig_comparison = go.Figure()
                    fig_comparison.add_trace(go.Scatter(
                        x=y_test_return.index, y=y_test_return.values,
                        name='Retorno Real', line=dict(color='blue', width=3)
                    ))
                    fig_comparison.add_trace(go.Scatter(
                        x=y_test_return.index, y=ensemble_pred,
                        name='Previsão Ensemble', line=dict(color='red', width=2, dash='dash')
                    ))
                    fig_comparison.update_layout(
                        title="Comparação: Retorno Real vs Previsão do Modelo (Dados de Teste)",
                        xaxis_title="Data",
                        yaxis_title="Retorno Esperado (%)"
                    )
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # --- Previsão para o Futuro ---
                    st.subheader("🎯 Previsão para os Próximos Dias")
                    
                    # Usar os dados mais recentes para prever
                    latest_features = X.iloc[-1:].values
                    future_predictions = {}
                    
                    for name, model in trained_models.items():
                        future_predictions[name] = model.predict(latest_features)[0]
                    
                    ensemble_future = np.mean(list(future_predictions.values()))
                    
                    # Calcular preços futuros
                    current_price = data['Close'].iloc[-1]
                    predicted_prices = []
                    
                    for days in range(1, 6):
                        if days == 1:
                            pred_return = ensemble_future * (days/5)  # Projeção linear
                        else:
                            pred_return = ensemble_future * (days/5)
                        
                        predicted_price = current_price * (1 + pred_return)
                        predicted_prices.append({
                            'Dias': days,
                            'Data': (data.index[-1] + pd.Timedelta(days=days)).strftime('%d/%m/%Y'),
                            'Preço Previsto': predicted_price,
                            'Variação %': pred_return * 100
                        })
                    
                    predictions_df = pd.DataFrame(predicted_prices)
                    
                    # Formatar exibição
                    st.dataframe(predictions_df.style.format({
                        'Preço Previsto': 'R$ {:.2f}',
                        'Variação %': '{:+.2f}%'
                    }), use_container_width=True)
                    
                    # --- Gráfico de Previsão ---
                    fig_forecast = go.Figure()
                    
                    # Histórico recente
                    historical_days = min(30, len(data))
                    hist_data = data['Close'].iloc[-historical_days:]
                    fig_forecast.add_trace(go.Scatter(
                        x=hist_data.index, y=hist_data.values,
                        name='Histórico', line=dict(color='blue', width=2)
                    ))
                    
                    # Previsões
                    future_dates = [data.index[-1] + pd.Timedelta(days=i) for i in range(1, 6)]
                    future_prices = predictions_df['Preço Previsto'].values
                    
                    fig_forecast.add_trace(go.Scatter(
                        x=future_dates, y=future_prices,
                        name='Previsão', line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=8)
                    ))
                    
                    # Intervalo de confiança (simulado)
                    confidence = abs(ensemble_future) * 0.5
                    upper_bound = [current_price * (1 + ensemble_future * (i/5) + confidence * (i/5)) for i in range(1, 6)]
                    lower_bound = [current_price * (1 + ensemble_future * (i/5) - confidence * (i/5)) for i in range(1, 6)]
                    
                    fig_forecast.add_trace(go.Scatter(
                        x=future_dates + future_dates[::-1],
                        y=upper_bound + lower_bound[::-1],
                        fill='toself',
                        fillcolor='rgba(255,0,0,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Intervalo de Confiança'
                    ))
                    
                    fig_forecast.update_layout(
                        title="Previsão de Preço para os Próximos 5 Dias",
                        xaxis_title="Data",
                        yaxis_title="Preço (R$)",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # --- Análise de Confiança ---
                    st.subheader("📈 Análise de Confiança da Previsão")
                    
                    # Calcular métricas de confiança
                    model_agreement = np.std(list(future_predictions.values()))
                    confidence_score = max(0, 1 - model_agreement * 5)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Concordância entre Modelos", f"{(1 - model_agreement) * 100:.1f}%")
                    col2.metric("Score de Confiança", f"{confidence_score * 100:.1f}%")
                    
                    if confidence_score > 0.7:
                        recomendacao = "ALTA CONFIANÇA"
                        cor = "green"
                    elif confidence_score > 0.5:
                        recomendacao = "MÉDIA CONFIANÇA" 
                        cor = "orange"
                    else:
                        recomendacao = "BAIXA CONFIANÇA"
                        cor = "red"
                    
                    col3.metric("Recomendação", recomendacao)
                    
                    # Disclaimer importante
                    st.warning("""
                    **⚠️ Disclaimer Importante:** - Previsões baseadas em machine learning são probabilísticas, não garantias
                    - Mercado financeiro é influenciado por fatores imprevisíveis
                    - Use como ferramenta auxiliar, não como única base de decisão
                    - Consulte sempre um advisor financeiro para investimentos
                    """)

    # --- Seção de Machine Learning Original (Volatilidade) ---
    with st.expander("🧠 Previsão de Volatilidade (Modelo Original)", expanded=False):
        st.write("""
        Esta seção utiliza um modelo de Machine Learning (Random Forest) para prever a volatilidade do ativo no próximo dia útil. 
        O modelo é treinado com base na volatilidade dos 5 dias anteriores.
        """)

        if st.button('Executar Análise Preditiva de Volatilidade'):
            df_model = data[['Volatility']].copy().dropna()
            if len(df_model) < 20: 
                st.warning("⚠️ Dados históricos insuficientes para treinar e avaliar o modelo de forma confiável.")
            else:
                for i in range(1, 6):
                    df_model[f'vol_lag_{i}'] = df_model['Volatility'].shift(i)
                df_model.dropna(inplace=True)

                X_vol = df_model.drop('Volatility', axis=1)
                y_vol = df_model['Volatility']
                X_train_vol, X_test_vol, y_train_vol, y_test_vol = train_test_split(X_vol, y_vol, test_size=0.2, shuffle=False)

                with st.spinner('Treinando o modelo de volatilidade...'):
                    model_vol = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    model_vol.fit(X_train_vol, y_train_vol)
                
                # --- Avaliação do Modelo ---
                st.subheader("Avaliando a Performance do Modelo de Volatilidade")
                y_pred_vol = model_vol.predict(X_test_vol)
                mae_vol = mean_absolute_error(y_test_vol, y_pred_vol)

                col1_vol, _ = st.columns(2)
                col1_vol.metric("Erro Médio Absoluto (MAE)", f"{mae_vol:.4f}", help="Indica o erro médio das previsões do modelo no período de teste.")
                
                fig_eval_vol = go.Figure()
                fig_eval_vol.add_trace(go.Scatter(x=y_test_vol.index, y=y_test_vol, name='Volatilidade Real', line=dict(color='blue')))
                fig_eval_vol.add_trace(go.Scatter(x=y_test_vol.index, y=y_pred_vol, name='Previsão do Modelo', line=dict(color='red', dash='dash')))
                fig_eval_vol.update_layout(title="Comparativo: Volatilidade Real vs. Previsão do Modelo (Dados de Teste)")
                st.plotly_chart(fig_eval_vol, use_container_width=True)

                # --- Previsão Final ---
                st.subheader("Previsão de Volatilidade para o Próximo Dia Útil")
                prediction_vol = model_vol.predict(X_vol.iloc[-1:].values)
                
                # Lógica de Data
                last_date = data.index[-1]
                next_day = last_date + pd.Timedelta(days=1)
                if next_day.weekday() == 5:  # Sábado
                    next_day += pd.Timedelta(days=2)
                elif next_day.weekday() == 6:  # Domingo
                    next_day += pd.Timedelta(days=1)
                next_day_str = next_day.strftime('%d/%m/%Y')

                predicted_vol = prediction_vol[0]
                
                # Lógica de classificação de volatilidade
                if predicted_vol < 0.30:
                    status_text = "Baixa Volatilidade"
                    status_color = "#28a745"  # Verde
                elif predicted_vol >= 0.60:
                    status_text = "Alta Volatilidade"
                    status_color = "#dc3545"  # Vermelho
                else:
                    status_text = "Média Volatilidade"
                    status_color = "#ffc107"  # Amarelo
                
                st.markdown(f"""
                <div style='border: 1px solid #444; border-radius: 10px; padding: 20px; text-align: center;'>
                    <p style='font-size: 1.1em; margin-bottom: 5px; color: #FAFAFA;'>Previsão de Volatilidade para <strong>{next_day_str}</strong></p>
                    <p style='font-size: 2.5em; font-weight: bold; color: {status_color}; margin: 0;'>{predicted_vol:.4f}</p>
                    <p style='font-size: 1.2em; font-weight: bold; color: {status_color}; margin-top: 5px;'>{status_text}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.info('**Disclaimer:** Este modelo é apenas para fins educacionais e não constitui uma recomendação de investimento.')

    # --- Nota de atualização ---
    last_update_date = data.index[-1].strftime('%d/%m/%Y')
    st.markdown("---")
    st.caption(f"📅 Última atualização dos preços: **{last_update_date}** — Dados fornecidos pelo Yahoo Finance (podem ter atraso).")

    # --- Rodapé de Autoria ---
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>Desenvolvido por Rodrigo Costa de Araujo | rodrigocosta@usp.br</p>", unsafe_allow_html=True)
