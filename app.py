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

def prepare_advanced_features(data, forecast_days=5):
    """
    Prepara features com lógica robusta para lidar com dados ausentes.
    """
    df = data[['Close', 'Volume', 'RSI', 'MM_Curta', 'MM_Longa', 'Volatility']].copy()
    periods = [1, 3, 5, 10, 20]

    for days in periods:
        df[f'return_{days}d'] = df['Close'].pct_change(days)
        df[f'high_{days}d'] = df['Close'].rolling(window=days).max()
        df[f'low_{days}d'] = df['Close'].rolling(window=days).min()
        df[f'volatility_{days}d'] = df['Close'].pct_change().rolling(window=days).std()

    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        for days in periods:
            df[f'volume_ma_{days}d'] = df['Volume'].rolling(window=days).mean()

    df['price_vs_ma20'] = df['Close'] / df['MM_Curta']
    df['price_vs_ma50'] = df['Close'] / df['MM_Longa']
    df['ma_cross'] = (df['MM_Curta'] > df['MM_Longa']).astype(int)
    df['target_future_return'] = df['Close'].shift(-forecast_days) / df['Close'] - 1
    df['target_direction'] = (df['target_future_return'] > 0).astype(int)

    potential_feature_columns = [col for col in df.columns if col.startswith(
        ('return_', 'volume_ma_', 'high_', 'low_', 'volatility_', 'price_vs_', 'ma_cross'))]
    potential_feature_columns.extend(['RSI', 'Volatility'])
    
    feature_columns = [col for col in potential_feature_columns if col in df.columns and not df[col].isnull().all()]
    target_columns = ['target_future_return', 'target_direction']
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=feature_columns + target_columns, inplace=True)

    return df, feature_columns

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
    # ... (código das abas 1, 2 e 3, que não precisa de alteração)
    
    with st.expander("🔮 Previsão de Preço Avançada (Machine Learning)", expanded=True):
        # ... (texto de descrição)
        
        if st.button('Executar Previsão de Preço Avançada'):
            with st.spinner('Processando dados e treinando modelos...'):
                advanced_data, used_features = prepare_advanced_features(data, forecast_days=5)

                if len(advanced_data) < 50:
                    st.warning(f"⚠️ Dados insuficientes para análise avançada. Disponíveis: {len(advanced_data)} dias.")
                else:
                    X = advanced_data[used_features]
                    y_return = advanced_data['target_future_return']
                    y_direction = advanced_data['target_direction']
                    st.info(f"📊 Dados disponíveis para treinamento: {len(X)} dias úteis. Features utilizadas: {len(used_features)}.")
                    
                    split_idx = int(len(X) * 0.8)
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train_return, y_test_return = y_return[:split_idx], y_return[split_idx:]
                    
                    models = create_advanced_model()
                    trained_models = {}
                    for name, model in models.items():
                        model.fit(X_train, y_train_return)
                        trained_models[name] = model

                    st.subheader("🎯 Previsão para os Próximos Dias")
                    latest_features = X.iloc[-1:].values
                    
                    future_predictions = {}
                    for name, model in trained_models.items():
                        future_predictions[name] = model.predict(latest_features)[0]
                    
                    ensemble_future_return = np.mean(list(future_predictions.values()))

                    # --- ✅ LÓGICA DE PREVISÃO CORRIGIDA ---
                    st.metric("Retorno Previsto para 5 Dias", f"{ensemble_future_return:+.2%}")

                    # Calcular a taxa diária composta equivalente
                    daily_compounded_rate = (1 + ensemble_future_return) ** (1/5) - 1
                    
                    current_price = data['Close'].iloc[-1]
                    predicted_prices_data = []
                    temp_price = current_price

                    for i in range(1, 6):
                        temp_price *= (1 + daily_compounded_rate)
                        total_return = (temp_price / current_price) - 1
                        predicted_prices_data.append({
                            'Dias': i,
                            'Data': (data.index[-1] + pd.Timedelta(days=i)).strftime('%d/%m/%Y'),
                            'Preço Previsto': temp_price,
                            'Variação %': total_return * 100
                        })
                    
                    predictions_df = pd.DataFrame(predicted_prices_data)
                    st.dataframe(predictions_df.style.format({
                        'Preço Previsto': 'R$ {:.2f}',
                        'Variação %': '{:+.2f}%'
                    }), use_container_width=True)


    # --- ✅ SEÇÃO DE VOLATILIDADE COM VISUAL RESTAURADO ---
    with st.expander("🧠 Previsão de Volatilidade (Modelo Simples)", expanded=False):
        st.write("""
        Esta seção utiliza um modelo de Machine Learning (Random Forest) para prever a volatilidade do ativo no próximo dia útil. 
        O modelo é treinado com base na volatilidade dos 5 dias anteriores.
        """)

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
                prediction_vol = model_vol.predict(X_vol.iloc[-1:].values)
                
                last_date = data.index[-1]
                next_day = last_date + pd.Timedelta(days=1)
                if next_day.weekday() >= 5: # Se for Sábado ou Domingo
                    next_day += pd.Timedelta(days=(7 - next_day.weekday()))
                next_day_str = next_day.strftime('%d/%m/%Y')

                predicted_vol = prediction_vol[0]
                
                if predicted_vol < 0.30:
                    status_text, status_color = "Baixa Volatilidade", "#28a745" # Verde
                elif predicted_vol >= 0.60:
                    status_text, status_color = "Alta Volatilidade", "#dc3545" # Vermelho
                else:
                    status_text, status_color = "Média Volatilidade", "#ffc107" # Amarelo
                
                # Visual restaurado
                st.markdown(f"""
                <div style='border: 1px solid #444; border-radius: 10px; padding: 20px; text-align: center;'>
                    <p style='font-size: 1.1em; margin-bottom: 5px; color: #FAFAFA;'>Previsão de Volatilidade para <strong>{next_day_str}</strong></p>
                    <p style='font-size: 2.5em; font-weight: bold; color: {status_color}; margin: 0;'>{predicted_vol:.4f}</p>
                    <p style='font-size: 1.2em; font-weight: bold; color: {status_color}; margin-top: 5px;'>{status_text}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.info('**Disclaimer:** Este modelo é apenas para fins educacionais e não constitui uma recomendação de investimento.')

    # --- Rodapé ---
    last_update_date = data.index[-1].strftime('%d/%m/%Y')
    st.markdown("---")
    st.caption(f"📅 Última atualização dos preços: **{last_update_date}** — Dados fornecidos pelo Yahoo Finance (podem ter atraso).")
    st.markdown("<p style='text-align: center; color: #888;'>Desenvolvido por Rodrigo Costa de Araujo | rodrigocosta@usp.br</p>", unsafe_allow_html=True)
