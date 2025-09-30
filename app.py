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
st.write('Analise preço, volatilidade e indicadores. Previsões com ML. Use com cuidado.')

# --- Sidebar parâmetros visuais e data range ---
st.sidebar.header('⚙️ Parâmetros de Análise')
viz_range = st.sidebar.selectbox("Período de visualização dos gráficos", ["Últimos 3 meses", "Últimos 6 meses", "Último 1 ano", "Todo período"], index=2)
viz_map = {"Últimos 3 meses": 63, "Últimos 6 meses": 126, "Último 1 ano": 252, "Todo período": None}
viz_days = viz_map[viz_range]

# --- Funções de Coleta e Processamento ---
@st.cache_data
def get_tickers_from_csv():
    file_path = 'acoes-listadas-b3.csv'
    try:
        df = pd.read_csv(file_path)
        df = df.rename(columns={'Ticker': 'ticker', 'Nome': 'nome'})
        df['display'] = df['nome'] + ' (' + df['ticker'] + ')'
        return df
    except FileNotFoundError:
        fallback_data = {'ticker': ['PETR4', 'VALE3', 'ITUB4', 'MGLU3'], 'nome': ['Petrobras', 'Vale', 'Itaú Unibanco', 'Magazine Luiza']}
        fallback_df = pd.DataFrame(fallback_data)
        fallback_df['display'] = fallback_df['nome'] + ' (' + fallback_df['ticker'] + ')'
        return fallback_df

@st.cache_data
def load_data(ticker, start, end):
    """
    Baixa dados do Yahoo Finance. Garante inclusão do dia 'end'.
    Retorna DataFrame com colunas simples e Volume presente.
    """
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
    df = data[['Close', 'Volume', 'RSI', 'MM_Curta', 'MM_Longa', 'Volatility']].copy()
    periods = [1, 3, 5, 10, 20]
    for days in periods:
        df[f'return_{days}d'] = df['Close'].pct_change(days)
        df[f'high_{days}d'] = df['Close'].rolling(window=days, min_periods=1).max()
        df[f'low_{days}d'] = df['Close'].rolling(window=days, min_periods=1).min()
        df[f'volatility_{days}d'] = df['Close'].pct_change().rolling(window=days, min_periods=1).std()
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        for days in periods:
            df[f'volume_ma_{days}d'] = df['Volume'].rolling(window=days, min_periods=1).mean()
    df['price_vs_ma20'] = df['Close'] / df['MM_Curta'].replace(0, np.nan)
    df['price_vs_ma50'] = df['Close'] / df['MM_Longa'].replace(0, np.nan)
    df['ma_cross'] = (df['MM_Curta'] > df['MM_Longa']).astype(int)
    df['target_future_return'] = df['Close'].shift(-forecast_days) / df['Close'] - 1
    df['target_direction'] = (df['target_future_return'] > 0).astype(int)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    potential_feature_columns = [col for col in df.columns if col.startswith(
        ('return_', 'volume_ma_', 'high_', 'low_', 'volatility_', 'price_vs_', 'ma_cross'))]
    potential_feature_columns.extend(['RSI', 'Volatility'])
    feature_columns = [col for col in potential_feature_columns if col in df.columns and not df[col].isnull().all()]
    required = feature_columns + ['target_future_return', 'target_direction']
    df.dropna(subset=required, inplace=True)
    return df, feature_columns

def create_advanced_model():
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    }
    return models

def ensemble_predict(models, X):
    predictions = [model.predict(X) for name, model in models.items()]
    return np.mean(predictions, axis=0)

# --- UI: seleção e datas ---
tickers_df = get_tickers_from_csv()
selected_display = st.sidebar.selectbox('Escolha a Ação', tickers_df['display'])
ticker_symbol = tickers_df[tickers_df['display'] == selected_display]['ticker'].iloc[0]
company_name = tickers_df[tickers_df['display'] == selected_display]['nome'].iloc[0]
ticker = f"{ticker_symbol}.SA"
start_date = st.sidebar.date_input("Data de Início", date(2019, 1, 1), format="DD/MM/YYYY")
end_date = st.sidebar.date_input("Data de Fim", date.today(), format="DD/MM/YYYY")

data = load_data(ticker, start_date, end_date)
ibov = load_data('^BVSP', start_date, end_date)

# Initialize session state containers
if 'advanced_result' not in st.session_state:
    st.session_state['advanced_result'] = None
if 'vol_result' not in st.session_state:
    st.session_state['vol_result'] = None

# --- Exibição Principal ---
if data.empty or len(data) < 60:
    st.error("❌ Nenhum dado encontrado ou dados insuficientes (mínimo 60 dias). Ajuste as datas ou ticker.")
else:
    data = calculate_indicators(data)

    # --- Cabeçalho e métricas ---
    st.subheader('📈 Visão Geral do Ativo')
    last_price = float(data['Close'].iloc[-1])
    prev_price = float(data['Close'].iloc[-2]) if len(data) >= 2 else last_price
    price_change = last_price - prev_price
    percent_change = ((price_change / prev_price) * 100) if prev_price != 0 else 0.0
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏢 Empresa", company_name)
    col2.metric("💹 Ticker", ticker_symbol)
    col3.metric("💰 Último Preço", f"R$ {last_price:.2f}")
    col4.metric("📊 Variação (Dia)", f"{price_change:+.2f} R$", f"{percent_change:+.2f}%")
    st.markdown("---")

    # --- Gráficos iniciais (preço, indicadores, volatilidade, comparativo IBOV) ---
    tab1, tab2, tab3 = st.tabs(["Preço e Indicadores", "Volatilidade", "Comparativo com IBOVESPA"])

    # Determine view slice
    if viz_days is None:
        view_slice = slice(None)
    else:
        view_slice = slice(-viz_days, None)

    with tab1:
        st.subheader('📉 Preço, Médias Móveis e Bandas de Bollinger')
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=data.index[view_slice], y=data['Close'][view_slice], name='Preço de Fechamento'))
        fig_price.add_trace(go.Scatter(x=data.index[view_slice], y=data['MM_Curta'][view_slice], name='Média Móvel 20p'))
        fig_price.add_trace(go.Scatter(x=data.index[view_slice], y=data['MM_Longa'][view_slice], name='Média Móvel 50p'))
        fig_price.add_trace(go.Scatter(x=data.index[view_slice], y=data['BB_Superior'][view_slice], name='Banda Superior', fill=None))
        fig_price.add_trace(go.Scatter(x=data.index[view_slice], y=data['BB_Inferior'][view_slice], name='Banda Inferior', fill='tonexty', opacity=0.1))
        st.plotly_chart(fig_price, use_container_width=True)

        st.subheader('📊 Índice de Força Relativa (RSI)')
        fig_rsi = px.line(data[view_slice], x=data[view_slice].index, y='RSI', title='RSI (Índice de Força Relativa)')
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrecompra")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobrevenda")
        st.plotly_chart(fig_rsi, use_container_width=True)

    with tab2:
        st.subheader('📈 Análise de Volatilidade')
        st.markdown("**Observação:** 'Janela de 30 dias' refere-se ao cálculo da volatilidade. Aqui você vê a volatilidade ao longo do tempo no período selecionado.")
        current_vol = float(data['Volatility'].iloc[-1]) if not pd.isna(data['Volatility'].iloc[-1]) else 0.0
        vol_median = float(data['Volatility'].median())
        col1_v, col2_v = st.columns([3, 1])
        with col1_v:
            fig_vol = px.line(data[view_slice], x=data[view_slice].index, y='Volatility', title='Volatilidade Anualizada (janela de 30 dias)')
            st.plotly_chart(fig_vol, use_container_width=True)
        with col2_v:
            st.metric("Volatilidade Atual", f"{current_vol:.3f}")
            st.metric("Volatilidade Mediana", f"{vol_median:.3f}")

    with tab3:
        st.subheader('🏁 Comparativo com o IBOVESPA')
        if not ibov.empty:
            common_index = data.index.intersection(ibov.index)
            comp_df = pd.DataFrame({
                'IBOVESPA': ibov.loc[common_index, 'Close'] / ibov.loc[common_index, 'Close'].iloc[0],
                ticker_symbol: data.loc[common_index, 'Close'] / data.loc[common_index, 'Close'].iloc[0]
            }, index=common_index)
            if not comp_df.empty:
                fig_comp = px.line(comp_df, x=comp_df.index, y=comp_df.columns, title='Performance Normalizada: Ação vs IBOVESPA')
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.warning("Períodos não coincidem entre ação e IBOVESPA.")
        else:
            st.warning("Não foi possível carregar IBOVESPA para comparação.")

    st.markdown("---")

    # --- Destaque: Previsão Preditiva de Preço (principal) ---
    st.markdown("<div style='background:#07203a;padding:14px;border-radius:8px;margin-bottom:10px'><h2 style='color:#fff;margin:0'>🔮 Previsão Preditiva de Preço (PRIORITÁRIO)</h2><p style='color:#ddd;margin:0.25rem 0 0 0'>Clique em Executar para treinar modelos com dados reais do Yahoo Finance e gerar previsões para os próximos 5 dias úteis.</p></div>", unsafe_allow_html=True)

    with st.container():
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button('Executar Previsão de Preço Avançada'):
                advanced_data, used_features = prepare_advanced_features(data, forecast_days=5)
                if len(advanced_data) < 60:
                    st.warning(f"⚠️ Dados insuficientes para análise avançada. Disponíveis: {len(advanced_data)} dias.")
                else:
                    X = advanced_data[used_features].copy()
                    y_return = advanced_data['target_future_return'].copy()
                    y_direction = advanced_data['target_direction'].copy()
                    split_idx = int(len(X) * 0.8)
                    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                    y_train_return, y_test_return = y_return.iloc[:split_idx], y_return.iloc[split_idx:]
                    y_train_dir, y_test_dir = y_direction.iloc[:split_idx], y_direction.iloc[split_idx:]
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    models = create_advanced_model()
                    trained_models = {}
                    return_predictions = {}
                    progress = st.progress(0)
                    status = st.empty()
                    for i, (name, model) in enumerate(models.items()):
                        status.text(f"Treinando {name}...")
                        model.fit(X_train_scaled, y_train_return)
                        trained_models[name] = model
                        return_predictions[name] = model.predict(X_test_scaled)
                        progress.progress(int(((i + 1) / len(models)) * 100))
                    status.text("Treinamento concluído!")
                    ensemble_pred = ensemble_predict(trained_models, X_test_scaled)

                    # Métricas
                    metrics_data = []
                    for name in models.keys():
                        mae = mean_absolute_error(y_test_return, return_predictions[name])
                        rmse = np.sqrt(mean_squared_error(y_test_return, return_predictions[name]))
                        r2 = r2_score(y_test_return, return_predictions[name])
                        acc_dir = accuracy_score(y_test_dir, (return_predictions[name] > 0).astype(int))
                        metrics_data.append({'Modelo': name, 'MAE': mae, 'RMSE': rmse, 'R²': r2, 'Acerto Direção': acc_dir})
                    metrics_df = pd.DataFrame(metrics_data)

                    # Previsões futuras
                    latest_features = X.iloc[-1:].copy()
                    latest_index = X.index[-1]
                    latest_scaled = scaler.transform(latest_features)
                    future_predictions = {name: float(model.predict(latest_scaled)[0]) for name, model in trained_models.items()}
                    ensemble_future_return = float(np.mean(list(future_predictions.values())))
                    capped_return = float(np.clip(ensemble_future_return, -0.5, 0.5))
                    daily_comp_rate = (1 + capped_return) ** (1 / 5) - 1
                    current_price = float(advanced_data['Close'].loc[latest_index])
                    predicted_prices_data = []
                    temp_price = current_price
                    for i_day in range(1, 6):
                        temp_price *= (1 + daily_comp_rate)
                        total_return = (temp_price / current_price) - 1
                        fut_date = (pd.to_datetime(latest_index) + BDay(i_day)).normalize()
                        predicted_prices_data.append({
                            'Dias': i_day,
                            'Data': fut_date.strftime('%Y-%m-%d'),
                            'Preço Previsto': temp_price,
                            'Variação': total_return,
                            'Ticker': ticker_symbol
                        })
                    predictions_df = pd.DataFrame(predicted_prices_data)

                    # Confiança
                    preds_array = np.array(list(future_predictions.values()))
                    std_preds = float(np.std(preds_array))
                    max_reasonable_std = 0.20
                    agreement = max(0.0, 1.0 - min(std_preds, max_reasonable_std) / max_reasonable_std)
                    magnitude_penalty = max(0.0, 1.0 - (abs(ensemble_future_return) / 0.5))
                    confidence_score = agreement * magnitude_penalty
                    concordance_pct = agreement * 100
                    confidence_pct = confidence_score * 100
                    if confidence_score > 0.75:
                        recomendacao = "ALTA CONFIANÇA"
                    elif confidence_score > 0.5:
                        recomendacao = "MÉDIA CONFIANÇA"
                    else:
                        recomendacao = "BAIXA CONFIANÇA"

                    # Store in session state for export/compare later
                    st.session_state['advanced_result'] = {
                        'timestamp': pd.Timestamp.now().isoformat(),
                        'ticker': ticker_symbol,
                        'used_features': used_features,
                        'metrics': metrics_df.to_dict(orient='records'),
                        'per_model_return_predictions': future_predictions,
                        'ensemble_future_return': ensemble_future_return,
                        'predictions_df': predictions_df.to_dict(orient='records'),
                        'scaler_mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
                        'scaler_scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None
                    }

                    # Exibição
                    st.subheader("📊 Performance dos Modelos (Dados de Teste)")
                    st.dataframe(metrics_df.style.format({'MAE': '{:.4f}', 'RMSE': '{:.4f}', 'R²': '{:.4f}', 'Acerto Direção': '{:.2%}'}), use_container_width=True)
                    fig_comparison = go.Figure()
                    fig_comparison.add_trace(go.Scatter(x=y_test_return.index, y=y_test_return.values, name='Retorno Real'))
                    fig_comparison.add_trace(go.Scatter(x=y_test_return.index, y=ensemble_pred, name='Previsão Ensemble', line=dict(dash='dash')))
                    fig_comparison.update_layout(title="Comparação: Retorno Real vs Previsão (Teste)", xaxis_title="Data", yaxis_title="Retorno")
                    st.plotly_chart(fig_comparison, use_container_width=True)

                    st.subheader("🎯 Previsão para os Próximos 5 Dias (compostos)")
                    st.dataframe(predictions_df.style.format({'Preço Previsto': 'R$ {:.2f}', 'Variação': '{:+.2%}'}), use_container_width=True)

                    st.subheader("📈 Projeção no gráfico")
                    fig_forecast = go.Figure()
                    hist_data = data['Close'].iloc[-60:]
                    fig_forecast.add_trace(go.Scatter(x=hist_data.index, y=hist_data.values, name='Histórico'))
                    future_dates = [pd.to_datetime(d['Data']) for d in predicted_prices_data]
                    future_prices = predictions_df['Preço Previsto'].values
                    fig_forecast.add_trace(go.Scatter(x=future_dates, y=future_prices, name='Previsão', line=dict(dash='dash')))
                    fig_forecast.update_layout(title="Projeção de Preço para os Próximos 5 Dias", xaxis_title="Data", yaxis_title="Preço (R$)")
                    st.plotly_chart(fig_forecast, use_container_width=True)

                    st.subheader("📈 Análise de Confiança da Previsão")
                    col1_conf, col2_conf, col3_conf = st.columns(3)
                    col1_conf.metric("Concordância entre Modelos", f"{concordance_pct:.1f}%")
                    col2_conf.metric("Score de Confiança", f"{confidence_pct:.1f}%")
                    col3_conf.metric("Recomendação", recomendacao)

                    st.subheader("Previsões por Modelo (retorno estimado nos próximos 5 dias)")
                    per_model = pd.DataFrame([{'Modelo': k, 'Retorno Estimado': v} for k, v in future_predictions.items()])
                    st.dataframe(per_model.style.format({'Retorno Estimado': '{:+.4f}'}), use_container_width=True)

                    st.warning("⚠️ Previsões são probabilísticas. Use apenas como apoio. Não é recomendação de investimento.")

        with col_b:
            # Export / Import utilities
            st.subheader("Exportar / Importar Análise")
            st.markdown("Exporta os resultados da análise atual (predições e metadados). Depois você pode importar para comparar com preços reais.")
            if st.button("Exportar Última Análise"):
                if st.session_state.get('advanced_result') is None:
                    st.warning("Não há análise avançada recente para exportar. Execute a previsão primeiro.")
                else:
                    adv = st.session_state['advanced_result']
                    # construir zip: predictions.csv, metrics.json, features.csv (opcional)
                    mem_zip = io.BytesIO()
                    with zipfile.ZipFile(mem_zip, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                        # predictions
                        preds_df = pd.DataFrame(adv['predictions_df'])
                        zf.writestr('predictions.csv', preds_df.to_csv(index=False))
                        # metrics
                        zf.writestr('metrics.json', json.dumps({'timestamp': adv['timestamp'], 'ticker': adv['ticker'], 'metrics': adv['metrics'], 'per_model_return_predictions': adv['per_model_return_predictions'], 'ensemble_future_return': adv['ensemble_future_return'], 'used_features': adv['used_features']}))
                        # optional: small snapshot of last features (to reproduce)
                        zf.writestr('meta.txt', f"Ticker:{adv['ticker']}\nExport time:{adv['timestamp']}\n")
                    mem_zip.seek(0)
                    st.download_button(label="Download ZIP da Análise", data=mem_zip.getvalue(), file_name=f"analise_{ticker_symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.zip", mime="application/zip")

            st.markdown("---")
            st.markdown("Importe um arquivo ZIP de análise exportado por esta ferramenta para comparar previsões com preços reais.")
            uploaded = st.file_uploader("Carregar ZIP de análise", type=["zip"])
            if uploaded is not None:
                try:
                    z = zipfile.ZipFile(io.BytesIO(uploaded.read()))
                    if 'predictions.csv' not in z.namelist():
                        st.error("ZIP inválido. Arquivo 'predictions.csv' não encontrado.")
                    else:
                        preds = pd.read_csv(io.BytesIO(z.read('predictions.csv')))
                        st.write("Predições carregadas:")
                        st.dataframe(preds, use_container_width=True)
                        if st.button("Comparar com preços reais (Yahoo Finance)"):
                            # obter intervalo de datas necessárias
                            min_date = pd.to_datetime(preds['Data'].min())
                            max_date = pd.to_datetime(preds['Data'].max())
                            df_actual = yf.download(f"{preds['Ticker'].iloc[0]}.SA", start=(min_date - pd.Timedelta(days=2)).strftime('%Y-%m-%d'), end=(max_date + pd.Timedelta(days=2)).strftime('%Y-%m-%d'), progress=False)
                            if df_actual.empty:
                                st.error("Não foi possível baixar preços reais para as datas requeridas.")
                            else:
                                df_actual.index = pd.to_datetime(df_actual.index).normalize()
                                compare_rows = []
                                for _, row in preds.iterrows():
                                    pred_date = pd.to_datetime(row['Data']).normalize()
                                    # buscar fechamento nesse dia; se não existir, buscar próximo dia útil seguinte
                                    actual_price = None
                                    if pred_date in df_actual.index:
                                        actual_price = float(df_actual.loc[pred_date, 'Close'])
                                    else:
                                        # tentar até +3 dias úteis
                                        for k in range(1,6):
                                            try_date = (pred_date + pd.Timedelta(days=k))
                                            if try_date in df_actual.index:
                                                actual_price = float(df_actual.loc[try_date, 'Close'])
                                                break
                                    compare_rows.append({
                                        'Data Prevista': pred_date.strftime('%Y-%m-%d'),
                                        'Preço Previsto': float(row['Preço Previsto']),
                                        'Preço Real (fechamento próximo disponível)': actual_price
                                    })
                                comp_df = pd.DataFrame(compare_rows)
                                comp_df['Erro Abs'] = comp_df.apply(lambda r: None if pd.isna(r['Preço Real (fechamento próximo disponível)']) else abs(r['Preço Real (fechamento próximo disponível)'] - r['Preço Previsto']), axis=1)
                                comp_df['Erro %'] = comp_df.apply(lambda r: None if pd.isna(r['Preço Real (fechamento próximo disponível)']) else (r['Preço Real (fechamento próximo disponível)'] / r['Preço Previsto'] - 1), axis=1)
                                st.subheader("Comparação Previsto x Real")
                                st.dataframe(comp_df.style.format({'Preço Previsto': 'R$ {:.2f}', 'Preço Real (fechamento próximo disponível)': 'R$ {:.2f}', 'Erro Abs': 'R$ {:.2f}', 'Erro %': '{:+.2%}'}), use_container_width=True)
                except Exception as e:
                    st.error(f"Erro ao processar ZIP: {e}")

    st.markdown("---")

    # --- Previsão de Volatilidade (simples) ---
    with st.expander("🧠 Previsão de Volatilidade (Modelo Simples)", expanded=False):
        st.write("Previsão da volatilidade para o próximo dia útil usando Random Forest simples.")
        if st.button('Executar Análise Preditiva de Volatilidade'):
            df_model = data[['Volatility']].copy().dropna()
            if len(df_model) < 30:
                st.warning("⚠️ Dados históricos insuficientes para treinar o modelo de volatilidade.")
            else:
                for i in range(1, 6):
                    df_model[f'vol_lag_{i}'] = df_model['Volatility'].shift(i)
                df_model.dropna(inplace=True)
                X_vol = df_model.drop('Volatility', axis=1)
                y_vol = df_model['Volatility']
                model_vol = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
                model_vol.fit(X_vol, y_vol)
                prediction_vol = float(model_vol.predict(X_vol.iloc[-1:].values)[0])
                last_date = pd.to_datetime(data.index[-1])
                next_day = last_date + BDay(1)
                next_day_str = next_day.strftime('%d/%m/%Y')
                if prediction_vol < 0.30:
                    status_text, status_color = "Baixa Volatilidade", "#28a745"
                elif prediction_vol >= 0.60:
                    status_text, status_color = "Alta Volatilidade", "#dc3545"
                else:
                    status_text, status_color = "Média Volatilidade", "#ffc107"
                st.markdown(f"""<div style='border:1px solid #444;border-radius:10px;padding:20px;text-align:center;background:#0f1724'><p style='font-size:1.1em;margin-bottom:5px;color:#FAFAFA'>Previsão de Volatilidade para <strong>{next_day_str}</strong></p><p style='font-size:2.0em;font-weight:bold;color:{status_color};margin:0'>{prediction_vol:.4f}</p><p style='font-size:1.0em;font-weight:bold;color:{status_color};margin-top:5px'>{status_text}</p></div>""", unsafe_allow_html=True)
                st.info('**Disclaimer:** Modelo educacional. Não é recomendação de investimento.')
                # store vol result
                st.session_state['vol_result'] = {'pred_vol': prediction_vol, 'date': next_day_str}

    last_update_date = pd.to_datetime(data.index[-1]).strftime('%d/%m/%Y')
    st.markdown("---")
    st.caption(f"📅 Última atualização dos preços: **{last_update_date}** — Dados fornecidos pelo Yahoo Finance.")
    st.markdown("<p style='text-align:center;color:#888'>Desenvolvido por Rodrigo Costa de Araujo | rodrigocosta@usp.br</p>", unsafe_allow_html=True)


