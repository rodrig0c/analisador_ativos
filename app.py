# app.py
# Versão Final com:
# - Remoção da coluna "Erro" da tabela de backtest.
# - Adição de cores na tabela de comparação de resultados (Previsto vs. Real).

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date
from pandas.tseries.offsets import BDay
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import io, zipfile, json, warnings
from math import isfinite

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Analisador de Ativos", layout="wide")

# --- Verificação de Bibliotecas Opcionais ---
HAS_XGB = False
HAS_TF = False
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    st.sidebar.warning("XGBoost não instalado. Para usar o modelo XGBoost, execute: pip install xgboost")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    HAS_TF = True
except ImportError:
    st.sidebar.warning("TensorFlow não instalado. Para usar o modelo LSTM, execute: pip install tensorflow")

# --- UI ---
st.title('📊 Assetz - Analisador de Ativos Avançado')
st.write('Um projeto de portfólio para análise preditiva de ativos, com foco em explicabilidade e backtesting de performance.')

st.sidebar.header('⚙️ Parâmetros')
start_default = date(2019, 1, 1)
start_date = st.sidebar.date_input("Data de Início", start_default, format="DD/MM/YYYY")
end_date = st.sidebar.date_input("Data de Fim", date.today(), format="DD/MM/YYYY")
view_period = st.sidebar.selectbox("Período de visualização", ["Últimos 3 meses", "Últimos 6 meses", "Último 1 ano", "Todo período"], index=2)
view_map = {"Últimos 3 meses": 63, "Últimos 6 meses": 126, "Último 1 ano": 252, "Todo período": None}
viz_days = view_map[view_period]

MIN_DAYS_CHARTS = 60
MIN_DAYS_ADVANCED = 180
FORECAST_DAYS = 5

# --- Funções Auxiliares ---
@st.cache_data
def get_tickers_from_csv():
    try:
        df = pd.read_csv('acoes-listadas-b3.csv')
        df = df.rename(columns={'Ticker':'ticker','Nome':'nome'})
        df['display'] = df['nome'] + ' (' + df['ticker'] + ')'
        return df
    except Exception:
        fallback = {'ticker':['PETR4','VALE3','ITUB4','MGLU3'],'nome':['Petrobras','Vale','Itaú','Magazine Luiza']}
        df = pd.DataFrame(fallback)
        df['display'] = df['nome'] + ' (' + df['ticker'] + ')'
        return df

@st.cache_data
def load_data(ticker, start, end):
    s = pd.to_datetime(start)
    e = pd.to_datetime(end) + pd.Timedelta(days=1)
    df = yf.download(ticker, start=s.strftime('%Y-%m-%d'), end=e.strftime('%Y-%m-%d'), progress=False)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if 'Volume' not in df.columns: df['Volume'] = 0
    df.index = pd.to_datetime(df.index)
    return df

def calculate_indicators(df):
    df = df.copy()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MM_Curta'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['MM_Longa'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['BB_Media'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['BB_Superior'] = df['BB_Media'] + 2 * df['Close'].rolling(window=20, min_periods=1).std()
    df['BB_Inferior'] = df['BB_Media'] - 2 * df['Close'].rolling(window=20, min_periods=1).std()
    df['Daily Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily Return'].rolling(window=30, min_periods=1).std() * (252**0.5)
    return df

def prepare_advanced_features(df, forecast_days=FORECAST_DAYS):
    d = df[['Close', 'Volume', 'RSI', 'MM_Curta', 'MM_Longa', 'Volatility']].copy()
    periods = [1, 3, 5, 10, 20]
    for p in periods:
        d[f'return_{p}d'] = d['Close'].pct_change(p)
        d[f'volatility_{p}d'] = d['Close'].pct_change().rolling(window=p, min_periods=1).std()
    if 'Volume' in d.columns and d['Volume'].sum() > 0:
        for p in periods:
            d[f'volume_ma_{p}d'] = d['Volume'].rolling(window=p, min_periods=1).mean()
    d['price_vs_ma20'] = d['Close'] / d['MM_Curta'].replace(0, np.nan)
    d['price_vs_ma50'] = d['Close'] / d['MM_Longa'].replace(0, np.nan)
    d['target_future_return'] = d['Close'].shift(-forecast_days) / d['Close'] - 1
    d['target_future_price'] = d['Close'].shift(-forecast_days)
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    potential = [c for c in d.columns if c.startswith(('return_', 'volume_ma_', 'volatility_', 'price_vs_'))]
    potential.extend(['RSI', 'Volatility'])
    features = [c for c in potential if c in d.columns and not d[c].isnull().all()]
    required = features + ['target_future_return', 'target_future_price']
    d.dropna(subset=required, inplace=True)
    return d, features

def create_models():
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'Neural Net': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    }
    if HAS_XGB:
        models['XGBoost'] = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    return models

def compute_price_metrics(y_true_price, y_pred_price):
    y_t = np.array(y_true_price, dtype=float)
    y_p = np.array(y_pred_price, dtype=float)
    mask = np.isfinite(y_p) & np.isfinite(y_t)
    if mask.sum() == 0: return {'MAE': None, 'RMSE': None, 'MAPE': None}
    y_t_m, y_p_m = y_t[mask], y_p[mask]
    mae = float(mean_absolute_error(y_t_m, y_p_m))
    rmse = float(np.sqrt(mean_squared_error(y_t_m, y_p_m)))
    denom = np.where(np.abs(y_t_m) < 1e-6, 1e-6, np.abs(y_t_m))
    mape = float(np.mean(np.abs((y_t_m - y_p_m) / denom)))
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def compute_return_hitrate(y_true_ret, y_pred_ret):
    y_t, y_p = np.array(y_true_ret, dtype=float), np.array(y_pred_ret, dtype=float)
    mask = np.isfinite(y_p) & np.isfinite(y_t)
    if mask.sum() == 0: return None
    return float(np.mean((np.sign(y_t[mask]) == np.sign(y_p[mask])).astype(float)))

def backtest_ensemble(adv_df, features, include_lstm=HAS_TF, progress_callback=None):
    X = adv_df[features]
    y_ret = adv_df['target_future_return']
    y_price = adv_df['target_future_price']
    split = int(len(X) * 0.8)
    if split < 2: raise ValueError("Período de treino insuficiente para backtest.")
    
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train_ret, y_test_ret = y_ret.iloc[:split], y_ret.iloc[split:]
    y_test_price = y_price.iloc[split:split+len(X_test)]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = create_models()
    trained, results = {}, {}
    total = len(models) + (1 if include_lstm else 0)
    done = 0
    trained_rf_model = None

    for name, model in models.items():
        try:
            model.fit(X_train_s, y_train_ret)
            if name == 'Random Forest': trained_rf_model = model
            preds_ret = model.predict(X_test_s)
            base_prices = adv_df['Close'].iloc[split:split+len(X_test)].values
            preds_price = (1 + preds_ret) * base_prices
            price_metrics = compute_price_metrics(y_test_price.values, preds_price)
            hit = compute_return_hitrate(y_test_ret.values, preds_ret)
            results[name] = {'price': price_metrics, 'hitrate': hit}
            trained[name] = {'pred_ret': preds_ret, 'pred_price': preds_price}
        except Exception as e:
            results[name] = {'error': str(e)}
            trained[name] = {'pred_ret': np.full(len(X_test), np.nan), 'pred_price': np.full(len(X_test), np.nan)}
        done += 1
        if progress_callback: progress_callback(done / total)

    if include_lstm:
        if HAS_TF:
            try:
                X_train_seq = X_train_s.reshape((X_train_s.shape[0], 1, -1))
                X_test_seq = X_test_s.reshape((X_test_s.shape[0], 1, -1))
                model = Sequential([
                    LSTM(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
                    Dropout(0.2), Dense(32, activation='relu'), Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(X_train_seq, y_train_ret.values, epochs=20, batch_size=16, verbose=0)
                preds_ret = model.predict(X_test_seq).flatten()
                base_prices = adv_df['Close'].iloc[split:split+len(X_test)].values
                preds_price = (1 + preds_ret) * base_prices
                results['LSTM'] = {'price': compute_price_metrics(y_test_price.values, preds_price), 
                                   'hitrate': compute_return_hitrate(y_test_ret.values, preds_ret)}
                trained['LSTM'] = {'pred_ret': preds_ret, 'pred_price': preds_price}
            except Exception as e:
                results['LSTM'] = {'error': f"LSTM error: {e}"}
                trained['LSTM'] = {'pred_ret': np.full(len(X_test), np.nan), 'pred_price': np.full(len(X_test), np.nan)}
        else:
            results['LSTM'] = {'error': 'TensorFlow não instalado'}
        done += 1
        if progress_callback: progress_callback(done / total)

    pred_price_matrix = np.vstack([v['pred_price'] for v in trained.values()])
    with np.errstate(all='ignore'):
        ensemble_price = np.nanmean(np.where(np.isfinite(pred_price_matrix), pred_price_matrix, np.nan), axis=0)
    base_prices = adv_df['Close'].iloc[split:split+len(X_test)].values
    ensemble_ret = ensemble_price / base_prices - 1
    results['Ensemble'] = {'price': compute_price_metrics(y_test_price.values, ensemble_price), 
                           'hitrate': compute_return_hitrate(y_test_ret.values, ensemble_ret)}

    feature_importance_df = None
    if trained_rf_model:
        importances = trained_rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

    df_plot = pd.DataFrame({'Data': adv_df.index[split:split+len(X_test)], 'RealPrice': y_test_price.values, 'PredPrice': ensemble_price})
    
    return {'results': results, 'trained': trained, 'df_plot': df_plot, 
            'ensemble_ret': ensemble_ret, 'y_test_ret': y_test_ret.values, 
            'feature_importance_df': feature_importance_df}

def confidence_from_price_mape(mape):
    if mape is None: return 0.0, "BAIXA CONFIANÇA", "#E74C3C"
    conf_pct = max(0.0, 1.0 - mape) * 100.0
    if mape < 0.05: return conf_pct, "ALTA CONFIANÇA", "#2ECC71"
    if mape < 0.10: return conf_pct, "MÉDIA CONFIANÇA", "#F1C40F"
    return conf_pct, "BAIXA CONFIANÇA", "#E74C3C"

# --- Fluxo Principal da Aplicação ---
tickers_df = get_tickers_from_csv()
selected_display = st.sidebar.selectbox('Escolha a Ação', tickers_df['display'])
ticker_symbol = tickers_df.loc[tickers_df['display'] == selected_display, 'ticker'].iloc[0]
company_name = tickers_df.loc[tickers_df['display'] == selected_display, 'nome'].iloc[0]
ticker = f"{ticker_symbol}.SA"

if st.session_state.get('last_ticker') != ticker_symbol:
    st.session_state['last_ticker'] = ticker_symbol
    st.session_state['advanced_result'] = None

data = load_data(ticker, start_date, end_date)
ibov = load_data('^BVSP', start_date, end_date)
if data.empty or len(data) < 1:
    st.error("❌ Não foi possível baixar dados para este ticker no período solicitado.")
    st.stop()
data = calculate_indicators(data)

# Substitua o bloco "Visão Geral do Ativo" pelo código abaixo

st.subheader('📈 Visão Geral do Ativo')

# --- NOVA LÓGICA PARA BUSCAR PREÇO ATUALIZADO ---
# Busca o dado intraday mais recente (com delay de ~15min)
try:
    live_data = yf.Ticker(ticker).history(period='1d', interval='1m')
    if not live_data.empty:
        last_price = live_data['Close'].iloc[-1]
    else: # Fallback para o fechamento do dia anterior se não houver dado intraday
        last_price = data['Close'].iloc[-1]
except Exception:
    last_price = data['Close'].iloc[-1] # Fallback em caso de erro

prev_price = data['Close'].iloc[-2] if len(data) >= 2 else last_price
price_change = last_price - prev_price
percent_change = (price_change / prev_price * 100) if prev_price != 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("🏢 Empresa", company_name)
c2.metric("💹 Ticker", ticker_symbol)
c3.metric("💰 Último Preço", f"R$ {last_price:.2f}")
c4.metric("📊 Variação (vs. Fech. Anterior)", f"{price_change:+.2f} R$", f"{percent_change:+.2f}%")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Preço e Indicadores", "Volatilidade", "Comparativo com IBOVESPA"])
view_slice = slice(-viz_days, None) if viz_days is not None else slice(None)

with tab1:
    st.subheader('Preço, Médias Móveis e Bandas de Bollinger')
    if len(data) < MIN_DAYS_CHARTS:
        st.warning(f"Dados insuficientes para gráficos históricos (mínimo {MIN_DAYS_CHARTS} dias).")
    else:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index[view_slice], y=data['BB_Superior'][view_slice],
            line=dict(width=0), showlegend=False, name='Banda Superior'
        ))
        fig.add_trace(go.Scatter(
            x=data.index[view_slice], y=data['BB_Inferior'][view_slice],
            fill='tonexty', fillcolor='rgba(0, 176, 246, 0.2)',
            line=dict(width=0), name='Bandas de Bollinger', showlegend=True
        ))
        
        fig.add_trace(go.Scatter(x=data.index[view_slice], y=data['MM_Longa'][view_slice], name='Média Móvel 50 Dias', line=dict(color='purple', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index[view_slice], y=data['MM_Curta'][view_slice], name='Banda Média (MM 20 Dias)', line=dict(color='yellow', width=1.5)))
        fig.add_trace(go.Scatter(x=data.index[view_slice], y=data['Close'][view_slice], name='Preço de Fechamento', line=dict(color='cyan', width=2)))
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader('Índice de Força Relativa (RSI)')
        fig_rsi = px.line(data[view_slice], x=data.index[view_slice], y='RSI', title='RSI')
        fig_rsi.add_hline(y=70, line_dash="dash", annotation_text="Sobrecompra")
        fig_rsi.add_hline(y=30, line_dash="dash", annotation_text="Sobrevenda")
        st.plotly_chart(fig_rsi, use_container_width=True)

with tab2:
    st.subheader('Volatilidade (janela de 30 dias)')
    if len(data) < MIN_DAYS_CHARTS:
        st.info("Volatilidade não disponível por dados históricos insuficientes.")
    else:
        fig_vol = px.line(data[view_slice], x=data.index[view_slice], y='Volatility', title='Volatilidade Anualizada')
        st.plotly_chart(fig_vol, use_container_width=True)
        current_vol = float(data['Volatility'].iloc[-1]) if pd.notna(data['Volatility'].iloc[-1]) else 0.0
        if current_vol >= 0.5: vol_label, vol_color = "ALTA VOLATILIDADE", "#E74C3C"
        elif current_vol >= 0.25: vol_label, vol_color = "VOLATILIDADE MÉDIA", "#F1C40F"
        else: vol_label, vol_color = "BAIXA VOLATILIDADE", "#2ECC71"
        st.markdown(f"<div style='background:#0b1220;padding:8px;border-radius:6px'><span style='color:{vol_color};font-size:18px;font-weight:700'>{vol_label}</span>  <span style='color:#ddd;margin-left:12px;font-size:16px'>Volatilidade atual: <strong>{current_vol:.4f}</strong></span></div>", unsafe_allow_html=True)

with tab3:
    st.subheader('Comparativo com IBOVESPA')
    if len(data) < MIN_DAYS_CHARTS or ibov.empty:
        st.info("Comparador IBOVESPA indisponível por dados insuficientes.")
    else:
        common_idx = data.index.intersection(ibov.index)
        comp_df = pd.DataFrame({
            'IBOVESPA': ibov.loc[common_idx, 'Close'] / ibov.loc[common_idx[0], 'Close'],
            ticker_symbol: data.loc[common_idx, 'Close'] / data.loc[common_idx[0], 'Close']
        })
        fig_comp = px.line(comp_df, x=comp_df.index, y=comp_df.columns, title='Performance Normalizada (base 1)')
        st.plotly_chart(fig_comp, use_container_width=True)
st.markdown("---")

st.subheader('🔮 Previsão Avançada e Análise de Performance')
st.write(f"Requer no mínimo {MIN_DAYS_ADVANCED} dias de histórico. Modelos usados: Random Forest, Gradient Boosting, SVR, Neural Net, XGBoost (se instalado) e LSTM (se instalado).")

if st.button('Executar Previsão Avançada', key='run_advanced'):
    adv_df, used_features = prepare_advanced_features(data, forecast_days=FORECAST_DAYS)
    if len(adv_df) < MIN_DAYS_ADVANCED:
        st.warning(f"Dados insuficientes para análise. Linhas válidas: {len(adv_df)}. Mínimo: {MIN_DAYS_ADVANCED}.")
    else:
        st.info(f"Utilizando {len(adv_df)} dias de dados para a análise.")
        progress_bar = st.progress(0, text="Iniciando backtest...")
        def prog(p): progress_bar.progress(min(1.0, p), text=f"Treinando modelos... {int(p*100)}%")
        
        with st.spinner("Executando backtest (80/20) e treinando modelos..."):
            bt = backtest_ensemble(adv_df, used_features, progress_callback=prog)
        progress_bar.progress(1.0, text="Análise completa!")
        st.session_state['advanced_result'] = (bt, used_features)

if 'advanced_result' in st.session_state and st.session_state['advanced_result']:
    bt, used_features = st.session_state['advanced_result']
    metrics = bt['results']
    
    rows = []
    for k, v in metrics.items():
        if 'error' in v:
            # Se deu erro, preenche métricas com None para virar N/A
            rows.append({'Modelo': k, 'MAE (R$)': None, 'RMSE (R$)': None, 'MAPE (%)': None, 'HitRate': None})
        else:
            price = v.get('price', {})
            rows.append({
                'Modelo': k,
                'MAE (R$)': price.get('MAE'),
                'RMSE (R$)': price.get('RMSE'),
                'MAPE (%)': price.get('MAPE') * 100 if price.get('MAPE') is not None else None,
                'HitRate': v.get('hitrate')
            })
    metrics_df = pd.DataFrame(rows)

    st.subheader("Resultados do Backtest (Período de Teste: 20% dos dados)")
    sty = metrics_df.style.format({
        'MAE (R$)': "R$ {:,.2f}", 'RMSE (R$)': "R$ {:,.2f}",
        'MAPE (%)': "{:.2f}%", 'HitRate': "{:.2%}",
    }, na_rep="N/A")
    st.dataframe(sty, use_container_width=True)

    ensemble_mape = metrics.get('Ensemble', {}).get('price', {}).get('MAPE')
    conf_pct, conf_label, conf_color = confidence_from_price_mape(ensemble_mape)
    st.markdown(f"<div style='background:#0b1220;padding:8px;border-radius:8px'><span style='color:#ddd;font-size:16px;font-weight:700'>Confiança (1 − MAPE_preço):</span> <span style='color:{conf_color};font-size:20px;font-weight:900;margin-left:12px'>{conf_label} ({conf_pct:.1f}%)</span></div>", unsafe_allow_html=True)
    st.markdown("---")

    with st.expander("🔍 Drivers do Modelo (Feature Importance)"):
        st.info("Este gráfico mostra quais variáveis o modelo Random Forest considerou mais importantes para fazer suas previsões.")
        if bt.get('feature_importance_df') is not None:
            fig_fi = px.bar(bt['feature_importance_df'].head(15), x='Importance', y='Feature', orientation='h', title='Top 15 Features Mais Importantes')
            st.plotly_chart(fig_fi.update_layout(yaxis={'categoryorder':'total ascending'}), use_container_width=True)
        else:
            st.warning("Não foi possível gerar a importância das features.")

    with st.expander("💰 Simulação de Estratégia (Backtest de Rentabilidade)"):
        st.info("Simula o retorno de R$ 10.000 comparando a estratégia do modelo (comprar se a previsão for de alta) com 'Comprar e Segurar' (Buy & Hold).")
        pred_returns, real_returns, dates = bt['ensemble_ret'], bt['y_test_ret'], bt['df_plot']['Data']
        strategy_daily_returns = np.where(pred_returns > 0, real_returns, 0)
        initial_capital = 10000
        plot_df = pd.DataFrame({
            'Data': dates,
            'Estratégia do Modelo': (1 + strategy_daily_returns).cumprod() * initial_capital,
            'Comprar e Segurar (Buy & Hold)': (1 + real_returns).cumprod() * initial_capital
        })
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=plot_df['Data'], y=plot_df['Estratégia do Modelo'], name='Estratégia do Modelo', line=dict(color='cyan')))
        fig_sim.add_trace(go.Scatter(x=plot_df['Data'], y=plot_df['Comprar e Segurar (Buy & Hold)'], name='Buy & Hold', line=dict(color='gray', dash='dash')))
        st.plotly_chart(fig_sim.update_layout(title='Evolução do Capital (R$)', xaxis_title='Data', yaxis_title='Capital (R$)'), use_container_width=True)

    st.subheader("Projeção de Preço para os Próximos 5 Dias")
    trained, current_price, current_date = bt['trained'], data['Close'].iloc[-1], pd.to_datetime(data.index[-1])
    valid_preds = [info['pred_ret'][-1] for info in trained.values() if info.get('pred_ret') is not None and len(info['pred_ret']) > 0 and np.isfinite(info['pred_ret'][-1])]
    ensemble_future_ret = np.mean(valid_preds) if valid_preds else 0.0
    daily_rate = (1 + np.clip(ensemble_future_ret, -0.5, 0.5)) ** (1/FORECAST_DAYS) - 1
    
    preds_display = [{'Data': (current_date + BDay(d)).strftime('%d/%m/%Y'), 'Preço Previsto': current_price * ((1 + daily_rate) ** d)} for d in range(1, FORECAST_DAYS + 1)]
    preds_df = pd.DataFrame(preds_display)
    for r in preds_df.to_dict(orient='records'):
        r['Variação'] = r['Preço Previsto'] / current_price - 1
        var_color = "#2ECC71" if r['Variação'] > 0 else "#E74C3C"
        st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center;padding:8px 6px;border-radius:6px;margin-bottom:6px;background:#0b1220'><div style='color:#ddd;font-size:16px'>{r['Data']}</div><div style='color:#00BFFF;font-size:28px;font-weight:900'>R$ {r['Preço Previsto']:,.2f}</div><div style='color:{var_color};font-size:16px'>{r['Variação']:+.2%}</div></div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📥 Exportar Análise")
    adv_result_serializable = {
        'ticker': ticker_symbol, 'timestamp': pd.Timestamp.now().isoformat(),
        'metrics': {k: (v if 'error' in v else {'price': v.get('price'), 'hitrate': v.get('hitrate')}) for k, v in bt['results'].items()},
        'features_used': used_features, 'confidence_pct': conf_pct
    }
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('predictions.csv', preds_df.to_csv(index=False))
        zf.writestr('metadata.json', json.dumps(adv_result_serializable, indent=4))
    mem.seek(0)
    st.download_button(label="Baixar Resultados da Análise (ZIP)", data=mem,
        file_name=f"analise_{ticker_symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.zip", mime="application/zip")


st.markdown("---")
st.subheader("📂 Importar e Comparar Previsões Exportadas")
uploaded = st.file_uploader("Carregar ZIP de análise exportada por esta ferramenta", type=["zip"])
if uploaded is not None:
    try:
        z = zipfile.ZipFile(io.BytesIO(uploaded.read()))
        if 'metadata.json' in z.namelist() and 'predictions.csv' in z.namelist():
            meta = json.loads(z.read('metadata.json'))
            preds = pd.read_csv(io.BytesIO(z.read('predictions.csv')))
            st.write(f"Análise importada para o ticker **{meta.get('ticker')}** de **{meta.get('timestamp')}**.")
            st.dataframe(preds)
            
            if st.button("Comparar Previsão Importada com Preços Reais"):
                ticker_to_check = f"{meta.get('ticker')}.SA"
                dates_to_check = pd.to_datetime(preds['Data'], dayfirst=True)
                start_check, end_check = dates_to_check.min() - BDay(5), dates_to_check.max() + BDay(5)
                
                actual_data = yf.download(ticker_to_check, start=start_check, end=end_check, progress=False)
                if isinstance(actual_data.columns, pd.MultiIndex):
                    actual_data.columns = actual_data.columns.get_level_values(0)

                if not actual_data.empty:
                    actual_data.index = pd.to_datetime(actual_data.index).normalize()
                    preds['Data'] = pd.to_datetime(preds['Data'], dayfirst=True).dt.normalize()
                    merged_df = pd.merge(preds, actual_data[['Close']], left_on='Data', right_index=True, how='left').rename(columns={'Close': 'Preço Real'})
                    
                    merged_df['Erro (R$)'] = merged_df['Preço Real'] - merged_df['Preço Previsto']
                    merged_df['Erro (%)'] = (merged_df['Erro (R$)'] / merged_df['Preço Previsto'])
                    
                    # --- NOVA FUNÇÃO DE ESTILO PARA A TABELA DE COMPARAÇÃO ---
                    def style_error_percent(val):
                        if pd.isna(val): return ''
                        color = '#E74C3C' # Vermelho para erros grandes
                        if abs(val) < 0.05: color = '#F1C40F' # Amarelo para erros médios
                        if abs(val) < 0.02: color = '#2ECC71' # Verde para erros pequenos
                        return f'color: {color}'

                    styled_df = merged_df.style.format({
                        'Preço Previsto': 'R$ {:,.2f}', 'Preço Real': 'R$ {:,.2f}',
                        'Erro (R$)': 'R$ {:,.2f}', 'Erro (%)': '{:,.2%}'
                    }, na_rep="N/A").apply(lambda s: s.map(style_error_percent), subset=['Erro (%)'])
                    
                    st.dataframe(styled_df)
                else:
                    st.warning("Não foi possível baixar os dados reais para comparação.")
        else:
            st.error("ZIP inválido. Arquivos 'metadata.json' ou 'predictions.csv' não encontrados.")
    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo ZIP: {e}")

# --- Rodapé ---
st.markdown("---")
last_update = pd.to_datetime(data.index[-1]).strftime('%d/%m/%Y %H:%M:%S')
st.caption(f"Última atualização dos preços: **{last_update}** — Dados: Yahoo Finance.")
st.markdown("<p style='text-align:center;color:#888'>Desenvolvido por Rodrigo Costa de Araujo | rodrigocosta@usp.br</p>", unsafe_allow_html=True)
