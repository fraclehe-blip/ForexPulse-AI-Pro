# ============================================================
# app.py — ForexPulse AI Pro
# Alertas LSTM + Optimización Genética para Scalping
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import datetime
import warnings
import threading
warnings.filterwarnings("ignore")

# ── Módulos propios ──────────────────────────────────────────
from indicators import add_all_indicators, count_confirmations, PAIRS, TIMEFRAMES
from lstm_model  import get_predictor, TF_AVAILABLE
from genetic_optimizer import GeneticOptimizer
from backtester  import Backtester

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN STREAMLIT
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ForexPulse AI Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado (tema oscuro profesional) ──────────────
st.markdown("""
<style>
  /* ── Fuentes ── */
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@700;800&family=Inter:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
  }

  h1, h2, h3 { font-family: 'Syne', sans-serif; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1425 0%, #111827 100%);
    border-right: 1px solid #1e293b;
  }

  /* ── Métricas ── */
  [data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 12px 16px !important;
  }
  [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem !important;
    font-weight: 700;
  }

  /* ── Señal card ── */
  .signal-buy {
    background: linear-gradient(135deg, #052e16 0%, #14532d 100%);
    border: 2px solid #22c55e;
    border-radius: 16px;
    padding: 20px 24px;
    margin: 10px 0;
    box-shadow: 0 0 30px rgba(34,197,94,0.2);
  }
  .signal-sell {
    background: linear-gradient(135deg, #2d0a0a 0%, #7f1d1d 100%);
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 20px 24px;
    margin: 10px 0;
    box-shadow: 0 0 30px rgba(239,68,68,0.2);
  }
  .signal-neutral {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 16px;
    padding: 20px 24px;
    margin: 10px 0;
  }

  /* ── Botones ── */
  .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(124,58,237,0.4);
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: #94a3b8;
  }
  .stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom-color: #38bdf8 !important;
  }

  /* ── Selectbox / inputs ── */
  .stSelectbox > div > div,
  .stSlider > div { color: #e2e8f0; }

  /* ── Separador ── */
  hr { border-color: #1e293b; }

  /* ── Código / mono ── */
  code { font-family: 'JetBrains Mono', monospace; color: #a5f3fc; }

  /* ── Tabla ── */
  .dataframe { font-size: 0.82rem; }

  /* ── Barra de título ── */
  .title-bar {
    background: linear-gradient(90deg, #0f172a, #1e293b);
    border-bottom: 1px solid #334155;
    padding: 8px 0;
    margin-bottom: 16px;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════

def init_state():
    defaults = {
        "predictor":        None,
        "trained_pair":     None,
        "trained_tf":       None,
        "opt_params":       None,
        "opt_results":      None,
        "signal_history":   [],
        "last_refresh":     0,
        "df_cache":         {},
        "telegram_token":   "",
        "telegram_chat_id": "",
        "risk_pct":         1.0,
        "capital":          10_000.0,
        "auto_refresh":     False,
        "refresh_interval": 60,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ══════════════════════════════════════════════════════════════
# DESCARGA DE DATOS
# ══════════════════════════════════════════════════════════════

@st.cache_data(ttl=60, show_spinner=False)
def fetch_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """Descarga datos de yfinance con fallback."""
    try:
        import yfinance as yf
        df = yf.download(symbol, interval=interval, period=period,
                         progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("DataFrame vacío")
        # Aplanar MultiIndex si existe
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception as e:
        st.warning(f"⚠️ Error descargando {symbol}: {e}. Generando datos sintéticos.")
        return _synthetic_data(interval)


def _synthetic_data(interval: str = "5m") -> pd.DataFrame:
    """Datos sintéticos como fallback para demostración."""
    np.random.seed(42)
    n = 500 if interval == "5m" else 300
    freq = "5min" if interval == "5m" else "15min"
    idx  = pd.date_range(end=pd.Timestamp.now(), periods=n, freq=freq)
    price = 1.0850
    closes = [price]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + np.random.normal(0, 0.0003)))
    closes = np.array(closes)
    spread = 0.0002
    return pd.DataFrame({
        "Open":   closes * (1 - spread / 2),
        "High":   closes * (1 + abs(np.random.normal(0, 0.0005, n))),
        "Low":    closes * (1 - abs(np.random.normal(0, 0.0005, n))),
        "Close":  closes,
        "Volume": np.random.randint(1000, 50000, n).astype(float),
    }, index=idx)


def get_live_price(symbol: str) -> float:
    """Obtiene el precio más reciente."""
    try:
        import yfinance as yf
        tk = yf.Ticker(symbol)
        hist = tk.history(period="1d", interval="1m")
        if not hist.empty:
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


# ══════════════════════════════════════════════════════════════
# MOTOR DE SEÑALES
# ══════════════════════════════════════════════════════════════

def compute_signal(df: pd.DataFrame, params: dict,
                   lstm_predictor) -> dict:
    """
    Genera la señal combinada (LSTM + indicadores técnicos).
    Retorna dict con toda la información de la señal.
    """
    df_ind = add_all_indicators(df, params)
    if len(df_ind) < 5:
        return {"direction": "NEUTRAL", "prob": 0.5, "confirms": 0}

    row         = df_ind.iloc[-1]
    lstm_prob   = lstm_predictor.predict_proba(df_ind)
    lstm_thresh = params.get("lstm_threshold", 0.72)
    min_conf    = int(round(params.get("min_confirms", 4)))
    adx_thresh  = params.get("adx_threshold", 22)
    sl_mult     = params.get("sl_atr_mult", 1.5)
    tp_mult     = params.get("tp_atr_mult", 2.5)
    atr         = max(row["ATR"], 1e-5)

    direction = "NEUTRAL"
    confirms  = 0

    if lstm_prob >= lstm_thresh and row["ADX"] >= adx_thresh:
        confirms = count_confirmations(row, "BUY", params)
        if confirms >= min_conf:
            direction = "BUY"
    elif lstm_prob <= (1 - lstm_thresh) and row["ADX"] >= adx_thresh:
        confirms = count_confirmations(row, "SELL", params)
        if confirms >= min_conf:
            direction = "SELL"
    else:
        # Determinar qué sería más probable para mostrar
        conf_buy  = count_confirmations(row, "BUY", params)
        conf_sell = count_confirmations(row, "SELL", params)
        confirms  = max(conf_buy, conf_sell)

    close = float(row["Close"])
    sl = tp = 0.0

    if direction == "BUY":
        sl = close - sl_mult * atr
        tp = close + tp_mult * atr
    elif direction == "SELL":
        sl = close + sl_mult * atr
        tp = close - tp_mult * atr

    # Fuerza de señal
    strength = "DÉBIL"
    if direction != "NEUTRAL":
        prob_dist = abs(lstm_prob - 0.5) * 2   # 0-1
        conf_ratio = confirms / 7
        combined  = 0.5 * prob_dist + 0.5 * conf_ratio
        if combined > 0.75:
            strength = "FUERTE"
        elif combined > 0.55:
            strength = "MODERADA"

    return {
        "direction":  direction,
        "prob":       lstm_prob,
        "confirms":   confirms,
        "strength":   strength,
        "close":      close,
        "sl":         sl,
        "tp":         tp,
        "atr":        float(atr),
        "adx":        float(row["ADX"]),
        "rsi":        float(row["RSI"]),
        "timestamp":  datetime.datetime.now(),
        "df_ind":     df_ind,
    }


# ══════════════════════════════════════════════════════════════
# ALERTAS
# ══════════════════════════════════════════════════════════════

def send_telegram(token: str, chat_id: str, message: str) -> bool:
    """Envía mensaje via Telegram Bot API."""
    try:
        import requests as req
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message,
                   "parse_mode": "Markdown"}
        r = req.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def build_telegram_message(pair: str, sig: dict) -> str:
    emoji = "🟢" if sig["direction"] == "BUY" else "🔴"
    dir_es = "COMPRA" if sig["direction"] == "BUY" else "VENTA"
    conf = "Alta" if sig["strength"] == "FUERTE" else "Media"
    return (
        f"{emoji} *SEÑAL DE {dir_es} — {pair}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Precio: `{sig['close']:.5f}`\n"
        f"🧠 Prob LSTM: `{sig['prob']*100:.1f}%`\n"
        f"📊 Confirmaciones: `{sig['confirms']}/7`\n"
        f"💪 Fuerza: `{sig['strength']}`\n"
        f"🎯 TP: `{sig['tp']:.5f}`\n"
        f"🛡️ SL: `{sig['sl']:.5f}`\n"
        f"📡 ADX: `{sig['adx']:.1f}`  |  RSI: `{sig['rsi']:.1f}`\n"
        f"🕐 {sig['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ _Solo educativo. No es consejo financiero._"
    )


# ══════════════════════════════════════════════════════════════
# GRÁFICO PLOTLY
# ══════════════════════════════════════════════════════════════

def build_chart(df_ind: pd.DataFrame, pair: str,
                signal: dict = None,
                lstm_forecast: list = None) -> go.Figure:
    """
    Construye el gráfico interactivo de velas + indicadores.
    """
    # Limitar a las últimas 150 velas para velocidad
    df = df_ind.tail(150).copy()
    idx = df.index

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.50, 0.18, 0.17, 0.15],
        vertical_spacing=0.02,
        subplot_titles=("", "RSI (14)", "MACD", "Stoch %K/%D"),
    )

    # ── Velas ──
    fig.add_trace(go.Candlestick(
        x=idx, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Precio",
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
        increasing_fillcolor="#166534",
        decreasing_fillcolor="#7f1d1d",
    ), row=1, col=1)

    # ── EMAs ──
    for col, color, name in [
        ("EMA_fast", "#38bdf8", "EMA 9"),
        ("EMA_slow", "#fb923c", "EMA 21"),
        ("BB_upper", "#6b7280", "BB Up"),
        ("BB_mid",   "#4b5563", "BB Mid"),
        ("BB_lower", "#6b7280", "BB Low"),
    ]:
        if col in df.columns:
            dash = "dot" if "BB" in col else "solid"
            width = 1 if "BB" in col else 1.5
            fig.add_trace(go.Scatter(
                x=idx, y=df[col], name=name,
                line=dict(color=color, width=width, dash=dash),
                opacity=0.85,
            ), row=1, col=1)

    # ── Forecast LSTM ──
    if lstm_forecast:
        last_idx = idx[-1]
        try:
            freq = pd.infer_freq(idx[-10:]) or "5T"
            future_idx = pd.date_range(start=last_idx, periods=len(lstm_forecast)+1,
                                       freq=freq)[1:]
            base_price = float(df["Close"].iloc[-1])
            atr_val    = float(df["ATR"].iloc[-1]) if "ATR" in df else 0.0002
            # Convertir probs en precio estimado
            fc_prices = [base_price]
            for p in lstm_forecast:
                delta = atr_val * (0.3 if p > 0.5 else -0.3)
                fc_prices.append(fc_prices[-1] + delta)

            fig.add_trace(go.Scatter(
                x=list(idx[-3:]) + list(future_idx),
                y=list(df["Close"].iloc[-3:]) + fc_prices[1:],
                name="Forecast LSTM",
                line=dict(color="#a78bfa", width=2, dash="dash"),
                opacity=0.7,
            ), row=1, col=1)
        except Exception:
            pass

    # ── Señal en el gráfico ──
    if signal and signal["direction"] != "NEUTRAL":
        color  = "#22c55e" if signal["direction"] == "BUY" else "#ef4444"
        symbol = "triangle-up" if signal["direction"] == "BUY" else "triangle-down"
        fig.add_trace(go.Scatter(
            x=[idx[-1]], y=[signal["close"]],
            mode="markers+text",
            marker=dict(size=18, color=color, symbol=symbol),
            text=[signal["direction"]], textposition="top center",
            textfont=dict(color=color, size=12),
            name=f"Señal {signal['direction']}",
        ), row=1, col=1)

    # ── RSI ──
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=idx, y=df["RSI"], name="RSI",
                                  line=dict(color="#a78bfa", width=1.5)),
                      row=2, col=1)
        for level, color in [(70, "#ef4444"), (30, "#22c55e"), (50, "#4b5563")]:
            fig.add_hline(y=level, line_dash="dot",
                          line_color=color, opacity=0.5, row=2, col=1)

    # ── MACD ──
    if "MACD" in df.columns:
        colors_hist = ["#22c55e" if v >= 0 else "#ef4444"
                       for v in df["MACD_hist"]]
        fig.add_trace(go.Bar(x=idx, y=df["MACD_hist"],
                             name="Hist", marker_color=colors_hist,
                             opacity=0.7), row=3, col=1)
        fig.add_trace(go.Scatter(x=idx, y=df["MACD"],
                                  name="MACD", line=dict(color="#38bdf8", width=1.2)),
                      row=3, col=1)
        fig.add_trace(go.Scatter(x=idx, y=df["MACD_sig"],
                                  name="Signal", line=dict(color="#fb923c", width=1.2)),
                      row=3, col=1)

    # ── Stochastics ──
    if "Stoch_K" in df.columns:
        fig.add_trace(go.Scatter(x=idx, y=df["Stoch_K"],
                                  name="%K", line=dict(color="#34d399", width=1.2)),
                      row=4, col=1)
        fig.add_trace(go.Scatter(x=idx, y=df["Stoch_D"],
                                  name="%D", line=dict(color="#fbbf24", width=1.2)),
                      row=4, col=1)
        for level in [80, 20]:
            fig.add_hline(y=level, line_dash="dot",
                          line_color="#6b7280", opacity=0.5, row=4, col=1)

    fig.update_layout(
        height=720,
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0d1117",
        font=dict(color="#e2e8f0", family="JetBrains Mono", size=11),
        legend=dict(bgcolor="#111827", bordercolor="#1e293b",
                    borderwidth=1, font_size=10),
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
    )
    for i in range(1, 5):
        fig.update_xaxes(
            gridcolor="#1e293b", zerolinecolor="#374151",
            showgrid=True, row=i, col=1,
        )
        fig.update_yaxes(
            gridcolor="#1e293b", zerolinecolor="#374151",
            showgrid=True, row=i, col=1,
        )

    fig.update_layout(title=dict(
        text=f"<b>{pair}</b> — Análisis Técnico + Predicción LSTM",
        font=dict(size=16, color="#38bdf8"),
        x=0.01,
    ))
    return fig


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 10px 0 20px 0'>
          <span style='font-family:Syne;font-size:1.4rem;font-weight:800;
            background:linear-gradient(90deg,#38bdf8,#a78bfa);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
            📈 ForexPulse AI Pro
          </span><br>
          <span style='font-size:0.72rem;color:#64748b'>
            LSTM + Genetic Optimization
          </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🎯 Configuración")
        pair = st.selectbox("Par de divisas", list(PAIRS.keys()), index=0)
        tf   = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=0)

        st.markdown("---")
        st.markdown("### 🔄 Auto-refresco")
        auto = st.toggle("Activar refresco automático",
                         value=st.session_state.auto_refresh)
        interval = st.slider("Intervalo (seg)", 30, 120, 60, 10)
        st.session_state.auto_refresh     = auto
        st.session_state.refresh_interval = interval

        st.markdown("---")
        st.markdown("### 💰 Gestión de Riesgo")
        capital  = st.number_input("Capital ($)", 1000, 1_000_000,
                                    int(st.session_state.capital), 500)
        risk_pct = st.slider("Riesgo por trade (%)", 0.5, 5.0,
                              st.session_state.risk_pct, 0.5)
        st.session_state.capital  = float(capital)
        st.session_state.risk_pct = risk_pct

        st.markdown("---")
        st.markdown("### 📬 Alertas Telegram")
        token   = st.text_input("Bot Token", value=st.session_state.telegram_token,
                                 type="password", placeholder="123456:ABC-DEF...")
        chat_id = st.text_input("Chat ID", value=st.session_state.telegram_chat_id,
                                 placeholder="-100123456789")
        st.session_state.telegram_token   = token
        st.session_state.telegram_chat_id = chat_id

        if token and chat_id:
            if st.button("🧪 Probar conexión"):
                ok = send_telegram(token, chat_id,
                                   "✅ *ForexPulse AI Pro* conectado correctamente.")
                if ok:
                    st.success("✅ Telegram OK")
                else:
                    st.error("❌ Error de conexión")

        st.markdown("---")
        tf_info = TF_AVAILABLE
        st.markdown(
            f"🧠 **TensorFlow**: {'✅ Disponible' if tf_info else '⚠️ No instalado'}"
        )
        st.caption("v1.0 · Uso educativo")

    return pair, tf, capital, risk_pct


# ══════════════════════════════════════════════════════════════
# TABS PRINCIPALES
# ══════════════════════════════════════════════════════════════

def tab_dashboard(pair: str, tf: str):
    """Tab principal: señales en tiempo real."""
    symbol = PAIRS[pair]
    tf_cfg = TIMEFRAMES[tf]

    col_refresh, col_status = st.columns([3, 1])
    with col_refresh:
        st.markdown(f"**{pair}** · `{tf}` · Datos en tiempo real via yfinance")
    with col_status:
        if st.button("🔄 Refrescar ahora"):
            st.cache_data.clear()
            st.rerun()

    # ── Descarga ──
    with st.spinner("Cargando datos..."):
        df_raw = fetch_data(symbol, tf_cfg["interval"], tf_cfg["period"])

    if df_raw.empty:
        st.error("No se pudieron obtener datos. Revisa la conexión.")
        return

    # ── Obtener o entrenar predictor ──
    retrain_needed = (
        st.session_state.predictor is None or
        st.session_state.trained_pair != pair or
        st.session_state.trained_tf   != tf
    )
    if retrain_needed:
        with st.spinner("🧠 Entrenando modelo LSTM... (primera vez, ~30-60s)"):
            predictor = get_predictor()
            df_ind    = add_all_indicators(df_raw)
            result    = predictor.train(df_ind, epochs=30, verbose=0)
            st.session_state.predictor    = predictor
            st.session_state.trained_pair = pair
            st.session_state.trained_tf   = tf
            acc = result.get("val_accuracy", 0) * 100
            if acc > 0:
                st.success(f"✅ Modelo entrenado — Val Accuracy: {acc:.1f}%")
    else:
        predictor = st.session_state.predictor

    # ── Parámetros ──
    params = (st.session_state.opt_params
              if st.session_state.opt_params
              else GeneticOptimizer.default_params())

    # ── Señal ──
    with st.spinner("Calculando señal..."):
        sig = compute_signal(df_raw, params, predictor)

    df_ind = sig.pop("df_ind")
    price  = sig["close"]

    # Precio live (intento rápido)
    live_price = get_live_price(symbol)
    if live_price == 0.0:
        live_price = price

    # ── KPIs superiores ──
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("💱 Precio Live", f"{live_price:.5f}")
    k2.metric("🧠 Prob LSTM", f"{sig['prob']*100:.1f}%",
              delta=f"{'↑' if sig['prob']>0.5 else '↓'} BUY" if sig['direction']!='NEUTRAL' else None)
    k3.metric("📊 ADX", f"{sig['adx']:.1f}",
              delta="Tendencia" if sig['adx'] > 25 else "Rango")
    k4.metric("📈 RSI", f"{sig['rsi']:.1f}")
    k5.metric("⚡ Confirmaciones", f"{sig['confirms']}/7")

    st.markdown("---")

    # ── Card de señal ──
    col_sig, col_chart_info = st.columns([1.2, 2.8])

    with col_sig:
        direction = sig["direction"]
        if direction == "BUY":
            css_cls = "signal-buy"
            emoji   = "🚨"
            label   = "SEÑAL FUERTE DE COMPRA"
            color   = "#22c55e"
        elif direction == "SELL":
            css_cls = "signal-sell"
            emoji   = "🚨"
            label   = "SEÑAL FUERTE DE VENTA"
            color   = "#ef4444"
        else:
            css_cls = "signal-neutral"
            emoji   = "⏸️"
            label   = "SIN SEÑAL — ESPERAR"
            color   = "#94a3b8"

        st.markdown(f"""
        <div class="{css_cls}">
          <div style='font-family:Syne;font-size:1.05rem;font-weight:800;
            color:{color};margin-bottom:8px'>{emoji} {label}</div>
          <div style='font-family:JetBrains Mono;font-size:1.1rem;
            font-weight:700;color:#f1f5f9'>{pair} @ {price:.5f}</div>
          <div style='margin-top:10px;font-size:0.88rem;line-height:1.8'>
            🧠 Prob LSTM: <b style='color:{color}'>{sig['prob']*100:.1f}%</b><br>
            💪 Fuerza: <b>{sig['strength']}</b><br>
            ✅ Confirms: <b>{sig['confirms']}/7</b>
          </div>
          {'<hr style="border-color:#334155;margin:10px 0">' if direction != 'NEUTRAL' else ''}
          {'f"<div style=\'font-size:0.85rem;line-height:1.8\'>🎯 TP: <b style=\'color:#22c55e\'>{sig[\'tp\']:.5f}</b><br>🛡️ SL: <b style=\'color:#ef4444\'>{sig[\'sl\']:.5f}</b></div>"' if direction != 'NEUTRAL' else ''}
        </div>
        """.replace(
            "f\"<div", f"<div" if direction != "NEUTRAL" else ""
        ), unsafe_allow_html=True)

        if direction != "NEUTRAL":
            st.markdown(f"""
            <div style='background:#111827;border:1px solid #1e293b;
              border-radius:10px;padding:12px;margin-top:8px;
              font-size:0.85rem;line-height:2.0;font-family:JetBrains Mono'>
              🎯 TP: <b style='color:#22c55e'>{sig['tp']:.5f}</b><br>
              🛡️ SL: <b style='color:#ef4444'>{sig['sl']:.5f}</b><br>
              📏 ATR: <b>{sig['atr']:.5f}</b><br>
              💵 Riesgo: <b>${st.session_state.capital * st.session_state.risk_pct/100:.0f}</b>
            </div>
            """, unsafe_allow_html=True)

            # Guardar en historial
            sig_record = {**sig, "pair": pair, "tf": tf, "direction": direction}
            hist = st.session_state.signal_history
            if (not hist or
                (hist[-1]["direction"] != direction or
                 abs(hist[-1]["close"] - price) > sig["atr"] * 0.3)):
                st.session_state.signal_history.insert(0, sig_record)
                st.session_state.signal_history = st.session_state.signal_history[:50]

                # Enviar Telegram
                if st.session_state.telegram_token and sig["strength"] == "FUERTE":
                    msg = build_telegram_message(pair, sig)
                    send_telegram(st.session_state.telegram_token,
                                  st.session_state.telegram_chat_id, msg)

        # ── Tamaño de posición sugerido ──
        if direction != "NEUTRAL" and sig["atr"] > 0:
            risk_usd   = st.session_state.capital * st.session_state.risk_pct / 100
            pos_size   = risk_usd / (params.get("sl_atr_mult", 1.5) * sig["atr"])
            st.markdown(f"""
            <div style='background:#1e3a5f22;border:1px solid #1e40af55;
              border-radius:8px;padding:10px;margin-top:8px;font-size:0.82rem'>
              📐 <b>Posición sugerida</b><br>
              Lotes ≈ <b>{pos_size:.2f}</b> unidades/pip<br>
              <span style='color:#64748b;font-size:0.75rem'>
                Basado en {st.session_state.risk_pct}% riesgo · ${st.session_state.capital:,.0f} capital
              </span>
            </div>
            """, unsafe_allow_html=True)

    with col_chart_info:
        # ── Forecast LSTM ──
        forecast = predictor.forecast_next_n(df_ind.tail(200), n=5)
        fig = build_chart(df_ind, pair, sig, forecast)
        st.plotly_chart(fig, use_container_width=True)

    # ── Historial de señales recientes ──
    if st.session_state.signal_history:
        st.markdown("### 📋 Señales Recientes")
        rows = []
        for s in st.session_state.signal_history[:10]:
            rows.append({
                "Hora":      s["timestamp"].strftime("%H:%M:%S"),
                "Par":       s.get("pair", pair),
                "Dirección": s["direction"],
                "Precio":    f"{s['close']:.5f}",
                "Prob %":    f"{s['prob']*100:.1f}",
                "Confirms":  s["confirms"],
                "Fuerza":    s["strength"],
            })
        df_hist = pd.DataFrame(rows)
        # Colorear filas
        def color_row(row):
            c = "#166534" if row["Dirección"] == "BUY" else \
                "#7f1d1d" if row["Dirección"] == "SELL" else "#1e293b"
            return [f"background-color:{c}"] * len(row)
        st.dataframe(
            df_hist.style.apply(color_row, axis=1),
            use_container_width=True, height=250,
        )

    # ── Auto-refresco ──
    if st.session_state.auto_refresh:
        elapsed = time.time() - st.session_state.last_refresh
        remaining = max(0, st.session_state.refresh_interval - int(elapsed))
        st.caption(f"⏱️ Próximo refresco en {remaining}s")
        if elapsed >= st.session_state.refresh_interval:
            st.session_state.last_refresh = time.time()
            st.cache_data.clear()
            st.rerun()
        time.sleep(1)
        st.rerun()


# ── Tab Backtesting ──────────────────────────────────────────

def tab_backtesting(pair: str, tf: str):
    st.markdown("### 🔬 Backtesting Profesional")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_windows = st.slider("Ventanas walk-forward", 2, 6, 4)
    with col2:
        train_ratio = st.slider("Ratio entrenamiento", 0.5, 0.8, 0.7, 0.05)
    with col3:
        run_bt = st.button("▶️ Ejecutar Backtest", type="primary")

    if not run_bt:
        st.info("💡 Configura los parámetros y ejecuta el backtest.")
        return

    symbol = PAIRS[pair]
    tf_cfg = TIMEFRAMES[tf]

    with st.spinner("Descargando datos..."):
        df_raw = fetch_data(symbol, tf_cfg["interval"], tf_cfg["period"])

    params    = st.session_state.opt_params or GeneticOptimizer.default_params()
    predictor = st.session_state.predictor

    with st.spinner("Ejecutando backtest walk-forward..."):
        bt = Backtester(st.session_state.capital, st.session_state.risk_pct)
        wf_results = bt.walk_forward(df_raw, params, n_windows, train_ratio, predictor)
        # Backtest global
        bt_global  = Backtester(st.session_state.capital, st.session_state.risk_pct)
        global_m   = bt_global.run(df_raw, params, predictor)

    # ── KPIs globales ──
    m = global_m
    st.markdown("#### 📊 Métricas Globales")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Win Rate",        f"{m['win_rate']:.1f}%",
              delta="OK" if m['win_rate'] > 55 else "Bajo")
    c2.metric("Profit Factor",   f"{m['profit_factor']:.2f}",
              delta="OK" if m['profit_factor'] > 1.3 else "Bajo")
    c3.metric("Trades",          str(m['n_trades']))
    c4.metric("Sharpe Ratio",    f"{m['sharpe']:.2f}")
    c5.metric("Max Drawdown",    f"{m['max_drawdown']:.1f}%")
    c6.metric("Retorno Total",   f"{m['total_return']:.1f}%")

    # ── Equity Curve ──
    if m["equity_curve"]:
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            y=m["equity_curve"],
            mode="lines",
            name="Equity",
            line=dict(color="#38bdf8", width=2),
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.1)",
        ))
        fig_eq.update_layout(
            title="Equity Curve",
            paper_bgcolor="#0a0e1a",
            plot_bgcolor="#0d1117",
            font=dict(color="#e2e8f0"),
            height=280,
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b"),
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    # ── Walk-Forward ──
    if wf_results:
        st.markdown("#### 🪟 Resultados Walk-Forward")
        wf_df = pd.DataFrame([{
            "Ventana": r["window"],
            "Inicio":  r.get("test_start", ""),
            "Fin":     r.get("test_end", ""),
            "Trades":  r["n_trades"],
            "Win Rate": f"{r['win_rate']:.1f}%",
            "PF":       f"{r['profit_factor']:.2f}",
            "Sharpe":   f"{r['sharpe']:.2f}",
            "DD":       f"{r['max_drawdown']:.1f}%",
            "Retorno":  f"{r['total_return']:.1f}%",
        } for r in wf_results])
        st.dataframe(wf_df, use_container_width=True)

    # ── Lista de trades ──
    if m["trades"]:
        st.markdown("#### 📋 Detalle de Trades")
        trades_df = pd.DataFrame([{
            "Entrada":    str(t.entry_time)[:16],
            "Dir":        t.direction,
            "Precio E":   f"{t.entry_price:.5f}",
            "Precio S":   f"{t.exit_price:.5f}",
            "PnL (ATR)":  f"{t.pnl_pips:.3f}",
            "Outcome":    t.outcome,
            "Prob LSTM":  f"{t.lstm_prob*100:.1f}%",
            "Confirms":   t.confirms,
        } for t in m["trades"][:100]])
        st.dataframe(trades_df, use_container_width=True, height=300)


# ── Tab Optimización Genética ────────────────────────────────

def tab_genetic(pair: str, tf: str):
    st.markdown("### 🧬 Optimización Genética de Parámetros")
    st.markdown("""
    El algoritmo genético busca la combinación óptima de parámetros de indicadores,
    umbrales de señal y multiplicadores de SL/TP para maximizar `win_rate × profit_factor`.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        pop_size  = st.slider("Tamaño población", 10, 60, 25, 5)
    with col2:
        gens      = st.slider("Generaciones", 5, 50, 15, 5)
    with col3:
        mut_rate  = st.slider("Tasa mutación", 0.05, 0.35, 0.15, 0.05)

    run_ga = st.button("🚀 Iniciar Optimización Genética", type="primary")

    if not run_ga:
        # Mostrar resultados previos si existen
        if st.session_state.opt_results:
            _display_ga_results(st.session_state.opt_results)
        else:
            st.info("⚙️ Configura y ejecuta la optimización genética.")
        return

    symbol = PAIRS[pair]
    tf_cfg = TIMEFRAMES[tf]

    with st.spinner("Descargando datos para optimización..."):
        df_raw = fetch_data(symbol, tf_cfg["interval"], tf_cfg["period"])

    predictor = st.session_state.predictor
    progress_bar = st.progress(0)
    status_text  = st.empty()

    def update_progress(gen, total, best_fit):
        pct = int(gen / total * 100)
        progress_bar.progress(pct)
        status_text.markdown(
            f"**Generación {gen}/{total}** · "
            f"Mejor fitness: `{best_fit:.4f}`"
        )

    ga = GeneticOptimizer(
        population_size = pop_size,
        generations     = gens,
        mutation_rate   = mut_rate,
    )

    with st.spinner("Optimizando... (puede tomar varios minutos)"):
        results = ga.run(df_raw, predictor, progress_callback=update_progress)

    progress_bar.progress(100)
    status_text.success("✅ Optimización completada")

    st.session_state.opt_params  = results["best_params"]
    st.session_state.opt_results = results

    _display_ga_results(results)


def _display_ga_results(results: dict):
    bp = results["best_params"]

    st.markdown("#### 🏆 Mejores Parámetros Encontrados")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Win Rate",      f"{results['best_win_rate']*100:.1f}%")
    c2.metric("Profit Factor", f"{results['best_profit_factor']:.2f}")
    c3.metric("N° Trades",     str(results["best_n_trades"]))
    c4.metric("Fitness",       f"{results['best_fitness']:.4f}")

    st.markdown("#### 🔧 Parámetros Optimizados")
    param_cols = st.columns(3)
    param_items = list(bp.items())
    chunk = len(param_items) // 3 + 1
    for i, col in enumerate(param_cols):
        for k, v in param_items[i*chunk:(i+1)*chunk]:
            display_v = f"{int(v)}" if k in ("ema_fast","ema_slow","rsi_period",
                        "bb_period","atr_period","stoch_k","stoch_d",
                        "adx_period","macd_fast","macd_slow","macd_signal",
                        "min_confirms") else f"{v:.3f}"
            col.markdown(f"**{k}**: `{display_v}`")

    # Curva de evolución
    if "history" in results:
        hist_df = pd.DataFrame(results["history"])
        fig_ev  = go.Figure()
        fig_ev.add_trace(go.Scatter(
            x=hist_df["generation"], y=hist_df["best_fitness"],
            name="Fitness", line=dict(color="#a78bfa", width=2),
            fill="tozeroy", fillcolor="rgba(167,139,250,0.1)",
        ))
        fig_ev.update_layout(
            title="Evolución del Fitness por Generación",
            paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1117",
            font=dict(color="#e2e8f0"), height=280,
            xaxis=dict(gridcolor="#1e293b", title="Generación"),
            yaxis=dict(gridcolor="#1e293b", title="Fitness"),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_ev, use_container_width=True)


# ── Tab Historial ────────────────────────────────────────────

def tab_history():
    st.markdown("### 📜 Historial de Señales")
    hist = st.session_state.signal_history
    if not hist:
        st.info("Aún no se han generado señales. Ve al Dashboard.")
        return

    df_h = pd.DataFrame([{
        "Hora":      s["timestamp"].strftime("%Y-%m-%d %H:%M"),
        "Par":       s.get("pair", ""),
        "TF":        s.get("tf", ""),
        "Dirección": s["direction"],
        "Precio":    f"{s['close']:.5f}",
        "TP":        f"{s['tp']:.5f}",
        "SL":        f"{s['sl']:.5f}",
        "Prob %":    f"{s['prob']*100:.1f}",
        "Confirms":  s["confirms"],
        "Fuerza":    s["strength"],
        "ADX":       f"{s['adx']:.1f}",
        "RSI":       f"{s['rsi']:.1f}",
    } for s in hist])

    col_f, col_d = st.columns([1, 3])
    with col_f:
        filt = st.selectbox("Filtrar", ["Todas", "BUY", "SELL"])
    if filt != "Todas":
        df_h = df_h[df_h["Dirección"] == filt]

    st.dataframe(df_h, use_container_width=True)

    if st.button("🗑️ Limpiar historial"):
        st.session_state.signal_history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════
# APP PRINCIPAL
# ══════════════════════════════════════════════════════════════

def main():
    # ── Header ──
    st.markdown("""
    <div class='title-bar'>
      <span style='font-family:Syne;font-size:1.6rem;font-weight:800;
        background:linear-gradient(90deg,#38bdf8 0%,#a78bfa 50%,#34d399 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        padding-left:12px'>
        📈 ForexPulse AI Pro
      </span>
      <span style='color:#64748b;font-size:0.82rem;margin-left:16px'>
        LSTM · Genetic Optimization · Scalping Alerts
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Disclaimer ──
    st.warning("""
    ⚠️ **AVISO LEGAL**: Esta herramienta es exclusivamente educativa y para backtesting.
    No constituye consejo financiero ni garantía de resultados.
    El trading de forex conlleva alto riesgo de pérdida de capital.
    """)

    pair, tf, capital, risk_pct = render_sidebar()

    tabs = st.tabs([
        "📡 Dashboard en Vivo",
        "🔬 Backtesting Avanzado",
        "🧬 Optimización Genética",
        "📜 Historial de Señales",
    ])

    with tabs[0]:
        tab_dashboard(pair, tf)
    with tabs[1]:
        tab_backtesting(pair, tf)
    with tabs[2]:
        tab_genetic(pair, tf)
    with tabs[3]:
        tab_history()


if __name__ == "__main__":
    main()
