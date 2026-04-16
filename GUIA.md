# ForexPulse AI Pro — Guía de Instalación y Uso
## LSTM + Optimización Genética para Scalping Forex

---

## 📁 Estructura de Archivos

```
forexpulse/
├── app.py                  ← App principal (Streamlit)
├── indicators.py           ← Indicadores técnicos (EMA, RSI, MACD, BB, ATR, Stoch, ADX)
├── lstm_model.py           ← Motor LSTM bidireccional (TensorFlow/Keras)
├── genetic_optimizer.py    ← Algoritmo genético (DEAP-compatible)
├── backtester.py           ← Motor de backtesting walk-forward
├── requirements.txt        ← Dependencias Python
└── GUIA.md                 ← Este archivo
```

---

## ⚡ Instalación Rápida

### 1. Crear entorno virtual (recomendado)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

> **Nota TensorFlow**: Si tienes GPU NVIDIA, instala `tensorflow-gpu` en lugar de `tensorflow`.
> Si solo quieres probar sin LSTM real, la app funciona con el predictor de respaldo automático.

### 3. Ejecutar
```bash
streamlit run app.py
```

Se abrirá automáticamente en: `http://localhost:8501`

---

## 🚀 Flujo de Uso Recomendado

### Paso 1 — Dashboard en Vivo
1. Selecciona par (ej. EUR/USD) y timeframe (5m o 15m) en el sidebar.
2. El modelo LSTM se entrena automáticamente al primer acceso (~30-60s).
3. Observa la señal generada: dirección, probabilidad y confirmaciones.
4. Activa el **refresco automático** (30-120s) para monitoreo continuo.

### Paso 2 — Optimización Genética (¡hazlo antes de tradear!)
1. Ve a la pestaña **Optimización Genética**.
2. Configura: población=25-30, generaciones=15-20.
3. Pulsa "Iniciar Optimización" y espera (5-15 min según hardware).
4. Los parámetros optimizados se aplican automáticamente al Dashboard.

### Paso 3 — Backtesting
1. Ve a **Backtesting Avanzado**.
2. Ejecuta con 4 ventanas walk-forward.
3. Revisa win rate, profit factor, Sharpe y drawdown.
4. Un **win rate > 55%** y **profit factor > 1.3** en out-of-sample es sólido.

### Paso 4 — Alertas Telegram
1. Crea un bot en [@BotFather](https://t.me/BotFather) → copia el token.
2. Obtén tu Chat ID via [@userinfobot](https://t.me/userinfobot).
3. Pégalos en el Sidebar → Prueba conexión.
4. Recibirás alertas automáticas cuando aparezca señal FUERTE.

---

## 🧠 Arquitectura del Sistema

### Motor LSTM
```
Input (60 velas × 19 features)
    ↓
BiLSTM(128) + BatchNorm + Dropout(0.25)
    ↓
BiLSTM(64)  + BatchNorm + Dropout(0.20)
    ↓
LSTM(32)    + BatchNorm + Dropout(0.15)
    ↓
Dense(16, ReLU) + Dropout(0.10)
    ↓
Dense(1, Sigmoid) → P(movimiento alcista)
```

### Lógica de Señal (5 condiciones)
Una señal se genera SOLO cuando:
1. `P_LSTM ≥ 74%` (configurable por GA)
2. `ADX ≥ 22` (tendencia suficiente)
3. `≥ 4 confirmaciones técnicas` de 7 posibles
4. Parámetros validados por optimización genética
5. No hay trade abierto activo

### Confirmaciones técnicas (7 posibles):
1. EMA fast > EMA slow (cruce)
2. RSI en zona correcta
3. MACD histogram dirección
4. Precio vs BB mid
5. Stochastic K vs D
6. ADX threshold
7. MACD line vs Signal

---

## 🧬 Algoritmo Genético

**Cromosoma** (20 genes):
- Períodos: EMA(fast/slow), RSI, BB, ATR, Stoch, ADX, MACD
- Umbrales: prob LSTM, confirmaciones mínimas, ADX threshold
- Riesgo: multiplicadores SL/TP (ATR-based)

**Operadores**:
- Selección: torneo (k=3)
- Cruce: un punto
- Mutación: gaussiana (σ = 10% del rango)
- Elitismo: top-3 pasan directamente

**Función fitness**:
```
fitness = win_rate × min(profit_factor, 5) × trade_bonus
trade_bonus = min(1.0, n_trades / 30)
```

---

## 📊 Métricas de Referencia (Objetivos Realistas)

| Métrica          | Mínimo aceptable | Bueno     | Excelente |
|------------------|-----------------|-----------|-----------|
| Win Rate         | > 50%           | > 58%     | > 65%     |
| Profit Factor    | > 1.2           | > 1.5     | > 2.0     |
| Sharpe Ratio     | > 0.8           | > 1.5     | > 2.0     |
| Max Drawdown     | < 20%           | < 12%     | < 7%      |
| N° Trades (60d)  | > 20            | > 40      | > 80      |

> ⚠️ **Importante**: Los resultados de backtest siempre superan el live. Espera ~20-30% de degradación al pasar a paper trading real.

---

## 🛡️ Recomendaciones Anti-Overfitting

### 1. Walk-Forward Validation
Siempre usa walk-forward (4+ ventanas). Desconfía de resultados con menos de 30 trades out-of-sample.

### 2. Separación temporal estricta
El código NO hace shuffle de los datos. La validación del LSTM usa siempre los últimos 15% temporalmente.

### 3. Regularización LSTM
- Dropout (0.10-0.25) en cada capa
- BatchNormalization
- EarlyStopping (patience=8) con `restore_best_weights=True`
- ReduceLROnPlateau para convergencia suave

### 4. Parámetros del GA
- Penalización por pocos trades (`trade_bonus`)
- `min_trades=30` antes de considerar válido un set
- Evita optimizar con menos de 3 meses de datos

### 5. Out-of-Sample siempre
Nunca uses el periodo de optimización para evaluar. Usa los últimos 20-30% de datos como test final.

### 6. Validación cruzada temporal (k-fold temporal)
```python
# Implementación sugerida para más robustez:
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

---

## 🔧 Personalización

### Añadir nuevo par de divisas
En `indicators.py`:
```python
PAIRS["EUR/GBP"] = "EURGBP=X"
```

### Añadir nuevo indicador
En `indicators.py`, función `add_all_indicators()`:
```python
df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
```
Y añade la condición de confirmación en `count_confirmations()`.

### Cambiar arquitectura LSTM
En `lstm_model.py`, método `build()`:
- Aumenta unidades para mayor capacidad (riesgo: más overfitting)
- Añade `Attention` layer para mejor captura de patrones largos
- Prueba `TCN` (Temporal Convolutional Network) como alternativa

### Email alerts (opcional)
```python
import smtplib
from email.mime.text import MIMEText

def send_email(to, subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'tu@email.com'
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
        s.login('tu@email.com', 'app_password')
        s.sendmail('tu@email.com', to, msg.as_string())
```

---

## ⚠️ Limitaciones Conocidas

1. **yfinance rate limits**: Si ves errores de descarga, la app usa datos sintéticos automáticamente. Espera 60s entre recargas.
2. **TensorFlow en Windows**: Puede requerir Visual C++ Redistributable. Alternativa: usar WSL2.
3. **Timeframe 1m**: No soportado por yfinance para períodos largos. Usa 5m o 15m.
4. **Volumen en Forex**: El volumen en yfinance para forex es proxy (tick volume), no volumen real de mercado.

---

## 📈 Próximas Mejoras Sugeridas

1. **Transformer/Attention**: Reemplazar BiLSTM por modelo Transformer para capturar dependencias más largas
2. **Multi-timeframe**: Confirmación en TF superior (ej. señal 5m + tendencia 1h)
3. **Sentiment analysis**: Integrar noticias forex via NewsAPI
4. **Ensemble**: Combinar LSTM + XGBoost + Random Forest para señal final
5. **Paper trading**: Integrar con OANDA API para paper trading real
6. **MLflow**: Tracking de experimentos para comparar modelos
7. **Docker**: Containerización para deployment en cloud

---

*ForexPulse AI Pro — Uso exclusivamente educativo*
*El trading de forex conlleva alto riesgo de pérdida de capital*
