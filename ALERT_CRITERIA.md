# 🚨 Alert System Criteria

## 📋 **Overview**
Системата автоматично показва критични alerts когато започне анализа, ако има значителна whale активност или корелационни аномалии.

## 🐋 **Whale Activity Alerts**

### 🎯 **Показва се когато Alert Score ≥ 8**

#### 📊 **Scoring System:**
- **🚨 EXTREME WHALE signals**: +10 точки
- **🐋 MEGA WHALE signals (≥2)**: +6 точки  
- **📊 Volume spike (≥3x normal)**: +5 точки
- **💥 Price impact (≥3%)**: +4 точки
- **⚡ Multiple whale signals (≥3)**: +3 точки

#### 🔍 **Detection Criteria:**
```
EXTREME WHALE: 5x+ volume spike + significant price movement
MEGA WHALE: 3x+ volume spike + whale-level activity
Volume Spike: Current volume / average volume ≥ 3.0
Price Impact: |price_change| ≥ 3%
Multiple Signals: Total whale signals in period ≥ 3
```

### 💡 **Examples:**
**Alert Score 12 (CRITICAL):**
- 1 EXTREME WHALE signal (+10)
- 4x volume spike (+5) 
- → Auto-показва whale alert

**Alert Score 6 (NO ALERT):**
- 2 MEGA WHALE signals (+6)
- → Под threshold, не показва

---

## 📊 **Correlation Analysis Alerts**

### 🎯 **Показва се когато Alert Score ≥ 6**

#### 📈 **Scoring System:**
- **🔗 Correlation breakdown (<0.3)**: +8 точки
- **📉 Negative correlation (<-0.5)**: +6 точки
- **⚡ Extreme performance gap (≥5%)**: +5 точки
- **👑 BNB market leadership**: +4 точки
- **🎯 Strong correlation signal (≥2)**: +4 точки
- **📈 Unusually high correlation (≥0.9)**: +3 точки

#### 🔍 **Detection Criteria:**
```
Correlation Breakdown: |BTC_corr| < 0.3 AND |ETH_corr| < 0.3
Negative Correlation: BTC_corr < -0.5 OR ETH_corr < -0.5
Performance Gap: |BNB_vs_BTC| ≥ 5% OR |BNB_vs_ETH| ≥ 5%
BNB Leadership: BNB leading 24h performance
High Correlation: |correlation| ≥ 0.9 (unusual)
Strong Signal: correlation_score from signals ≥ 2
```

### 💡 **Examples:**
**Alert Score 8 (CRITICAL):**
- Correlation breakdown (+8)
- → BNB moving completely independently

**Alert Score 11 (CRITICAL):**
- Negative correlation with BTC (-0.6) (+6)
- BNB outperforming by 6% (+5)
- → Unusual market dynamics

**Alert Score 4 (NO ALERT):**
- BNB leadership (+4)
- → Интересно, но не критично

---

## ⚙️ **Configuration**

### 🐋 **Whale Alert Thresholds:**
```python
alert_thresholds = {
    "critical_volume_spike": 3.0,      # 3x+ volume
    "mega_whale_activity": 50000,      # 50K+ BNB
    "multiple_whale_signals": 3,       # 3+ signals
    "price_impact": 0.03,              # 3%+ change
    "unusual_activity_score": 8        # Min score
}
```

### 📊 **Correlation Alert Thresholds:**
```python
alert_thresholds = {
    "correlation_breakdown": 0.3,      # Drop below 0.3
    "negative_correlation": -0.5,      # Strong negative
    "extreme_performance_gap": 5.0,    # 5%+ difference
    "leadership_change": True,         # BNB leading
    "correlation_spike": 0.9,          # Very high corr
    "independence_threshold": 3.0      # 3%+ independent
}
```

## 🎯 **Why These Thresholds?**

### 🐋 **Whale Alerts (Score ≥ 8):**
- **Serious money movement**: 50K+ BNB = $40M+ at $800
- **Market impact**: 3x+ volume spikes are rare and significant
- **Price action**: 3%+ moves with volume indicate institutional activity

### 📊 **Correlation Alerts (Score ≥ 6):**
- **Market decoupling**: BNB independent movement is unusual
- **Negative correlation**: Extremely rare, indicates special events
- **Performance gaps**: 5%+ differences suggest fundamental changes

## 📱 **How It Works**

1. **📊 Main Analysis**: Sistema проверява alerts автоматично
2. **🚨 Alert Detection**: Scoring algorithms оценяват критичността  
3. **📋 Display**: Alerts се показват ПРЕДИ signal scores
4. **💡 Context**: User може да провери детайлите в специалните менюта

## 🔄 **Alert Frequency**

- **🐋 Whale**: Проверява се за последните 24 часа
- **📊 Correlation**: Проверява се real-time (50 periods)
- **⚡ Performance**: Lightweight проверки, не забавят анализа
- **🛡️ Error Handling**: Alert failures не счупват main analysis

---

## 📐 **Fibonacci Analysis Alerts**

### 🎯 **Показва се когато Alert Score ≥ 6**

#### 📊 **Scoring System:**
- **⭐ Golden Pocket (61.8% zone)**: +8 точки
- **🎯 Extreme zones (0%, 100%)**: +7 точки  
- **📐 Major Fib levels (38.2%, 50%, 61.8%, 78.6%)**: +6 точки
- **🛡️ Strong support at Fib level**: +5 точки
- **⚡ Strong resistance at Fib level**: +5 точки
- **🚀 Extension levels (161.8%, 261.8%)**: +4 точки
- **🟢/🔴 Strong BUY/SELL signals**: +3 точки

#### 🔍 **Detection Criteria:**
```
Golden Pocket: Price within $15 of 61.8% level in uptrend
Major Levels: Price within $8 of key Fibonacci levels
Support/Resistance: is_at_support or is_at_resistance = True
Extension Levels: Price within $15 of breakout levels
Extreme Zones: Price within $10 of swing high/low
```

### 💡 **Examples:**
**Alert Score 8 (CRITICAL):**
- Golden Pocket detection (+8)
- → Perfect bounce/reversal zone

**Alert Score 11 (CRITICAL):**
- Approaching swing high (+7)
- Strong resistance (+5)
- → Major resistance test

---

## 📊 **Technical Indicators Alerts**

### 🎯 **Показва се когато Alert Score ≥ 7**

#### 📈 **Scoring System:**
- **💎 Triple confluence (RSI+Bollinger+MACD)**: +9 точки
- **📉 RSI extreme oversold (≤25)**: +8 точки
- **📈 RSI extreme overbought (≥75)**: +8 точки
- **⚡ Fresh MACD crossover**: +7 точки
- **🚀 Strong MACD momentum (H>5)**: +6 точки
- **🎯 Bollinger squeeze (<3%)**: +6 точки
- **📉 RSI oversold (≤30)**: +5 точки
- **📈 RSI overbought (≥70)**: +5 точки
- **⚡ Price outside Bollinger bands**: +5 точки
- **🔥 Volume surge + MACD trend**: +4 точки

#### 🔍 **Detection Criteria:**
```
RSI Extreme: RSI ≤ 25 OR RSI ≥ 75
MACD Crossover: Fresh trend change (prev ≠ current)
Strong Momentum: MACD histogram > 5 OR < -5
Bollinger Squeeze: (upper - lower) / middle < 3%
Triple Confluence: RSI extreme + Bollinger extreme + MACD trend
Volume Surge: Recent volume +50% vs previous + MACD trend
```

### 💡 **Examples:**
**Alert Score 9 (CRITICAL):**
- Triple bullish confluence (+9)
- → RSI oversold + Bollinger oversold + MACD bullish

**Alert Score 13 (CRITICAL):**
- RSI extreme oversold 23 (+8)
- Fresh MACD bullish crossover (+7)
- → Strong reversal setup

**Alert Score 6 (NO ALERT):**
- RSI overbought 72 (+5)
- → Important but not critical

---

## 🔄 **Alert Integration**

### 📊 **Complete Alert System:**
```
🐋 Whale Alerts: Score ≥ 8
📊 Correlation Alerts: Score ≥ 6  
📐 Fibonacci Alerts: Score ≥ 6
📊 Technical Alerts: Score ≥ 7
🤖 ML Prediction Alerts: Score ≥ 7
```

### 📱 **Priority Display:**
1. **🚨 Highest Priority**: Technical confluences, Golden Pocket
2. **⚡ High Priority**: MACD crossovers, whale activity
3. **📊 Medium Priority**: Correlation breakdowns, major Fib levels

### 🎯 **Alert Frequency:**
- **📐 Fibonacci**: Real-time (with current price)
- **📊 Technical**: Real-time (with latest candle data)
- **🐋 Whale**: Last 24 hours
- **📊 Correlation**: Last 50 periods (1h intervals)

---

## 🤖 **Machine Learning Prediction Alerts**

### 🎯 **Показва се когато Alert Score ≥ 7**

#### 🧠 **Scoring System:**
- **⚡ Extreme prediction (≥10%)**: +8 точки
- **🚀 Significant move (≥5%)**: +6 точки
- **🤝 High model agreement (≥80%)**: +5 точки
- **🎯 High confidence (≥85%)**: +4 точки
- **🔥 Multiple models agree (≥3 models)**: +3 точки

#### 🔍 **Detection Criteria:**
```
Extreme Prediction: |predicted_change| ≥ 10%
Significant Move: |predicted_change| ≥ 5%
Model Agreement: % of models agreeing on direction ≥ 80%
High Confidence: Prediction confidence score ≥ 85%
Multiple Models: ≥3 trained models making predictions
```

### 💡 **Examples:**
**Alert Score 8 (CRITICAL):**
- Extreme bullish prediction: +12% (+8)
- → AI predicts major price increase

**Alert Score 11 (CRITICAL):**
- Significant bearish move: -7% (+6)
- High model agreement: 90% (+5)
- → Strong consensus на price drop

**Alert Score 6 (NO ALERT):**
- Significant move: +5% (+6)
- → Important but under threshold

### 🧠 **ML Model Types:**
- **Random Forest**: Tree-based ensemble learning
- **Gradient Boosting**: Sequential improvement algorithm
- **Linear Regression**: Statistical relationship modeling
- **LSTM Networks**: Deep learning time series (future)

### 📊 **Feature Engineering:**
```
Price Features: Returns, ratios, moving averages
Technical Indicators: RSI, MACD, Bollinger Bands
Volume Analysis: Volume ratios, price-volume
Time Features: Hour, day of week, seasonality
Lag Features: 1h, 2h, 3h, 6h, 12h, 24h historical data
```

---

**💡 Цел**: Показва само най-важните събития, които потребителя трябва да знае веднага, без да го bombardира с тривиални alerts.
