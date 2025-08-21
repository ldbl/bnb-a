# 🚀 BNB Advanced Trading Analyzer

Advanced technical analysis tool for BNB/USDT trading with multiple indicators, Elliott Wave analysis, and Fibonacci retracements.

## ✨ Features

### 📊 **Technical Indicators**
- **RSI (Relative Strength Index)** - Momentum oscillator
- **MACD** - Moving Average Convergence Divergence  
- **Bollinger Bands** - Volatility indicator
- **EMA** - Exponential Moving Average

### 🌊 **Advanced Analysis**
- **Elliott Wave Theory** - Pattern recognition and wave counting
- **Fibonacci Retracements** - Support/resistance levels
- **Multi-timeframe Analysis** - Week, Month, 3M, Year views
- **Support/Resistance Detection** - Key price levels

### 🎯 **Signal Generation** 
- **Bull/Bear Scoring System** - Weighted indicator analysis
- **Confidence Levels** - Signal reliability percentage
- **Position Size Recommendations** - Risk management
- **Target/Stop Loss Calculation** - Entry/exit levels

### 🎨 **User Interface**
- **Colorful Console Output** - Easy-to-read formatting
- **ASCII Charts** - Price and volume visualization
- **Interactive Menu** - Multiple analysis views
- **Real-time Data** - Live Binance API integration

## 🏗️ **Architecture**

### 📁 **Modular Structure**
```
bnb-a/
├── main.py                # 🚀 Main application entry point
├── indicators.py          # 📊 Technical indicators (RSI, MACD, Bollinger)
├── elliott_wave.py        # 🌊 Unified Elliott Wave analysis (visual + algorithmic)
├── fib.py                # 📐 Fibonacci retracements & extensions
├── ichimoku_module.py    # ☁️ Ichimoku Cloud analysis (multi-period)
├── whale_tracker.py      # 🐋 Whale tracking & large volume analysis
├── sentiment_module.py   # 🎭 Fear & Greed + social media sentiment
├── correlation_module.py # 📊 BNB correlation analysis with BTC/ETH
├── ml_predictor.py       # 🤖 Machine Learning price predictions
├── trend_reversal.py     # 🔄 Classic reversal pattern detection
├── data_fetcher.py       # 📡 Binance API communication
├── signal_generator.py   # 🎯 Trading signal logic & scoring system
├── display.py            # 🎨 UI formatting and colors
├── cache_manager.py      # ⚡ API response caching
├── charts.py             # 📈 ASCII chart generation
├── config.py             # ⚙️ Configuration settings
├── logger.py             # 📝 Logging system
└── requirements.txt      # 📦 Python dependencies
```

## 🚀 **Quick Start**

### 1. **Installation**
```bash
# Clone or download the project
cd bnb-a

# Install dependencies
pip install -r requirements.txt
```

### 2. **Run the Analyzer**
```bash
python3 main.py
```

### 3. **Menu Options**
- **1. Refresh analysis** - Update with latest data
- **2. Show detailed Fibonacci analysis** - Full Fib retracements & extensions
- **3. Show Elliott Wave analysis** - Unified visual + algorithmic Elliott Wave  
- **4. Show Ichimoku Cloud analysis** - Multi-period Ichimoku с TK Cross
- **5. Show Whale Tracking analysis** - Whale activity за различни периоди
- **6. Show Sentiment Analysis** - Fear & Greed + social media sentiment
- **7. Show Correlation Analysis (BTC/ETH)** - Daily+ correlation с главните криптовалути
- **8. Show ML Predictions (Strategic Forecasts)** - Daily+ strategic AI predictions
- **9. Show Trend Reversal Analysis** - Classic reversal patterns & signals
- **10. Show market summary** - Comprehensive market data
- **11. Toggle colors** - Enable/disable colored output
- **12. Exit** - Close the application

## 📊 **Sample Output**

```
🚀 BNB ADVANCED TRADING ANALYSIS
============================================================

📊 CURRENT STATUS
Price: $842.95
Time: 2025-08-21 20:47:35

📊 MARKET SUMMARY
Price: $842.98
24h Change: -0.35%
24h Range: $837.96 - $883.86
24h Volume: 422,510 BNB

📈 INDICATORS
RSI(14): 63.95 (Neutral)
MACD: NEUTRAL (M:25.27 S:25.27)
Bollinger: NEUTRAL
Elliott Wave: WAVE_1
Fibonacci: WAIT (UPTREND)
  └─ Closest level: 23.6% at $847.55

⏰ MULTI-TIMEFRAME ANALYSIS
Week: BEARISH (3.72%) | RSI:61.49
Month: BULLISH (0.14%) | RSI:40.99
3 Months: BULLISH (14.47%) | RSI:42.49
Year: BULLISH (44.28%) | RSI:77.37

📐 ENHANCED FIBONACCI ANALYSIS
Action: WAIT
Trend: UPTREND
Closest Level: 23.6% at $847.55 (4.6 away)
Golden Pocket: 🟢 ABOVE GOLDEN POCKET

🌊 MULTI-PERIOD ELLIOTT WAVES
6 месеца: WAVE_5_EXTENSION (75%)
  └─ Status: 🟢 TRENDING | Next: FINAL PUSH
1 година: WAVE_5_IN_PROGRESS (80%)
  └─ Status: 🟡 LATE STAGE | Next: WAVE_5_COMPLETION
1.5 години: WAVE_5_COMPLETION (95%)
  └─ Status: 🔴 CYCLE TOP | Next: ABC CORRECTION

🎯 SIGNAL SCORES
Bullish Score: 0
Bearish Score: 3
Confidence: 89%

💡 RECOMMENDATION
Action: STRONG SELL
Primary Target: $775.51
Extended Target: $716.51
Stop Loss: $885.10
Support Target: $732.00
Position Size: 33% short position
Reason: Overbought RSI
```

## ⚙️ **Configuration**

### 📝 **Key Settings** (config.py)
- **RSI_PERIOD**: 14 (default)
- **MACD_FAST/SLOW**: 12/26 periods
- **BOLLINGER_PERIOD**: 20 periods
- **CACHE_TTL**: 30 seconds for data caching
- **POSITION_SIZE**: Risk management rules

### 🎨 **Display Options**
- **USE_COLORS**: True/False
- **CHART_WIDTH/HEIGHT**: ASCII chart dimensions
- **DISPLAY_PRECISION**: Decimal places for prices

## 🔧 **Technical Details**

### 📡 **Data Sources**
- **Binance API**: Real-time price and volume data
- **Multiple Timeframes**: 1h, 4h, 1d, 1w intervals
- **Historical Data**: Up to 100 periods per timeframe

### 🧮 **Calculation Methods**
- **RSI**: Standard 14-period momentum calculation
- **MACD**: 12/26/9 EMA-based convergence divergence
- **Bollinger**: 20-period SMA ± 2 standard deviations
- **Elliott Wave**: Pivot point detection and wave counting
- **Fibonacci**: Automatic swing high/low detection

### 🎯 **Signal Generation System**

#### 📊 **Scoring Algorithm**
Всички сигнали се генерират чрез **многофакторна система за точки**, която комбинира:

**🟢 Bull Score (Бичи точки):**
- **RSI < 30**: +2 точки (oversold зона)
- **RSI < 40**: +1 точка (приближава oversold)
- **MACD Bullish**: +2 точки (+ допълнителна +1 ако histogram > 0)
- **Bollinger Oversold**: +2 точки (цена под долната лента)
- **Elliott Wave 2**: +2 точки (най-добра зона за покупка)
- **Elliott Wave 3**: +1 точка (силен тренд)
- **Fibonacci BUY**: +2 точки (STRONG_BUY = +3)
- **Golden Pocket**: +2 точки (61.8% Fib ниво)
- **Близо до Support**: +2 точки (в рамките на $20)
- **Ниска цена**: +3 точки (под $650)
- **Високия volume + растеж**: +1 точка
- **Correlation Enhancement**: +2 точки за independent bullish movement

**🔴 Bear Score (Мечи точки):**
- **RSI > 70**: +2 точки (overbought зона)
- **RSI > 60**: +1 точка (приближава overbought)
- **MACD Bearish**: +2 точки (+ допълнителна +1 ако histogram < 0)
- **Bollinger Overbought**: +2 точки (цена над горната лента)
- **Elliott Wave 5**: +2 точки (край на цикъл)
- **Fibonacci SELL**: +2 точки (STRONG_SELL = +3)
- **Близо до Resistance**: +2 точки (в рамките на $20)
- **Висока цена**: +3 точки (над $850)
- **Високия volume + спад**: +1 точка
- **Correlation Enhancement**: +2 точки за independent bearish movement

#### 🎯 **Decision Matrix (Матрица за решения)**

```
Bull Score > Bear Score + 2  →  STRONG BUY
Bull Score > Bear Score      →  BUY  
Bear Score > Bull Score + 2  →  STRONG SELL
Bear Score > Bull Score      →  SELL
Иначе                        →  WAIT
```

#### 📈 **Confidence Calculation**
```python
score_diff = abs(bull_score - bear_score)
max_score = max(bull_score, bear_score)
confidence = min(50 + (score_diff * 10) + (max_score * 3), 95%)
```

**Обяснение:**
- **Базова увереност**: 50%
- **Разлика в точките**: +10% за всяка точка разлика
- **Максимален score**: +3% за всяка точка от най-високия score
- **Максимум**: 95% (никога 100% заради рисковете)

#### 🧮 **Практически Примери**

**Пример 1: STRONG BUY Signal**
```
Текуща цена: $620
RSI: 28 (oversold)           → +2 точки
MACD: Bullish crossover      → +2 точки  
Elliott Wave: Wave 2         → +2 точки
Fibonacci: Golden Pocket     → +2 точки
Близо до support $600        → +2 точки
────────────────────────────────────────
Bull Score: 10 | Bear Score: 0
Confidence: 50 + (10*10) + (10*3) = 95%
Action: STRONG BUY
```

**Пример 2: STRONG SELL Signal**
```
Текуща цена: $875
RSI: 78 (overbought)         → +2 точки
Elliott Wave: Wave 5         → +2 точки
Fibonacci: STRONG_SELL       → +3 точки
Висока цена (>$850)          → +3 точки
Близо до resistance $880     → +2 точки
────────────────────────────────────────
Bull Score: 0 | Bear Score: 12
Confidence: 50 + (12*10) + (12*3) = 95%
Action: STRONG SELL
```

**Пример 3: WAIT Signal**
```
Текуща цена: $780
RSI: 52 (neutral)            → 0 точки
MACD: Neutral                → 0 точки
Elliott Wave: Wave 1         → 0 точки
Fibonacci: WAIT              → 0 точки
────────────────────────────────────────
Bull Score: 1 | Bear Score: 1
Confidence: 50 + (0*10) + (1*3) = 53%
Action: WAIT
```

#### 🆕 **Новото в Signal System (2025)**

**📐 Enhanced Fibonacci Analysis:**
- **Golden Pocket Detection**: Автоматично откритие на 61.8% зона
- **Multi-level Support/Resistance**: Показва 3 най-близки Fib нива
- **Trend Confirmation**: Комбинира Fib с trend direction

**🌊 Multi-Period Elliott Waves:**
- **6 месеца**: Wave 5 Extension анализ
- **1 година**: Wave 5 In Progress tracking  
- **1.5 години**: Complete Cycle анализ с 95% точност
- **Visual + Algorithmic**: Комбинира ръчен и автоматичен анализ

**🐋 Whale Tracking:**
- **Volume Spike Detection**: 2x, 3x, 5x+ обычен volume
- **Multi-Period Whale Analysis**: 24h, 3d, 1w whale активност
- **Order Book Walls**: Големи buy/sell стени детекция

**🎭 Sentiment Analysis:**
- **Fear & Greed Index**: Пазарни емоции scoring
- **Social Media Sentiment**: Twitter, Reddit, Telegram анализ
- **News Sentiment**: Положителни/отрицателни новини tracking
- **Composite Score**: Обединен sentiment от всички източници

**☁️ Ichimoku Cloud:**
- **Multi-Period Analysis**: 3, 6, 12 месеца перспектива
- **Cloud Position**: Above/Below/In Cloud статус
- **TK Cross Detection**: Tenkan/Kijun кръстосване сигнали

**📊 Correlation Analysis:**
- **BTC/ETH Correlation**: Real-time корелационни коефициенти
- **Market Leadership**: Кой актив води пазарните движения
- **Independent Movement**: Детекция на независими BNB движения
- **Multi-Timeframe**: 24h, 1w, 1m, 3m корелационен анализ
- **Signal Enhancement**: +2 точки за силна независима performance

**🤖 Machine Learning Predictions:**
- **Multi-Model Ensemble**: Random Forest, Gradient Boost, Linear Regression
- **LSTM Neural Networks**: Deep learning за time series prediction
- **Strategic Feature Engineering**: Long-term trend analysis, cycle detection
- **Daily+ Horizons**: 1d, 1w, 1m, 6m, 1y strategic forecasts
- **Confidence Scoring**: Model agreement и prediction reliability
- **Investment Focus**: Daily+ timeframes за strategic decision making

**🔄 Trend Reversal Detection:**
- **Classic Candlestick Patterns**: Doji, Hammer, Shooting Star, Engulfing
- **Technical Divergences**: RSI, MACD, Volume divergences
- **Support/Resistance Breaks**: Key level breakouts с volume confirmation  
- **Multi-Timeframe Analysis**: 1w, 2w, 1m, 3m reversal detection
- **Conviction Scoring**: HIGH/MODERATE/LOW conviction levels
- **Automated Alerts**: Critical reversal signals в alert system

## 📈 **Advanced Features**

### 🔄 **Caching System**
- **API Response Caching**: Reduces API calls
- **TTL-based Expiry**: Automatic cache invalidation
- **Performance Optimization**: Faster repeated analysis

### 📊 **ASCII Charts**
- **Price Charts**: Visual price movement representation
- **RSI Indicators**: Oversold/overbought visualization  
- **Volume Charts**: Trading volume trends
- **Support/Resistance**: Key level visualization

### 📝 **Logging**
- **Debug Logging**: Detailed execution information
- **Performance Tracking**: Function execution times
- **Error Handling**: Comprehensive error logging
- **Signal History**: Trading signal records

## ⚠️ **Disclaimer**

This tool is for **educational and analysis purposes only**. 

**NOT FINANCIAL ADVICE** - Always do your own research and consider:
- Market volatility and risks
- Your financial situation
- Professional financial advice
- Proper risk management

## 📞 **Support**

For issues or improvements:
1. Check the logs/ directory for error details
2. Verify internet connection for API access
3. Ensure Python 3.7+ compatibility
4. Review configuration settings

## 🔄 **Updates and Improvements**

### 🆕 **Latest Enhancements (2025)**
- ✅ **Unified Elliott Wave Analysis** - Visual + Algorithmic approach
- ✅ **Enhanced Fibonacci Display** - Golden Pocket, Multi-level Support/Resistance  
- ✅ **Multi-Period Elliott Waves** - 6м, 1г, 1.5г perspective в началния екран
- ✅ **Ichimoku Cloud Module** - Multi-period analysis (3, 6, 12 months)
- ✅ **Whale Tracking System** - Volume spike detection & whale sentiment
- ✅ **Sentiment Analysis** - Fear & Greed + Social Media + News sentiment
- ✅ **Correlation Analysis Module** - BTC/ETH correlation + market leadership detection
- ✅ **Machine Learning Predictor** - Strategic AI forecasts с daily+ focus
- ✅ **Trend Reversal Detector** - Classic pattern recognition с multi-timeframe analysis
- ✅ **Enhanced Alert System** - Complete alert system за всички модули
- ✅ **Enhanced Signal System** - Improved scoring with ML + correlation integration
- ✅ **Optimized Whale Tracker** - Klines-based analysis (faster performance)

### 🎯 **Daily+ Focus (2025)**

**НОВА ФИЛОСОФИЯ**: Премахнахме къснотърговските timeframes (1h, 4h) и се фокусираме върху **strategic investing**:

- ✅ **Primary Timeframes**: 1d, 1w, 1M, 3M (strategic focus)
- ✅ **ML Strategic Analysis**: Long-term cycle detection, trend strength, investment zones
- ✅ **Daily+ Correlation**: Multi-month correlation patterns
- ✅ **Strategic Fibonacci**: Daily+ swing point detection
- ✅ **Investment-Grade Ichimoku**: Multi-period analysis (1-6 месеца)
- ✅ **Cycle-Based Recommendations**: Portfolio allocation based на market cycle position

### 📋 **Previous Enhancements**
- ✅ Modular architecture refactoring
- ✅ Multi-timeframe RSI calculation
- ✅ Caching system for performance
- ✅ ASCII chart visualization
- ✅ Comprehensive logging system
- ✅ Configuration management

---

**Happy Trading! 📈🚀**
# bnb-a
