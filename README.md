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
├── main.py              # Main application entry point
├── indicators.py        # Technical indicators (RSI, MACD, Bollinger)
├── elliott_wave.py      # Elliott Wave analysis
├── fib.py              # Fibonacci retracements
├── data_fetcher.py     # Binance API communication
├── signal_generator.py # Trading signal logic
├── display.py          # UI formatting and colors
├── cache_manager.py    # API response caching
├── charts.py           # ASCII chart generation
├── config.py           # Configuration settings
├── logger.py           # Logging system
└── requirements.txt    # Python dependencies
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
- **2. Show detailed Fibonacci analysis** - Full Fib levels
- **3. Show market summary** - Comprehensive market data  
- **4. Toggle colors** - Enable/disable colored output
- **5. Exit** - Close the application

## 📊 **Sample Output**

```
🚀 BNB ADVANCED TRADING ANALYSIS
============================================================

📊 CURRENT STATUS
Price: $840.70
Time: 2025-08-21 19:37:18

📈 INDICATORS
RSI(14): 63.95 (Neutral)
MACD: NEUTRAL (M:25.09 S:25.09)
Bollinger: NEUTRAL
Elliott Wave: WAVE_1
Fibonacci: WAIT (UPTREND)
  └─ Closest level: 23.6% at $847.55

🎯 SIGNAL SCORES
Bullish Score: 0
Bearish Score: 3
Confidence: 89%

💡 RECOMMENDATION
Action: STRONG SELL
Target: $773.44
Stop Loss: $882.74
Position Size: 33% short position
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

### 🎯 **Signal Logic**
- **Bull Score**: Points for bullish indicators
- **Bear Score**: Points for bearish indicators  
- **Confidence**: Based on score difference and alignment
- **Actions**: STRONG BUY/BUY/WAIT/SELL/STRONG SELL

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

Recent enhancements:
- ✅ Modular architecture refactoring
- ✅ Enhanced Fibonacci integration
- ✅ Multi-timeframe RSI calculation
- ✅ Caching system for performance
- ✅ ASCII chart visualization
- ✅ Comprehensive logging system
- ✅ Configuration management

---

**Happy Trading! 📈🚀**
# bnb-a
