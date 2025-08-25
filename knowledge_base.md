# Knowledge Base: BNB Cryptocurrency Prediction System

## 📊 What We've Already Covered

### ✅ Project Goals & Objectives
- **Hybrid monthly (5-10%) and quarterly (20-40%) amplitudes**
- **Target accuracy: >85%**
- **Target returns: >5% monthly, >25% quarterly**
- **Risk management: <10% monthly drawdown, <15% quarterly drawdown**

### ✅ Technical Requirements
- **PyTorch 2.0+ (no TensorFlow)**
- **pandas 2.0+**
- **TA-Lib for MACD, EMA, RSI, ROC**
- **Daily data (~540 points, 2024-02-01 to 2025-08-31)**
- **Close prices: [606.91, 578.49, 532.90, 567.26, 701, 834.96]**

### ✅ Core Modules Identified
- **data_fetcher.py** - CCXT for Binance API
- **helformer_model.py** - Holt-Winters + seasonal periods
- **signal_generator.py** - Hybrid monthly/quarterly logic
- **swing_risk_manager.py** - Risk management class
- **fib.py** - Fibonacci retracement levels
- **swing_backtester.py** - Backtrader implementation
- **email_reporter.py** - Automated signal notifications

### ✅ Haiduk Code (8 Rules)
- **Rule #0**: No over-engineering
- **Rule #1**: Anchor ($600-650 entry)
- **Rule #2**: Patience (3-7 days bottom, 2-4 weeks top)
- **Rule #3**: Steps (1/3 capital, no all-in)
- **Rule #4**: Leverage (max 2x)
- **Rule #5**: Exit ($750-780 targets)
- **Rule #6**: One battle (focus)
- **Rule #7**: Retreat (below $550)
- **Rule #8**: Team (system approach)

### ✅ 'Horo' Philosophy
- **Two steps forward, one step back**
- **Enter/exit in rhythm with the market**

### ✅ Testing & Validation
- **Test periods: August-September 2024 (+6.4%), Q3-Q4 2024 (+31%)**
- **Data validation: ~540 daily points**
- **Performance metrics: Sharpe ratio, max drawdown**

### ✅ Error Prevention
- **NaN handling: np.nan_to_num or fillna(0)**
- **Shape validation: assert x.shape[2] == 5**
- **Data type consistency**

## 🚀 What Needs to be Implemented

### 🔧 Core Infrastructure
- [ ] **PyTorch 2.0+ setup** (replace TensorFlow)
- [ ] **GPU optimization** with torch.cuda.is_available()
- [ ] **Performance optimization** (avoid unnecessary loops)
- [ ] **Efficient data structures** and algorithms

### 📊 Data & Features
- [ ] **Daily data pipeline** (~540 points, 2024-2025)
- [ ] **On-chain metrics integration** (Glassnode API)
- [ ] **Whale movement tracking** (>1000 BNB)
- [ ] **Real-time data feeds**

### 🤖 ML Models
- [ ] **Helformer enhancement** for swing trading
- [ ] **Holt-Winters integration** (seasonal_periods=1/3)
- [ ] **Forecast horizons** (720h monthly, 2880h quarterly)
- [ ] **Accuracy validation** (>85%)

### 📈 Technical Indicators
- [ ] **Monthly strategy**: EMA10/50, RSI7, ROC1, MACD12/26/9
- [ ] **Quarterly strategy**: EMA50/200, RSI14, ROC3, MACD50/200/9
- [ ] **TA-Lib integration** for all indicators
- **Buy conditions**: EMA_fast > EMA_slow, RSI < 40, ROC > 0, MACD > Signal, price $600-650
- **Sell conditions**: EMA_fast < EMA_slow, RSI > 70, MACD < Signal, price $750-780

### 🎯 Risk Management
- [ ] **SwingRiskManager class** (capital=10000, risk_per_trade=0.02)
- [ ] **ATR calculation** (period=7 monthly, 14 quarterly)
- [ ] **Stop-loss logic** (multiplier=3 monthly, 2 quarterly)
- [ ] **Position sizing** (1/3 capital, leverage up to 2x)
- [ ] **Retreat mechanism** (below $550)

### 📐 Fibonacci Analysis
- [ ] **Retracement levels** (23.6%, 38.2%, 61.8%)
- [ ] **Monthly cycles** (5-10%)
- [ ] **Quarterly cycles** (20-40%)
- [ ] **Test data**: Q3-Q4 2024 (Low $407.52, High $793.35)
- [ ] **Target levels**: Entry $600-650, Exit $750-780

### 🔄 Backtesting
- [ ] **Backtrader implementation**
- [ ] **Hybrid monthly/quarterly cycles**
- [ ] **Performance metrics** (Sharpe ratio, max drawdown)
- [ ] **Historical validation** (August-September 2024, Q3-Q4 2024)

### 📧 Automation
- [ ] **Email notifications** (smtplib)
- [ ] **Buy/Sell signal alerts**
- [ ] **Integration with email_reporter.py**

### 🧪 Testing & Validation
- [ ] **Unit testing** (unittest.testcase)
- [ ] **Accuracy testing**
- [ ] **Return/drawdown validation**
- [ ] **Comprehensive test coverage**

### 🔗 Module Integration
- [ ] **signal_generator.py** combines helformer_model.py + fib.py
- [ ] **Clean interfaces** between modules
- [ ] **Minimal dependencies**
- [ ] **System compatibility**

## 📋 Implementation Priority

### **Week 1: Foundation**
1. PyTorch 2.0+ setup
2. Daily data pipeline
3. Basic Helformer enhancement

### **Week 2: Core Features**
1. Technical indicators (TA-Lib)
2. Risk management class
3. Fibonacci analysis

### **Week 3: Integration**
1. Signal generation
2. Module integration
3. Basic testing

### **Week 4: Advanced Features**
1. On-chain metrics
2. Backtesting
3. Email automation

## 🎯 Success Metrics

### **Technical Performance**
- **Accuracy**: >85%
- **Monthly returns**: >5%
- **Quarterly returns**: >25%
- **Drawdown**: <10% monthly, <15% quarterly

### **System Performance**
- **Data processing**: <5 seconds for 540 points
- **GPU utilization**: >80% when available
- **Memory efficiency**: <2GB RAM usage
- **API response**: <1 second

### **Trading Performance**
- **Entry precision**: ±$10 from target levels
- **Exit timing**: ±2 days from optimal
- **Risk management**: 100% stop-loss execution
- **Position sizing**: Accurate to ±5%

---
*This knowledge base documents the current state and implementation roadmap for the BNB prediction system.*
*Last updated: 2025-08-22*
*Next review: Week of 2025-08-25*
