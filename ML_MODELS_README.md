# 🤖 ML Модели - BNB Enhanced Analysis System

## **🚀 REVOLUTIONARY 2025 MODELS**

### **1. Helformer (Holt-Winters + Transformer)**
**📍 Файл**: `helformer_model.py`  
**🎯 Предвижда**: Multi-horizon forecasting (цена, обем, волатилност)  
**⏰ Хоризонт**: 24 часа, 7 дни, 30 дни  
**🔧 Как се тренира**: Option 8 → 5  
**💾 Запазва**: `ml_models_bnb_enhanced/helformer_*_transformer.h5`  
**✨ Особености**: 
- Комбинира Holt-Winters с Transformer архитектура
- Breakthrough performance за time series
- Multi-target prediction (цена, обем, волатилност)

### **2. ADE-TFT (Advanced Deep Learning-Enhanced Temporal Fusion Transformer)**
**📍 Файл**: `temporal_fusion_transformer.py`  
**🎯 Предвижда**: Multi-horizon forecasting с attention механизми  
**⏰ Хоризонт**: 24 стъпки напред  
**🔧 Как се тренира**: Option 8 → 6  
**💾 Запазва**: `ml_models_bnb_enhanced/tft_*_tft.h5`  
**✨ Особености**:
- Superior multi-horizon forecasting
- Attention-based feature importance
- Cross-venue arbitrage detection

### **3. Performer Neural Network with BiLSTM**
**📍 Файл**: `performer_bilstm_model.py`  
**🎯 Предвижда**: Long-term price movements с FAVOR+ attention  
**⏰ Хоризонт**: 7, 30, 90 дни  
**🔧 Как се тренира**: Option 8 → 7  
**💾 Запазва**: `ml_models_bnb_enhanced/performer_*_bilstm.h5`  
**✨ Особености**:
- FAVOR+ attention за linear complexity
- BiLSTM за sequence modeling
- 87 on-chain metrics integration

---

## **🔧 Класически ML Модели**

### **4. BNB Enhanced ML (Gradient Boosting)**
**📍 Файл**: `bnb_enhanced_ml.py`  
**🎯 Предвижда**: Reversal patterns за 5, 10, 20, 30, 90 дни  
**⏰ Хоризонт**: Multiple timeframes  
**🔧 Как се тренира**: Option 8 → 4  
**💾 Запазва**: `ml_models_bnb_enhanced/bnb_enhanced_*.pkl`  
**✨ Особености**:
- Gradient Boosting с feature engineering
- Multi-timeframe predictions
- Market microstructure analysis

---

## **📊 Как да Тренирате Моделите**

### **Option 8: ML Model Management**
```
8. ML Model Management
   ├── 4. Train BNB Enhanced ML
   ├── 5. Train Helformer
   ├── 6. Train TFT
   └── 7. Train Performer
```

### **Трениране на Revolutionary Модели**
1. **Helformer**: `python3 main.py` → Option 8 → 5
2. **TFT**: `python3 main.py` → Option 8 → 6  
3. **Performer**: `python3 main.py` → Option 8 → 7

### **Трениране на Enhanced ML**
- `python3 main.py` → Option 8 → 4

---

## **🎯 Какво Предвижда Всеки Модел**

### **Helformer**
- **Цена**: Next 24h, 7d, 30d
- **Обем**: Volume prediction
- **Волатилност**: Volatility forecasting
- **Trend**: Direction and strength

### **TFT**
- **Multi-horizon**: 24 steps ahead
- **Feature Importance**: Attention weights
- **Cross-validation**: Time series CV
- **Ensemble**: Multiple predictions

### **Performer**
- **Long-term**: 7, 30, 90 days
- **On-chain**: 87 blockchain metrics
- **Attention**: FAVOR+ mechanism
- **BiLSTM**: Sequence modeling

### **BNB Enhanced**
- **Reversals**: 5, 10, 20, 30, 90 days
- **Patterns**: Technical analysis
- **Market Micro**: Order book dynamics
- **Correlation**: BTC/ETH relationships

---

## **💾 Модел Persistence**

### **TensorFlow Models (.h5)**
- **Helformer**: `helformer_*_transformer.h5`
- **TFT**: `tft_*_tft.h5`
- **Performer**: `performer_*_bilstm.h5`

### **Scikit-learn Models (.pkl)**
- **BNB Enhanced**: `bnb_enhanced_*.pkl`

### **Metadata (.json)**
- Training parameters
- Feature names
- Performance metrics
- Model version

---

## **🔍 Feature Engineering**

### **Technical Indicators**
- RSI, MACD, Bollinger Bands
- Moving averages (SMA, EMA, WMA)
- Volume indicators
- Momentum oscillators

### **On-Chain Metrics**
- Network activity
- Transaction volume
- Wallet distribution
- Mining difficulty

### **Market Microstructure**
- Order book dynamics
- Liquidity measures
- Cross-venue arbitrage
- Market maker activity

---

## **📈 Performance Metrics**

### **Accuracy Metrics**
- Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Confusion Matrix

### **Time Series Metrics**
- MAE, MSE, RMSE
- MAPE, SMAPE
- Directional Accuracy

### **Trading Metrics**
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor

---

## **🚀 Production Deployment**

### **Real-time Predictions**
- FastAPI endpoints
- WebSocket streaming
- Redis caching
- Kubernetes scaling

### **Model Monitoring**
- MLflow tracking
- Performance drift detection
- Auto-retraining
- A/B testing

### **Risk Management**
- Position sizing
- Stop-loss automation
- Portfolio optimization
- Risk metrics dashboard

---

## **📚 Допълнителни Ресурси**

- **ALERT_CRITERIA.md**: Alert thresholds
- **REVERSAL_PATTERNS.md**: Pattern recognition
- **EMAIL_SETUP.md**: Notification system
- **config.py**: Configuration management
- **market_config.py**: Market parameters

---

## **🔄 Auto-Retraining**

### **Schedule**
- **Daily**: Performance monitoring
- **Weekly**: Model evaluation
- **Monthly**: Full retraining
- **Quarterly**: Architecture updates

### **Triggers**
- Performance degradation
- Market regime change
- New data availability
- Model drift detection

---

*Last Updated: 2025-08-22*  
*Version: 2.0 - Revolutionary 2025 Models*
