# ⚡ Quick Reference - BNB Revolutionary AI System

Бърз справочник за всички команди и функции.

## 🚀 Essential Commands

### 📊 Базови Прогнози (Ready to Use)
```bash
# Ensemble прогноза (най-добра точност)
python3 predict_revolutionary_2025.py --model ensemble --periods 24

# Бърза Helformer прогноза 
python3 predict_revolutionary_2025.py --model helformer --periods 24

# Всички модели едновременно
python3 predict_revolutionary_2025.py --model all --periods 24
```

### 🎯 Тренинг на модели
```bash
# Всички модели
python3 train_revolutionary_models.py --models all --periods 7

# Само най-добрите
python3 train_revolutionary_models.py --models helformer performer --periods 7

# Бърз тест
python3 train_revolutionary_models.py --models enhanced --periods 7
```

### 🔗 Test On-Chain Features
```bash
# Тест на 82.44% accuracy boost (без API ключове)
python3 test_onchain_accuracity_boost.py

# Setup на API конфигурация
python3 onchain_api_config.py
```

---

## 🎯 Model-Specific Commands

### 🧠 Enhanced ML
```bash
# Тренинг
python3 train_revolutionary_models.py --models enhanced --periods 7 30 90

# Прогноза
python3 predict_revolutionary_2025.py --model enhanced --periods 24
```

### 🔥 Helformer (Best Performance)
```bash
# Тренинг
python3 train_revolutionary_models.py --models helformer --periods 10

# Прогноза
python3 predict_revolutionary_2025.py --model helformer --periods 24
```

### 🚀 TFT (Multi-Horizon)
```bash
# Тренинг
python3 train_revolutionary_models.py --models tft --periods 24

# Прогноза
python3 predict_revolutionary_2025.py --model tft --periods 48
```

### ⚡ Performer (Fastest)
```bash
# Тренинг
python3 train_revolutionary_models.py --models performer --periods 7

# Прогноза  
python3 predict_revolutionary_2025.py --model performer --periods 24
```

---

## 🔧 Advanced Options

### 🎚️ Confidence & Validation
```bash
# Само високо-уверени сигнали
python3 predict_revolutionary_2025.py --model ensemble --confidence_threshold 0.8

# С validation
python3 predict_revolutionary_2025.py --model helformer --validation

# Advanced features
python3 predict_revolutionary_2025.py --model all --advanced_features
```

### 💾 Export Results
```bash
# Запис в JSON
python3 predict_revolutionary_2025.py --model ensemble --export_signals results.json

# Verbose mode
python3 train_revolutionary_models.py --models all --periods 7 --verbose
```

---

## 📊 Quick Status Check

### ✅ System Health
```bash
# Test accuracy boost (working без API keys)
python3 test_onchain_accuracity_boost.py

# Check available models
ls ml_models_bnb_enhanced/

# Check logs
tail -f logs/bnb_analyzer_*.log
```

### 📈 Performance Monitoring
```bash
# Check training results
python3 train_revolutionary_models.py --models enhanced --periods 7 --verbose

# Compare models
python3 predict_revolutionary_2025.py --model all --periods 24
```

---

## 🆘 Troubleshooting

### Problem: "Model not found"
```bash
# Solution: Train the model first
python3 train_revolutionary_models.py --models helformer --periods 7
```

### Problem: "TensorFlow not found" / Revolutionary models not working
```bash
# Solution: Install TensorFlow for Revolutionary models
pip install tensorflow>=2.13.0

# Test installation
python3 -c "import tensorflow as tf; print('TensorFlow OK')"

# Then train Revolutionary models
python3 train_revolutionary_models.py --models all --periods 7
```

### Problem: Import errors
```bash
# Solution: Install requirements
pip install -r requirements.txt
```

### Problem: No API keys
```bash
# Solution: System works with simulation
python3 test_onchain_accuracity_boost.py
```

---

## 🎯 Use Case Scenarios

### 📈 Short-term Trading (1 week)
```bash
# Best: Performer (fastest)
python3 predict_revolutionary_2025.py --model performer --periods 7
```

### 📊 Medium-term Trading (1 month)
```bash
# Best: Helformer (most accurate)
python3 predict_revolutionary_2025.py --model helformer --periods 30
```

### 📉 Long-term Trading (3 months)
```bash
# Best: TFT (multi-horizon)  
python3 predict_revolutionary_2025.py --model tft --periods 90
```

### 🎭 Maximum Accuracy
```bash
# Best: Ensemble (combines all)
python3 predict_revolutionary_2025.py --model ensemble --periods 24 --advanced_features
```

---

## 📚 File Structure Quick Reference

```
bnb-a/
├── 🚀 MAIN SCRIPTS
│   ├── train_revolutionary_models.py    # Train all models
│   ├── predict_revolutionary_2025.py    # Make predictions  
│   └── test_onchain_accuracity_boost.py # Test accuracy boost
│
├── 🧠 AI MODELS
│   ├── bnb_enhanced_ml.py              # Core ML system
│   ├── helformer_model.py              # Helformer architecture
│   ├── temporal_fusion_transformer.py  # TFT model
│   └── performer_bilstm_model.py       # Performer model
│
├── 🔗 ON-CHAIN DATA
│   ├── onchain_metrics_provider.py     # 87 metrics provider
│   ├── onchain_api_config.py          # API configuration
│   └── enhanced_feature_engineering.py # Feature engineering
│
├── 📊 DOCUMENTATION
│   ├── README.md                       # Main guide
│   ├── MODELS_GUIDE.md                # Detailed models guide
│   └── QUICK_REFERENCE.md             # This file
│
└── 📁 DATA & MODELS
    ├── ml_models_bnb_enhanced/         # Trained models
    ├── logs/                          # System logs
    └── requirements.txt               # Dependencies
```

---

## 🎪 Demo Commands

### 🚀 Full System Demo
```bash
# 1. Test accuracy boost
python3 test_onchain_accuracity_boost.py

# 2. Train best models
python3 train_revolutionary_models.py --models helformer performer --periods 7

# 3. Make predictions
python3 predict_revolutionary_2025.py --model ensemble --periods 24 --advanced_features

# 4. Export results
python3 predict_revolutionary_2025.py --model all --periods 24 --export_signals demo_results.json
```

### ⚡ Quick Test (5 minutes)
```bash
# 1. Enhanced ML only (fastest to train)
python3 train_revolutionary_models.py --models enhanced --periods 7

# 2. Quick prediction
python3 predict_revolutionary_2025.py --model enhanced --periods 24

# 3. Test accuracy
python3 test_onchain_accuracity_boost.py
```

---

## 💡 Pro Tips

### 🎯 Optimal Settings
- **Short-term** (1 week): `--model performer --periods 7`
- **Medium-term** (1 month): `--model helformer --periods 30` 
- **Long-term** (3 months): `--model tft --periods 90`
- **Best accuracy**: `--model ensemble --periods 30 --advanced_features`

### ⚡ Performance
- **Fastest training**: `enhanced` model
- **Fastest prediction**: `performer` model  
- **Best accuracy**: `ensemble` with `helformer`
- **Most features**: `--advanced_features` flag

### 🔗 On-Chain Data
- **Works without API keys** (simulation mode)
- **Free tiers available** for most providers
- **82.44% accuracy boost** with real on-chain data
- **Test first**: `python3 test_onchain_accuracity_boost.py`

---

## 🚀 Next Steps

1. **Install TensorFlow** (for Revolutionary models): `pip install tensorflow>=2.13.0`
2. **Start with test**: `python3 test_onchain_accuracity_boost.py`
3. **Train models**: `python3 train_revolutionary_models.py --models all --periods 7`  
4. **Make predictions**: `python3 predict_revolutionary_2025.py --model ensemble --periods 24`
5. **Setup API keys** for real on-chain data (optional)
6. **Deploy in production** with automated retraining

For detailed explanations, check `MODELS_GUIDE.md` 📚
