# 🤖 BNB ML Trend Reversal System

Комплексна система за машинно обучение, специализирана в детекция на обръщане на тренда за BNB/USDT.

## 📋 Съдържание

- [🎯 Какво прави системата](#-какво-прави-системата)
- [📊 Където се пазят данните](#-където-се-пазят-данните)
- [🧠 Как да обучим моделите](#-как-да-обучим-моделите)
- [🔮 Как да правим прогнози](#-как-да-правим-прогнози)
- [📈 Features и индикатори](#-features-и-индикатори)
- [⚙️ Конфигурация](#-конфигурация)

## 🎯 Какво прави системата

### **Основни възможности:**
- **Trend Reversal Detection** - Предсказва обръщане от възходящ към низходящ тренд и обратно
- **Multi-Model Ensemble** - Използва Random Forest, Gradient Boosting и Logistic Regression
- **Persistent Storage** - Моделите се запазват на диск и могат да се зареждат
- **Feature Engineering** - 12 специализирани features за reversal detection
- **Risk Assessment** - Confidence scoring и risk level оценка

### **Типове предсказания:**
- `0` = **No Reversal** - Тренда продължава
- `1` = **Bullish Reversal** - Обръщане към възходящ тренд (BUY signal)
- `2` = **Bearish Reversal** - Обръщане към низходящ тренд (SELL signal)

## 📊 Където се пазят данните

### **Структура на файловете:**
```
bnb-a/
├── ml_models/                          # 💾 Persistent model storage
│   ├── random_forest_reversal_10d.pkl  # Обучен Random Forest модел
│   ├── random_forest_reversal_10d_metadata.json  # Метаданни на модела
│   ├── gradient_boost_reversal_10d.pkl # Gradient Boosting модел
│   ├── logistic_reversal_10d.pkl       # Logistic Regression модел
│   └── ...                             # Други хоризонти (5d, 20d, etc.)
│
├── ml_enhanced.py                      # 🧠 Основен ML engine
├── train_ml_models.py                  # 🎓 Training script
├── predict_reversals.py                # 🔮 Prediction script
└── ML_GUIDE.md                         # 📚 Тази документация
```

### **Metadata формат:**
```json
{
  "created_at": "2025-08-21T23:04:10.922533",
  "feature_columns": ["rsi", "macd", "bb_position", ...],
  "training_samples": 784,
  "accuracy": 0.9949238578680203,
  "model_type": "random_forest",
  "reversal_threshold": 0.05
}
```

## 🧠 Как да обучим моделите

### **Лесен начин (препоръчителен):**
```bash
# Основно обучение с default настройки
python3 train_ml_models.py

# Обучение с повече данни
python3 train_ml_models.py --data-limit 5000

# Обучение за специфични хоризонти
python3 train_ml_models.py --periods 5 10 15 20

# Тест режим (без запазване)
python3 train_ml_models.py --no-save
```

### **Програмен начин:**
```python
from ml_enhanced import TrendReversalML
from ml_predictor import MLPredictor

# Инициализация
ml_system = TrendReversalML()
base_predictor = MLPredictor()

# Зареждане на данни
training_data = base_predictor.fetch_training_data("1h", 2000)

# Обучение за различни хоризонти
for periods in [5, 10, 20]:
    result = ml_system.train_reversal_models(training_data, periods)
    print(f"Trained {result['models_trained']} models for {periods}h")
```

### **Параметри за обучение:**

| Параметър | Default | Описание |
|-----------|---------|----------|
| `data_limit` | 2000 | Брой samples за обучение |
| `periods` | [5,10,20] | Хоризонти за предсказание (часове) |
| `reversal_threshold` | 0.05 | 5% движение за reversal |
| `trend_lookback` | 10 | Периоди за trend detection |

## 🔮 Как да правим прогнози

### **Лесен начин:**
```bash
# Основна прогноза за 10 часа напред
python3 predict_reversals.py

# Прогноза за конкретен хоризонт
python3 predict_reversals.py --periods 20

# Детайлна прогноза с individual model results
python3 predict_reversals.py --periods 10 --details
```

### **Програмен начин:**
```python
from ml_enhanced import TrendReversalML
from ml_predictor import MLPredictor

# Инициализация
ml_system = TrendReversalML()
base_predictor = MLPredictor()

# Зареждане на текущи данни
recent_data = base_predictor.fetch_training_data("1h", 200)

# Прогноза
prediction = ml_system.predict_reversal(recent_data, periods_ahead=10)

print(f"Prediction: {prediction['prediction_label']}")
print(f"Confidence: {prediction['confidence']:.1%}")
```

### **Интерпретация на резултатите:**

| Prediction | Icon | Значение | Trading Action |
|------------|------|----------|----------------|
| No Reversal | 🟡➡️ | Тренда продължава | HOLD |
| Bullish Reversal | 🟢📈 | Обръщане нагоре | LONG/BUY |
| Bearish Reversal | 🔴📉 | Обръщане надолу | SHORT/SELL |

### **Risk Levels:**

| Confidence | Risk Level | Описание |
|------------|------------|----------|
| ≥ 80% | 🟢 LOW | Високо доверие |
| 60-79% | 🟡 MEDIUM | Умерено доверие |
| < 60% | 🔴 HIGH | Ниско доверие |

## 📈 Features и индикатори

Системата използва **12 специализирани features** за reversal detection:

### **Technical Indicators (4):**
- `rsi` - Relative Strength Index (14 period)
- `macd` - MACD line (12/26 EMA)
- `macd_histogram` - MACD histogram
- `bb_position` - Bollinger Bands position (0-1)

### **Trend Analysis (3):**
- `trend_5` - 5-period trend slope
- `trend_10` - 10-period trend slope  
- `trend_20` - 20-period trend slope

### **Volatility & Volume (3):**
- `price_volatility` - 20-period rolling volatility
- `volume_ratio` - Volume vs 20-period average
- `volume_spike` - Volume spike detection (>2x average)

### **Pattern Recognition (2):**
- `doji` - Doji candlestick pattern
- `hammer` - Hammer candlestick pattern

## ⚙️ Конфигурация

### **Reversal Thresholds:**
```python
reversal_thresholds = {
    "bullish_reversal": 0.05,   # 5%+ upward move
    "bearish_reversal": -0.05,  # 5%+ downward move  
    "trend_lookback": 10,       # Periods for trend detection
    "prediction_periods": [5, 10, 20]  # Available horizons
}
```

### **Model Configurations:**
```python
model_configs = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    },
    "gradient_boost": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6
    },
    "logistic": {
        "max_iter": 1000,
        "random_state": 42
    }
}
```

## 🎯 Практически примери

### **Scenario 1: Daily Trading**
```bash
# Сутрин - обучи моделите с fresh data
python3 train_ml_models.py --data-limit 1000 --periods 5 10

# Преди търговия - провери за reversals
python3 predict_reversals.py --periods 5 --details
```

### **Scenario 2: Swing Trading**
```bash
# Обучи за по-дълги хоризонти
python3 train_ml_models.py --periods 10 20 50

# Проверка за medium-term reversals
python3 predict_reversals.py --periods 20
```

### **Scenario 3: Risk Management**
```python
# Автоматизиран risk check
prediction = ml_system.predict_reversal(data, 10)

if prediction['confidence'] >= 0.8:
    if prediction['ensemble_prediction'] == 2:  # Bearish reversal
        print("🚨 HIGH CONFIDENCE BEARISH REVERSAL - Consider stop loss!")
```

## 🔧 Поддръжка и troubleshooting

### **Общи проблеми:**

1. **"No trained models found"**
   ```bash
   # Решение: Обучи моделите първо
   python3 train_ml_models.py
   ```

2. **"No models trained for X periods"**
   ```bash
   # Решение: Обучи за този хоризонт
   python3 train_ml_models.py --periods X
   ```

3. **Low accuracy модели**
   ```bash
   # Решение: Използвай повече данни
   python3 train_ml_models.py --data-limit 5000
   ```

### **Performance Tips:**

- **Преобучавай моделите** всеки ден с fresh data
- **Използвай ensemble predictions** вместо individual models
- **Комбинирай с технически анализ** за по-добри резултати
- **Backtest стратегиите** преди real trading

## ⚠️ Важни бележки

- **Не е финансов съвет** - използвай на собствен риск
- **Модели могат да overfitting** - валидирай резултатите
- **Market conditions се променят** - преобучавай редовно
- **Използвай risk management** - винаги с stop loss
- **Комбинирай с други анализи** - не разчитай само на ML

---

💡 **За въпроси и поддръжка:** Проверете логовете в `logs/` директорията
