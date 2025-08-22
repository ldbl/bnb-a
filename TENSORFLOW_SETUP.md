# 🤖 TensorFlow Setup for Revolutionary Models

Инструкции за инсталиране на TensorFlow за да работят Revolutionary моделите.

## ⚠️ Грешката която виждате

```
❌ Helformer error: Failed to load Helformer model: name 'HelformerModel' is not defined
❌ TFT error: Failed to load TFT model: name 'TemporalFusionTransformer' is not defined  
❌ Performer error: Performer BiLSTM model not available
```

**Причина:** TensorFlow не е инсталиран

## 🔧 Решение

### 1. Инсталирайте TensorFlow:

```bash
# За CPU версия (препоръчително)
pip install tensorflow>=2.13.0

# За GPU версия (ако имате NVIDIA GPU)
pip install tensorflow-gpu>=2.13.0

# Или с conda
conda install tensorflow>=2.13.0
```

### 2. Проверете инсталацията:

```bash
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

**Очакван резултат:**
```
TensorFlow version: 2.13.0 (или по-нова)
```

### 3. Тестване на Revolutionary моделите:

```bash
# Test Helformer import
python3 -c "from helformer_model import HelformerModel; print('✅ Helformer OK')"

# Test TFT import  
python3 -c "from temporal_fusion_transformer import TemporalFusionTransformer; print('✅ TFT OK')"

# Test Performer import
python3 -c "from performer_bilstm_model import PerformerBiLSTM; print('✅ Performer OK')"
```

## 🚀 След инсталация

### 1. Тренирайте моделите:

```bash
# Всички Revolutionary модели
python3 train_revolutionary_models.py --models all --periods 7

# Или поотделно
python3 train_revolutionary_models.py --models helformer --periods 10
python3 train_revolutionary_models.py --models tft --periods 24
python3 train_revolutionary_models.py --models performer --periods 7
```

### 2. Тестване в main.py:

```bash
python3 main.py
# Изберете: 8 → 5 (Helformer) / 6 (TFT) / 7 (Performer) / 8 (Ensemble)
```

### 3. Директни прогнози:

```bash
# Revolutionary predictions
python3 predict_revolutionary_2025.py --model helformer --periods 30
python3 predict_revolutionary_2025.py --model tft --periods 30  
python3 predict_revolutionary_2025.py --model performer --periods 30
python3 predict_revolutionary_2025.py --model ensemble --periods 30
```

## 💡 Troubleshooting

### Problem: "No module named 'tensorflow'"
```bash
# Solution
pip install tensorflow>=2.13.0
```

### Problem: TensorFlow версия твърде стара
```bash
# Upgrade TensorFlow
pip install --upgrade tensorflow>=2.13.0
```

### Problem: Конфликти с други библиотеки
```bash
# Създайте virtual environment
python3 -m venv venv_bnb
source venv_bnb/bin/activate  # Linux/Mac
# или
venv_bnb\Scripts\activate     # Windows

# Инсталирайте requirements
pip install -r requirements.txt
```

### Problem: Import errors след инсталация
```bash
# Рестартирайте Python/terminal и тествайте отново
python3 -c "import tensorflow as tf; print('TensorFlow OK')"
```

## 🎯 Minimum Requirements

- **Python**: 3.8+
- **TensorFlow**: 2.13.0+
- **RAM**: 8GB+ (16GB препоръчително)
- **Storage**: 2GB+ свободно място

## ⚡ Performance Tips

### За по-бърза тренировка:
```bash
# Използвайте CPU optimization
export TF_CPP_MIN_LOG_LEVEL=2

# Ограничете памет (ако имате проблеми)
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

### За production deployment:
```bash
# Оптимизирайте TensorFlow
pip install tensorflow-serving-api
```

## 🏆 След успешна инсталация

Ще можете да използвате всички Revolutionary функции:

- ✅ **Helformer**: 925% excess return potential
- ✅ **TFT**: Multi-horizon forecasting 
- ✅ **Performer**: Linear complexity O(N)
- ✅ **Ensemble**: Комбинирани прогнози
- ✅ **Integrated в main.py**: Опции 5-8

🚀 Готово за breakthrough cryptocurrency prediction performance!
