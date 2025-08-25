# Насоки за ldbl/bnb-a

## 🎯 Общ преглед
- Проектът е модулен, позволява работа по отделни файлове (`helformer_model.py`, `signal_generator.py`, `swing_risk_manager.py`, `fib.py`, `data_fetcher.py`, `swing_backtester.py`).
- Всеки модул приема pandas DataFrame с Date, Open, High, Low, Close, Volume от `data_fetcher.py` (daily, ~540 точки, 2024-02-01 до 2025-08-31).
- Следвай Хайдушкия кодекс и „хоро" философията (две напред, една назад) за дисциплина в swing trading.

## 📊 BNB Контекст (25 август 2025)
- **Текуща цена**: ~$834.96, overbought (RSI > 70)
- **Очаквана корекция**: $750-800 (септември 2025)
- **Цели**:
  - **Monthly**: Buy на ~$800, Sell на ~$850 (5-10% return)
  - **Quarterly**: Buy на ~$800, Sell на $900-1000 (20-40% return)

## 🚀 Насоки за работа

### 1. **data_fetcher.py**
- Настрой за daily данни (interval='1d', 2024-02-01 до 2025-08-31)
- Обработи pagination с CCXT за Binance API
- Тествай за август-септември 2024 (+6.4%)
- Върни pandas DataFrame с Date, Open, High, Low, Close, Volume

### 2. **helformer_model.py**
- Адаптирай за hybrid monthly (seasonal_periods=1, 5-10%) и quarterly (seasonal_periods=3, 20-40%) с PyTorch 2.0+
- Настрой set_forecast_horizon(720h за monthly, 2880h за quarterly)
- Избягвай shape mismatches (assert x.shape[2] == 5) и NaN (np.nan_to_num)
- Тествай за >85% accuracy
- Интегрирай прогнози за signal_generator.py

### 3. **signal_generator.py**
- Имплементирай hybrid сигнали с TA-Lib:
  - **Monthly**: EMA10/50, RSI7, ROC1, MACD12/26/9
  - **Quarterly**: EMA50/200, RSI14, ROC3, MACD50/200/9
- **Buy условия**: EMA_fast > EMA_slow, RSI < 40, ROC > 0, MACD > Signal, цена в ниво ($600-650)
- **Sell условия**: EMA_fast < EMA_slow, RSI > 70, MACD < Signal, цена в цел ($750-780)
- Следвай Rule #1 (котвата), #5 (излизане), #6 (една битка) от Хайдушкия кодекс
- Интегрирай прогнози от helformer_model.py и нива от fib.py

### 4. **swing_risk_manager.py**
- Клас `SwingRiskManager(capital=10000, risk_per_trade=0.02)`
- **ATR**: period=7 (monthly) или 14 (quarterly)
- **Stop-loss**: multiplier=3 (monthly) или 2 (quarterly)
- **Position size**: 1/3 капитал, ливъридж до 2х (Rule #3, #4)
- Отстъпление при пробив под $550 (Rule #7)
- Цел: drawdown <10% monthly, <15% quarterly

### 5. **fib.py**
- Изчисли Fibonacci нива (23.6%, 38.2%, 61.8%) за вход/изход ($600-650, $750-780)
- Тествай за Q3-Q4 2024 (Low $407.52, High $793.35) и септември 2025 (~$750-800)
- Интегрирай в signal_generator.py за потвърждение на сигнали
- Следвай Rule #1 (котвата) и Rule #5 (излизане)

### 6. **swing_backtester.py**
- Използвай Backtrader за тестване на сигнали
- Валидирай за август-септември 2024 (+6.4%) и Q3-Q4 2024 (+31%)
- Добави Sharpe ratio и max drawdown
- Тествай hybrid monthly (5-10%) и quarterly (20-40%) цикли

### 7. **email_reporter.py**
- Имплементирай email известия за Buy/Sell сигнали с smtplib
- Тествай с daily BNB данни за август-септември 2024 (+6.4%) и Q3-Q4 2024 (+31%)
- Следвай Rule #5 (излизане на такт) и Rule #6 (една битка)

## 🥋 ХАЙДУШКИ КОДЕКС

### **Rule #0:**
**Без over-engineering. Keep it simple.**

### **Rule #1:**
**Котвата е закон.**
– Влизаме само на ясно ниво (примерно $600–650).
– Никакви гонитби, никакъв FOMO.

### **Rule #2:**
**Търпение над всичко.**
– Дъното обръща за 3–7 дни → бързо действие.
– Върховете се търкалят 2–4 седмици → време за наблюдение.

### **Rule #3:**
**Стъпки, не скокове.**
– Влизаме с 1/3.
– Усредняваме само при потвърдено движение.
– Никога all-in.

### **Rule #4:**
**Ливъридж до 2х, буфер задължителен.**
– Капитанът не оставя дружината на ликвидация.

### **Rule #5:**
**Излизане на такт.**
– Вземаме печалба на повторяеми цели ($750–780).
– Не гоним върхове, не пускаме корени в позиции.

### **Rule #6:**
**Една битка наведнъж.**
– Никакви паралелни входове.
– Никакви отмъстителни трейдове.

### **Rule #7:**
**Когато планът се счупи → отстъпление.**
– Ако седмична свещ затвори под $550 → прегрупиране, не упорство.

### **Rule #8:**
**Дружината над всичко.**
– Всеки хайдутин има роля: едни смятат, други проверяват, капитанът решава.

### **📌 Кратко мото:**
**„Търпение, математика, и едно хоро в правилния такт."**

## 🕺 'Хоро' Философия
- **Две стъпки напред, една назад**
- Влизай/излизай в такт с пазара
- Не се бори с тренда, танцувай с него

## 📈 Технически Изисквания

### **Data Pipeline**
- **Source**: Binance API via CCXT
- **Interval**: 1d (daily)
- **Period**: 2024-02-01 до 2025-08-31
- **Points**: ~540 daily data points
- **Format**: pandas DataFrame (Date, Open, High, Low, Close, Volume)

### **Performance Targets**
- **Accuracy**: >85%
- **Monthly returns**: >5%
- **Quarterly returns**: >25%
- **Risk**: <10% monthly drawdown, <15% quarterly drawdown

### **Technical Stack**
- **ML Framework**: PyTorch 2.0+ (без TensorFlow)
- **Data Processing**: pandas 2.0+
- **Technical Indicators**: TA-Lib
- **Backtesting**: Backtrader
- **Email**: smtplib

## 🔧 Development Workflow

### **Phase 1: Foundation**
1. Setup PyTorch 2.0+ environment
2. Implement data_fetcher.py with daily data
3. Basic Helformer enhancement

### **Phase 2: Core Features**
1. Technical indicators (TA-Lib)
2. Risk management class
3. Fibonacci analysis

### **Phase 3: Integration**
1. Signal generation
2. Module integration
3. Basic testing

### **Phase 4: Advanced Features**
1. On-chain metrics
2. Backtesting
3. Email automation

## 📝 Notes
- Всички модули трябва да работят с еднакви data format
- Избягвай NaN стойности в индикаторите
- Валидирай shape consistency (assert x.shape[2] == 5)
- Следвай Хайдушкия кодекс за дисциплина
- Прилагай 'хоро' философията за timing

---
*Тези насоки осигуряват консистентна работа по модулите и следват установените принципи.*
*Last updated: 2025-08-22*
*Next review: Week of 2025-08-25*
