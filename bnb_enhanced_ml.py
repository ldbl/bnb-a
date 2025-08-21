#!/usr/bin/env python3
"""
BNB Enhanced ML System
Analyze BNB using data and patterns learned from top 10 cryptocurrencies
"""

import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import SelectKBest, f_classif
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from logger import get_logger

class BNBEnhancedML:
    """BNB analyzer using patterns learned from top 10 cryptocurrencies"""
    
    def __init__(self, model_dir: str = "ml_models_bnb_enhanced"):
        self.logger = get_logger(__name__)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Top 10 cryptocurrencies for learning patterns
        self.learning_cryptos = {
            "BTCUSDT": {"name": "Bitcoin", "weight": 0.40, "category": "store_of_value"},
            "ETHUSDT": {"name": "Ethereum", "weight": 0.20, "category": "smart_contracts"},
            "BNBUSDT": {"name": "BNB", "weight": 0.08, "category": "exchange"},  # Our target
            "XRPUSDT": {"name": "XRP", "weight": 0.05, "category": "payments"},
            "SOLUSDT": {"name": "Solana", "weight": 0.05, "category": "high_performance"},
            "ADAUSDT": {"name": "Cardano", "weight": 0.04, "category": "academic"},
            "AVAXUSDT": {"name": "Avalanche", "weight": 0.03, "category": "defi"},
            "DOTUSDT": {"name": "Polkadot", "weight": 0.03, "category": "interop"},
            "LINKUSDT": {"name": "Chainlink", "weight": 0.03, "category": "oracle"},
            "MATICUSDT": {"name": "Polygon", "weight": 0.03, "category": "scaling"}
        }
        
        self.base_url = "https://api.binance.com/api/v3"
        self.models = {}
        self.scalers = {}
        self.pattern_library = {}  # Learned patterns from all cryptos
        
        # Feature engineering parameters
        self.lookback_windows = [5, 10, 20, 50, 100]
        self.fibonacci_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.volume_thresholds = [1.5, 2.0, 2.5, 3.0, 4.0]
        
        self.logger.info(f"BNBEnhancedML initialized - will learn from {len(self.learning_cryptos)} cryptos")
    
    def fetch_learning_data(self, interval: str = "1d", limit: int = 1000) -> Dict[str, pd.DataFrame]:
        """Fetch data from all top 10 cryptos for pattern learning"""
        
        self.logger.info(f"Fetching learning data from {len(self.learning_cryptos)} cryptocurrencies...")
        
        learning_data = {}
        
        for symbol, info in self.learning_cryptos.items():
            try:
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "limit": min(limit, 1000)
                }
                
                self.logger.debug(f"Fetching {symbol} ({info['name']})...")
                response = requests.get(f"{self.base_url}/klines", params=params, timeout=15)
                
                if response.status_code == 200:
                    klines = response.json()
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(klines, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore'
                    ])
                    
                    # Convert to proper data types
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades']
                    for col in numeric_columns:
                        df[col] = pd.to_numeric(df[col])
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Add metadata
                    df['symbol'] = symbol
                    df['crypto_name'] = info['name']
                    df['category'] = info['category']
                    df['market_weight'] = info['weight']
                    
                    learning_data[symbol] = df
                    self.logger.debug(f"✅ {symbol}: {len(df)} candles")
                    
                else:
                    self.logger.error(f"❌ Failed to fetch {symbol}: {response.status_code}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error fetching {symbol}: {e}")
        
        self.logger.info(f"✅ Collected learning data from {len(learning_data)} cryptocurrencies")
        return learning_data
    
    def extract_universal_patterns(self, crypto_data: Dict[str, pd.DataFrame]) -> Dict:
        """Extract patterns that work across all cryptocurrencies"""
        
        self.logger.info("Extracting universal patterns from all cryptocurrencies...")
        
        universal_patterns = {
            "fibonacci_effectiveness": {},
            "volume_spike_patterns": {},
            "candlestick_patterns": {},
            "momentum_patterns": {},
            "correlation_patterns": {},
            "market_regime_patterns": {}
        }
        
        # Analyze each cryptocurrency for patterns
        for symbol, df in crypto_data.items():
            if len(df) < 100:
                continue
                
            crypto_name = df['crypto_name'].iloc[0]
            self.logger.debug(f"Analyzing patterns in {crypto_name}...")
            
            # 1. Test Fibonacci level effectiveness
            fib_effectiveness = self._test_fibonacci_levels(df)
            universal_patterns["fibonacci_effectiveness"][symbol] = fib_effectiveness
            
            # 2. Volume spike patterns
            volume_patterns = self._analyze_volume_spikes(df)
            universal_patterns["volume_spike_patterns"][symbol] = volume_patterns
            
            # 3. Candlestick patterns
            candle_patterns = self._analyze_candlestick_patterns(df)
            universal_patterns["candlestick_patterns"][symbol] = candle_patterns
            
            # 4. Momentum patterns
            momentum_patterns = self._analyze_momentum_patterns(df)
            universal_patterns["momentum_patterns"][symbol] = momentum_patterns
        
        # Extract universal insights
        universal_insights = self._extract_universal_insights(universal_patterns)
        
        # Store pattern library
        self.pattern_library = {
            "patterns": universal_patterns,
            "insights": universal_insights,
            "learning_date": datetime.now().isoformat()
        }
        
        self.logger.info(f"✅ Extracted {len(universal_insights)} universal insights")
        return universal_insights
    
    def _test_fibonacci_levels(self, df: pd.DataFrame) -> Dict:
        """Test effectiveness of Fibonacci retracement levels"""
        
        fib_results = {}
        
        for window in [20, 50, 100]:
            if len(df) <= window:
                continue
                
            # Calculate rolling high/low
            rolling_high = df['high'].rolling(window).max()
            rolling_low = df['low'].rolling(window).min()
            price_range = rolling_high - rolling_low
            
            for ratio in self.fibonacci_ratios:
                fib_level = rolling_low + (price_range * ratio)
                
                # Test how often price bounces from this level
                distance_to_fib = abs(df['close'] - fib_level) / fib_level
                near_fib = distance_to_fib < 0.02  # Within 2%
                
                # Check what happens after touching fib level
                reversal_count = 0
                total_touches = 0
                
                for i in range(len(df) - 10):
                    if near_fib.iloc[i]:
                        total_touches += 1
                        
                        # Check if price reversed in next 5-10 periods
                        future_prices = df['close'].iloc[i+1:i+11]
                        if len(future_prices) > 0:
                            if ratio < 0.5:  # Support levels
                                if future_prices.max() > df['close'].iloc[i] * 1.03:  # 3%+ bounce
                                    reversal_count += 1
                            else:  # Resistance levels
                                if future_prices.min() < df['close'].iloc[i] * 0.97:  # 3%+ drop
                                    reversal_count += 1
                
                effectiveness = reversal_count / total_touches if total_touches > 5 else 0
                
                fib_results[f"fib_{ratio}_{window}"] = {
                    "effectiveness": effectiveness,
                    "total_touches": total_touches,
                    "reversals": reversal_count
                }
        
        return fib_results
    
    def _analyze_volume_spikes(self, df: pd.DataFrame) -> Dict:
        """Analyze volume spike patterns and their predictive power"""
        
        volume_ma = df['volume'].rolling(20).mean()
        volume_patterns = {}
        
        for threshold in self.volume_thresholds:
            volume_spikes = df['volume'] > (volume_ma * threshold)
            
            # Analyze what happens after volume spikes
            bullish_after_spike = 0
            bearish_after_spike = 0
            total_spikes = 0
            
            for i in range(len(df) - 10):
                if volume_spikes.iloc[i]:
                    total_spikes += 1
                    
                    # Check price movement in next 5 periods
                    current_price = df['close'].iloc[i]
                    future_price = df['close'].iloc[i+5] if i+5 < len(df) else current_price
                    
                    price_change = (future_price - current_price) / current_price
                    
                    if price_change > 0.03:  # 3%+ bullish
                        bullish_after_spike += 1
                    elif price_change < -0.03:  # 3%+ bearish
                        bearish_after_spike += 1
            
            if total_spikes > 0:
                volume_patterns[f"spike_{threshold}x"] = {
                    "total_occurrences": total_spikes,
                    "bullish_probability": bullish_after_spike / total_spikes,
                    "bearish_probability": bearish_after_spike / total_spikes,
                    "predictive_power": max(bullish_after_spike, bearish_after_spike) / total_spikes
                }
        
        return volume_patterns
    
    def _analyze_candlestick_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze candlestick pattern effectiveness"""
        
        # Calculate basic candlestick metrics
        body_size = abs(df['close'] - df['open'])
        full_range = df['high'] - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        
        body_ratio = body_size / (full_range + 1e-8)
        upper_shadow_ratio = upper_shadow / (full_range + 1e-8)
        lower_shadow_ratio = lower_shadow / (full_range + 1e-8)
        
        patterns = {}
        
        # Doji pattern (small body)
        doji_pattern = body_ratio < 0.1
        patterns["doji"] = self._test_pattern_effectiveness(df, doji_pattern, "doji")
        
        # Hammer pattern (long lower shadow, small body)
        hammer_pattern = (lower_shadow_ratio > 0.6) & (body_ratio < 0.3)
        patterns["hammer"] = self._test_pattern_effectiveness(df, hammer_pattern, "hammer")
        
        # Engulfing pattern (simplified)
        bullish_engulfing = (
            (df['close'] > df['open']) &  # Current is bullish
            (df['close'].shift(1) < df['open'].shift(1)) &  # Previous was bearish
            (df['close'] > df['open'].shift(1)) &  # Current close > previous open
            (df['open'] < df['close'].shift(1))  # Current open < previous close
        )
        patterns["bullish_engulfing"] = self._test_pattern_effectiveness(df, bullish_engulfing, "bullish_engulfing")
        
        return patterns
    
    def _test_pattern_effectiveness(self, df: pd.DataFrame, pattern: pd.Series, pattern_name: str) -> Dict:
        """Test how effective a candlestick pattern is for prediction"""
        
        bullish_success = 0
        bearish_success = 0
        total_occurrences = 0
        
        for i in range(len(df) - 10):
            if pattern.iloc[i]:
                total_occurrences += 1
                
                # Check what happens in next 5 periods
                current_price = df['close'].iloc[i]
                future_price = df['close'].iloc[i+5] if i+5 < len(df) else current_price
                
                price_change = (future_price - current_price) / current_price
                
                if price_change > 0.03:  # 3%+ bullish
                    bullish_success += 1
                elif price_change < -0.03:  # 3%+ bearish
                    bearish_success += 1
        
        return {
            "total_occurrences": total_occurrences,
            "bullish_success_rate": bullish_success / total_occurrences if total_occurrences > 0 else 0,
            "bearish_success_rate": bearish_success / total_occurrences if total_occurrences > 0 else 0,
            "overall_effectiveness": max(bullish_success, bearish_success) / total_occurrences if total_occurrences > 0 else 0
        }
    
    def _analyze_momentum_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze momentum-based patterns"""
        
        momentum_patterns = {}
        
        for window in [7, 14, 21]:
            if len(df) <= window:
                continue
                
            # Calculate momentum
            momentum = df['close'] / df['close'].shift(window)
            
            # Test different momentum thresholds
            for threshold in [1.05, 1.10, 1.15]:  # 5%, 10%, 15% momentum
                strong_momentum = momentum > threshold
                weak_momentum = momentum < (2 - threshold)  # Inverse for bearish
                
                # Test continuation vs reversal after strong momentum
                momentum_patterns[f"momentum_{window}_{threshold}"] = {
                    "bullish": self._test_momentum_continuation(df, strong_momentum),
                    "bearish": self._test_momentum_continuation(df, weak_momentum)
                }
        
        return momentum_patterns
    
    def _test_momentum_continuation(self, df: pd.DataFrame, momentum_signal: pd.Series) -> Dict:
        """Test if momentum continues or reverses"""
        
        continuation_count = 0
        reversal_count = 0
        total_signals = 0
        
        for i in range(len(df) - 10):
            if momentum_signal.iloc[i]:
                total_signals += 1
                
                # Check if momentum continues in next 5 periods
                current_price = df['close'].iloc[i]
                future_price = df['close'].iloc[i+5] if i+5 < len(df) else current_price
                
                price_change = (future_price - current_price) / current_price
                
                # Determine if momentum continued or reversed
                if abs(price_change) > 0.03:
                    if (momentum_signal.iloc[i] and price_change > 0) or (not momentum_signal.iloc[i] and price_change < 0):
                        continuation_count += 1
                    else:
                        reversal_count += 1
        
        return {
            "total_signals": total_signals,
            "continuation_rate": continuation_count / total_signals if total_signals > 0 else 0,
            "reversal_rate": reversal_count / total_signals if total_signals > 0 else 0
        }
    
    def _extract_universal_insights(self, patterns: Dict) -> List[str]:
        """Extract insights that work across multiple cryptocurrencies"""
        
        insights = []
        
        # Analyze Fibonacci effectiveness across cryptos
        fib_effectiveness = {}
        for symbol, fib_data in patterns["fibonacci_effectiveness"].items():
            for fib_level, data in fib_data.items():
                if fib_level not in fib_effectiveness:
                    fib_effectiveness[fib_level] = []
                fib_effectiveness[fib_level].append(data["effectiveness"])
        
        # Find universally effective Fibonacci levels
        for fib_level, effectiveness_list in fib_effectiveness.items():
            if len(effectiveness_list) >= 5:  # At least 5 cryptos
                avg_effectiveness = np.mean(effectiveness_list)
                if avg_effectiveness > 0.6:  # 60%+ effectiveness
                    insights.append(f"🔢 {fib_level.replace('_', ' ')} level is effective across cryptos ({avg_effectiveness:.1%} success rate)")
        
        # Analyze volume spike patterns
        volume_effectiveness = {}
        for symbol, vol_data in patterns["volume_spike_patterns"].items():
            for spike_level, data in vol_data.items():
                if spike_level not in volume_effectiveness:
                    volume_effectiveness[spike_level] = []
                volume_effectiveness[spike_level].append(data["predictive_power"])
        
        for spike_level, power_list in volume_effectiveness.items():
            if len(power_list) >= 5:
                avg_power = np.mean(power_list)
                if avg_power > 0.5:  # 50%+ predictive power
                    insights.append(f"📊 {spike_level.replace('_', ' ')} volume spikes predict moves ({avg_power:.1%} accuracy)")
        
        # Analyze candlestick patterns
        candle_effectiveness = {}
        for symbol, candle_data in patterns["candlestick_patterns"].items():
            for pattern_name, data in candle_data.items():
                if pattern_name not in candle_effectiveness:
                    candle_effectiveness[pattern_name] = []
                candle_effectiveness[pattern_name].append(data["overall_effectiveness"])
        
        for pattern_name, eff_list in candle_effectiveness.items():
            if len(eff_list) >= 5:
                avg_eff = np.mean(eff_list)
                if avg_eff > 0.4:  # 40%+ effectiveness
                    insights.append(f"🕯️ {pattern_name} pattern works across cryptos ({avg_eff:.1%} success rate)")
        
        return insights
    
    def create_bnb_enhanced_features(self, bnb_data: pd.DataFrame, market_patterns: Dict) -> pd.DataFrame:
        """Create BNB features enhanced with learned patterns from all cryptos"""
        
        self.logger.info("Creating BNB features enhanced with multi-crypto intelligence...")
        
        features_df = bnb_data.copy()
        
        # 1. BASIC BNB FEATURES
        for window in self.lookback_windows:
            if len(bnb_data) > window:
                features_df[f'return_{window}'] = bnb_data['close'].pct_change(window)
                features_df[f'volatility_{window}'] = bnb_data['close'].rolling(window).std()
                features_df[f'volume_ratio_{window}'] = bnb_data['volume'] / bnb_data['volume'].rolling(window).mean()
        
        # 2. ENHANCED FIBONACCI FEATURES (based on learned effectiveness)
        for window in [20, 50, 100]:
            if len(bnb_data) > window:
                rolling_high = bnb_data['high'].rolling(window).max()
                rolling_low = bnb_data['low'].rolling(window).min()
                price_range = rolling_high - rolling_low
                
                for ratio in self.fibonacci_ratios:
                    # Use learned effectiveness weights
                    fib_key = f"fib_{ratio}_{window}"
                    effectiveness_weight = 1.0  # Default
                    
                    # Apply learned weights if available
                    if "insights" in market_patterns:
                        for insight in market_patterns["insights"]:
                            if fib_key.replace('_', ' ') in insight:
                                # Extract effectiveness from insight
                                if "60%" in insight:
                                    effectiveness_weight = 1.5
                                elif "70%" in insight:
                                    effectiveness_weight = 2.0
                    
                    fib_level = rolling_low + (price_range * ratio)
                    features_df[f'weighted_fib_{ratio}_{window}'] = (
                        ((bnb_data['close'] - fib_level) / fib_level) * effectiveness_weight
                    )
                    
                    # Enhanced fib proximity detection
                    features_df[f'near_fib_{ratio}_{window}'] = (
                        abs(bnb_data['close'] - fib_level) / fib_level < 0.02
                    ).astype(int) * effectiveness_weight
        
        # 3. ENHANCED VOLUME FEATURES (based on learned spike patterns)
        volume_ma = bnb_data['volume'].rolling(20).mean()
        for threshold in self.volume_thresholds:
            # Apply learned effectiveness weights
            effectiveness_weight = 1.0
            spike_key = f"spike_{threshold}x"
            
            if "insights" in market_patterns:
                for insight in market_patterns["insights"]:
                    if spike_key.replace('_', ' ') in insight:
                        if "50%" in insight:
                            effectiveness_weight = 1.5
                        elif "60%" in insight:
                            effectiveness_weight = 2.0
            
            features_df[f'weighted_volume_spike_{threshold}x'] = (
                (bnb_data['volume'] > (volume_ma * threshold)).astype(int) * effectiveness_weight
            )
        
        # 4. ENHANCED CANDLESTICK FEATURES
        body_size = abs(bnb_data['close'] - bnb_data['open'])
        full_range = bnb_data['high'] - bnb_data['low']
        
        features_df['body_ratio'] = body_size / (full_range + 1e-8)
        features_df['upper_shadow_ratio'] = (bnb_data['high'] - bnb_data[['open', 'close']].max(axis=1)) / (full_range + 1e-8)
        features_df['lower_shadow_ratio'] = (bnb_data[['open', 'close']].min(axis=1) - bnb_data['low']) / (full_range + 1e-8)
        
        # Enhanced pattern detection with learned weights
        pattern_weights = {"doji": 1.0, "hammer": 1.0, "bullish_engulfing": 1.0}
        
        if "insights" in market_patterns:
            for insight in market_patterns["insights"]:
                for pattern in pattern_weights.keys():
                    if pattern in insight:
                        if "40%" in insight:
                            pattern_weights[pattern] = 1.2
                        elif "50%" in insight:
                            pattern_weights[pattern] = 1.5
        
        # Apply enhanced pattern detection
        features_df['weighted_doji'] = (
            (features_df['body_ratio'] < 0.1).astype(int) * pattern_weights["doji"]
        )
        
        features_df['weighted_hammer'] = (
            ((features_df['lower_shadow_ratio'] > 0.6) & (features_df['body_ratio'] < 0.3)).astype(int) * pattern_weights["hammer"]
        )
        
        # 5. MARKET REGIME FEATURES (based on learned patterns)
        # BTC correlation (BNB follows market leader patterns)
        btc_data = self._fetch_btc_reference()
        if btc_data is not None and len(btc_data) == len(bnb_data):
            features_df['btc_correlation'] = bnb_data['close'].rolling(20).corr(btc_data['close'])
            features_df['relative_strength_vs_btc'] = bnb_data['close'] / btc_data['close']
        
        # Clean data
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_df.dropna(inplace=True)
        
        self.logger.info(f"✅ Created {len(features_df.columns)} enhanced BNB features")
        return features_df
    
    def _fetch_btc_reference(self) -> Optional[pd.DataFrame]:
        """Fetch BTC data for correlation analysis"""
        try:
            params = {
                "symbol": "BTCUSDT",
                "interval": "1d",
                "limit": 500
            }
            
            response = requests.get(f"{self.base_url}/klines", params=params, timeout=10)
            if response.status_code == 200:
                klines = response.json()
                df = pd.DataFrame(klines)
                df['close'] = pd.to_numeric(df[4])  # Close price is column 4
                return df
        except:
            pass
        return None
    
    def train_bnb_enhanced_model(self, periods_ahead: int = 10) -> Dict:
        """Train BNB model enhanced with patterns from all cryptocurrencies"""
        
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not available"}
        
        self.logger.info("Training BNB model with multi-crypto enhanced intelligence...")
        
        # Step 1: Learn patterns from all cryptocurrencies
        self.logger.info("Step 1: Learning patterns from top 10 cryptocurrencies...")
        crypto_data = self.fetch_learning_data("1h", 1000)
        
        if len(crypto_data) < 5:
            return {"error": "Insufficient learning data from cryptocurrencies"}
        
        # Extract universal patterns
        universal_insights = self.extract_universal_patterns(crypto_data)
        
        # Step 2: Get BNB data and apply enhanced features
        self.logger.info("Step 2: Creating enhanced BNB features...")
        bnb_data = crypto_data.get("BNBUSDT")
        if bnb_data is None:
            return {"error": "No BNB data available"}
        
        # Create enhanced features
        features_df = self.create_bnb_enhanced_features(bnb_data, self.pattern_library)
        
        # Create labels
        labels = self._create_enhanced_labels(features_df, periods_ahead)
        
        # Prepare training data
        common_index = features_df.index.intersection(labels.index)
        features_clean = features_df.loc[common_index]
        labels_clean = labels.loc[common_index]
        
        # Select numeric features
        numeric_features = features_clean.select_dtypes(include=[np.number]).columns.tolist()
        X = features_clean[numeric_features].values
        y = labels_clean.values
        
        # Remove NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        if len(X) < 100:
            return {"error": "Insufficient clean training data"}
        
        self.logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=min(50, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train enhanced models
        models = {}
        results = {}
        
        model_configs = {
            "enhanced_rf": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
            "enhanced_gb": GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, random_state=42),
            "enhanced_lr": LogisticRegression(random_state=42, max_iter=1500, C=0.1)
        }
        
        for model_name, model in model_configs.items():
            try:
                self.logger.info(f"Training {model_name}...")
                
                model.fit(X_train_scaled, y_train)
                
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                models[model_name] = model
                results[model_name] = {
                    "train_accuracy": train_score,
                    "test_accuracy": test_score
                }
                
                self.logger.info(f"{model_name}: Train={train_score:.3f}, Test={test_score:.3f}")
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        # Store model in memory
        model_key = f"bnb_enhanced_{periods_ahead}"
        self.models[model_key] = {
            "models": models,
            "scaler": scaler,
            "feature_selector": selector,
            "selected_features": [numeric_features[i] for i in selector.get_support(indices=True)],
            "pattern_library": self.pattern_library,
            "universal_insights": universal_insights
        }
        
        # Save models to disk for persistence
        try:
            import pickle
            import json
            from datetime import datetime
            
            model_path = self.model_dir / f"bnb_enhanced_{periods_ahead}.pkl"
            metadata_path = self.model_dir / f"bnb_enhanced_{periods_ahead}_metadata.json"
            
            # Save complete model data
            with open(model_path, 'wb') as f:
                pickle.dump(self.models[model_key], f)
            
            # Save metadata
            metadata = {
                "created_at": datetime.now().isoformat(),
                "periods_ahead": periods_ahead,
                "models_trained": len(models),
                "training_samples": len(X_train),
                "learning_cryptos": len(crypto_data),
                "enhanced_features": len(numeric_features),
                "results": results
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"✅ Models saved to disk: {model_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save models to disk: {e}")
        
        return {
            "success": True,
            "models_trained": len(models),
            "training_samples": len(X_train),
            "learning_cryptos": len(crypto_data),
            "universal_insights": len(universal_insights),
            "enhanced_features": len(numeric_features),
            "results": results
        }
    
    def _create_enhanced_labels(self, df: pd.DataFrame, periods_ahead: int) -> pd.Series:
        """Create enhanced labels for BNB prediction"""
        
        labels = pd.Series(0, index=df.index)
        
        for i in range(len(df) - periods_ahead):
            current_price = df['close'].iloc[i] if 'close' in df.columns else df.iloc[i, 3]  # Fallback
            
            # Get future prices
            future_idx = min(i + periods_ahead, len(df) - 1)
            future_prices = df['close'].iloc[i+1:future_idx+1] if 'close' in df.columns else df.iloc[i+1:future_idx+1, 3]
            
            if len(future_prices) > 0:
                max_gain = (future_prices.max() - current_price) / current_price
                max_loss = (future_prices.min() - current_price) / current_price
                
                # Enhanced thresholds
                if max_gain >= 0.05:  # 5%+ bullish
                    labels.iloc[i] = 1
                elif max_loss <= -0.05:  # 5%+ bearish
                    labels.iloc[i] = 2
        
        return labels
    
    def predict_bnb_enhanced(self, periods_ahead: int = 10) -> Dict:
        """Make enhanced BNB prediction using multi-crypto intelligence"""
        
        model_key = f"bnb_enhanced_{periods_ahead}"
        
        # Try to load from memory first, then from disk
        if model_key not in self.models:
            # Try to load from disk
            try:
                import pickle
                model_path = self.model_dir / f"bnb_enhanced_{periods_ahead}.pkl"
                
                if model_path.exists():
                    self.logger.info(f"Loading trained model from disk: {model_path}")
                    with open(model_path, 'rb') as f:
                        self.models[model_key] = pickle.load(f)
                else:
                    return {"error": f"No trained model for {periods_ahead} periods ahead"}
            except Exception as e:
                self.logger.error(f"Failed to load model from disk: {e}")
                return {"error": f"No trained model for {periods_ahead} periods ahead"}
        
        try:
            # Get fresh BNB data  
            bnb_params = {"symbol": "BNBUSDT", "interval": "1d", "limit": 200}
            response = requests.get(f"{self.base_url}/klines", params=bnb_params, timeout=10)
            
            if response.status_code != 200:
                return {"error": "Failed to fetch current BNB data"}
            
            klines = response.json()
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Create enhanced features
            model_info = self.models[model_key]
            features_df = self.create_bnb_enhanced_features(df, model_info["pattern_library"])
            
            # Get latest features
            selected_features = model_info["selected_features"]
            latest_features = features_df[selected_features].iloc[-1:].values
            
            # Scale features
            scaled_features = model_info["scaler"].transform(latest_features)
            
            # Make predictions
            predictions = {}
            confidence_scores = []
            
            for model_name, model in model_info["models"].items():
                prediction = model.predict(scaled_features)[0]
                probability = model.predict_proba(scaled_features)[0] if hasattr(model, 'predict_proba') else None
                
                predictions[model_name] = {
                    "prediction": int(prediction),
                    "probability": probability.tolist() if probability is not None else None
                }
                
                if probability is not None:
                    confidence_scores.append(probability.max())
            
            # Ensemble prediction
            prediction_votes = [pred["prediction"] for pred in predictions.values()]
            ensemble_prediction = max(set(prediction_votes), key=prediction_votes.count) if prediction_votes else 0
            
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            
            current_price = df['close'].iloc[-1]
            
            return {
                "ensemble_prediction": ensemble_prediction,
                "prediction_label": ["No Reversal", "Bullish Reversal", "Bearish Reversal"][ensemble_prediction],
                "confidence": avg_confidence,
                "current_bnb_price": current_price,
                "periods_ahead": periods_ahead,
                "individual_predictions": predictions,
                "universal_insights": model_info.get("universal_insights", []),
                "enhanced_analysis": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced prediction failed: {e}")
            return {"error": str(e)}


# Example usage
if __name__ == "__main__":
    print("🎯 BNB ENHANCED ML - Learning from Top 10 Cryptocurrencies")
    print("=" * 70)
    
    # Initialize system
    bnb_ml = BNBEnhancedML()
    
    # Train enhanced model
    print("🧠 Training BNB model with multi-crypto intelligence...")
    result = bnb_ml.train_bnb_enhanced_model(10)
    
    if "success" in result:
        print(f"✅ Success: {result['models_trained']} models trained")
        print(f"📊 Learning data from: {result['learning_cryptos']} cryptocurrencies")
        print(f"🎯 Universal insights discovered: {result['universal_insights']}")
        print(f"🔧 Enhanced features: {result['enhanced_features']}")
        
        # Make prediction
        print(f"\n🔮 Making enhanced BNB prediction...")
        prediction = bnb_ml.predict_bnb_enhanced(10)
        
        if "error" not in prediction:
            print(f"💰 Current BNB Price: ${prediction['current_bnb_price']:.2f}")
            print(f"🎯 Prediction: {prediction['prediction_label']}")
            print(f"🎲 Confidence: {prediction['confidence']:.1%}")
            print(f"📊 Enhanced with multi-crypto intelligence: ✅")
        else:
            print(f"❌ Prediction error: {prediction['error']}")
            
    else:
        print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
