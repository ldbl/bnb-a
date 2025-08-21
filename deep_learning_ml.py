#!/usr/bin/env python3
"""
Deep Learning ML System for BNB
Enhanced ML with automatic pattern discovery from massive historical data
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
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.feature_selection import SelectKBest, f_classif
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Deep learning imports
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from logger import get_logger

class DeepLearningML:
    """Advanced ML system with automatic pattern discovery"""
    
    def __init__(self, model_dir: str = "ml_models_deep"):
        self.logger = get_logger(__name__)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.base_url = "https://api.binance.com/api/v3"
        self.symbol = "BNBUSDT"
        
        # Massive data collection parameters
        self.max_data_points = 5000  # Much more data
        self.feature_lookback_windows = [5, 10, 20, 50, 100, 200]  # Multiple timeframes
        self.pattern_detection_windows = [3, 5, 7, 10, 14, 21]  # Various pattern lengths
        
        # Automatic pattern discovery
        self.price_level_percentiles = np.arange(0.1, 1.0, 0.05)  # Every 5% level
        self.fibonacci_like_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
        self.volume_spike_thresholds = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        self.logger.info("DeepLearningML initialized for automatic pattern discovery")
    
    def fetch_massive_historical_data(self, intervals: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Fetch massive amounts of historical data across multiple timeframes"""
        
        if intervals is None:
            intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        self.logger.info(f"Fetching massive historical data across {len(intervals)} timeframes...")
        
        all_data = {}
        
        for interval in intervals:
            try:
                # Calculate how much data we can get for each interval
                if interval == "1m":
                    limit = 1000  # Binance limit
                elif interval == "5m":
                    limit = 1000
                elif interval == "15m":
                    limit = 1000
                elif interval == "1h":
                    limit = 1000
                elif interval == "4h":
                    limit = 1000
                elif interval == "1d":
                    limit = 1000
                else:
                    limit = 1000
                
                self.logger.info(f"Fetching {interval} data (limit: {limit})...")
                
                params = {
                    "symbol": self.symbol,
                    "interval": interval,
                    "limit": limit
                }
                
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
                    
                    all_data[interval] = df
                    self.logger.info(f"✅ {interval}: {len(df)} candles")
                    
                else:
                    self.logger.error(f"❌ Failed to fetch {interval}: {response.status_code}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error fetching {interval}: {e}")
        
        total_data_points = sum(len(df) for df in all_data.values())
        self.logger.info(f"✅ Total data points collected: {total_data_points}")
        
        return all_data
    
    def create_automatic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create extensive features for automatic pattern discovery"""
        
        self.logger.info("Creating extensive feature set for pattern discovery...")
        
        features_df = df.copy()
        
        # 1. BASIC PRICE FEATURES (Multiple timeframes)
        for window in self.feature_lookback_windows:
            if len(df) > window:
                # Price movements
                features_df[f'return_{window}'] = df['close'].pct_change(window)
                features_df[f'high_return_{window}'] = (df['high'] - df['close'].shift(window)) / df['close'].shift(window)
                features_df[f'low_return_{window}'] = (df['low'] - df['close'].shift(window)) / df['close'].shift(window)
                
                # Volatility measures
                features_df[f'volatility_{window}'] = df['close'].rolling(window).std()
                features_df[f'price_range_{window}'] = (df['high'] - df['low']).rolling(window).mean()
                
                # Moving averages and positions
                features_df[f'sma_{window}'] = df['close'].rolling(window).mean()
                features_df[f'price_vs_sma_{window}'] = df['close'] / features_df[f'sma_{window}']
                
                # High/Low positions
                features_df[f'high_percentile_{window}'] = df['high'].rolling(window).rank(pct=True)
                features_df[f'low_percentile_{window}'] = df['low'].rolling(window).rank(pct=True)
        
        # 2. AUTOMATIC FIBONACCI-LIKE LEVEL DETECTION
        for window in [20, 50, 100]:
            if len(df) > window:
                rolling_high = df['high'].rolling(window).max()
                rolling_low = df['low'].rolling(window).min()
                price_range = rolling_high - rolling_low
                
                for ratio in self.fibonacci_like_ratios:
                    level = rolling_low + (price_range * ratio)
                    features_df[f'fib_level_{ratio}_{window}'] = level
                    features_df[f'distance_to_fib_{ratio}_{window}'] = (df['close'] - level) / level
                    features_df[f'price_near_fib_{ratio}_{window}'] = (
                        abs(df['close'] - level) / level < 0.02
                    ).astype(int)
        
        # 3. AUTOMATIC CANDLESTICK PATTERN FEATURES
        # Body and shadow ratios
        body_size = abs(df['close'] - df['open'])
        full_range = df['high'] - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        
        features_df['body_ratio'] = body_size / full_range
        features_df['upper_shadow_ratio'] = upper_shadow / full_range
        features_df['lower_shadow_ratio'] = lower_shadow / full_range
        features_df['shadow_ratio'] = (upper_shadow + lower_shadow) / full_range
        
        # Automatic pattern detection (let ML find the patterns)
        for window in self.pattern_detection_windows:
            if len(df) > window:
                # Consecutive patterns
                features_df[f'consecutive_green_{window}'] = (
                    (df['close'] > df['open']).rolling(window).sum()
                )
                features_df[f'consecutive_red_{window}'] = (
                    (df['close'] < df['open']).rolling(window).sum()
                )
                
                # Body size patterns
                features_df[f'avg_body_size_{window}'] = body_size.rolling(window).mean()
                features_df[f'body_size_std_{window}'] = body_size.rolling(window).std()
                
                # Volume patterns
                features_df[f'avg_volume_{window}'] = df['volume'].rolling(window).mean()
                features_df[f'volume_trend_{window}'] = (
                    df['volume'].rolling(window).apply(
                        lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) > 1 else 0
                    )
                )
        
        # 4. AUTOMATIC SUPPORT/RESISTANCE LEVEL DISCOVERY
        for window in [50, 100, 200]:
            if len(df) > window:
                # Price level clustering (automatic S/R detection)
                price_levels = pd.cut(df['close'].rolling(window), bins=20, labels=False)
                features_df[f'price_level_cluster_{window}'] = price_levels
                
                # Time spent at price levels
                for percentile in [0.1, 0.25, 0.5, 0.75, 0.9]:
                    level = df['close'].rolling(window).quantile(percentile)
                    features_df[f'time_at_level_{percentile}_{window}'] = (
                        (abs(df['close'] - level) / level < 0.01).rolling(window).sum()
                    )
        
        # 5. VOLUME SPIKE DETECTION (Automatic whale detection)
        volume_ma = df['volume'].rolling(20).mean()
        for threshold in self.volume_spike_thresholds:
            features_df[f'volume_spike_{threshold}x'] = (
                df['volume'] > (volume_ma * threshold)
            ).astype(int)
        
        # 6. AUTOMATIC DIVERGENCE DETECTION
        # Price vs Volume divergence
        for window in [10, 20, 30]:
            if len(df) > window:
                price_trend = df['close'].rolling(window).apply(
                    lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) > 1 else 0
                )
                volume_trend = df['volume'].rolling(window).apply(
                    lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) > 1 else 0
                )
                features_df[f'price_volume_divergence_{window}'] = price_trend - volume_trend
        
        # 7. MOMENTUM AND OSCILLATOR FEATURES (Let ML discover optimal periods)
        for period in [7, 14, 21, 30, 50]:
            if len(df) > period:
                # Custom momentum
                features_df[f'momentum_{period}'] = df['close'] / df['close'].shift(period)
                
                # Price acceleration
                features_df[f'acceleration_{period}'] = (
                    features_df[f'momentum_{period}'] - features_df[f'momentum_{period}'].shift(period)
                )
                
                # Volume-weighted momentum
                volume_weight = df['volume'] / df['volume'].rolling(period).sum()
                features_df[f'volume_weighted_momentum_{period}'] = (
                    (df['close'].pct_change() * volume_weight).rolling(period).sum()
                )
        
        # 8. AUTOMATIC CYCLE DETECTION
        # Let ML find cyclic patterns
        for cycle_length in [7, 14, 21, 30, 50, 100]:
            if len(df) > cycle_length:
                features_df[f'cycle_position_{cycle_length}'] = (
                    df.index.dayofweek if cycle_length <= 7 else
                    (df.index.day % cycle_length)
                )
                
                # Seasonal patterns
                features_df[f'seasonal_return_{cycle_length}'] = (
                    df['close'].pct_change(cycle_length)
                )
        
        # 9. POLYNOMIAL FEATURES (Let ML find complex relationships)
        # This creates interactions between features automatically
        price_features = ['close', 'volume', 'high', 'low']
        for i, feature1 in enumerate(price_features):
            for feature2 in price_features[i+1:]:
                features_df[f'{feature1}_x_{feature2}'] = df[feature1] * df[feature2]
                features_df[f'{feature1}_div_{feature2}'] = df[feature1] / (df[feature2] + 1e-8)
        
        # 10. TIME-BASED FEATURES
        features_df['hour'] = df.index.hour
        features_df['day_of_week'] = df.index.dayofweek
        features_df['day_of_month'] = df.index.day
        features_df['month'] = df.index.month
        features_df['quarter'] = df.index.quarter
        
        # 11. STATISTICAL FEATURES
        for window in [20, 50, 100]:
            if len(df) > window:
                # Skewness and kurtosis
                features_df[f'skewness_{window}'] = df['close'].rolling(window).skew()
                features_df[f'kurtosis_{window}'] = df['close'].rolling(window).kurt()
                
                # Percentile positions
                features_df[f'percentile_rank_{window}'] = (
                    df['close'].rolling(window).rank(pct=True)
                )
        
        # Clean data
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        features_df.dropna(inplace=True)
        
        self.logger.info(f"✅ Created {len(features_df.columns)} automatic features")
        return features_df
    
    def create_enhanced_labels(self, df: pd.DataFrame, periods_ahead: int = 10) -> pd.Series:
        """Create enhanced labels with multiple criteria for trend reversal"""
        
        labels = pd.Series(0, index=df.index)
        
        for i in range(len(df) - periods_ahead):
            current_price = df['close'].iloc[i]
            future_prices = df['close'].iloc[i+1:i+periods_ahead+1]
            
            # Enhanced labeling criteria
            max_future = future_prices.max()
            min_future = future_prices.min()
            
            # Calculate gains/losses
            max_gain = (max_future - current_price) / current_price
            max_loss = (min_future - current_price) / current_price
            
            # Multi-threshold labeling (let ML learn the optimal thresholds)
            thresholds = [0.03, 0.05, 0.07, 0.10]  # 3%, 5%, 7%, 10%
            
            # Use the most significant move
            if max_gain >= 0.05:  # 5%+ bullish move
                labels.iloc[i] = 1
            elif max_loss <= -0.05:  # 5%+ bearish move
                labels.iloc[i] = 2
            
            # Enhanced labeling for strong signals
            if max_gain >= 0.10:  # 10%+ very bullish
                labels.iloc[i] = 3
            elif max_loss <= -0.10:  # 10%+ very bearish
                labels.iloc[i] = 4
        
        self.logger.info(f"Label distribution: {labels.value_counts().to_dict()}")
        return labels
    
    def train_deep_learning_models(self, df: pd.DataFrame, periods_ahead: int = 10) -> Dict:
        """Train advanced ML models with automatic feature selection"""
        
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not available"}
        
        self.logger.info(f"Training deep learning models with massive feature set...")
        
        # Create extensive features
        features_df = self.create_automatic_features(df)
        if features_df is None or len(features_df) < 100:
            return {"error": "Insufficient feature data"}
        
        # Create enhanced labels
        labels = self.create_enhanced_labels(features_df, periods_ahead)
        
        # Align features and labels
        common_index = features_df.index.intersection(labels.index)
        features_clean = features_df.loc[common_index]
        labels_clean = labels.loc[common_index]
        
        if len(features_clean) < 200:
            return {"error": "Insufficient training data after cleaning"}
        
        # Select numeric features only
        numeric_features = features_clean.select_dtypes(include=[np.number]).columns.tolist()
        X = features_clean[numeric_features].values
        y = labels_clean.values
        
        # Remove any remaining NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        self.logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        self.logger.info(f"Label distribution: {np.bincount(y)}")
        
        # Feature selection (let ML pick the most important features)
        selector = SelectKBest(score_func=f_classif, k=min(100, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = [numeric_features[i] for i in selector.get_support(indices=True)]
        self.logger.info(f"Selected {len(selected_features)} most important features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {}
        results = {}
        
        # Enhanced model configurations
        model_configs = {
            "random_forest_deep": RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            "gradient_boost_deep": GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            ),
            "logistic_enhanced": LogisticRegression(
                random_state=42,
                max_iter=2000,
                C=0.01,
                penalty='l1',
                solver='liblinear'
            )
        }
        
        # Train models
        for model_name, model in model_configs.items():
            try:
                self.logger.info(f"Training {model_name}...")
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                # Store results
                models[model_name] = model
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    self.feature_importance[model_name] = {
                        "features": selected_features,
                        "importance": importance.tolist()
                    }
                
                results[model_name] = {
                    "train_accuracy": train_score,
                    "test_accuracy": test_score,
                    "feature_count": len(selected_features)
                }
                
                self.logger.info(f"{model_name}: Train={train_score:.3f}, Test={test_score:.3f}")
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        # Store everything
        model_key = f"deep_learning_{periods_ahead}"
        self.models[model_key] = {
            "models": models,
            "scaler": scaler,
            "feature_selector": selector,
            "selected_features": selected_features,
            "periods_ahead": periods_ahead
        }
        
        return {
            "success": True,
            "models_trained": len(models),
            "training_samples": len(X_train),
            "total_features_created": X.shape[1],
            "selected_features": len(selected_features),
            "results": results,
            "automatic_pattern_discovery": True
        }
    
    def analyze_discovered_patterns(self, model_key: str) -> Dict:
        """Analyze what patterns the ML discovered automatically"""
        
        if model_key not in self.models:
            return {"error": "Model not found"}
        
        analysis = {
            "discovered_patterns": {},
            "feature_importance_analysis": {},
            "automatic_insights": []
        }
        
        model_info = self.models[model_key]
        
        # Analyze feature importance across models
        for model_name, importance_data in self.feature_importance.items():
            if model_name in model_info["models"]:
                features = importance_data["features"]
                importance = importance_data["importance"]
                
                # Top 10 most important features
                top_indices = np.argsort(importance)[-10:][::-1]
                top_features = [(features[i], importance[i]) for i in top_indices]
                
                analysis["feature_importance_analysis"][model_name] = top_features
                
                # Analyze what types of patterns were discovered
                pattern_types = {
                    "fibonacci_like": [],
                    "volume_patterns": [],
                    "momentum_patterns": [],
                    "candlestick_patterns": [],
                    "time_patterns": [],
                    "cycle_patterns": []
                }
                
                for feature, imp in top_features:
                    if any(fib in feature for fib in ["fib", "0.236", "0.382", "0.618"]):
                        pattern_types["fibonacci_like"].append((feature, imp))
                    elif "volume" in feature:
                        pattern_types["volume_patterns"].append((feature, imp))
                    elif any(mom in feature for mom in ["momentum", "acceleration", "return"]):
                        pattern_types["momentum_patterns"].append((feature, imp))
                    elif any(candle in feature for candle in ["body", "shadow", "consecutive"]):
                        pattern_types["candlestick_patterns"].append((feature, imp))
                    elif any(time_f in feature for time_f in ["hour", "day", "month"]):
                        pattern_types["time_patterns"].append((feature, imp))
                    elif "cycle" in feature:
                        pattern_types["cycle_patterns"].append((feature, imp))
                
                analysis["discovered_patterns"][model_name] = pattern_types
        
        # Generate insights
        insights = []
        
        # Check if ML discovered Fibonacci-like patterns
        fib_count = sum(len(patterns["fibonacci_like"]) 
                       for patterns in analysis["discovered_patterns"].values())
        if fib_count > 0:
            insights.append(f"🔢 ML automatically discovered {fib_count} Fibonacci-like patterns as important")
        
        # Check volume pattern discovery
        volume_count = sum(len(patterns["volume_patterns"]) 
                          for patterns in analysis["discovered_patterns"].values())
        if volume_count > 0:
            insights.append(f"📊 ML found {volume_count} volume-based patterns for prediction")
        
        # Check momentum patterns
        momentum_count = sum(len(patterns["momentum_patterns"]) 
                           for patterns in analysis["discovered_patterns"].values())
        if momentum_count > 0:
            insights.append(f"⚡ ML identified {momentum_count} momentum/trend patterns")
        
        analysis["automatic_insights"] = insights
        
        return analysis


# Example usage
if __name__ == "__main__":
    print("🧠 DEEP LEARNING ML - AUTOMATIC PATTERN DISCOVERY")
    print("=" * 70)
    
    # Initialize system
    deep_ml = DeepLearningML()
    
    # Fetch massive historical data
    print("📊 Fetching massive historical data...")
    all_data = deep_ml.fetch_massive_historical_data(["1h", "4h", "1d"])
    
    if all_data:
        # Use 1h data for training (most data points)
        main_data = all_data.get("1h")
        if main_data is not None and len(main_data) > 100:
            print(f"✅ Training with {len(main_data)} data points")
            
            # Train models
            result = deep_ml.train_deep_learning_models(main_data, 10)
            
            if "success" in result:
                print(f"✅ Success: {result['models_trained']} models trained")
                print(f"📊 Features created: {result['total_features_created']}")
                print(f"🎯 Features selected: {result['selected_features']}")
                
                # Analyze discovered patterns
                patterns = deep_ml.analyze_discovered_patterns("deep_learning_10")
                
                print(f"\n🔍 AUTOMATICALLY DISCOVERED PATTERNS:")
                for insight in patterns.get("automatic_insights", []):
                    print(f"   {insight}")
                
            else:
                print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
        else:
            print("❌ Insufficient data for training")
    else:
        print("❌ Could not fetch historical data")
