#!/usr/bin/env python3
"""
Machine Learning Predictor Module
Advanced price prediction using multiple ML models and ensemble methods
"""

import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import warnings
# Suppress only specific ML-related warnings to keep important diagnostics visible
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn') 
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')

# ML imports with fallbacks
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class MLPredictor:
    """Machine Learning predictor for BNB price forecasting"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.prediction_horizons = {
            # Primary strategic horizons (Daily+ focus)
            "1d": 24,     # 1 day ahead
            "1w": 168,    # 1 week ahead
            "1m": 720,    # 1 month ahead (30 days)
            "3m": 2160,   # 3 months ahead
            "6m": 4320,   # 6 months ahead  
            "1y": 8760,   # 1 year ahead
            
            # Legacy short-term (for compatibility)
            "1h": 1,      # 1 hour ahead
            "4h": 4       # 4 hours ahead
        }
        
        # Model configurations
        self.model_configs = {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "gradient_boost": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "random_state": 42
            },
            "linear": {
                "alpha": 1.0
            }
        }
        
        # Alert thresholds for ML predictions
        self.alert_thresholds = {
            "significant_move": 0.05,      # 5%+ predicted move
            "model_agreement": 0.8,        # 80%+ models agree on direction
            "high_confidence": 0.85,       # 85%+ prediction confidence
            "extreme_prediction": 0.10,    # 10%+ predicted move
            "trend_reversal": True          # Prediction opposite to current trend
        }
    
    def fetch_training_data(self, interval: str = "1d", limit: int = 365) -> Optional[pd.DataFrame]:
        """Fetch historical data for training ML models"""
        
        try:
            params = {
                "symbol": "BNBUSDT",
                "interval": interval,
                "limit": min(limit, 1000)  # Binance API limit
            }
            
            response = requests.get(f"{self.base_url}/klines", params=params)
            if response.status_code != 200:
                return None
            
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
            
            return df
            
        except Exception as e:
            print(f"Error fetching training data: {e}")
            return None
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators and features for ML models"""
        
        if df is None or len(df) < 50:
            return None
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['open_close_ratio'] = df['open'] / df['close']
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ma_ratio_{period}'] = df['close'] / df[f'ma_{period}']
        
        # Technical indicators
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        rolling_mean = df['close'].rolling(window=20).mean()
        rolling_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = rolling_mean + (rolling_std * 2)
        df['bb_lower'] = rolling_mean - (rolling_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['price_volume'] = df['close'] * df['volume']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # Long-term trend features (for strategic analysis)
        df['trend_strength'] = df['close'].rolling(window=50).apply(
            lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) > 1 else 0
        )
        
        # Cycle analysis features
        df['monthly_return'] = df['close'].pct_change(30)  # 30-day return
        df['quarterly_return'] = df['close'].pct_change(90)  # 90-day return
        
        # Support/resistance levels
        df['distance_to_ath'] = (df['close'].expanding().max() - df['close']) / df['close']
        df['distance_to_atl'] = (df['close'] - df['close'].expanding().min()) / df['close']
        
        # Market structure
        df['higher_highs'] = (df['high'] > df['high'].shift(20)).rolling(5).sum()
        df['higher_lows'] = (df['low'] > df['low'].shift(20)).rolling(5).sum()
        
        # Volume profile
        df['volume_trend'] = df['volume'].rolling(window=30).apply(
            lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) > 1 else 0
        )
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        return df
    
    def prepare_ml_data(self, df: pd.DataFrame, target_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for ML training"""
        
        if df is None or len(df) < 100:
            return None, None, []
        
        # Define feature columns (exclude target and non-feature columns)
        exclude_columns = ['open', 'high', 'low', 'close', 'volume', 'close_time', 
                          'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Create target variable (future price change)
        df['target'] = df['close'].shift(-target_horizon) / df['close'] - 1
        
        # Remove rows where target is NaN
        df_clean = df[:-target_horizon].copy()
        
        X = df_clean[feature_columns].values
        y = df_clean['target'].values
        
        # Remove any remaining NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        return X, y, feature_columns
    
    def train_sklearn_models(self, X: np.ndarray, y: np.ndarray, horizon: str) -> Dict:
        """Train scikit-learn models"""
        
        if not SKLEARN_AVAILABLE:
            return {"error": "Scikit-learn not available"}
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers[horizon] = scaler
            
            models = {}
            results = {}
            
            # Random Forest
            rf = RandomForestRegressor(**self.model_configs["random_forest"])
            rf.fit(X_train_scaled, y_train)
            rf_pred = rf.predict(X_test_scaled)
            models['random_forest'] = rf
            results['random_forest'] = {
                'mae': mean_absolute_error(y_test, rf_pred),
                'mse': mean_squared_error(y_test, rf_pred),
                'feature_importance': rf.feature_importances_
            }
            
            # Gradient Boosting
            gb = GradientBoostingRegressor(**self.model_configs["gradient_boost"])
            gb.fit(X_train_scaled, y_train)
            gb_pred = gb.predict(X_test_scaled)
            models['gradient_boost'] = gb
            results['gradient_boost'] = {
                'mae': mean_absolute_error(y_test, gb_pred),
                'mse': mean_squared_error(y_test, gb_pred),
                'feature_importance': gb.feature_importances_
            }
            
            # Linear Regression
            lr = Ridge(**self.model_configs["linear"])
            lr.fit(X_train_scaled, y_train)
            lr_pred = lr.predict(X_test_scaled)
            models['linear'] = lr
            results['linear'] = {
                'mae': mean_absolute_error(y_test, lr_pred),
                'mse': mean_squared_error(y_test, lr_pred)
            }
            
            self.models[horizon] = models
            
            return {
                'models': models,
                'results': results,
                'scaler': scaler,
                'test_data': (X_test_scaled, y_test)
            }
            
        except Exception as e:
            return {"error": f"Error training sklearn models: {e}"}
    
    def create_lstm_model(self, sequence_length: int = 60, features: int = 10) -> Optional[object]:
        """Create LSTM model for time series prediction"""
        
        if not TENSORFLOW_AVAILABLE:
            return None
        
        try:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, features)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            return model
            
        except Exception as e:
            print(f"Error creating LSTM model: {e}")
            return None
    
    def make_predictions(self, horizon: str = "1d") -> Dict:
        """Make predictions using trained models"""
        
        try:
            # Get recent data
            df = self.fetch_training_data("1h", 200)
            if df is None:
                return {"error": "Unable to fetch recent data"}
            
            # Create features
            df_features = self.create_features(df)
            if df_features is None:
                return {"error": "Unable to create features"}
            
            current_price = df_features['close'].iloc[-1]
            
            # Get the latest feature vector
            feature_columns = [col for col in df_features.columns 
                             if col not in ['open', 'high', 'low', 'close', 'volume', 'close_time', 
                                          'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']]
            
            latest_features = df_features[feature_columns].iloc[-1:].values
            
            predictions = {}
            model_agreement = 0
            
            # Make predictions if models are trained
            if horizon in self.models and horizon in self.scalers:
                scaled_features = self.scalers[horizon].transform(latest_features)
                
                for model_name, model in self.models[horizon].items():
                    try:
                        pred_change = model.predict(scaled_features)[0]
                        pred_price = current_price * (1 + pred_change)
                        
                        predictions[model_name] = {
                            'predicted_change': pred_change,
                            'predicted_price': pred_price,
                            'direction': 'bullish' if pred_change > 0 else 'bearish'
                        }
                    except Exception as e:
                        predictions[model_name] = {"error": f"Prediction error: {e}"}
                
                # Calculate model agreement
                directions = [p.get('direction') for p in predictions.values() if 'direction' in p]
                if directions:
                    bullish_count = directions.count('bullish')
                    model_agreement = max(bullish_count, len(directions) - bullish_count) / len(directions)
            
            # Ensemble prediction (average of all valid predictions)
            valid_predictions = [p['predicted_change'] for p in predictions.values() 
                               if 'predicted_change' in p]
            
            if valid_predictions:
                ensemble_change = np.mean(valid_predictions)
                ensemble_price = current_price * (1 + ensemble_change)
                ensemble_std = np.std(valid_predictions)
                
                predictions['ensemble'] = {
                    'predicted_change': ensemble_change,
                    'predicted_price': ensemble_price,
                    'confidence': max(0, 1 - ensemble_std * 10),  # Simple confidence metric
                    'direction': 'bullish' if ensemble_change > 0 else 'bearish',
                    'model_agreement': model_agreement
                }
            
            return {
                'current_price': current_price,
                'horizon': horizon,
                'predictions': predictions,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'models_used': list(predictions.keys())
            }
            
        except Exception as e:
            return {"error": f"Error making predictions: {e}"}
    
    def check_critical_ml_alerts(self, horizon: str = "1d") -> Dict:
        """Check for critical ML prediction alerts"""
        
        try:
            predictions = self.make_predictions(horizon)
            
            if "error" in predictions:
                return {"show_alert": False, "reason": predictions["error"]}
            
            critical_signals = []
            alert_score = 0
            
            ensemble = predictions.get("predictions", {}).get("ensemble", {})
            if not ensemble:
                return {"show_alert": False, "reason": "No ensemble prediction available"}
            
            predicted_change = ensemble.get("predicted_change", 0)
            confidence = ensemble.get("confidence", 0)
            model_agreement = ensemble.get("model_agreement", 0)
            direction = ensemble.get("direction", "neutral")
            
            # Significant move prediction
            if abs(predicted_change) >= self.alert_thresholds["significant_move"]:
                direction_emoji = "🚀" if predicted_change > 0 else "💥"
                critical_signals.append(f"{direction_emoji} Significant {direction} move predicted: {predicted_change*100:.1f}%")
                alert_score += 6
            
            # High model agreement
            if model_agreement >= self.alert_thresholds["model_agreement"]:
                critical_signals.append(f"🤝 High model agreement ({model_agreement*100:.0f}%) on {direction} direction")
                alert_score += 5
            
            # High confidence prediction
            if confidence >= self.alert_thresholds["high_confidence"]:
                critical_signals.append(f"🎯 High confidence prediction ({confidence*100:.0f}%)")
                alert_score += 4
            
            # Extreme prediction
            if abs(predicted_change) >= self.alert_thresholds["extreme_prediction"]:
                critical_signals.append(f"⚡ EXTREME price movement predicted: {predicted_change*100:.1f}%")
                alert_score += 8
            
            # Multiple models ensemble strength
            model_count = len([p for p in predictions["predictions"].values() if "predicted_change" in p])
            if model_count >= 3 and model_agreement >= 0.7:
                critical_signals.append(f"🔥 {model_count} models agree on {direction} direction")
                alert_score += 3
            
            # Determine if alert should be shown
            show_alert = alert_score >= 7  # Threshold for ML alerts
            
            return {
                "show_alert": show_alert,
                "alert_score": alert_score,
                "critical_signals": critical_signals,
                "ml_data": {
                    "horizon": horizon,
                    "predicted_change": predicted_change,
                    "predicted_price": ensemble.get("predicted_price", 0),
                    "confidence": confidence,
                    "model_agreement": model_agreement,
                    "direction": direction,
                    "models_used": predictions.get("models_used", [])
                }
            }
            
        except Exception as e:
            return {"show_alert": False, "reason": f"Error checking ML alerts: {e}"}
    
    def train_models_for_horizon(self, horizon: str = "1d") -> Dict:
        """Train all available models for a specific prediction horizon"""
        
        print(f"\n🤖 Training ML models for {horizon} predictions...")
        
        # Get training data
        df = self.fetch_training_data("1h", 1000)
        if df is None:
            return {"error": "Unable to fetch training data"}
        
        # Create features
        df_features = self.create_features(df)
        if df_features is None:
            return {"error": "Unable to create features"}
        
        # Prepare ML data
        target_horizon = self.prediction_horizons.get(horizon, 24)
        X, y, feature_columns = self.prepare_ml_data(df_features, target_horizon)
        
        if X is None:
            return {"error": "Unable to prepare ML data"}
        
        self.feature_columns = feature_columns
        
        results = {}
        
        # Train sklearn models
        if SKLEARN_AVAILABLE:
            sklearn_results = self.train_sklearn_models(X, y, horizon)
            results['sklearn'] = sklearn_results
        else:
            results['sklearn'] = {"error": "Scikit-learn not available"}
        
        # TODO: Add LSTM training when TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            results['lstm'] = {"status": "LSTM training not implemented yet"}
        else:
            results['lstm'] = {"error": "TensorFlow not available"}
        
        print(f"✅ Model training completed for {horizon}")
        return results
    
    def display_ml_analysis(self, horizon: str = "1d"):
        """Display comprehensive ML analysis"""
        
        print(f"\n🤖 MACHINE LEARNING PRICE PREDICTION")
        print("=" * 60)
        
        # Make predictions
        predictions = self.make_predictions(horizon)
        
        if "error" in predictions:
            print(f"❌ Error: {predictions['error']}")
            return
        
        current_price = predictions["current_price"]
        pred_data = predictions["predictions"]
        
        print(f"📊 CURRENT STATUS")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Prediction Horizon: {horizon}")
        print(f"Timestamp: {predictions['timestamp']}")
        
        if "ensemble" in pred_data:
            ensemble = pred_data["ensemble"]
            pred_price = ensemble["predicted_price"]
            pred_change = ensemble["predicted_change"]
            confidence = ensemble["confidence"]
            agreement = ensemble["model_agreement"]
            
            print(f"\n🎯 ENSEMBLE PREDICTION")
            print(f"Predicted Price: ${pred_price:.2f}")
            print(f"Predicted Change: {pred_change*100:+.2f}%")
            print(f"Direction: {ensemble['direction'].upper()}")
            print(f"Confidence: {confidence*100:.1f}%")
            print(f"Model Agreement: {agreement*100:.1f}%")
        
        print(f"\n🔮 INDIVIDUAL MODEL PREDICTIONS")
        print("-" * 40)
        
        for model_name, pred in pred_data.items():
            if model_name == "ensemble":
                continue
            
            if "error" in pred:
                print(f"{model_name}: ❌ {pred['error']}")
            else:
                change = pred["predicted_change"]
                price = pred["predicted_price"]
                direction = pred["direction"]
                emoji = "🟢" if direction == "bullish" else "🔴"
                print(f"{model_name}: {emoji} ${price:.2f} ({change*100:+.1f}%)")
        
        # Check for alerts
        alert_data = self.check_critical_ml_alerts(horizon)
        if alert_data.get("show_alert"):
            print(f"\n🚨 ML PREDICTION ALERTS")
            print("-" * 30)
            for signal in alert_data.get("critical_signals", []):
                print(f"   {signal}")
            print(f"   Alert Score: {alert_data.get('alert_score', 0)}/20")
        
        print("\n" + "=" * 60)
    
    def analyze_long_term_trends(self) -> Dict:
        """Analyze long-term strategic trends for 1m, 6m, 1y horizons"""
        
        print(f"\n📊 LONG-TERM STRATEGIC ANALYSIS")
        print("=" * 50)
        
        try:
            # Get longer historical data
            df = self.fetch_training_data("1d", 500)  # Get more data for long-term
            if df is None:
                return {"error": "Unable to fetch long-term data"}
            
            df_features = self.create_features(df)
            if df_features is None:
                return {"error": "Unable to create long-term features"}
            
            current_price = df_features['close'].iloc[-1]
            
            # Strategic analysis
            strategic_analysis = {
                "current_price": current_price,
                "analysis_date": datetime.now().strftime('%Y-%m-%d'),
                "strategic_zones": {},
                "trend_analysis": {},
                "cycle_position": {},
                "long_term_targets": {}
            }
            
            # Trend strength analysis
            trend_strength = df_features['trend_strength'].iloc[-1]
            monthly_return = df_features['monthly_return'].iloc[-1] * 100
            quarterly_return = df_features['quarterly_return'].iloc[-1] * 100
            
            strategic_analysis["trend_analysis"] = {
                "trend_strength": trend_strength,
                "monthly_performance": monthly_return,
                "quarterly_performance": quarterly_return,
                "trend_direction": "BULLISH" if trend_strength > 0.3 else "BEARISH" if trend_strength < -0.3 else "SIDEWAYS"
            }
            
            # Calculate strategic price levels
            ath = df_features['close'].max()
            atl = df_features['close'].min()
            current_range = ath - atl
            
            # Long-term target zones
            strategic_analysis["strategic_zones"] = {
                "all_time_high": ath,
                "all_time_low": atl,
                "current_position": (current_price - atl) / (ath - atl) * 100,  # % of range
                "resistance_zone": ath * 0.95,  # 5% below ATH
                "support_zone": atl * 1.05,     # 5% above ATL
                "golden_zone": atl + (current_range * 0.618)  # 61.8% retracement
            }
            
            # Cycle position analysis
            distance_to_ath = (ath - current_price) / current_price * 100
            distance_to_atl = (current_price - atl) / current_price * 100
            
            if distance_to_ath < 20:
                cycle_phase = "LATE_BULL_MARKET"
                risk_level = "HIGH"
            elif distance_to_atl < 50:
                cycle_phase = "EARLY_BULL_MARKET" 
                risk_level = "LOW"
            else:
                cycle_phase = "MID_CYCLE"
                risk_level = "MEDIUM"
            
            strategic_analysis["cycle_position"] = {
                "phase": cycle_phase,
                "risk_level": risk_level,
                "distance_to_ath_pct": distance_to_ath,
                "distance_to_atl_pct": distance_to_atl
            }
            
            # Long-term price targets based on historical patterns
            if trend_strength > 0.2:  # Bullish trend
                target_1m = current_price * (1 + 0.15)  # 15% monthly target
                target_6m = current_price * (1 + 0.50)  # 50% 6-month target  
                target_1y = current_price * (1 + 1.20)  # 120% yearly target
            elif trend_strength < -0.2:  # Bearish trend
                target_1m = current_price * (1 - 0.10)  # -10% monthly target
                target_6m = current_price * (1 - 0.30)  # -30% 6-month target
                target_1y = current_price * (1 - 0.40)  # -40% yearly target  
            else:  # Sideways
                target_1m = current_price * (1 + 0.05)  # 5% monthly target
                target_6m = current_price * (1 + 0.20)  # 20% 6-month target
                target_1y = current_price * (1 + 0.40)  # 40% yearly target
            
            strategic_analysis["long_term_targets"] = {
                "1_month": {"price": target_1m, "change_pct": (target_1m/current_price - 1)*100},
                "6_months": {"price": target_6m, "change_pct": (target_6m/current_price - 1)*100},  
                "1_year": {"price": target_1y, "change_pct": (target_1y/current_price - 1)*100}
            }
            
            return strategic_analysis
            
        except Exception as e:
            return {"error": f"Error in long-term analysis: {e}"}
    
    def display_strategic_analysis(self):
        """Display long-term strategic analysis"""
        
        analysis = self.analyze_long_term_trends()
        
        if "error" in analysis:
            print(f"❌ Error: {analysis['error']}")
            return
        
        print(f"📊 STRATEGIC INVESTMENT ANALYSIS")
        print("=" * 50)
        print(f"Current Price: ${analysis['current_price']:.2f}")
        print(f"Analysis Date: {analysis['analysis_date']}")
        
        # Trend Analysis
        trend = analysis["trend_analysis"]
        print(f"\n📈 TREND ANALYSIS")
        print("-" * 30)
        print(f"Trend Direction: {trend['trend_direction']}")
        print(f"Trend Strength: {trend['trend_strength']:.3f}")
        print(f"Monthly Performance: {trend['monthly_performance']:+.1f}%")
        print(f"Quarterly Performance: {trend['quarterly_performance']:+.1f}%")
        
        # Cycle Position
        cycle = analysis["cycle_position"] 
        print(f"\n🌊 MARKET CYCLE POSITION")
        print("-" * 30)
        print(f"Cycle Phase: {cycle['phase']}")
        print(f"Risk Level: {cycle['risk_level']}")
        print(f"Distance to ATH: {cycle['distance_to_ath_pct']:.1f}%")
        print(f"Distance from ATL: {cycle['distance_to_atl_pct']:.1f}%")
        
        # Strategic Zones
        zones = analysis["strategic_zones"]
        print(f"\n🎯 STRATEGIC PRICE ZONES")
        print("-" * 30)
        print(f"All-Time High: ${zones['all_time_high']:.2f}")
        print(f"Golden Zone (61.8%): ${zones['golden_zone']:.2f}")
        print(f"Current Position: {zones['current_position']:.1f}% of range")
        print(f"Support Zone: ${zones['support_zone']:.2f}")
        print(f"All-Time Low: ${zones['all_time_low']:.2f}")
        
        # Long-term Targets
        targets = analysis["long_term_targets"]
        print(f"\n🎯 LONG-TERM PRICE TARGETS")
        print("-" * 30)
        
        for period, target in targets.items():
            price = target["price"]
            change = target["change_pct"]
            emoji = "🚀" if change > 0 else "💥"
            print(f"{period.replace('_', ' ').title()}: {emoji} ${price:.2f} ({change:+.1f}%)")
        
        # Investment Recommendation
        print(f"\n💡 STRATEGIC RECOMMENDATION")
        print("-" * 30)
        
        risk_level = cycle['risk_level']
        phase = cycle['phase']
        
        if risk_level == "LOW" and phase == "EARLY_BULL_MARKET":
            print("🟢 ACCUMULATE - Early bull market phase")
            print("💰 Suggested allocation: 70-80% of crypto portfolio")
            print("⏱️ Time horizon: 6-12 months")
        elif risk_level == "MEDIUM":
            print("🟡 HOLD/DCA - Mid-cycle consolidation")  
            print("💰 Suggested allocation: 40-60% of crypto portfolio")
            print("⏱️ Time horizon: 3-6 months")
        else:
            print("🔴 REDUCE RISK - Late cycle or high risk")
            print("💰 Suggested allocation: 10-30% of crypto portfolio") 
            print("⏱️ Time horizon: 1-3 months")
        
        print("\n" + "=" * 50)
    
    def run_full_ml_analysis(self):
        """Run complete ML analysis with training and predictions"""
        
        print("\n🤖 COMPREHENSIVE ML ANALYSIS")
        print("=" * 60)
        
        # Check dependencies
        missing_deps = []
        if not SKLEARN_AVAILABLE:
            missing_deps.append("scikit-learn")
        if not TENSORFLOW_AVAILABLE:
            missing_deps.append("tensorflow")
        
        if missing_deps:
            print(f"⚠️ Missing dependencies: {', '.join(missing_deps)}")
            print("📥 To install: pip install scikit-learn tensorflow")
            print("📊 Running with limited functionality...\n")
        
        # Train models for different horizons
        for horizon in ["1h", "1d"]:
            print(f"\n📈 ANALYSIS FOR {horizon.upper()} PREDICTIONS")
            print("-" * 40)
            
            # Train models if not already trained
            if horizon not in self.models:
                training_result = self.train_models_for_horizon(horizon)
                if "error" in training_result:
                    print(f"❌ Training failed: {training_result['error']}")
                    continue
            
            # Display predictions
            self.display_ml_analysis(horizon)
            
            print()  # Add spacing


# Example usage
if __name__ == "__main__":
    predictor = MLPredictor()
    
    print("🤖 ML PREDICTOR - INTERACTIVE MODE")
    print("=" * 50)
    
    try:
        # Run full analysis
        predictor.run_full_ml_analysis()
        
    except KeyboardInterrupt:
        print("\n\n👋 ML Analysis stopped by user.")
