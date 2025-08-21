#!/usr/bin/env python3
"""
Enhanced ML System for Trend Reversal Detection
Advanced machine learning with model persistence and reversal pattern recognition
"""

import os
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from logger import get_logger
from ml_predictor import MLPredictor


class TrendReversalML:
    """Enhanced ML system specifically for trend reversal detection"""
    
    def __init__(self, model_dir: str = "ml_models"):
        self.logger = get_logger(__name__)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.training_history = []
        
        # Trend reversal specific parameters
        self.reversal_thresholds = {
            "bullish_reversal": 0.05,   # 5%+ upward move after downtrend
            "bearish_reversal": -0.05,  # 5%+ downward move after uptrend
            "trend_lookback": 10,       # Periods to look back for trend
            "prediction_periods": [5, 10, 20]  # Predict reversals N periods ahead
        }
        
        # Feature engineering config
        self.feature_config = {
            "price_features": ["close", "high", "low", "volume"],
            "technical_indicators": ["rsi", "macd", "bb_position", "ema_ratio"],
            "pattern_features": ["doji", "hammer", "engulfing", "morning_star"],
            "trend_features": ["trend_strength", "trend_duration", "volatility"]
        }
        
        self.logger.info(f"TrendReversalML initialized with model directory: {self.model_dir}")
    
    def save_model(self, model_name: str, model_data: Dict) -> bool:
        """Save trained model to disk"""
        try:
            model_path = self.model_dir / f"{model_name}.pkl"
            metadata_path = self.model_dir / f"{model_name}_metadata.json"
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model_data['model'], f)
            
            # Save metadata
            metadata = {
                "created_at": datetime.now().isoformat(),
                "feature_columns": model_data.get('feature_columns', []),
                "training_samples": model_data.get('training_samples', 0),
                "accuracy": model_data.get('accuracy', 0),
                "model_type": model_data.get('model_type', 'unknown'),
                "reversal_threshold": model_data.get('reversal_threshold', 0.05)
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Model {model_name} saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model {model_name}: {e}")
            return False
    
    def load_model(self, model_name: str) -> Optional[Dict]:
        """Load trained model from disk"""
        try:
            model_path = self.model_dir / f"{model_name}.pkl"
            metadata_path = self.model_dir / f"{model_name}_metadata.json"
            
            if not model_path.exists():
                self.logger.warning(f"Model {model_name} not found")
                return None
            
            # Load model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load metadata
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            self.logger.info(f"Model {model_name} loaded successfully")
            return {
                "model": model,
                "metadata": metadata
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            return None
    
    def create_reversal_labels(self, df: pd.DataFrame, periods_ahead: int = 10) -> pd.Series:
        """
        Create labels for trend reversal detection
        
        Labels:
        0 = No reversal
        1 = Bullish reversal (bottom)
        2 = Bearish reversal (top)
        """
        labels = pd.Series(0, index=df.index)
        
        for i in range(len(df) - periods_ahead):
            current_price = df['close'].iloc[i]
            future_prices = df['close'].iloc[i+1:i+periods_ahead+1]
            
            # Calculate trend before current point
            lookback = min(self.reversal_thresholds["trend_lookback"], i)
            if lookback < 3:
                continue
                
            past_prices = df['close'].iloc[i-lookback:i]
            trend_slope = np.polyfit(range(len(past_prices)), past_prices, 1)[0]
            
            # Calculate future performance
            max_future = future_prices.max()
            min_future = future_prices.min()
            
            max_gain = (max_future - current_price) / current_price
            max_loss = (min_future - current_price) / current_price
            
            # Label bullish reversals (after downtrend)
            if (trend_slope < 0 and  # Was in downtrend
                max_gain >= self.reversal_thresholds["bullish_reversal"]):  # Significant upward move
                labels.iloc[i] = 1
            
            # Label bearish reversals (after uptrend)
            elif (trend_slope > 0 and  # Was in uptrend
                  max_loss <= self.reversal_thresholds["bearish_reversal"]):  # Significant downward move
                labels.iloc[i] = 2
        
        return labels
    
    def create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive features for reversal detection"""
        features_df = df.copy()
        
        # Price-based features
        features_df['price_change'] = df['close'].pct_change()
        features_df['price_volatility'] = df['close'].rolling(20).std()
        features_df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Technical indicators (simplified versions)
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features_df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        features_df['macd'] = ema12 - ema26
        features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()
        features_df['macd_histogram'] = features_df['macd'] - features_df['macd_signal']
        
        # Bollinger Bands position
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        features_df['bb_upper'] = sma20 + (2 * std20)
        features_df['bb_lower'] = sma20 - (2 * std20)
        features_df['bb_position'] = (df['close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
        
        # Trend features
        for period in [5, 10, 20]:
            slope = df['close'].rolling(period).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == period else np.nan)
            features_df[f'trend_{period}'] = slope
        
        # Pattern recognition (simplified)
        # Doji pattern
        body_size = abs(df['close'] - df['open'])
        wick_size = df['high'] - df['low']
        features_df['doji'] = (body_size / wick_size < 0.1).astype(int)
        
        # Hammer pattern
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        features_df['hammer'] = ((lower_wick > 2 * body_size) & (upper_wick < body_size)).astype(int)
        
        # Volume patterns
        features_df['volume_spike'] = (df['volume'] > df['volume'].rolling(10).mean() * 2).astype(int)
        
        return features_df
    
    def prepare_training_data(self, df: pd.DataFrame, periods_ahead: int = 10) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data for reversal detection"""
        
        # Create features
        features_df = self.create_enhanced_features(df)
        
        # Create labels
        labels = self.create_reversal_labels(df, periods_ahead)
        
        # Select feature columns (exclude NaN-prone columns)
        feature_columns = [
            'rsi', 'macd', 'macd_histogram', 'bb_position',
            'trend_5', 'trend_10', 'trend_20',
            'price_volatility', 'volume_ratio', 'volume_spike',
            'doji', 'hammer'
        ]
        
        # Clean data
        features_df = features_df[feature_columns].dropna()
        labels = labels.loc[features_df.index]
        
        X = features_df.values
        y = labels.values
        
        self.logger.info(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        self.logger.info(f"Label distribution: {np.bincount(y)}")
        
        return X, y, feature_columns
    
    def train_reversal_models(self, df: pd.DataFrame, periods_ahead: int = 10) -> Dict:
        """Train multiple models for trend reversal detection"""
        
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not available"}
        
        self.logger.info(f"Training reversal models for {periods_ahead} periods ahead")
        
        # Prepare data
        X, y, feature_columns = self.prepare_training_data(df, periods_ahead)
        self.feature_columns = feature_columns
        
        if len(X) < 100:
            return {"error": "Insufficient training data"}
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        models = {}
        results = {}
        
        # Model configurations
        model_configs = {
            "random_forest": RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            ),
            "gradient_boost": GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
            ),
            "logistic": LogisticRegression(
                random_state=42, max_iter=1000
            )
        }
        
        # Train each model
        for model_name, model in model_configs.items():
            try:
                self.logger.info(f"Training {model_name} model...")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
                
                # Store results
                models[model_name] = model
                self.scalers[f"{model_name}_{periods_ahead}"] = scaler
                
                results[model_name] = {
                    "train_accuracy": train_score,
                    "test_accuracy": test_score,
                    "predictions": y_pred.tolist(),
                    "feature_importance": (
                        model.feature_importances_.tolist() 
                        if hasattr(model, 'feature_importances_') else None
                    )
                }
                
                # Save model
                model_data = {
                    "model": model,
                    "scaler": scaler,
                    "feature_columns": feature_columns,
                    "training_samples": len(X_train),
                    "accuracy": test_score,
                    "model_type": model_name,
                    "reversal_threshold": self.reversal_thresholds["bullish_reversal"],
                    "periods_ahead": periods_ahead
                }
                
                self.save_model(f"{model_name}_reversal_{periods_ahead}d", model_data)
                
                self.logger.info(f"{model_name}: Train={train_score:.3f}, Test={test_score:.3f}")
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        self.models[f"reversal_{periods_ahead}"] = models
        
        # Save training history
        training_record = {
            "timestamp": datetime.now().isoformat(),
            "periods_ahead": periods_ahead,
            "training_samples": len(X),
            "features": len(feature_columns),
            "results": results
        }
        self.training_history.append(training_record)
        
        return {
            "success": True,
            "models_trained": len(models),
            "training_samples": len(X),
            "feature_count": len(feature_columns),
            "results": results
        }
    
    def predict_reversal(self, current_data: pd.DataFrame, periods_ahead: int = 10) -> Dict:
        """Predict trend reversal probability"""
        
        model_key = f"reversal_{periods_ahead}"
        if model_key not in self.models:
            return {"error": f"No trained models for {periods_ahead} periods ahead"}
        
        try:
            # Create features for current data
            features_df = self.create_enhanced_features(current_data)
            latest_features = features_df[self.feature_columns].iloc[-1:].values
            
            predictions = {}
            confidence_scores = []
            
            # Get predictions from all models
            for model_name, model in self.models[model_key].items():
                scaler_key = f"{model_name}_{periods_ahead}"
                if scaler_key in self.scalers:
                    scaled_features = self.scalers[scaler_key].transform(latest_features)
                    
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
            
            # Average confidence
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            
            # Interpretation
            reversal_types = {0: "No Reversal", 1: "Bullish Reversal", 2: "Bearish Reversal"}
            
            return {
                "ensemble_prediction": ensemble_prediction,
                "prediction_label": reversal_types[ensemble_prediction],
                "confidence": avg_confidence,
                "individual_predictions": predictions,
                "periods_ahead": periods_ahead,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return {"error": str(e)}
    
    def get_model_status(self) -> Dict:
        """Get status of all trained models"""
        
        status = {
            "models_trained": len(self.models),
            "models_available": [],
            "training_history": len(self.training_history),
            "model_directory": str(self.model_dir)
        }
        
        # List available model files
        model_files = list(self.model_dir.glob("*.pkl"))
        status["saved_models"] = [f.stem for f in model_files]
        
        # Current loaded models
        for model_key, models in self.models.items():
            status["models_available"].append({
                "key": model_key,
                "algorithms": list(models.keys())
            })
        
        return status


# Example usage and training
if __name__ == "__main__":
    print("🤖 ENHANCED ML TREND REVERSAL SYSTEM")
    print("=" * 50)
    
    # Initialize system
    ml_system = TrendReversalML()
    
    # Get training data
    base_predictor = MLPredictor()
    training_data = base_predictor.fetch_training_data("1h", 2000)
    
    if training_data is not None:
        print(f"📊 Training data: {len(training_data)} samples")
        
        # Train models for different horizons
        for periods in [5, 10, 20]:
            print(f"\n🧠 Training models for {periods} periods ahead...")
            result = ml_system.train_reversal_models(training_data, periods)
            
            if "success" in result:
                print(f"✅ Success: {result['models_trained']} models trained")
                print(f"📊 Training samples: {result['training_samples']}")
                print(f"🎯 Features: {result['feature_count']}")
            else:
                print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
        
        # Show model status
        print("\n📋 MODEL STATUS:")
        status = ml_system.get_model_status()
        print(f"Models trained: {status['models_trained']}")
        print(f"Saved models: {len(status['saved_models'])}")
        
    else:
        print("❌ Could not fetch training data")
