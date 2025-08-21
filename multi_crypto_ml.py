#!/usr/bin/env python3
"""
Multi-Cryptocurrency ML System
Advanced machine learning with top 10 cryptocurrencies for enhanced pattern recognition
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
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from logger import get_logger
from ml_enhanced import TrendReversalML

class MultiCryptoML:
    """Enhanced ML system for multi-cryptocurrency analysis"""
    
    def __init__(self, model_dir: str = "ml_models_multi"):
        self.logger = get_logger(__name__)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Top 10 cryptocurrencies by market cap (as of 2024)
        self.crypto_symbols = {
            "BTCUSDT": {"name": "Bitcoin", "weight": 0.40},
            "ETHUSDT": {"name": "Ethereum", "weight": 0.20}, 
            "BNBUSDT": {"name": "BNB", "weight": 0.08},
            "XRPUSDT": {"name": "XRP", "weight": 0.05},
            "SOLUSDT": {"name": "Solana", "weight": 0.05},
            "ADAUSDT": {"name": "Cardano", "weight": 0.04},
            "AVAXUSDT": {"name": "Avalanche", "weight": 0.03},
            "DOTUSDT": {"name": "Polkadot", "weight": 0.03},
            "LINKUSDT": {"name": "Chainlink", "weight": 0.03},
            "MATICUSDT": {"name": "Polygon", "weight": 0.03}
        }
        
        self.base_url = "https://api.binance.com/api/v3"
        self.models = {}
        self.scalers = {}
        self.market_data = {}
        
        # Multi-crypto specific parameters
        self.correlation_window = 20
        self.dominance_window = 50
        self.sector_rotation_threshold = 0.15  # 15% divergence
        
        self.logger.info(f"MultiCryptoML initialized for {len(self.crypto_symbols)} cryptocurrencies")
    
    def fetch_multi_crypto_data(self, interval: str = "1h", limit: int = 1000) -> Dict[str, pd.DataFrame]:
        """Fetch data for all cryptocurrencies"""
        
        self.logger.info(f"Fetching data for {len(self.crypto_symbols)} cryptocurrencies...")
        multi_data = {}
        
        for symbol, info in self.crypto_symbols.items():
            try:
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "limit": min(limit, 1000)
                }
                
                self.logger.debug(f"Fetching {symbol} data...")
                response = requests.get(f"{self.base_url}/klines", params=params, timeout=10)
                
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
                    
                    # Add symbol info
                    df['symbol'] = symbol
                    df['crypto_name'] = info['name']
                    df['market_weight'] = info['weight']
                    
                    multi_data[symbol] = df
                    self.logger.debug(f"✅ {symbol}: {len(df)} candles")
                    
                else:
                    self.logger.error(f"❌ Failed to fetch {symbol}: {response.status_code}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error fetching {symbol}: {e}")
        
        self.market_data = multi_data
        self.logger.info(f"✅ Fetched data for {len(multi_data)} cryptocurrencies")
        return multi_data
    
    def create_cross_asset_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create advanced cross-asset features for ML"""
        
        if not data:
            return None
        
        self.logger.info("Creating cross-asset features...")
        
        # Align all data to common timestamps
        common_timestamps = None
        for symbol, df in data.items():
            if common_timestamps is None:
                common_timestamps = df.index
            else:
                common_timestamps = common_timestamps.intersection(df.index)
        
        if len(common_timestamps) < 100:
            self.logger.error("Insufficient common data points")
            return None
        
        # Create unified DataFrame
        combined_features = pd.DataFrame(index=common_timestamps)
        
        # 1. INDIVIDUAL ASSET FEATURES
        for symbol, df in data.items():
            df_aligned = df.loc[common_timestamps]
            prefix = symbol.replace('USDT', '')
            
            # Basic price features
            combined_features[f'{prefix}_close'] = df_aligned['close']
            combined_features[f'{prefix}_volume'] = df_aligned['volume']
            combined_features[f'{prefix}_returns'] = df_aligned['close'].pct_change()
            
            # Technical indicators
            combined_features[f'{prefix}_rsi'] = self._calculate_rsi(df_aligned['close'])
            combined_features[f'{prefix}_volatility'] = df_aligned['close'].rolling(20).std()
        
        # 2. CROSS-ASSET CORRELATION FEATURES
        price_columns = [col for col in combined_features.columns if col.endswith('_close')]
        correlation_matrix = combined_features[price_columns].rolling(window=self.correlation_window).corr()
        
        # BTC correlations (market leadership)
        if 'BTC_close' in price_columns:
            for col in price_columns:
                if col != 'BTC_close':
                    asset_name = col.replace('_close', '')
                    combined_features[f'{asset_name}_btc_corr'] = (
                        combined_features[f'{asset_name}_close'].rolling(self.correlation_window)
                        .corr(combined_features['BTC_close'])
                    )
        
        # 3. MARKET DOMINANCE FEATURES
        if 'BTC_close' in combined_features.columns:
            total_market_cap = 0
            for symbol, info in self.crypto_symbols.items():
                symbol_prefix = symbol.replace('USDT', '')
                if f'{symbol_prefix}_close' in combined_features.columns:
                    # Simplified market cap (price * weight as proxy)
                    combined_features[f'{symbol_prefix}_mcap_proxy'] = (
                        combined_features[f'{symbol_prefix}_close'] * info['weight']
                    )
                    total_market_cap += combined_features[f'{symbol_prefix}_mcap_proxy']
            
            # BTC dominance
            if total_market_cap is not None:
                combined_features['btc_dominance'] = (
                    combined_features['BTC_mcap_proxy'] / total_market_cap
                )
                combined_features['btc_dominance_change'] = combined_features['btc_dominance'].pct_change()
        
        # 4. RELATIVE STRENGTH FEATURES
        for symbol in self.crypto_symbols.keys():
            prefix = symbol.replace('USDT', '')
            if f'{prefix}_close' in combined_features.columns and 'BTC_close' in combined_features.columns:
                # Relative strength vs BTC
                combined_features[f'{prefix}_vs_btc'] = (
                    combined_features[f'{prefix}_close'] / combined_features['BTC_close']
                )
                combined_features[f'{prefix}_vs_btc_change'] = (
                    combined_features[f'{prefix}_vs_btc'].pct_change()
                )
        
        # 5. VOLUME FLOW FEATURES
        volume_columns = [col for col in combined_features.columns if col.endswith('_volume')]
        total_volume = combined_features[volume_columns].sum(axis=1)
        
        for col in volume_columns:
            asset_name = col.replace('_volume', '')
            combined_features[f'{asset_name}_volume_share'] = (
                combined_features[col] / total_volume
            )
        
        # 6. DIVERGENCE FEATURES
        # Detect when assets move differently from the market average
        return_columns = [col for col in combined_features.columns if col.endswith('_returns')]
        if len(return_columns) > 1:
            market_avg_return = combined_features[return_columns].mean(axis=1)
            
            for col in return_columns:
                asset_name = col.replace('_returns', '')
                combined_features[f'{asset_name}_divergence'] = (
                    combined_features[col] - market_avg_return
                )
        
        # 7. SECTOR ROTATION SIGNALS
        # Large cap vs small cap performance
        large_cap_assets = ['BTC', 'ETH', 'BNB']  # Top 3
        small_cap_assets = ['ADA', 'AVAX', 'DOT', 'LINK', 'MATIC']  # Lower market cap
        
        large_cap_returns = []
        small_cap_returns = []
        
        for asset in large_cap_assets:
            if f'{asset}_returns' in combined_features.columns:
                large_cap_returns.append(combined_features[f'{asset}_returns'])
        
        for asset in small_cap_assets:
            if f'{asset}_returns' in combined_features.columns:
                small_cap_returns.append(combined_features[f'{asset}_returns'])
        
        if large_cap_returns and small_cap_returns:
            combined_features['large_cap_avg'] = pd.concat(large_cap_returns, axis=1).mean(axis=1)
            combined_features['small_cap_avg'] = pd.concat(small_cap_returns, axis=1).mean(axis=1)
            combined_features['sector_rotation'] = (
                combined_features['small_cap_avg'] - combined_features['large_cap_avg']
            )
        
        # Clean data
        combined_features.dropna(inplace=True)
        
        self.logger.info(f"✅ Created {len(combined_features.columns)} cross-asset features")
        self.logger.info(f"📊 Data points: {len(combined_features)}")
        
        return combined_features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI for a price series"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def create_multi_asset_labels(self, features_df: pd.DataFrame, target_asset: str = "BNB", 
                                periods_ahead: int = 10) -> pd.Series:
        """
        Create labels for multi-asset trend reversal detection
        Enhanced with cross-asset confirmation
        """
        
        target_col = f'{target_asset}_close'
        if target_col not in features_df.columns:
            self.logger.error(f"Target asset {target_asset} not found in features")
            return None
        
        labels = pd.Series(0, index=features_df.index)
        target_prices = features_df[target_col]
        
        # Enhanced labeling with cross-asset confirmation
        for i in range(len(features_df) - periods_ahead):
            current_price = target_prices.iloc[i]
            future_prices = target_prices.iloc[i+1:i+periods_ahead+1]
            
            # Calculate trend before current point
            lookback = min(20, i)  # Longer lookback for multi-asset
            if lookback < 5:
                continue
            
            past_prices = target_prices.iloc[i-lookback:i]
            trend_slope = np.polyfit(range(len(past_prices)), past_prices, 1)[0]
            
            # Calculate future performance
            max_future = future_prices.max()
            min_future = future_prices.min()
            max_gain = (max_future - current_price) / current_price
            max_loss = (min_future - current_price) / current_price
            
            # Cross-asset confirmation signals
            confirmation_score = 0
            
            # BTC correlation confirmation
            if f'{target_asset}_btc_corr' in features_df.columns:
                btc_corr = features_df[f'{target_asset}_btc_corr'].iloc[i]
                if not pd.isna(btc_corr):
                    if abs(btc_corr) > 0.7:  # Strong correlation
                        confirmation_score += 1
            
            # Volume confirmation
            if f'{target_asset}_volume_share' in features_df.columns:
                volume_share = features_df[f'{target_asset}_volume_share'].iloc[i]
                if not pd.isna(volume_share):
                    # Higher than average volume share
                    avg_volume_share = features_df[f'{target_asset}_volume_share'].rolling(50).mean().iloc[i]
                    if volume_share > avg_volume_share * 1.2:
                        confirmation_score += 1
            
            # Divergence confirmation
            if f'{target_asset}_divergence' in features_df.columns:
                divergence = features_df[f'{target_asset}_divergence'].iloc[i]
                if not pd.isna(divergence):
                    if abs(divergence) > 0.02:  # Significant divergence
                        confirmation_score += 1
            
            # Enhanced thresholds with confirmation
            base_threshold = 0.05  # 5%
            confirmed_threshold = 0.03  # 3% if confirmed by other signals
            
            threshold = confirmed_threshold if confirmation_score >= 2 else base_threshold
            
            # Label bullish reversals (after downtrend with confirmation)
            if (trend_slope < 0 and max_gain >= threshold):
                labels.iloc[i] = 1
            
            # Label bearish reversals (after uptrend with confirmation)
            elif (trend_slope > 0 and max_loss <= -threshold):
                labels.iloc[i] = 2
        
        return labels
    
    def train_multi_crypto_models(self, target_asset: str = "BNB", periods_ahead: int = 10) -> Dict:
        """Train ML models using multi-cryptocurrency data"""
        
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not available"}
        
        self.logger.info(f"Training multi-crypto models for {target_asset}")
        
        # Fetch multi-crypto data
        if not self.market_data:
            multi_data = self.fetch_multi_crypto_data("1h", 1500)
            if not multi_data:
                return {"error": "Failed to fetch multi-crypto data"}
        else:
            multi_data = self.market_data
        
        # Create cross-asset features
        features_df = self.create_cross_asset_features(multi_data)
        if features_df is None:
            return {"error": "Failed to create cross-asset features"}
        
        # Create enhanced labels
        labels = self.create_multi_asset_labels(features_df, target_asset, periods_ahead)
        if labels is None:
            return {"error": "Failed to create labels"}
        
        # Select relevant features for target asset
        target_prefix = target_asset
        feature_columns = []
        
        # Include target asset features
        for col in features_df.columns:
            if (col.startswith(target_prefix) or 
                col in ['btc_dominance', 'btc_dominance_change', 'sector_rotation'] or
                col.endswith('_btc_corr') or
                col.endswith('_divergence') or
                col.endswith('_volume_share')):
                feature_columns.append(col)
        
        # Clean data
        features_clean = features_df[feature_columns].dropna()
        labels_clean = labels.loc[features_clean.index]
        
        if len(features_clean) < 100:
            return {"error": "Insufficient clean data for training"}
        
        X = features_clean.values
        y = labels_clean.values
        
        self.logger.info(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        self.logger.info(f"Label distribution: {np.bincount(y)}")
        
        # Train models
        models = {}
        results = {}
        
        model_configs = {
            "random_forest": RandomForestClassifier(
                n_estimators=150, max_depth=12, random_state=42
            ),
            "gradient_boost": GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=8, random_state=42
            ),
            "logistic": LogisticRegression(
                random_state=42, max_iter=1000, C=0.1
            )
        }
        
        for model_name, model in model_configs.items():
            try:
                self.logger.info(f"Training {model_name} for {target_asset}...")
                
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
                
                # Store results
                models[model_name] = model
                scaler_key = f"{model_name}_{target_asset}_{periods_ahead}"
                self.scalers[scaler_key] = scaler
                
                results[model_name] = {
                    "train_accuracy": train_score,
                    "test_accuracy": test_score,
                    "feature_count": len(feature_columns)
                }
                
                self.logger.info(f"{model_name}: Train={train_score:.3f}, Test={test_score:.3f}")
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {e}")
                results[model_name] = {"error": str(e)}
        
        model_key = f"multi_{target_asset}_{periods_ahead}"
        self.models[model_key] = {
            "models": models,
            "feature_columns": feature_columns,
            "target_asset": target_asset,
            "periods_ahead": periods_ahead
        }
        
        return {
            "success": True,
            "target_asset": target_asset,
            "models_trained": len(models),
            "training_samples": len(X),
            "feature_count": len(feature_columns),
            "results": results,
            "cross_asset_features": True
        }
    
    def predict_multi_crypto_reversal(self, target_asset: str = "BNB", periods_ahead: int = 10) -> Dict:
        """Make trend reversal prediction using multi-crypto analysis"""
        
        model_key = f"multi_{target_asset}_{periods_ahead}"
        if model_key not in self.models:
            return {"error": f"No trained models for {target_asset} with {periods_ahead} periods"}
        
        try:
            # Fetch recent data
            recent_data = self.fetch_multi_crypto_data("1h", 300)
            if not recent_data:
                return {"error": "Failed to fetch recent multi-crypto data"}
            
            # Create features
            features_df = self.create_cross_asset_features(recent_data)
            if features_df is None:
                return {"error": "Failed to create recent features"}
            
            # Get model info
            model_info = self.models[model_key]
            models = model_info["models"]
            feature_columns = model_info["feature_columns"]
            
            # Extract latest features
            latest_features = features_df[feature_columns].iloc[-1:].values
            
            predictions = {}
            confidence_scores = []
            
            # Get predictions from all models
            for model_name, model in models.items():
                scaler_key = f"{model_name}_{target_asset}_{periods_ahead}"
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
            
            # Get current market context
            current_price = recent_data[f"{target_asset}USDT"]['close'].iloc[-1]
            
            # Cross-asset context
            context = {}
            if 'BTCUSDT' in recent_data:
                btc_price = recent_data['BTCUSDT']['close'].iloc[-1]
                context['btc_price'] = btc_price
                
                # Calculate current correlation if possible
                if len(features_df) > 20:
                    target_col = f'{target_asset}_close'
                    btc_col = 'BTC_close'
                    if target_col in features_df.columns and btc_col in features_df.columns:
                        recent_corr = features_df[target_col].tail(20).corr(features_df[btc_col].tail(20))
                        context['btc_correlation'] = recent_corr
            
            # Market dominance if available
            if 'btc_dominance' in features_df.columns:
                context['btc_dominance'] = features_df['btc_dominance'].iloc[-1]
            
            # Interpretation
            reversal_types = {0: "No Reversal", 1: "Bullish Reversal", 2: "Bearish Reversal"}
            
            return {
                "ensemble_prediction": ensemble_prediction,
                "prediction_label": reversal_types[ensemble_prediction],
                "confidence": avg_confidence,
                "individual_predictions": predictions,
                "target_asset": target_asset,
                "current_price": current_price,
                "periods_ahead": periods_ahead,
                "market_context": context,
                "multi_crypto_analysis": True,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Multi-crypto prediction failed: {e}")
            return {"error": str(e)}
    
    def get_market_overview(self) -> Dict:
        """Get overview of multi-crypto market state"""
        
        if not self.market_data:
            recent_data = self.fetch_multi_crypto_data("1h", 100)
        else:
            recent_data = self.market_data
        
        if not recent_data:
            return {"error": "No market data available"}
        
        overview = {
            "timestamp": datetime.now().isoformat(),
            "assets_analyzed": len(recent_data),
            "market_summary": {}
        }
        
        for symbol, df in recent_data.items():
            if len(df) > 0:
                current_price = df['close'].iloc[-1]
                daily_change = (df['close'].iloc[-1] / df['close'].iloc[-24] - 1) * 100 if len(df) >= 24 else 0
                
                overview["market_summary"][symbol] = {
                    "price": current_price,
                    "daily_change_pct": daily_change,
                    "volume": df['volume'].iloc[-1],
                    "crypto_name": df['crypto_name'].iloc[0]
                }
        
        return overview


# Example usage and training
if __name__ == "__main__":
    print("🌐 MULTI-CRYPTOCURRENCY ML SYSTEM")
    print("=" * 60)
    
    # Initialize system
    multi_ml = MultiCryptoML()
    
    # Train models for different assets
    target_assets = ["BNB", "ETH", "SOL"]
    
    for asset in target_assets:
        print(f"\n🎯 Training models for {asset}...")
        result = multi_ml.train_multi_crypto_models(asset, 10)
        
        if "success" in result:
            print(f"✅ Success: {result['models_trained']} models trained")
            print(f"📊 Training samples: {result['training_samples']}")
            print(f"🎯 Cross-asset features: {result['feature_count']}")
        else:
            print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
    
    # Show market overview
    print(f"\n📊 MARKET OVERVIEW:")
    overview = multi_ml.get_market_overview()
    if "error" not in overview:
        for symbol, data in overview["market_summary"].items():
            change = data["daily_change_pct"]
            emoji = "🟢" if change > 0 else "🔴" if change < 0 else "🟡"
            print(f"{emoji} {data['crypto_name']}: ${data['price']:.2f} ({change:+.1f}%)")
