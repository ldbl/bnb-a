#!/usr/bin/env python3
"""
Analyze why 90-day BNB prediction is strongly bearish (95.5%)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime
# import matplotlib.pyplot as plt
# import seaborn as sns

from bnb_enhanced_ml import BNBEnhancedML
from logger import get_logger

class BearishAnalyzer:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.bnb_ml = BNBEnhancedML()
        self.model_dir = Path("ml_models_bnb_enhanced")
        
    def analyze_90d_bearish_signal(self):
        """Comprehensive analysis of the 90-day bearish prediction"""
        
        print("🔍 ANALYZING 90-DAY BEARISH PREDICTION")
        print("=" * 60)
        
        # 1. Load the 90-day model
        model_path = self.model_dir / "bnb_enhanced_90.pkl"
        if not model_path.exists():
            print("❌ 90-day model not found. Train it first!")
            return
            
        print("📊 Loading 90-day model...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        # 2. Get current BNB data
        print("📈 Fetching current BNB data...")
        bnb_data = self._get_current_bnb_data()
        
        # 3. Create features
        print("🔧 Creating enhanced features...")
        features_df = self.bnb_ml.create_bnb_enhanced_features(bnb_data, model_data["pattern_library"])
        
        # 4. Feature importance analysis
        print("\n🎯 FEATURE IMPORTANCE ANALYSIS:")
        print("-" * 40)
        self._analyze_feature_importance(model_data, features_df)
        
        # 5. Current market conditions
        print("\n📊 CURRENT MARKET CONDITIONS:")
        print("-" * 40)
        self._analyze_current_conditions(bnb_data, features_df)
        
        # 6. Historical pattern analysis
        print("\n📈 HISTORICAL PATTERN ANALYSIS:")
        print("-" * 40)
        self._analyze_historical_patterns(bnb_data)
        
        # 7. Cross-crypto validation
        print("\n🌐 CROSS-CRYPTO VALIDATION:")
        print("-" * 40)
        self._analyze_cross_crypto_signals()
        
        # 8. Risk factors
        print("\n⚠️ IDENTIFIED RISK FACTORS:")
        print("-" * 40)
        self._identify_risk_factors(bnb_data, features_df)
        
        print("\n" + "=" * 60)
        print("✅ Analysis completed!")
        
    def _get_current_bnb_data(self):
        """Get recent BNB data for analysis"""
        import requests
        
        params = {
            "symbol": "BNBUSDT",
            "interval": "1d",
            "limit": 200
        }
        
        response = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=10)
        
        if response.status_code == 200:
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
            
            # Add metadata
            df['symbol'] = 'BNBUSDT'
            df['crypto_name'] = 'BNB'
            df['category'] = 'Exchange Token'
            df['market_weight'] = 5.0
            
            return df
        else:
            raise Exception(f"Failed to fetch BNB data: {response.status_code}")
    
    def _analyze_feature_importance(self, model_data, features_df):
        """Analyze which features are driving the bearish prediction"""
        
        # Get feature importance from Random Forest
        rf_model = model_data["models"].get("enhanced_rf")
        if rf_model and hasattr(rf_model, 'feature_importances_'):
            
            feature_names = model_data["selected_features"]
            importances = rf_model.feature_importances_
            
            # Sort by importance
            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print("🏆 TOP 10 MOST IMPORTANT FEATURES:")
            for i, (feature, importance) in enumerate(feature_importance[:10]):
                current_value = features_df[feature].iloc[-1] if feature in features_df.columns else "N/A"
                print(f"   {i+1:2d}. {feature:25s}: {importance:.4f} (current: {current_value:.4f})")
                
            # Analyze bearish indicators
            bearish_features = []
            for feature, importance in feature_importance[:15]:
                if feature in features_df.columns:
                    current_value = features_df[feature].iloc[-1]
                    historical_mean = features_df[feature].mean()
                    
                    # Check if current value suggests bearish trend
                    if 'rsi' in feature.lower() and current_value > 70:
                        bearish_features.append(f"{feature}: {current_value:.2f} (overbought)")
                    elif 'macd' in feature.lower() and current_value < 0:
                        bearish_features.append(f"{feature}: {current_value:.4f} (negative)")
                    elif 'ema' in feature.lower() and current_value < historical_mean:
                        bearish_features.append(f"{feature}: {current_value:.2f} (below average)")
            
            if bearish_features:
                print("\n🔴 BEARISH INDICATORS DETECTED:")
                for indicator in bearish_features[:5]:
                    print(f"   • {indicator}")
        
    def _analyze_current_conditions(self, bnb_data, features_df):
        """Analyze current market conditions"""
        
        current_price = bnb_data['close'].iloc[-1]
        price_7d = bnb_data['close'].iloc[-7] if len(bnb_data) >= 7 else current_price
        price_30d = bnb_data['close'].iloc[-30] if len(bnb_data) >= 30 else current_price
        
        change_7d = (current_price - price_7d) / price_7d * 100
        change_30d = (current_price - price_30d) / price_30d * 100
        
        print(f"💰 Current BNB Price: ${current_price:.2f}")
        print(f"📊 7-day change: {change_7d:+.2f}%")
        print(f"📊 30-day change: {change_30d:+.2f}%")
        
        # Volume analysis
        current_volume = bnb_data['volume'].iloc[-1]
        avg_volume = bnb_data['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume
        
        print(f"📈 Volume vs 20-day avg: {volume_ratio:.2f}x")
        
        # Technical indicators
        if 'rsi_14' in features_df.columns:
            rsi = features_df['rsi_14'].iloc[-1]
            print(f"📊 RSI(14): {rsi:.1f} {'(Overbought)' if rsi > 70 else '(Oversold)' if rsi < 30 else '(Neutral)'}")
            
        if 'macd' in features_df.columns:
            macd = features_df['macd'].iloc[-1]
            print(f"📊 MACD: {macd:.4f} {'(Bearish)' if macd < 0 else '(Bullish)'}")
    
    def _analyze_historical_patterns(self, bnb_data):
        """Analyze historical patterns that might suggest bearish trend"""
        
        # Check for consecutive red days
        daily_changes = bnb_data['close'].pct_change()
        recent_changes = daily_changes.iloc[-10:]
        
        red_days = (recent_changes < 0).sum()
        green_days = (recent_changes > 0).sum()
        
        print(f"📈 Last 10 days: {green_days} green, {red_days} red")
        
        # Check for lower highs and lower lows
        recent_highs = bnb_data['high'].iloc[-30:]
        recent_lows = bnb_data['low'].iloc[-30:]
        
        if len(recent_highs) >= 20:
            recent_high = recent_highs.iloc[-10:].max()
            earlier_high = recent_highs.iloc[-30:-10].max()
            
            recent_low = recent_lows.iloc[-10:].min()
            earlier_low = recent_lows.iloc[-30:-10].min()
            
            if recent_high < earlier_high:
                print("🔴 Lower highs detected (bearish pattern)")
            if recent_low < earlier_low:
                print("🔴 Lower lows detected (bearish pattern)")
                
        # Support/Resistance levels
        current_price = bnb_data['close'].iloc[-1]
        support_levels = self._find_support_levels(bnb_data)
        resistance_levels = self._find_resistance_levels(bnb_data)
        
        print(f"📊 Nearest support: ${min(support_levels, key=lambda x: abs(x - current_price)):.2f}")
        print(f"📊 Nearest resistance: ${min(resistance_levels, key=lambda x: abs(x - current_price)):.2f}")
    
    def _find_support_levels(self, data, window=20):
        """Find support levels"""
        lows = data['low']
        support_levels = []
        
        for i in range(window, len(lows) - window):
            if lows.iloc[i] == lows.iloc[i-window:i+window].min():
                support_levels.append(lows.iloc[i])
                
        return sorted(set(support_levels))[-5:]  # Last 5 support levels
    
    def _find_resistance_levels(self, data, window=20):
        """Find resistance levels"""
        highs = data['high']
        resistance_levels = []
        
        for i in range(window, len(highs) - window):
            if highs.iloc[i] == highs.iloc[i-window:i+window].max():
                resistance_levels.append(highs.iloc[i])
                
        return sorted(set(resistance_levels))[-5:]  # Last 5 resistance levels
    
    def _analyze_cross_crypto_signals(self):
        """Check if other cryptos also show bearish signals"""
        
        # This would require fetching data for other cryptos
        # For now, we'll implement a simplified version
        print("🧠 Multi-crypto intelligence suggests:")
        print("   • BTC trend analysis: [Would need live data]")
        print("   • ETH correlation: [Would need live data]")
        print("   • Market-wide sentiment: [Would need live data]")
        print("   • Cross-validation with top 10 cryptos: Enhanced model learned patterns")
    
    def _identify_risk_factors(self, bnb_data, features_df):
        """Identify specific risk factors contributing to bearish outlook"""
        
        risk_factors = []
        
        # Price momentum
        recent_returns = bnb_data['close'].pct_change().iloc[-30:]
        if recent_returns.mean() < 0:
            risk_factors.append(f"Negative 30-day momentum: {recent_returns.mean()*100:.2f}%")
        
        # Volatility
        volatility = recent_returns.std() * np.sqrt(365) * 100
        if volatility > 50:  # High volatility threshold
            risk_factors.append(f"High volatility: {volatility:.1f}% annualized")
        
        # Volume decline
        volume_trend = bnb_data['volume'].iloc[-10:].mean() / bnb_data['volume'].iloc[-30:-10].mean()
        if volume_trend < 0.8:
            risk_factors.append(f"Declining volume: {(1-volume_trend)*100:.1f}% drop")
        
        if risk_factors:
            for i, factor in enumerate(risk_factors, 1):
                print(f"   {i}. {factor}")
        else:
            print("   ✅ No major risk factors identified in basic analysis")

if __name__ == "__main__":
    analyzer = BearishAnalyzer()
    analyzer.analyze_90d_bearish_signal()
