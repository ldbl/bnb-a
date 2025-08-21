#!/usr/bin/env python3
"""
Fibonacci Retracement Calculator Module
Import this into your main BNB analyzer
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class FibonacciAnalyzer:
    def __init__(self):
        self.ratios = {
            "0%": 0.0,
            "23.6%": 0.236,
            "38.2%": 0.382,
            "50%": 0.5,
            "61.8%": 0.618,
            "78.6%": 0.786,
            "100%": 1.0,
            "161.8%": 1.618,  # Extension
            "261.8%": 2.618,  # Extension
            "423.6%": 4.236   # Extension
        }
        self.base_url = "https://api.binance.com/api/v3"
    
    def get_price_data(self, interval: str = "1d", limit: int = 100):
        """Fetch price data from Binance"""
        try:
            params = {"symbol": "BNBUSDT", "interval": interval, "limit": limit}
            response = requests.get(f"{self.base_url}/klines", params=params)
            klines = response.json()
            return {
                "highs": [float(k[2]) for k in klines],
                "lows": [float(k[3]) for k in klines],
                "closes": [float(k[4]) for k in klines],
                "times": [datetime.fromtimestamp(k[0]/1000) for k in klines]
            }
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def find_swing_points(self, data: Dict, lookback: int = 10) -> Dict:
        """Find swing high and swing low"""
        highs = data["highs"]
        lows = data["lows"]
        
        # Find highest high in lookback period
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        
        swing_high = max(recent_highs)
        swing_low = min(recent_lows)
        
        # Find indices
        high_index = len(highs) - lookback + recent_highs.index(swing_high)
        low_index = len(lows) - lookback + recent_lows.index(swing_low)
        
        return {
            "swing_high": swing_high,
            "swing_low": swing_low,
            "high_date": data["times"][high_index],
            "low_date": data["times"][low_index],
            "is_uptrend": high_index > low_index  # High came after low = uptrend
        }
    
    def calculate_fibonacci_levels(self, swing_high: float, swing_low: float, is_uptrend: bool) -> Dict:
        """Calculate Fibonacci retracement and extension levels"""
        range_size = swing_high - swing_low
        levels = {}
        
        if is_uptrend:
            # In uptrend, we retrace from high
            for name, ratio in self.ratios.items():
                if ratio <= 1.0:  # Retracement
                    level = swing_high - (range_size * ratio)
                else:  # Extension
                    level = swing_high + (range_size * (ratio - 1))
                levels[name] = round(level, 2)
        else:
            # In downtrend, we retrace from low
            for name, ratio in self.ratios.items():
                if ratio <= 1.0:  # Retracement
                    level = swing_low + (range_size * ratio)
                else:  # Extension
                    level = swing_low - (range_size * (ratio - 1))
                levels[name] = round(level, 2)
        
        return levels
    
    def find_closest_fib_level(self, current_price: float, levels: Dict) -> Dict:
        """Find closest Fibonacci level to current price"""
        closest = None
        min_distance = float('inf')
        
        for name, level in levels.items():
            distance = abs(current_price - level)
            if distance < min_distance:
                min_distance = distance
                closest = {"name": name, "level": level, "distance": round(distance, 2)}
        
        # Determine if price is at a key level
        key_levels = ["38.2%", "50%", "61.8%"]
        is_key_level = closest["name"] in key_levels and closest["distance"] < 10
        
        return {
            "closest": closest,
            "is_at_key_level": is_key_level,
            "percentage_to_level": round((closest["distance"] / current_price) * 100, 2)
        }
    
    def analyze_multiple_timeframes(self) -> Dict:
        """Analyze Fibonacci levels across multiple timeframes"""
        timeframes = {
            "Daily": {"interval": "1d", "limit": 30},
            "Weekly": {"interval": "1w", "limit": 12},
            "Monthly": {"interval": "1M", "limit": 6}
        }
        
        analysis = {}
        
        for tf_name, tf_config in timeframes.items():
            data = self.get_price_data(tf_config["interval"], tf_config["limit"])
            if not data:
                continue
            
            swing_points = self.find_swing_points(data, lookback=tf_config["limit"])
            fib_levels = self.calculate_fibonacci_levels(
                swing_points["swing_high"],
                swing_points["swing_low"],
                swing_points["is_uptrend"]
            )
            
            analysis[tf_name] = {
                "swing_high": swing_points["swing_high"],
                "swing_low": swing_points["swing_low"],
                "trend": "UPTREND" if swing_points["is_uptrend"] else "DOWNTREND",
                "levels": fib_levels
            }
        
        return analysis
    
    def get_fibonacci_signals(self, current_price: float) -> Dict:
        """Generate trading signals based on Fibonacci levels"""
        # Get daily Fibonacci levels for main analysis
        data = self.get_price_data("1d", 50)
        if not data:
            return {"signal": "NO_DATA"}
        
        swing_points = self.find_swing_points(data, lookback=20)
        fib_levels = self.calculate_fibonacci_levels(
            swing_points["swing_high"],
            swing_points["swing_low"],
            swing_points["is_uptrend"]
        )
        
        closest = self.find_closest_fib_level(current_price, fib_levels)
        
        # Generate signals
        signals = {
            "current_price": current_price,
            "swing_high": swing_points["swing_high"],
            "swing_low": swing_points["swing_low"],
            "trend": "UPTREND" if swing_points["is_uptrend"] else "DOWNTREND",
            "fibonacci_levels": fib_levels,
            "closest_level": closest["closest"],
            "is_at_support": False,
            "is_at_resistance": False,
            "action": "WAIT"
        }
        
        # Determine if at support/resistance
        if swing_points["is_uptrend"]:
            # In uptrend, Fib levels act as support
            if closest["closest"]["name"] in ["38.2%", "50%", "61.8%"] and closest["closest"]["distance"] < 10:
                signals["is_at_support"] = True
                signals["action"] = "BUY"
                signals["target"] = fib_levels["0%"]  # Target is swing high
                signals["stop_loss"] = fib_levels["78.6%"]
        else:
            # In downtrend, Fib levels act as resistance
            if closest["closest"]["name"] in ["38.2%", "50%", "61.8%"] and closest["closest"]["distance"] < 10:
                signals["is_at_resistance"] = True
                signals["action"] = "SELL"
                signals["target"] = fib_levels["0%"]  # Target is swing low
                signals["stop_loss"] = fib_levels["78.6%"]
        
        # Golden pocket check (61.8% - 65% area)
        if swing_points["is_uptrend"]:
            golden_pocket = fib_levels["61.8%"]
            if abs(current_price - golden_pocket) < 15:
                signals["golden_pocket"] = True
                signals["action"] = "STRONG_BUY"
                signals["note"] = "Price at Golden Pocket - High probability bounce zone!"
        
        return signals
    
    def display_analysis(self):
        """Display formatted Fibonacci analysis"""
        # Get current price
        try:
            response = requests.get(f"{self.base_url}/ticker/price", params={"symbol": "BNBUSDT"})
            current_price = float(response.json()['price'])
        except:
            current_price = 850
        
        print("\n" + "="*60)
        print("📐 FIBONACCI RETRACEMENT ANALYSIS")
        print("="*60)
        
        # Single timeframe analysis
        signals = self.get_fibonacci_signals(current_price)
        
        print(f"\n📊 CURRENT SITUATION")
        print(f"Price: ${current_price}")
        print(f"Trend: {signals['trend']}")
        print(f"Swing High: ${signals['swing_high']}")
        print(f"Swing Low: ${signals['swing_low']}")
        
        print(f"\n📏 FIBONACCI LEVELS")
        for name, level in signals['fibonacci_levels'].items():
            distance = current_price - level
            symbol = "↑" if distance > 0 else "↓" if distance < 0 else "="
            
            # Highlight current level
            if abs(distance) < 10:
                print(f"→ {name}: ${level} {symbol} YOU ARE HERE!")
            else:
                print(f"  {name}: ${level} {symbol} ${abs(distance):.2f} away")
        
        print(f"\n🎯 SIGNAL")
        print(f"Action: {signals['action']}")
        if signals.get('golden_pocket'):
            print(f"⭐ {signals['note']}")
        if signals['action'] != "WAIT":
            print(f"Target: ${signals.get('target', 'N/A')}")
            print(f"Stop Loss: ${signals.get('stop_loss', 'N/A')}")
        
        # Multi-timeframe analysis
        print(f"\n⏰ MULTI-TIMEFRAME FIBONACCI")
        mtf = self.analyze_multiple_timeframes()
        for tf_name, tf_data in mtf.items():
            print(f"\n{tf_name}:")
            print(f"  Range: ${tf_data['swing_low']} - ${tf_data['swing_high']}")
            print(f"  Key levels:")
            for level_name in ["38.2%", "50%", "61.8%"]:
                print(f"    {level_name}: ${tf_data['levels'][level_name]}")
        
        print("\n" + "="*60)

# Standalone usage
if __name__ == "__main__":
    fib = FibonacciAnalyzer()
    fib.display_analysis()

# Integration example:
# Save this as fibonacci_module.py
# Then in your main analyzer:
#
# from fibonacci_module import FibonacciAnalyzer
# 
# fib = FibonacciAnalyzer()
# signals = fib.get_fibonacci_signals(current_price)
# print(f"Fibonacci says: {signals['action']}")