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
    
    def calculate_dynamic_fibonacci_levels(self, prices: List[float], window: int = 3) -> Dict:
        """Calculate dynamic Fibonacci levels based on rolling average lows"""
        import numpy as np
        
        # Convert to numpy array and handle NaN
        prices_array = np.array(prices)
        prices_array = np.nan_to_num(prices_array, nan=0.0)
        
        # Calculate rolling average of lows (using close prices as proxy)
        if len(prices_array) >= window:
            rolling_lows = []
            for i in range(window, len(prices_array)):
                window_prices = prices_array[i-window:i]
                rolling_lows.append(np.mean(window_prices))
            
            # Use recent rolling lows for dynamic levels
            recent_low = np.mean(rolling_lows[-5:]) if len(rolling_lows) >= 5 else np.mean(rolling_lows)
            recent_high = np.max(prices_array[-20:])  # Last 20 periods
            
            # Calculate dynamic Fibonacci levels
            range_size = recent_high - recent_low
            
            dynamic_levels = {
                "dynamic_low": round(recent_low, 2),
                "dynamic_high": round(recent_high, 2),
                "buy_zone_50": round(recent_low + (range_size * 0.5), 2),
                "buy_zone_618": round(recent_low + (range_size * 0.618), 2),
                "sell_zone_1382": round(recent_high + (range_size * 0.382), 2),
                "sell_zone_1618": round(recent_high + (range_size * 0.618), 2),
                "risk_reward_ratio": round((recent_high - recent_low) / (recent_high * 0.1), 2)  # 10% risk
            }
            
            return dynamic_levels
        else:
            return {"error": "Insufficient data for dynamic calculation"}
    
    def get_dynamic_swing_signals(self, current_price: float, prices: List[float]) -> Dict:
        """Get dynamic swing trading signals based on Fibonacci levels"""
        try:
            # Get dynamic Fibonacci levels
            dynamic_levels = self.calculate_dynamic_fibonacci_levels(prices)
            
            if "error" in dynamic_levels:
                return {"error": dynamic_levels["error"]}
            
            # Calculate current position relative to dynamic levels
            buy_zone_50 = dynamic_levels["buy_zone_50"]
            buy_zone_618 = dynamic_levels["buy_zone_618"]
            sell_zone_1382 = dynamic_levels["sell_zone_1382"]
            sell_zone_1618 = dynamic_levels["sell_zone_1618"]
            
            # Generate swing trading signals
            signals = {
                "current_price": current_price,
                "dynamic_levels": dynamic_levels,
                "buy_signals": [],
                "sell_signals": [],
                "risk_management": {}
            }
            
            # Buy signals (50-61.8% Fibonacci retracement)
            if current_price <= buy_zone_618:
                signals["buy_signals"].append({
                    "type": "STRONG_BUY",
                    "level": "61.8% retracement",
                    "price": buy_zone_618,
                    "confidence": "HIGH"
                })
            elif current_price <= buy_zone_50:
                signals["buy_signals"].append({
                    "type": "BUY",
                    "level": "50% retracement",
                    "price": buy_zone_50,
                    "confidence": "MEDIUM"
                })
            
            # Sell signals (138.2-161.8% Fibonacci extension)
            if current_price >= sell_zone_1382:
                signals["sell_signals"].append({
                    "type": "STRONG_SELL",
                    "level": "138.2% extension",
                    "price": sell_zone_1382,
                    "confidence": "HIGH"
                })
            elif current_price >= sell_zone_1618:
                signals["buy_signals"].append({
                    "type": "SELL",
                    "level": "161.8% extension",
                    "price": sell_zone_1618,
                    "confidence": "MEDIUM"
                })
            
            # Risk management (Rule #7: отстъпление под $550)
            stop_loss = max(550, dynamic_levels["dynamic_low"] * 0.95)  # 5% below dynamic low
            signals["risk_management"] = {
                "stop_loss": round(stop_loss, 2),
                "risk_reward_ratio": dynamic_levels["risk_reward_ratio"],
                "position_sizing": "1/3 capital (Rule #3)",
                "max_leverage": 2
            }
            
            return signals
            
        except Exception as e:
            return {"error": f"Error calculating dynamic signals: {str(e)}"}
    
    def calculate_historical_support_resistance(self, prices: List[float], volumes: List[float] = None, 
                                             lookback_periods: int = 100) -> Dict:
        """Calculate historical support and resistance levels from price data"""
        import numpy as np
        
        try:
            # Convert to numpy array and handle NaN
            prices_array = np.array(prices[-lookback_periods:])
            prices_array = np.nan_to_num(prices_array, nan=0.0)
            
            # Find local minima (support) and maxima (resistance)
            support_levels = []
            resistance_levels = []
            
            # Use peak detection for support/resistance
            for i in range(2, len(prices_array) - 2):
                # Support: local minimum
                if (prices_array[i] < prices_array[i-1] and 
                    prices_array[i] < prices_array[i-2] and
                    prices_array[i] < prices_array[i+1] and 
                    prices_array[i] < prices_array[i+2]):
                    support_levels.append({
                        'price': round(prices_array[i], 2),
                        'index': i,
                        'strength': self._calculate_level_strength(prices_array, i, 'support')
                    })
                
                # Resistance: local maximum
                if (prices_array[i] > prices_array[i-1] and 
                    prices_array[i] > prices_array[i-2] and
                    prices_array[i] > prices_array[i+1] and 
                    prices_array[i] > prices_array[i+2]):
                    resistance_levels.append({
                        'price': round(prices_array[i], 2),
                        'index': i,
                        'strength': self._calculate_level_strength(prices_array, i, 'resistance')
                    })
            
            # Sort by strength and get top 3
            support_levels.sort(key=lambda x: x['strength'], reverse=True)
            resistance_levels.sort(key=lambda x: x['strength'], reverse=True)
            
            # Group nearby levels (within 2% of each other)
            support_levels = self._group_nearby_levels(support_levels[:10], 0.02)
            resistance_levels = self._group_nearby_levels(resistance_levels[:10], 0.02)
            
            return {
                'support_levels': support_levels[:3],
                'resistance_levels': resistance_levels[:3],
                'analysis_periods': lookback_periods
            }
            
        except Exception as e:
            return {"error": f"Error calculating support/resistance: {str(e)}"}
    
    def _calculate_level_strength(self, prices, level_index: int, level_type: str) -> float:
        """Calculate the strength of a support/resistance level"""
        try:
            level_price = prices[level_index]
            strength = 0.0
            
            # Count touches (how many times price approached this level)
            touch_count = 0
            for i in range(len(prices)):
                if abs(prices[i] - level_price) / level_price < 0.01:  # Within 1%
                    touch_count += 1
            
            # Volume confirmation (if available)
            volume_factor = 1.0  # Default if no volume data
            
            # Distance from current price
            current_price = prices[-1]
            distance_factor = 1.0 / (1.0 + abs(current_price - level_price) / current_price)
            
            # Calculate final strength
            strength = (touch_count * 0.4 + volume_factor * 0.3 + distance_factor * 0.3)
            
            return round(strength, 2)
            
        except Exception as e:
            return 1.0  # Default strength
    
    def _group_nearby_levels(self, levels: List[Dict], threshold: float) -> List[Dict]:
        """Group nearby support/resistance levels"""
        if not levels:
            return []
        
        grouped = []
        used_indices = set()
        
        for i, level in enumerate(levels):
            if i in used_indices:
                continue
                
            group = [level]
            used_indices.add(i)
            
            # Find nearby levels
            for j, other_level in enumerate(levels[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                price_diff = abs(level['price'] - other_level['price']) / level['price']
                if price_diff <= threshold:
                    group.append(other_level)
                    used_indices.add(j)
            
            # Average the grouped levels
            avg_price = sum(l['price'] for l in group) / len(group)
            avg_strength = sum(l['strength'] for l in group) / len(group)
            
            grouped.append({
                'price': round(avg_price, 2),
                'strength': round(avg_strength, 2),
                'touches': len(group),
                'original_levels': group
            })
        
        return grouped
    
    def get_fibonacci_position_analysis(self, current_price: float, prices: List[float]) -> Dict:
        """Get comprehensive analysis of current price position relative to Fibonacci levels"""
        try:
            # Get dynamic Fibonacci levels
            dynamic_levels = self.calculate_dynamic_fibonacci_levels(prices)
            
            if "error" in dynamic_levels:
                return {"error": dynamic_levels["error"]}
            
            # Get historical support/resistance
            support_resistance = self.calculate_historical_support_resistance(prices)
            
            if "error" in support_resistance:
                return {"error": support_resistance["error"]}
            
            # Calculate current position relative to all levels
            analysis = {
                'current_price': current_price,
                'fibonacci_levels': dynamic_levels,
                'support_levels': [],
                'resistance_levels': [],
                'current_position': {},
                'recommendations': []
            }
            
            # Analyze support levels
            for level in support_resistance['support_levels']:
                level_price = level['price']
                distance = ((current_price - level_price) / current_price) * 100
                
                # Find closest Fibonacci level
                fib_level = self._find_closest_fibonacci_level(level_price, dynamic_levels)
                
                level_analysis = {
                    'price': level_price,
                    'strength': level['strength'],
                    'touches': level['touches'],
                    'distance_from_current': round(distance, 2),
                    'fibonacci_position': fib_level,
                    'is_above_current': level_price > current_price
                }
                
                analysis['support_levels'].append(level_analysis)
            
            # Analyze resistance levels
            for level in support_resistance['resistance_levels']:
                level_price = level['price']
                distance = ((level_price - current_price) / current_price) * 100
                
                # Find closest Fibonacci level
                fib_level = self._find_closest_fibonacci_level(level_price, dynamic_levels)
                
                level_analysis = {
                    'price': level_price,
                    'strength': level['strength'],
                    'touches': level['touches'],
                    'distance_from_current': round(distance, 2),
                    'fibonacci_position': fib_level,
                    'is_above_current': level_price > current_price
                }
                
                analysis['resistance_levels'].append(level_analysis)
            
            # Current position analysis
            analysis['current_position'] = self._analyze_current_position(current_price, dynamic_levels)
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_position_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            return {"error": f"Error in Fibonacci position analysis: {str(e)}"}
    
    def _find_closest_fibonacci_level(self, price: float, fib_levels: Dict) -> Dict:
        """Find the closest Fibonacci level to a given price"""
        try:
            closest_level = None
            min_distance = float('inf')
            
            # Check all Fibonacci levels
            for level_name, level_price in fib_levels.items():
                if isinstance(level_price, (int, float)) and level_price > 0:
                    distance = abs(price - level_price)
                    if distance < min_distance:
                        min_distance = distance
                        closest_level = {
                            'name': level_name,
                            'price': level_price,
                            'distance': round(distance, 2),
                            'percentage': round((distance / price) * 100, 2)
                        }
            
            return closest_level or {'name': 'None', 'price': 0, 'distance': 0, 'percentage': 0}
            
        except Exception as e:
            return {'name': 'Error', 'price': 0, 'distance': 0, 'percentage': 0}
    
    def _analyze_current_position(self, current_price: float, fib_levels: Dict) -> Dict:
        """Analyze current price position relative to Fibonacci levels"""
        try:
            # Find closest Fibonacci level
            closest_fib = self._find_closest_fibonacci_level(current_price, fib_levels)
            
            # Determine position type
            if closest_fib['name'] == 'buy_zone_618':
                position_type = "STRONG_BUY_ZONE"
                confidence = "HIGH"
            elif closest_fib['name'] == 'buy_zone_50':
                position_type = "BUY_ZONE"
                confidence = "MEDIUM"
            elif closest_fib['name'] == 'sell_zone_1382':
                position_type = "STRONG_SELL_ZONE"
                confidence = "HIGH"
            elif closest_fib['name'] == 'sell_zone_1618':
                position_type = "SELL_ZONE"
                confidence = "MEDIUM"
            else:
                position_type = "NEUTRAL"
                confidence = "LOW"
            
            return {
                'position_type': position_type,
                'confidence': confidence,
                'closest_fibonacci': closest_fib,
                'recommendation': self._get_position_recommendation(position_type, confidence)
            }
            
        except Exception as e:
            return {
                'position_type': 'ERROR',
                'confidence': 'LOW',
                'closest_fibonacci': {'name': 'Error', 'price': 0, 'distance': 0, 'percentage': 0},
                'recommendation': 'Error in analysis'
            }
    
    def _get_position_recommendation(self, position_type: str, confidence: str) -> str:
        """Get trading recommendation based on position type and confidence"""
        recommendations = {
            'STRONG_BUY_ZONE': 'Strong buy signal - Enter long position',
            'BUY_ZONE': 'Buy signal - Consider long entry',
            'STRONG_SELL_ZONE': 'Strong sell signal - Exit long or enter short',
            'SELL_ZONE': 'Sell signal - Consider taking profits',
            'NEUTRAL': 'Wait for clearer signals'
        }
        
        return recommendations.get(position_type, 'No recommendation available')
    
    def _generate_position_recommendations(self, analysis: Dict) -> List[str]:
        """Generate comprehensive trading recommendations"""
        recommendations = []
        
        try:
            # Support level recommendations
            for level in analysis['support_levels']:
                if level['distance_from_current'] < 5:  # Within 5%
                    recommendations.append(f"Support at ${level['price']} nearby (Fibonacci: {level['fibonacci_position']['name']})")
            
            # Resistance level recommendations
            for level in analysis['resistance_levels']:
                if level['distance_from_current'] < 5:  # Within 5%
                    recommendations.append(f"Resistance at ${level['price']} nearby (Fibonacci: {level['fibonacci_position']['name']})")
            
            # Current position recommendations
            current_pos = analysis['current_position']
            if current_pos['position_type'] != 'NEUTRAL':
                recommendations.append(f"Current position: {current_pos['recommendation']}")
            
            # Risk management
            recommendations.append("Risk management: Use 1/3 position sizing, max 2x leverage")
            
            return recommendations
            
        except Exception as e:
            return [f"Error generating recommendations: {str(e)}"]
        
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
    
    def check_critical_fibonacci_alerts(self, current_price: float) -> Dict:
        """Check for critical Fibonacci situations that warrant immediate attention"""
        
        try:
            signals = self.get_fibonacci_signals(current_price)
            
            if signals.get("signal") == "NO_DATA":
                return {"show_alert": False, "reason": "No Fibonacci data available"}
            
            critical_signals = []
            alert_score = 0
            
            closest = signals.get("closest_level", {})
            fib_levels = signals.get("fibonacci_levels", {})
            trend = signals.get("trend", "")
            
            # Golden Pocket detection (61.8% zone)
            if signals.get("golden_pocket"):
                critical_signals.append("⭐ GOLDEN POCKET - High probability bounce zone!")
                alert_score += 8
            
            # Major Fibonacci level proximity
            major_levels = ["38.2%", "50%", "61.8%", "78.6%"]
            if closest.get("name") in major_levels and closest.get("distance", 100) < 8:
                level_name = closest["name"]
                distance = closest["distance"]
                critical_signals.append(f"📐 At major Fib level {level_name} (${distance:.1f} away)")
                alert_score += 6
            
            # Strong support/resistance at Fib levels
            if signals.get("is_at_support"):
                critical_signals.append(f"🛡️ Strong Fibonacci support at {closest.get('name', 'N/A')}")
                alert_score += 5
            elif signals.get("is_at_resistance"):
                critical_signals.append(f"⚡ Strong Fibonacci resistance at {closest.get('name', 'N/A')}")
                alert_score += 5
            
            # Extension levels (breakout scenarios)
            extension_levels = ["161.8%", "261.8%"]
            if closest.get("name") in extension_levels and closest.get("distance", 100) < 15:
                critical_signals.append(f"🚀 Near Fibonacci extension {closest['name']} - Breakout zone")
                alert_score += 4
            
            # Extreme Fibonacci zones (near 0% or 100%)
            extreme_levels = ["0%", "100%"]
            if closest.get("name") in extreme_levels and closest.get("distance", 100) < 10:
                zone_type = "swing high" if closest["name"] == "0%" else "swing low"
                critical_signals.append(f"🎯 Approaching {zone_type} ({closest['name']}) - Critical zone")
                alert_score += 7
            
            # Action-based alerts
            action = signals.get("action", "WAIT")
            if action in ["STRONG_BUY", "STRONG_SELL"]:
                action_emoji = "🟢" if "BUY" in action else "🔴"
                critical_signals.append(f"{action_emoji} Strong Fibonacci {action} signal")
                alert_score += 3
            
            # Determine if alert should be shown
            show_alert = alert_score >= 6  # Threshold for Fibonacci alerts
            
            return {
                "show_alert": show_alert,
                "alert_score": alert_score,
                "critical_signals": critical_signals,
                "fibonacci_data": {
                    "action": action,
                    "trend": trend,
                    "closest_level": closest,
                    "golden_pocket": signals.get("golden_pocket", False),
                    "swing_high": signals.get("swing_high", 0),
                    "swing_low": signals.get("swing_low", 0)
                }
            }
            
        except Exception as e:
            return {"show_alert": False, "reason": f"Error checking Fibonacci alerts: {e}"}
    
    def get_critical_fibonacci_alert_text(self, alert_data: Dict) -> str:
        """Generate formatted alert text for critical Fibonacci activity"""
        
        if not alert_data.get("show_alert"):
            return ""
        
        signals = alert_data.get("critical_signals", [])
        fib_data = alert_data.get("fibonacci_data", {})
        
        alert_text = f"\n📐 CRITICAL FIBONACCI ALERT\n"
        alert_text += "=" * 45 + "\n"
        
        for signal in signals:
            alert_text += f"{signal}\n"
        
        alert_text += f"\nAction: {fib_data.get('action', 'WAIT')}"
        alert_text += f"\nTrend: {fib_data.get('trend', 'Unknown')}"
        
        closest = fib_data.get("closest_level", {})
        if closest:
            alert_text += f"\nClosest Level: {closest.get('name', 'N/A')} at ${closest.get('level', 0)}"
        
        alert_text += f"\n\nAlert Score: {alert_data.get('alert_score', 0)}/20"
        alert_text += "\n💡 Consider: Check Fibonacci analysis for entry/exit levels"
        alert_text += "\n" + "=" * 45
        
        return alert_text
    
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