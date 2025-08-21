#!/usr/bin/env python3
"""
Technical Indicators Module
Contains RSI, MACD, Bollinger Bands, and EMA calculations
"""

from typing import Dict, List
import numpy as np


class TechnicalIndicators:
    """Class containing all technical indicator calculations"""
    
    @staticmethod
    def calculate_ema(data: List[float], period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return data[-1] if data else 0
        
        multiplier = 2 / (period + 1)
        ema = sum(data[:period]) / period
        
        for price in data[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            diff = prices[i] - prices[i-1]
            gains.append(max(diff, 0))
            losses.append(abs(min(diff, 0)))
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 2)
    
    @staticmethod
    def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow:
            return {"macd": 0, "signal": 0, "histogram": 0, "trend": "NEUTRAL"}
        
        ema_fast = TechnicalIndicators.calculate_ema(prices, fast)
        ema_slow = TechnicalIndicators.calculate_ema(prices, slow)
        macd = ema_fast - ema_slow
        macd_signal = TechnicalIndicators.calculate_ema([macd], signal)
        histogram = macd - macd_signal
        
        trend = "BULLISH" if macd > macd_signal else "BEARISH" if macd < macd_signal else "NEUTRAL"
        
        return {
            "macd": round(macd, 2),
            "signal": round(macd_signal, 2),
            "histogram": round(histogram, 2),
            "trend": trend
        }
    
    @staticmethod
    def calculate_bollinger(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Dict:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return {"upper": 0, "middle": 0, "lower": 0, "position": "NEUTRAL"}
        
        recent_prices = prices[-period:]
        sma = sum(recent_prices) / period
        std = np.std(recent_prices)
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        current = prices[-1]
        
        if current > upper:
            position = "OVERBOUGHT"
        elif current < lower:
            position = "OVERSOLD"
        else:
            position = "NEUTRAL"
        
        return {
            "upper": round(upper, 2),
            "middle": round(sma, 2),
            "lower": round(lower, 2),
            "position": position
        }
    
    @staticmethod
    def calculate_all_indicators(prices: List[float]) -> Dict:
        """Calculate all indicators at once for convenience"""
        return {
            "rsi": TechnicalIndicators.calculate_rsi(prices),
            "macd": TechnicalIndicators.calculate_macd(prices),
            "bollinger": TechnicalIndicators.calculate_bollinger(prices)
        }
    
    @staticmethod
    def check_critical_indicator_alerts(prices: List[float], volumes: List[float] = None) -> Dict:
        """Check for critical technical indicator situations that warrant immediate attention"""
        
        try:
            if len(prices) < 20:
                return {"show_alert": False, "reason": "Insufficient price data for indicators"}
            
            # Calculate all indicators
            rsi = TechnicalIndicators.calculate_rsi(prices)
            macd = TechnicalIndicators.calculate_macd(prices)
            bollinger = TechnicalIndicators.calculate_bollinger(prices)
            
            critical_signals = []
            alert_score = 0
            current_price = prices[-1]
            
            # RSI extreme conditions
            if rsi <= 25:
                critical_signals.append(f"📉 RSI EXTREME OVERSOLD ({rsi:.1f}) - Potential reversal")
                alert_score += 8
            elif rsi <= 30:
                critical_signals.append(f"📉 RSI oversold ({rsi:.1f}) - Watch for bounce")
                alert_score += 5
            elif rsi >= 75:
                critical_signals.append(f"📈 RSI EXTREME OVERBOUGHT ({rsi:.1f}) - Potential correction")
                alert_score += 8
            elif rsi >= 70:
                critical_signals.append(f"📈 RSI overbought ({rsi:.1f}) - Caution advised")
                alert_score += 5
            
            # MACD critical conditions
            if macd["trend"] == "BULLISH" and macd["histogram"] > 5:
                critical_signals.append(f"🚀 Strong MACD bullish momentum (H:{macd['histogram']:.1f})")
                alert_score += 6
            elif macd["trend"] == "BEARISH" and macd["histogram"] < -5:
                critical_signals.append(f"💥 Strong MACD bearish momentum (H:{macd['histogram']:.1f})")
                alert_score += 6
            
            # MACD crossover detection (recent trend change)
            if len(prices) >= 2:
                prev_macd = TechnicalIndicators.calculate_macd(prices[:-1])
                if (prev_macd["trend"] != macd["trend"] and 
                    macd["trend"] in ["BULLISH", "BEARISH"]):
                    direction = "🟢 BULLISH" if macd["trend"] == "BULLISH" else "🔴 BEARISH"
                    critical_signals.append(f"⚡ Fresh MACD {direction} crossover")
                    alert_score += 7
            
            # Bollinger Bands squeeze/expansion
            bb_width = bollinger["upper"] - bollinger["lower"]
            bb_width_pct = (bb_width / bollinger["middle"]) * 100
            
            if bb_width_pct < 3:  # Very tight bands
                critical_signals.append(f"🎯 Bollinger Bands SQUEEZE ({bb_width_pct:.1f}%) - Breakout imminent")
                alert_score += 6
            elif bollinger["position"] == "OVERBOUGHT" and current_price > bollinger["upper"]:
                critical_signals.append(f"⚡ Price ABOVE Bollinger Upper Band - Extreme condition")
                alert_score += 5
            elif bollinger["position"] == "OVERSOLD" and current_price < bollinger["lower"]:
                critical_signals.append(f"⚡ Price BELOW Bollinger Lower Band - Extreme condition")
                alert_score += 5
            
            # Multi-indicator confluence
            if rsi <= 30 and bollinger["position"] == "OVERSOLD" and macd["trend"] == "BULLISH":
                critical_signals.append("💎 TRIPLE BULLISH CONFLUENCE - RSI+Bollinger+MACD")
                alert_score += 9
            elif rsi >= 70 and bollinger["position"] == "OVERBOUGHT" and macd["trend"] == "BEARISH":
                critical_signals.append("⚠️ TRIPLE BEARISH CONFLUENCE - RSI+Bollinger+MACD")
                alert_score += 9
            
            # Volume confirmation (if available)
            if volumes and len(volumes) >= 2:
                recent_volume = sum(volumes[-3:]) / 3  # 3-period average
                prev_volume = sum(volumes[-6:-3]) / 3   # Previous 3-period average
                volume_increase = (recent_volume / prev_volume - 1) * 100
                
                if volume_increase > 50 and macd["trend"] == "BULLISH":
                    critical_signals.append(f"🔥 Volume surge (+{volume_increase:.0f}%) + Bullish MACD")
                    alert_score += 4
                elif volume_increase > 50 and macd["trend"] == "BEARISH":
                    critical_signals.append(f"💥 Volume surge (+{volume_increase:.0f}%) + Bearish MACD")
                    alert_score += 4
            
            # Determine if alert should be shown
            show_alert = alert_score >= 7  # Threshold for indicator alerts
            
            return {
                "show_alert": show_alert,
                "alert_score": alert_score,
                "critical_signals": critical_signals,
                "indicator_data": {
                    "rsi": rsi,
                    "macd": macd,
                    "bollinger": bollinger,
                    "bb_width_pct": bb_width_pct,
                    "current_price": current_price
                }
            }
            
        except Exception as e:
            return {"show_alert": False, "reason": f"Error checking indicator alerts: {e}"}
    
    @staticmethod
    def get_critical_indicator_alert_text(alert_data: Dict) -> str:
        """Generate formatted alert text for critical technical indicator activity"""
        
        if not alert_data.get("show_alert"):
            return ""
        
        signals = alert_data.get("critical_signals", [])
        indicator_data = alert_data.get("indicator_data", {})
        
        alert_text = f"\n📊 CRITICAL TECHNICAL INDICATOR ALERT\n"
        alert_text += "=" * 50 + "\n"
        
        for signal in signals:
            alert_text += f"{signal}\n"
        
        alert_text += f"\nRSI: {indicator_data.get('rsi', 0):.1f}"
        alert_text += f"\nMACD: {indicator_data.get('macd', {}).get('trend', 'Unknown')}"
        alert_text += f"\nBollinger: {indicator_data.get('bollinger', {}).get('position', 'Unknown')}"
        
        alert_text += f"\n\nAlert Score: {alert_data.get('alert_score', 0)}/20"
        alert_text += "\n💡 Consider: Check technical analysis for confirmation"
        alert_text += "\n" + "=" * 50
        
        return alert_text


# Example usage
if __name__ == "__main__":
    # Test data
    test_prices = [100, 102, 101, 105, 107, 106, 109, 111, 108, 112, 115, 113, 117, 119, 116, 120]
    
    indicators = TechnicalIndicators()
    
    print("Technical Indicators Test:")
    print(f"RSI: {indicators.calculate_rsi(test_prices)}")
    print(f"MACD: {indicators.calculate_macd(test_prices)}")
    print(f"Bollinger: {indicators.calculate_bollinger(test_prices)}")
