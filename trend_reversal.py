#!/usr/bin/env python3
"""
Trend Reversal Detection Module
Analyzes classic reversal patterns and signals across multiple timeframes
"""

import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time


class TrendReversalDetector:
    """Detects trend reversal patterns and signals"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        
        # Reversal pattern detection parameters
        self.min_volume_ratio = 1.5  # Minimum volume increase for confirmation
        self.divergence_lookback = 14  # Periods to look back for divergences
        self.pattern_confirmation_periods = 3  # Periods for pattern confirmation
        
        # Alert thresholds for reversal signals
        self.alert_thresholds = {
            "strong_reversal": 15,      # Score threshold for strong reversal
            "moderate_reversal": 10,    # Score threshold for moderate reversal
            "multiple_timeframes": 3,   # Number of timeframes showing reversal
            "high_conviction": 20       # Score for very high conviction reversal
        }
        
    def fetch_klines_data(self, interval: str = "1d", limit: int = 100) -> List:
        """Fetch historical klines data"""
        try:
            params = {
                "symbol": "BNBUSDT",
                "interval": interval,
                "limit": min(limit, 1000)
            }
            
            response = requests.get(f"{self.base_url}/klines", params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error fetching klines: {e}")
            return []
    
    def process_klines_data(self, klines: List) -> Dict:
        """Process klines into OHLCV format"""
        if not klines:
            return {}
        
        data = {
            "timestamps": [datetime.fromtimestamp(k[0]/1000) for k in klines],
            "opens": [float(k[1]) for k in klines],
            "highs": [float(k[2]) for k in klines],
            "lows": [float(k[3]) for k in klines],
            "closes": [float(k[4]) for k in klines],
            "volumes": [float(k[5]) for k in klines]
        }
        
        return data
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """Calculate RSI for divergence detection"""
        if len(prices) < period + 1:
            return []
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        rsi_values = []
        
        for i in range(period, len(deltas)):
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
            
            # Update averages for next iteration
            if i < len(deltas) - 1:
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        return rsi_values
    
    def detect_candlestick_patterns(self, data: Dict) -> Dict:
        """Detect classic candlestick reversal patterns"""
        
        if len(data.get("opens", [])) < 3:
            return {"patterns": [], "signals": []}
        
        opens = data["opens"]
        highs = data["highs"] 
        lows = data["lows"]
        closes = data["closes"]
        volumes = data["volumes"]
        
        patterns = []
        reversal_signals = []
        
        # Analyze last few candles for patterns
        for i in range(2, min(len(closes), 10)):  # Check last 10 candles
            curr_idx = -(i + 1)
            
            o, h, l, c = opens[curr_idx], highs[curr_idx], lows[curr_idx], closes[curr_idx]
            body_size = abs(c - o)
            upper_shadow = h - max(o, c)
            lower_shadow = min(o, c) - l
            candle_range = h - l
            
            # Avoid division by zero
            if candle_range == 0:
                continue
            
            # 1. DOJI Pattern (indecision)
            if body_size < (candle_range * 0.1):  # Body is less than 10% of range
                patterns.append({
                    "pattern": "DOJI",
                    "type": "INDECISION",
                    "strength": "MODERATE",
                    "position": i,
                    "description": "Market indecision - potential reversal"
                })
            
            # 2. HAMMER Pattern (bullish reversal)
            if (lower_shadow > body_size * 2 and  # Long lower shadow
                upper_shadow < body_size * 0.3 and  # Small upper shadow
                c < o):  # Red candle at bottom
                
                patterns.append({
                    "pattern": "HAMMER", 
                    "type": "BULLISH_REVERSAL",
                    "strength": "STRONG",
                    "position": i,
                    "description": "Hammer - strong bullish reversal signal"
                })
                reversal_signals.append("BULLISH")
            
            # 3. SHOOTING STAR Pattern (bearish reversal)
            if (upper_shadow > body_size * 2 and  # Long upper shadow
                lower_shadow < body_size * 0.3 and  # Small lower shadow  
                c < o):  # Red candle at top
                
                patterns.append({
                    "pattern": "SHOOTING_STAR",
                    "type": "BEARISH_REVERSAL", 
                    "strength": "STRONG",
                    "position": i,
                    "description": "Shooting Star - strong bearish reversal signal"
                })
                reversal_signals.append("BEARISH")
            
            # 4. ENGULFING PATTERNS (need previous candle)
            if i < len(closes) - 1:
                prev_idx = curr_idx + 1
                prev_o, prev_c = opens[prev_idx], closes[prev_idx]
                
                # Bullish Engulfing
                if (prev_c < prev_o and  # Previous candle was red
                    c > o and           # Current candle is green
                    o < prev_c and      # Current open below previous close
                    c > prev_o):        # Current close above previous open
                    
                    patterns.append({
                        "pattern": "BULLISH_ENGULFING",
                        "type": "BULLISH_REVERSAL",
                        "strength": "VERY_STRONG", 
                        "position": i,
                        "description": "Bullish Engulfing - very strong reversal"
                    })
                    reversal_signals.append("BULLISH")
                
                # Bearish Engulfing  
                elif (prev_c > prev_o and  # Previous candle was green
                      c < o and           # Current candle is red
                      o > prev_c and      # Current open above previous close
                      c < prev_o):        # Current close below previous open
                    
                    patterns.append({
                        "pattern": "BEARISH_ENGULFING",
                        "type": "BEARISH_REVERSAL",
                        "strength": "VERY_STRONG",
                        "position": i, 
                        "description": "Bearish Engulfing - very strong reversal"
                    })
                    reversal_signals.append("BEARISH")
        
        return {
            "patterns": patterns,
            "signals": reversal_signals,
            "pattern_count": len(patterns),
            "reversal_signals": len(reversal_signals)
        }
    
    def detect_technical_divergences(self, data: Dict) -> Dict:
        """Detect technical indicator divergences"""
        
        if len(data.get("closes", [])) < 20:
            return {"divergences": [], "signals": []}
        
        closes = data["closes"]
        volumes = data["volumes"]
        
        # Calculate RSI
        rsi_values = self.calculate_rsi(closes)
        if len(rsi_values) < 10:
            return {"divergences": [], "signals": []}
        
        divergences = []
        reversal_signals = []
        
        # Look for RSI divergences in recent periods
        recent_periods = min(10, len(rsi_values))
        
        for i in range(5, recent_periods):
            price_segment = closes[-(i+5):-i] if i > 0 else closes[-10:]
            rsi_segment = rsi_values[-(i+5):-i] if i > 0 else rsi_values[-10:]
            
            if len(price_segment) < 5 or len(rsi_segment) < 5:
                continue
            
            # Check for bullish divergence (price makes lower low, RSI makes higher low)
            price_trend = (price_segment[-1] - price_segment[0]) / price_segment[0]
            rsi_trend = rsi_segment[-1] - rsi_segment[0]
            
            if price_trend < -0.02 and rsi_trend > 2:  # Price down 2%+, RSI up 2+ points
                divergences.append({
                    "type": "BULLISH_DIVERGENCE",
                    "indicator": "RSI",
                    "strength": "STRONG",
                    "description": "RSI Bullish Divergence - price down, RSI up",
                    "price_change": price_trend * 100,
                    "rsi_change": rsi_trend
                })
                reversal_signals.append("BULLISH")
            
            # Check for bearish divergence (price makes higher high, RSI makes lower high)
            elif price_trend > 0.02 and rsi_trend < -2:  # Price up 2%+, RSI down 2+ points
                divergences.append({
                    "type": "BEARISH_DIVERGENCE", 
                    "indicator": "RSI",
                    "strength": "STRONG",
                    "description": "RSI Bearish Divergence - price up, RSI down",
                    "price_change": price_trend * 100,
                    "rsi_change": rsi_trend
                })
                reversal_signals.append("BEARISH")
        
        # Volume divergence
        if len(volumes) >= 10:
            recent_volumes = volumes[-10:]
            recent_prices = closes[-10:]
            
            volume_trend = (recent_volumes[-1] - recent_volumes[0]) / recent_volumes[0]
            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            # Volume divergence (price up but volume down, or vice versa)
            if abs(price_trend) > 0.03 and volume_trend * price_trend < -0.1:
                divergences.append({
                    "type": "VOLUME_DIVERGENCE",
                    "indicator": "VOLUME", 
                    "strength": "MODERATE",
                    "description": f"Volume divergence - price trend not confirmed by volume",
                    "price_change": price_trend * 100,
                    "volume_change": volume_trend * 100
                })
        
        return {
            "divergences": divergences,
            "signals": reversal_signals,
            "divergence_count": len(divergences)
        }
    
    def analyze_support_resistance_breaks(self, data: Dict) -> Dict:
        """Analyze support/resistance level breaks"""
        
        if len(data.get("closes", [])) < 20:
            return {"breaks": [], "signals": []}
        
        closes = data["closes"]
        highs = data["highs"]
        lows = data["lows"]
        volumes = data["volumes"]
        
        breaks = []
        reversal_signals = []
        
        # Find recent significant levels
        recent_data = 20
        recent_highs = highs[-recent_data:]
        recent_lows = lows[-recent_data:]
        recent_closes = closes[-recent_data:]
        recent_volumes = volumes[-recent_data:]
        
        # Identify key levels
        resistance_level = max(recent_highs[:-3])  # Exclude last 3 for break detection
        support_level = min(recent_lows[:-3])
        
        current_price = closes[-1]
        current_volume = volumes[-1]
        avg_volume = np.mean(recent_volumes[:-1])
        
        # Check for resistance break (bullish)
        if (current_price > resistance_level and
            current_volume > avg_volume * 1.2):  # With volume confirmation
            
            breaks.append({
                "type": "RESISTANCE_BREAK",
                "direction": "BULLISH",
                "level": resistance_level,
                "current_price": current_price,
                "break_percentage": ((current_price - resistance_level) / resistance_level) * 100,
                "volume_confirmation": True,
                "strength": "STRONG"
            })
            reversal_signals.append("BULLISH")
        
        # Check for support break (bearish)
        elif (current_price < support_level and
              current_volume > avg_volume * 1.2):  # With volume confirmation
            
            breaks.append({
                "type": "SUPPORT_BREAK", 
                "direction": "BEARISH",
                "level": support_level,
                "current_price": current_price,
                "break_percentage": ((support_level - current_price) / support_level) * 100,
                "volume_confirmation": True,
                "strength": "STRONG"
            })
            reversal_signals.append("BEARISH")
        
        return {
            "breaks": breaks,
            "signals": reversal_signals,
            "resistance_level": resistance_level,
            "support_level": support_level,
            "current_price": current_price
        }
    
    def calculate_reversal_score(self, patterns: Dict, divergences: Dict, breaks: Dict) -> Dict:
        """Calculate overall reversal probability score"""
        
        total_score = 0
        bullish_signals = 0
        bearish_signals = 0
        signal_details = []
        
        # Score candlestick patterns
        for pattern in patterns.get("patterns", []):
            if pattern["strength"] == "VERY_STRONG":
                score = 8
            elif pattern["strength"] == "STRONG":
                score = 5
            else:
                score = 2
                
            total_score += score
            
            if "BULLISH" in pattern["type"]:
                bullish_signals += score
            elif "BEARISH" in pattern["type"]:
                bearish_signals += score
            
            signal_details.append(f"{pattern['pattern']} ({score} points)")
        
        # Score divergences
        for div in divergences.get("divergences", []):
            if div["strength"] == "STRONG":
                score = 6
            else:
                score = 3
                
            total_score += score
            
            if "BULLISH" in div["type"]:
                bullish_signals += score
            elif "BEARISH" in div["type"]:
                bearish_signals += score
            
            signal_details.append(f"{div['type']} ({score} points)")
        
        # Score support/resistance breaks
        for break_signal in breaks.get("breaks", []):
            if break_signal["volume_confirmation"]:
                score = 7
            else:
                score = 4
                
            total_score += score
            
            if break_signal["direction"] == "BULLISH":
                bullish_signals += score
            else:
                bearish_signals += score
            
            signal_details.append(f"{break_signal['type']} ({score} points)")
        
        # Determine overall direction
        if bullish_signals > bearish_signals + 3:
            direction = "BULLISH_REVERSAL"
            conviction = "HIGH" if total_score >= 15 else "MODERATE" if total_score >= 10 else "LOW"
        elif bearish_signals > bullish_signals + 3:
            direction = "BEARISH_REVERSAL"
            conviction = "HIGH" if total_score >= 15 else "MODERATE" if total_score >= 10 else "LOW"
        else:
            direction = "MIXED_SIGNALS"
            conviction = "LOW"
        
        return {
            "total_score": total_score,
            "bullish_score": bullish_signals,
            "bearish_score": bearish_signals,
            "direction": direction,
            "conviction": conviction,
            "signal_details": signal_details,
            "show_alert": total_score >= self.alert_thresholds["moderate_reversal"]
        }
    
    def analyze_timeframe_reversal(self, interval: str, limit: int, period_name: str) -> Dict:
        """Analyze reversal signals for a specific timeframe"""
        
        print(f"\n📊 Analyzing {period_name} ({interval})...")
        
        # Fetch data
        klines = self.fetch_klines_data(interval, limit)
        if not klines:
            return {"error": f"No data for {period_name}"}
        
        data = self.process_klines_data(klines)
        if not data:
            return {"error": f"Failed to process data for {period_name}"}
        
        # Detect patterns and signals
        patterns = self.detect_candlestick_patterns(data)
        divergences = self.detect_technical_divergences(data)
        breaks = self.analyze_support_resistance_breaks(data)
        
        # Calculate overall score
        reversal_analysis = self.calculate_reversal_score(patterns, divergences, breaks)
        
        return {
            "timeframe": interval,
            "period_name": period_name,
            "patterns": patterns,
            "divergences": divergences,
            "breaks": breaks,
            "reversal_analysis": reversal_analysis,
            "current_price": data["closes"][-1] if data["closes"] else 0,
            "analysis_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def multi_timeframe_reversal_analysis(self) -> Dict:
        """Analyze reversal signals across multiple timeframes"""
        
        print(f"\n🔄 MULTI-TIMEFRAME TREND REVERSAL ANALYSIS")
        print("=" * 60)
        
        timeframes = [
            ("1d", 7, "Last Week"),
            ("1d", 14, "Last 2 Weeks"), 
            ("1d", 30, "Last Month"),
            ("1w", 12, "Last 3 Months")
        ]
        
        results = {}
        overall_signals = {"bullish": 0, "bearish": 0, "mixed": 0}
        high_conviction_signals = []
        
        for interval, limit, period_name in timeframes:
            
            result = self.analyze_timeframe_reversal(interval, limit, period_name)
            
            if "error" not in result:
                results[period_name] = result
                
                # Track overall signals
                direction = result["reversal_analysis"]["direction"]
                conviction = result["reversal_analysis"]["conviction"]
                
                if "BULLISH" in direction:
                    overall_signals["bullish"] += 1
                    if conviction == "HIGH":
                        high_conviction_signals.append(f"{period_name}: BULLISH")
                elif "BEARISH" in direction:
                    overall_signals["bearish"] += 1
                    if conviction == "HIGH":
                        high_conviction_signals.append(f"{period_name}: BEARISH")
                else:
                    overall_signals["mixed"] += 1
                
                # Display results
                self.display_timeframe_results(result)
            else:
                print(f"❌ {result['error']}")
            
            time.sleep(0.1)  # Rate limiting
        
        # Overall assessment
        self.display_overall_reversal_assessment(overall_signals, high_conviction_signals)
        
        return {
            "timeframe_results": results,
            "overall_signals": overall_signals,
            "high_conviction_signals": high_conviction_signals,
            "analysis_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def display_timeframe_results(self, result: Dict):
        """Display reversal analysis results for a timeframe"""
        
        period = result["period_name"]
        reversal = result["reversal_analysis"]
        patterns = result["patterns"]
        divergences = result["divergences"]
        breaks = result["breaks"]
        
        print(f"\n📈 {period}:")
        print("-" * 30)
        print(f"Current Price: ${result['current_price']:.2f}")
        print(f"Reversal Score: {reversal['total_score']} points")
        print(f"Direction: {reversal['direction']}")
        print(f"Conviction: {reversal['conviction']}")
        
        if patterns["patterns"]:
            print(f"\n🕯️ Candlestick Patterns ({len(patterns['patterns'])}):")
            for pattern in patterns["patterns"][-3:]:  # Show last 3
                print(f"   • {pattern['pattern']} - {pattern['description']}")
        
        if divergences["divergences"]:
            print(f"\n📊 Technical Divergences ({len(divergences['divergences'])}):")
            for div in divergences["divergences"]:
                print(f"   • {div['description']}")
        
        if breaks["breaks"]:
            print(f"\n💥 Support/Resistance Breaks:")
            for break_signal in breaks["breaks"]:
                print(f"   • {break_signal['type']}: ${break_signal['level']:.2f} ({break_signal['break_percentage']:+.1f}%)")
        
        if reversal["show_alert"]:
            print(f"\n⚠️ REVERSAL ALERT: {reversal['conviction']} conviction signal!")
    
    def display_overall_reversal_assessment(self, signals: Dict, high_conviction: List):
        """Display overall reversal assessment"""
        
        print(f"\n🎯 OVERALL REVERSAL ASSESSMENT")
        print("=" * 40)
        print(f"Bullish Timeframes: {signals['bullish']}")
        print(f"Bearish Timeframes: {signals['bearish']}")
        print(f"Mixed Signals: {signals['mixed']}")
        
        if high_conviction:
            print(f"\n🚨 HIGH CONVICTION SIGNALS:")
            for signal in high_conviction:
                print(f"   • {signal}")
        
        # Overall recommendation
        if signals["bullish"] >= 3:
            print(f"\n🟢 OVERALL TREND: BULLISH REVERSAL LIKELY")
            print(f"💡 Recommendation: Consider long positions")
        elif signals["bearish"] >= 3:
            print(f"\n🔴 OVERALL TREND: BEARISH REVERSAL LIKELY") 
            print(f"💡 Recommendation: Consider short positions or exit longs")
        else:
            print(f"\n🟡 OVERALL TREND: MIXED SIGNALS")
            print(f"💡 Recommendation: Wait for clearer confirmation")
        
        print(f"\n⏰ Analysis completed at {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
    
    def check_critical_reversal_alerts(self) -> Dict:
        """Check for critical reversal signals that warrant immediate attention"""
        
        try:
            # Quick analysis of recent data
            result = self.analyze_timeframe_reversal("1d", 14, "Recent 2 Weeks")
            
            if "error" in result:
                return {"show_alert": False, "reason": result["error"]}
            
            reversal = result["reversal_analysis"]
            critical_signals = []
            alert_score = reversal["total_score"]
            
            # Check for high conviction reversal
            if reversal["conviction"] == "HIGH":
                critical_signals.append(f"🚨 HIGH CONVICTION {reversal['direction']}")
                alert_score += 5
            
            # Check for multiple pattern confirmations
            pattern_count = result["patterns"]["pattern_count"]
            divergence_count = result["divergences"]["divergence_count"] 
            break_count = len(result["breaks"]["breaks"])
            
            total_signals = pattern_count + divergence_count + break_count
            if total_signals >= 3:
                critical_signals.append(f"📊 MULTIPLE CONFIRMATIONS ({total_signals} signals)")
                alert_score += 3
            
            # Check for strong patterns
            for pattern in result["patterns"]["patterns"]:
                if pattern["strength"] in ["STRONG", "VERY_STRONG"]:
                    critical_signals.append(f"🕯️ {pattern['pattern']} detected")
            
            # Show alert if score is high enough
            show_alert = alert_score >= self.alert_thresholds["moderate_reversal"]
            
            return {
                "show_alert": show_alert,
                "alert_score": alert_score,
                "critical_signals": critical_signals,
                "reversal_data": {
                    "direction": reversal["direction"],
                    "conviction": reversal["conviction"],
                    "total_score": reversal["total_score"],
                    "current_price": result["current_price"]
                }
            }
            
        except Exception as e:
            return {"show_alert": False, "reason": f"Error checking reversal alerts: {e}"}
    
    def get_critical_reversal_alert_text(self, alert_data: Dict) -> str:
        """Format reversal alert message for display"""
        
        if not alert_data.get("show_alert"):
            return ""
        
        reversal_data = alert_data.get("reversal_data", {})
        direction = reversal_data.get("direction", "UNKNOWN")
        conviction = reversal_data.get("conviction", "LOW")
        score = reversal_data.get("total_score", 0)
        
        alert_text = f"🔄 TREND REVERSAL ALERT ({conviction} conviction)\n"
        alert_text += f"   Direction: {direction}\n"
        alert_text += f"   Score: {score}/25\n"
        
        for signal in alert_data.get("critical_signals", []):
            alert_text += f"   {signal}\n"
        
        return alert_text


if __name__ == "__main__":
    # Test the reversal detector
    detector = TrendReversalDetector()
    results = detector.multi_timeframe_reversal_analysis()
