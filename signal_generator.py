#!/usr/bin/env python3
"""
Signal Generator Module
Contains trading signal logic and scoring system
"""

from datetime import datetime
from typing import Dict, List
from indicators import TechnicalIndicators
from elliott_wave import ElliottWaveAnalyzer
from fib import FibonacciAnalyzer
from correlation_module import CorrelationAnalyzer


class TradingSignalGenerator:
    """Class for generating comprehensive trading signals"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.elliott = ElliottWaveAnalyzer()
        self.fibonacci = FibonacciAnalyzer()
        self.correlation = CorrelationAnalyzer()
        
        # Predefined support and resistance levels
        self.support_levels = [732, 680, 644, 600, 550, 519]
        self.resistance_levels = [869, 850, 820, 800, 780]
        
        # Historical cycle data
        self.cycles = [
            {"bottom": 363, "top": 641, "next_bottom": 408, "date": "2024-03"},
            {"bottom": 408, "top": 793, "next_bottom": 519, "date": "2024-08"},
            {"bottom": 519, "top": 869, "next_bottom": None, "date": "2025-02"}
        ]
    
    def calculate_bull_bear_score(self, current_price: float, prices: List[float], 
                                  volumes: List[float]) -> Dict:
        """Calculate bullish and bearish scores based on multiple factors"""
        
        # Calculate all indicators
        rsi = self.indicators.calculate_rsi(prices)
        macd = self.indicators.calculate_macd(prices)
        bollinger = self.indicators.calculate_bollinger(prices)
        elliott = self.elliott.detect_elliott_wave(prices)
        fibonacci = self.fibonacci.get_fibonacci_signals(current_price)
        
        bull_score = 0
        bear_score = 0
        
        # Price level analysis
        if current_price < 650:
            bull_score += 3
        elif current_price > 850:
            bear_score += 3
        
        # Support/Resistance proximity
        closest_support = min(self.support_levels, key=lambda x: abs(x - current_price))
        closest_resistance = min(self.resistance_levels, key=lambda x: abs(x - current_price))
        
        if abs(current_price - closest_support) < 20:
            bull_score += 2
        if abs(current_price - closest_resistance) < 20:
            bear_score += 2
        
        # RSI analysis
        if rsi < 30:
            bull_score += 2
        elif rsi < 40:
            bull_score += 1
        elif rsi > 70:
            bear_score += 2
        elif rsi > 60:
            bear_score += 1
        
        # MACD analysis
        if macd["trend"] == "BULLISH":
            bull_score += 2
            if macd["histogram"] > 0:
                bull_score += 1
        elif macd["trend"] == "BEARISH":
            bear_score += 2
            if macd["histogram"] < 0:
                bear_score += 1
        
        # Bollinger Bands analysis
        if bollinger["position"] == "OVERSOLD":
            bull_score += 2
        elif bollinger["position"] == "OVERBOUGHT":
            bear_score += 2
        
        # Elliott Wave analysis
        wave = elliott.get("wave", "")
        if "5" in wave:
            bear_score += 2
        elif "2" in wave:
            bull_score += 2
        elif "3" in wave:
            bull_score += 1
        
        # Volume trend analysis
        if len(volumes) >= 5:
            recent_vol = sum(volumes[-5:]) / 5
            avg_vol = sum(volumes) / len(volumes)
            
            if recent_vol > avg_vol * 1.2:
                if prices[-1] > prices[-2]:
                    bull_score += 1
                else:
                    bear_score += 1
        
        # Fibonacci signals
        fib_action = fibonacci.get("action", "WAIT")
        if fib_action == "BUY":
            bull_score += 2
        elif fib_action == "STRONG_BUY":
            bull_score += 3
        elif fib_action == "SELL":
            bear_score += 2
        elif fib_action == "STRONG_SELL":
            bear_score += 3
        
        # Golden pocket bonus
        if fibonacci.get("golden_pocket"):
            bull_score += 2
        
        # Correlation analysis (lightweight version for scoring)
        try:
            correlation_data = self.correlation.run_correlation_analysis("1h", 50)
            if correlation_data and "signals" in correlation_data:
                corr_signals = correlation_data["signals"]
                correlation_score = corr_signals.get("correlation_score", 0)
                
                # Add correlation score to appropriate side
                if correlation_score > 0:
                    bull_score += min(correlation_score, 2)  # Cap at +2
                elif correlation_score < 0:
                    bear_score += min(abs(correlation_score), 2)  # Cap at +2
        except:
            # If correlation analysis fails, continue without it
            correlation_data = None
        
        return {
            "bull_score": bull_score,
            "bear_score": bear_score,
            "indicators": {
                "RSI": rsi,
                "MACD": macd,
                "Bollinger": bollinger,
                "Elliott": elliott,
                "Fibonacci": fibonacci,
                "Correlation": correlation_data
            }
        }
    
    def determine_position_size(self, signal_strength: str, confidence: int) -> str:
        """Determine position size based on signal strength and confidence"""
        if confidence < 60:
            return "10% - Low confidence"
        
        position_sizes = {
            "STRONG BUY": "50% with 2x leverage" if confidence > 80 else "30% with 1.5x leverage",
            "BUY": "25% with 2x leverage" if confidence > 75 else "20% with 1x leverage",
            "STRONG SELL": "33% short position" if confidence > 80 else "20% short position",
            "SELL": "15% short position" if confidence > 75 else "10% short position",
            "WAIT": "0% - Wait for better setup"
        }
        
        return position_sizes.get(signal_strength, "10% - Conservative")
    
    def calculate_targets_and_stops(self, current_price: float, action: str, 
                                   fibonacci_data: Dict) -> Dict:
        """Calculate target and stop loss levels"""
        targets = {}
        
        if action in ["STRONG BUY", "BUY"]:
            # Bullish targets
            if fibonacci_data.get("target"):
                targets["primary_target"] = fibonacci_data["target"]
            else:
                targets["primary_target"] = round(current_price * 1.08, 2)
            
            targets["extended_target"] = round(current_price * 1.15, 2)
            targets["stop_loss"] = round(current_price * 0.95, 2)
            
            # Use closest resistance as secondary target
            resistance_above = [r for r in self.resistance_levels if r > current_price]
            if resistance_above:
                targets["resistance_target"] = min(resistance_above)
            
        elif action in ["STRONG SELL", "SELL"]:
            # Bearish targets
            if fibonacci_data.get("target"):
                targets["primary_target"] = fibonacci_data["target"]
            else:
                targets["primary_target"] = round(current_price * 0.92, 2)
            
            targets["extended_target"] = round(current_price * 0.85, 2)
            targets["stop_loss"] = round(current_price * 1.05, 2)
            
            # Use closest support as secondary target
            support_below = [s for s in self.support_levels if s < current_price]
            if support_below:
                targets["support_target"] = max(support_below)
        
        return targets
    
    def generate_comprehensive_signal(self, current_price: float, prices: List[float], 
                                    volumes: List[float], mtf_analysis: Dict) -> Dict:
        """Generate the main trading signal with all analysis"""
        
        # Calculate scores and get all indicator data
        scoring_data = self.calculate_bull_bear_score(current_price, prices, volumes)
        bull_score = scoring_data["bull_score"]
        bear_score = scoring_data["bear_score"]
        indicators = scoring_data["indicators"]
        
        # Calculate confidence based on score difference and indicator alignment
        score_diff = abs(bull_score - bear_score)
        max_score = max(bull_score, bear_score)
        confidence = min(50 + (score_diff * 10) + (max_score * 3), 95)
        
        # Determine action based on scores
        if bull_score > bear_score + 2:
            action = "STRONG BUY"
        elif bull_score > bear_score:
            action = "BUY"
        elif bear_score > bull_score + 2:
            action = "STRONG SELL"
        elif bear_score > bull_score:
            action = "SELL"
        else:
            action = "WAIT"
        
        # Calculate targets and stops
        targets = self.calculate_targets_and_stops(
            current_price, action, indicators["Fibonacci"]
        )
        
        # Determine position size
        position_size = self.determine_position_size(action, confidence)
        
        # Create comprehensive signal
        signal = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "price": current_price,
            "action": action,
            "confidence": confidence,
            "bull_score": bull_score,
            "bear_score": bear_score,
            "position_size": position_size,
            "indicators": indicators,
            "timeframes": mtf_analysis,
            "targets": targets
        }
        
        # Add specific recommendations based on action
        if action == "WAIT":
            signal["reason"] = self._get_wait_reason(indicators, bull_score, bear_score)
        else:
            signal["reason"] = self._get_action_reason(action, indicators)
        
        return signal
    
    def _get_wait_reason(self, indicators: Dict, bull_score: int, bear_score: int) -> str:
        """Generate reason for WAIT signal"""
        reasons = []
        
        if abs(bull_score - bear_score) <= 1:
            reasons.append("Mixed signals across indicators")
        
        rsi = indicators["RSI"]
        if 45 <= rsi <= 55:
            reasons.append("RSI in neutral zone")
        
        if indicators["MACD"]["trend"] == "NEUTRAL":
            reasons.append("MACD showing no clear trend")
        
        if indicators["Bollinger"]["position"] == "NEUTRAL":
            reasons.append("Price in middle of Bollinger Bands")
        
        return " | ".join(reasons) if reasons else "Wait for clearer setup"
    
    def _get_action_reason(self, action: str, indicators: Dict) -> str:
        """Generate reason for BUY/SELL signal"""
        reasons = []
        
        if action in ["BUY", "STRONG BUY"]:
            if indicators["RSI"] < 40:
                reasons.append("Oversold RSI")
            if indicators["MACD"]["trend"] == "BULLISH":
                reasons.append("MACD bullish crossover")
            if indicators["Fibonacci"].get("golden_pocket"):
                reasons.append("Golden Pocket bounce zone")
            if "2" in indicators["Elliott"].get("wave", ""):
                reasons.append("Elliott Wave 2 pullback")
        
        elif action in ["SELL", "STRONG SELL"]:
            if indicators["RSI"] > 60:
                reasons.append("Overbought RSI")
            if indicators["MACD"]["trend"] == "BEARISH":
                reasons.append("MACD bearish crossover")
            if "5" in indicators["Elliott"].get("wave", ""):
                reasons.append("Elliott Wave 5 completion")
        
        return " | ".join(reasons) if reasons else f"{action} signal generated"
    
    def get_enhanced_fibonacci_info(self, current_price: float) -> Dict:
        """Get enhanced Fibonacci information for main screen"""
        fibonacci = self.fibonacci.get_fibonacci_signals(current_price)
        
        # Enhanced Fibonacci analysis
        fib_info = {
            "action": fibonacci.get("action", "WAIT"),
            "trend": fibonacci.get("trend", "UNKNOWN"),
            "closest_level": fibonacci.get("closest_level", "N/A"),
            "distance": fibonacci.get("distance", 0),
            "golden_pocket": fibonacci.get("golden_pocket", False),
            "retracement_zone": fibonacci.get("retracement_zone", "UNKNOWN")
        }
        
        # Get detailed levels
        levels = fibonacci.get("levels", {})
        if levels:
            # Find support and resistance levels
            fib_info["support_levels"] = []
            fib_info["resistance_levels"] = []
            
            for level_name, level_price in levels.items():
                if level_price < current_price:
                    fib_info["support_levels"].append(f"{level_name}: ${level_price:.0f}")
                else:
                    fib_info["resistance_levels"].append(f"{level_name}: ${level_price:.0f}")
            
            # Limit to 3 closest levels each
            fib_info["support_levels"] = fib_info["support_levels"][:3]
            fib_info["resistance_levels"] = fib_info["resistance_levels"][:3]
        
        # Golden Pocket analysis
        if fib_info["golden_pocket"]:
            fib_info["pocket_status"] = "🟡 IN GOLDEN POCKET"
        elif fib_info["retracement_zone"] == "DEEP":
            fib_info["pocket_status"] = "🔴 BELOW GOLDEN POCKET"
        else:
            fib_info["pocket_status"] = "🟢 ABOVE GOLDEN POCKET"
        
        return fib_info
    
    def get_multi_period_elliott_waves(self) -> Dict:
        """Get Elliott Wave analysis for multiple periods"""
        periods = {
            "6_months": {"interval": "1d", "limit": 180, "description": "6 месеца"},
            "1_year": {"interval": "1w", "limit": 52, "description": "1 година"},
            "1_5_years": {"interval": "1w", "limit": 78, "description": "1.5 години"}
        }
        
        elliott_periods = {}
        
        for period_key, period_data in periods.items():
            try:
                # Get historical data (simplified - would use actual API call)
                # For demo, we'll use the visual analysis structure for longer periods
                if period_key == "1_5_years":
                    # Use visual Elliott analysis for 1.5 years
                    elliott_periods[period_key] = {
                        "description": period_data["description"],
                        "wave": "WAVE_5_COMPLETION",
                        "confidence": 95,
                        "status": "🔴 CYCLE TOP",
                        "next_move": "ABC CORRECTION",
                        "analysis": "Visual analysis - complete 5-wave cycle"
                    }
                elif period_key == "1_year":
                    # Algorithmic for 1 year (Waves 3-4-5)
                    elliott_periods[period_key] = {
                        "description": period_data["description"],
                        "wave": "WAVE_5_IN_PROGRESS",
                        "confidence": 80,
                        "status": "🟡 LATE STAGE",
                        "next_move": "WAVE_5_COMPLETION",
                        "analysis": "Waves 3-4-5 sequence"
                    }
                else:  # 6 months
                    # Recent Wave 5 development
                    elliott_periods[period_key] = {
                        "description": period_data["description"],
                        "wave": "WAVE_5_EXTENSION",
                        "confidence": 75,
                        "status": "🟢 TRENDING",
                        "next_move": "FINAL PUSH",
                        "analysis": "Wave 5 extension phase"
                    }
            except Exception as e:
                elliott_periods[period_key] = {
                    "description": period_data["description"],
                    "wave": "ERROR",
                    "confidence": 0,
                    "status": "❌ N/A",
                    "next_move": "UNKNOWN",
                    "analysis": f"Error: {e}"
                }
        
        return elliott_periods


# Example usage
if __name__ == "__main__":
    # Test signal generator
    generator = TradingSignalGenerator()
    
    # Sample data
    test_prices = [850, 852, 848, 855, 851, 857, 853, 860, 856, 863]
    test_volumes = [1000, 1200, 900, 1100, 1050, 1300, 950, 1400, 1000, 1500]
    current_price = 860
    
    signal = generator.generate_comprehensive_signal(
        current_price, test_prices, test_volumes, {}
    )
    
    print("Signal Generator Test:")
    print(f"Action: {signal['action']}")
    print(f"Confidence: {signal['confidence']}%")
    print(f"Bull Score: {signal['bull_score']}")
    print(f"Bear Score: {signal['bear_score']}")
    print(f"Reason: {signal['reason']}")
