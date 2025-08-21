#!/usr/bin/env python3
"""
Elliott Wave Analysis Module
Contains Elliott Wave pattern detection and analysis
"""

from typing import Dict, List


class ElliottWaveAnalyzer:
    """Class for Elliott Wave pattern analysis"""
    
    def __init__(self):
        self.wave_descriptions = {
            1: "Wave 1 - Initial impulse",
            2: "Wave 2 - Pullback (buy opportunity)",
            3: "Wave 3 - Strongest trend",
            4: "Wave 4 - Consolidation",
            5: "Wave 5 - Final push (prepare for reversal)"
        }
    
    def find_pivot_points(self, prices: List[float], lookback: int = 2) -> List[Dict]:
        """Find local highs and lows (pivot points)"""
        if len(prices) < lookback * 2 + 1:
            return []
        
        pivots = []
        
        for i in range(lookback, len(prices) - lookback):
            # Check for local high
            if (prices[i] > max(prices[i-lookback:i]) and 
                prices[i] > max(prices[i+1:i+lookback+1])):
                pivots.append({
                    "type": "HIGH",
                    "price": prices[i],
                    "index": i
                })
            
            # Check for local low
            elif (prices[i] < min(prices[i-lookback:i]) and 
                  prices[i] < min(prices[i+1:i+lookback+1])):
                pivots.append({
                    "type": "LOW",
                    "price": prices[i],
                    "index": i
                })
        
        return pivots
    
    def count_waves(self, pivots: List[Dict]) -> int:
        """Count wave patterns from pivot points"""
        if len(pivots) < 2:
            return 1
        
        # Simple wave counting based on pivot alternation
        wave_count = 1
        
        for i in range(1, len(pivots)):
            if pivots[i]["type"] != pivots[i-1]["type"]:
                wave_count += 1
        
        return min(wave_count, 5)  # Max 5 waves in impulse
    
    def detect_trend_direction(self, pivots: List[Dict], prices: List[float]) -> str:
        """Determine if we're in uptrend or downtrend"""
        if len(pivots) < 2:
            return "SIDEWAYS"
        
        # Compare first and last significant pivots
        first_pivot = pivots[0]
        last_pivot = pivots[-1]
        
        # Also check current price vs starting price
        price_trend = prices[-1] > prices[0]
        
        if last_pivot["price"] > first_pivot["price"] and price_trend:
            return "UPTREND"
        elif last_pivot["price"] < first_pivot["price"] and not price_trend:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
    
    def calculate_fibonacci_projections(self, pivots: List[Dict]) -> Dict:
        """Calculate potential wave targets using Fibonacci ratios"""
        if len(pivots) < 3:
            return {"target": None, "extension": None}
        
        # Use last 3 pivots for projection
        recent_pivots = pivots[-3:]
        
        # Calculate wave 1 length
        wave1_length = abs(recent_pivots[1]["price"] - recent_pivots[0]["price"])
        
        # Common Fibonacci projections
        projections = {
            "0.618": recent_pivots[-1]["price"] + (wave1_length * 0.618),
            "1.0": recent_pivots[-1]["price"] + wave1_length,
            "1.618": recent_pivots[-1]["price"] + (wave1_length * 1.618)
        }
        
        return {
            "target": round(projections["1.0"], 2),
            "extension": round(projections["1.618"], 2),
            "all_levels": {k: round(v, 2) for k, v in projections.items()}
        }
    
    def detect_elliott_wave(self, prices: List[float]) -> Dict:
        """Main Elliott Wave detection function"""
        if len(prices) < 20:
            return {
                "wave": "INSUFFICIENT_DATA",
                "description": "Need more data for Elliott Wave analysis",
                "confidence": 0
            }
        
        # Find pivot points
        pivots = self.find_pivot_points(prices, lookback=2)
        
        if len(pivots) < 3:
            return {
                "wave": "FORMING",
                "description": "Pattern still forming - need more pivots",
                "confidence": 20
            }
        
        # Determine trend and wave count
        trend = self.detect_trend_direction(pivots, prices)
        wave_count = self.count_waves(pivots)
        
        # Calculate projections
        projections = self.calculate_fibonacci_projections(pivots)
        
        # Analyze recent pattern for specific wave identification
        if len(pivots) >= 5:
            recent_pivots = pivots[-5:]
            
            # Check for completed 5-wave pattern
            if (trend == "UPTREND" and 
                recent_pivots[0]["type"] == "LOW" and 
                recent_pivots[-1]["type"] == "HIGH"):
                
                if recent_pivots[-1]["price"] > recent_pivots[0]["price"]:
                    return {
                        "wave": "WAVE_5",
                        "description": "Possible end of 5-wave impulse - prepare for correction",
                        "target": projections.get("target"),
                        "confidence": 75,
                        "trend": trend,
                        "next_move": "CORRECTION_EXPECTED"
                    }
            
            # Check for correction pattern
            elif (trend == "DOWNTREND" and 
                  recent_pivots[0]["type"] == "HIGH" and 
                  recent_pivots[-1]["type"] == "LOW"):
                
                return {
                    "wave": "CORRECTION",
                    "description": "ABC correction in progress",
                    "target": projections.get("target"),
                    "confidence": 65,
                    "trend": trend,
                    "next_move": "IMPULSE_EXPECTED"
                }
        
        # Default wave analysis
        wave_number = min(wave_count % 5 + 1, 5) if wave_count > 0 else 1
        
        # Adjust confidence based on pattern clarity
        confidence = min(50 + (len(pivots) * 5), 90)
        
        return {
            "wave": f"WAVE_{wave_number}",
            "description": self.wave_descriptions.get(wave_number, "Unknown wave"),
            "target": projections.get("target"),
            "confidence": confidence,
            "trend": trend,
            "pivot_count": len(pivots),
            "projections": projections.get("all_levels", {})
        }
    
    def get_wave_signals(self, prices: List[float]) -> Dict:
        """Get trading signals based on Elliott Wave analysis"""
        wave_data = self.detect_elliott_wave(prices)
        
        signals = {
            "action": "WAIT",
            "confidence": wave_data.get("confidence", 0),
            "reason": "Elliott Wave analysis",
            "wave_info": wave_data
        }
        
        wave = wave_data.get("wave", "")
        
        # Wave-based signals
        if "WAVE_2" in wave:
            signals.update({
                "action": "BUY",
                "reason": "Wave 2 pullback - good entry point",
                "confidence": min(wave_data.get("confidence", 0) + 10, 95)
            })
        
        elif "WAVE_3" in wave:
            signals.update({
                "action": "HOLD",
                "reason": "Wave 3 - strongest trend continuation",
                "confidence": wave_data.get("confidence", 0)
            })
        
        elif "WAVE_5" in wave:
            signals.update({
                "action": "PREPARE_SELL",
                "reason": "Wave 5 - trend may be ending soon",
                "confidence": wave_data.get("confidence", 0)
            })
        
        elif "CORRECTION" in wave:
            signals.update({
                "action": "WAIT",
                "reason": "Correction in progress - wait for completion",
                "confidence": wave_data.get("confidence", 0)
            })
        
        return signals


# Example usage
if __name__ == "__main__":
    # Test with sample price data
    test_prices = [
        100, 105, 102, 108, 104, 112, 109, 116, 113, 120, 
        117, 115, 119, 122, 118, 125, 121, 128, 124, 130
    ]
    
    analyzer = ElliottWaveAnalyzer()
    result = analyzer.detect_elliott_wave(test_prices)
    signals = analyzer.get_wave_signals(test_prices)
    
    print("Elliott Wave Analysis Test:")
    print(f"Wave: {result['wave']}")
    print(f"Description: {result['description']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Signal: {signals['action']} - {signals['reason']}")
