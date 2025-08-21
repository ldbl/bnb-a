#!/usr/bin/env python3
"""
Market Configuration Module
Centralizes hardcoded values for support/resistance levels, cycles, and market structure
"""

from datetime import datetime
from typing import Dict, List


class BNBMarketConfig:
    """Centralized configuration for BNB market analysis"""
    
    # Support and Resistance Levels (Updated regularly based on market analysis)
    SUPPORT_LEVELS = [732, 680, 644, 600, 550, 519]
    RESISTANCE_LEVELS = [869, 850, 820, 800, 780]
    
    # Historical Market Cycles
    MARKET_CYCLES = [
        {"bottom": 363, "top": 641, "next_bottom": 408, "date": "2024-03"},
        {"bottom": 408, "top": 793, "next_bottom": 519, "date": "2024-08"},
        {"bottom": 519, "top": 869, "next_bottom": None, "date": "2025-02"}
    ]
    
    # Elliott Wave Visual Structure (1.5 year analysis)
    VISUAL_ELLIOTT_STRUCTURE = {
        "wave_1": {"start": 191, "end": 731, "period": "Feb-Jul 2024", "gain": 282.7},
        "wave_2": {"start": 731, "end": 380, "period": "Jul-Aug 2024", "decline": -48.0},
        "wave_3": {"start": 380, "end": 731, "period": "Aug-Nov 2024", "gain": 92.4},
        "wave_4": {"start": 731, "end": 600, "period": "Nov-Dec 2024", "decline": -17.9},
        "wave_5": {"start": 600, "period": "Dec 2024-Now"}  # Current wave
    }
    
    # Technical Analysis Configuration
    TECHNICAL_CONFIG = {
        "support_resistance_proximity": 20,  # Price distance to trigger signal
        "fibonacci_levels": {
            "retracements": [0.236, 0.382, 0.5, 0.618, 0.786],
            "extensions": [1.272, 1.414, 1.618, 2.0, 2.618]
        },
        "golden_pocket": {"min": 0.618, "max": 0.786},
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "volume_spike_threshold": 2.0  # 2x average volume
    }
    
    # Whale Activity Thresholds
    WHALE_CONFIG = {
        "large_transaction_thresholds": {
            "tier_1": 1000000,  # $1M USD
            "tier_2": 5000000,  # $5M USD  
            "tier_3": 10000000  # $10M USD
        },
        "volume_spike_multiplier": 3.0,  # 3x average volume
        "whale_detection_periods": ["24h", "3d", "7d"]
    }
    
    # Alert Thresholds
    ALERT_THRESHOLDS = {
        "whale_activity": 7,
        "correlation_anomaly": 6,
        "fibonacci_level": 6,
        "technical_indicator": 7,
        "ml_prediction": 6,
        "trend_reversal": 8
    }
    
    @classmethod
    def get_current_support_resistance(cls, current_price: float) -> Dict:
        """Get relevant support and resistance levels for current price"""
        support = [level for level in cls.SUPPORT_LEVELS if level < current_price]
        resistance = [level for level in cls.RESISTANCE_LEVELS if level > current_price]
        
        # Sort to get closest levels first
        support.sort(reverse=True)  # Highest support first
        resistance.sort()  # Lowest resistance first
        
        return {
            "support": support[:3],  # Top 3 support levels
            "resistance": resistance[:3],  # Top 3 resistance levels
            "closest_support": support[0] if support else None,
            "closest_resistance": resistance[0] if resistance else None
        }
    
    @classmethod
    def get_wave_structure_for_price(cls, current_price: float) -> Dict:
        """Get Elliott Wave structure with current price"""
        structure = cls.VISUAL_ELLIOTT_STRUCTURE.copy()
        
        # Update Wave 5 with current price
        if "wave_5" in structure:
            wave_5_start = structure["wave_5"]["start"]
            structure["wave_5"]["end"] = current_price
            structure["wave_5"]["gain"] = ((current_price - wave_5_start) / wave_5_start) * 100
        
        return structure
    
    @classmethod
    def get_proximity_to_levels(cls, current_price: float) -> Dict:
        """Check proximity to important levels"""
        levels = cls.get_current_support_resistance(current_price)
        proximity_threshold = cls.TECHNICAL_CONFIG["support_resistance_proximity"]
        
        proximity_alerts = []
        
        # Check support proximity
        if levels["closest_support"]:
            distance_to_support = current_price - levels["closest_support"]
            if distance_to_support <= proximity_threshold:
                proximity_alerts.append({
                    "type": "support",
                    "level": levels["closest_support"],
                    "distance": distance_to_support
                })
        
        # Check resistance proximity
        if levels["closest_resistance"]:
            distance_to_resistance = levels["closest_resistance"] - current_price
            if distance_to_resistance <= proximity_threshold:
                proximity_alerts.append({
                    "type": "resistance", 
                    "level": levels["closest_resistance"],
                    "distance": distance_to_resistance
                })
        
        return {
            "near_important_level": len(proximity_alerts) > 0,
            "proximity_alerts": proximity_alerts
        }
    
    @classmethod
    def update_levels(cls, new_support: List[float] = None, new_resistance: List[float] = None):
        """Update support/resistance levels (for manual configuration updates)"""
        if new_support:
            cls.SUPPORT_LEVELS = sorted(new_support, reverse=True)
        if new_resistance:
            cls.RESISTANCE_LEVELS = sorted(new_resistance)
    
    @classmethod
    def get_config_summary(cls) -> str:
        """Get a summary of current configuration"""
        summary = f"""
📊 BNB MARKET CONFIGURATION SUMMARY
{'='*50}

💰 Support Levels: {', '.join(f'${x}' for x in cls.SUPPORT_LEVELS[:5])}
⚡ Resistance Levels: {', '.join(f'${x}' for x in cls.RESISTANCE_LEVELS[:5])}

🌊 Elliott Waves: {len(cls.VISUAL_ELLIOTT_STRUCTURE)} wave structure
📈 Market Cycles: {len(cls.MARKET_CYCLES)} historical cycles tracked

🔧 Technical Settings:
   • S/R Proximity Alert: {cls.TECHNICAL_CONFIG['support_resistance_proximity']}
   • RSI Oversold/Overbought: {cls.TECHNICAL_CONFIG['rsi_oversold']}/{cls.TECHNICAL_CONFIG['rsi_overbought']}
   • Volume Spike Threshold: {cls.TECHNICAL_CONFIG['volume_spike_threshold']}x

🐋 Whale Thresholds: ${cls.WHALE_CONFIG['large_transaction_thresholds']['tier_1']:,} / ${cls.WHALE_CONFIG['large_transaction_thresholds']['tier_2']:,} / ${cls.WHALE_CONFIG['large_transaction_thresholds']['tier_3']:,}

🚨 Alert Thresholds: Whale({cls.ALERT_THRESHOLDS['whale_activity']}), Technical({cls.ALERT_THRESHOLDS['technical_indicator']}), Reversal({cls.ALERT_THRESHOLDS['trend_reversal']})

Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
{'='*50}
        """
        return summary.strip()


# Example usage and testing
if __name__ == "__main__":
    config = BNBMarketConfig()
    
    # Test current price analysis
    test_price = 850.0
    
    print("🧪 Testing Market Configuration")
    print("=" * 40)
    
    # Test support/resistance
    sr_levels = config.get_current_support_resistance(test_price)
    print(f"Current Price: ${test_price}")
    print(f"Support: {sr_levels['support']}")
    print(f"Resistance: {sr_levels['resistance']}")
    
    # Test proximity alerts
    proximity = config.get_proximity_to_levels(test_price)
    print(f"Near Important Level: {proximity['near_important_level']}")
    
    # Print full config summary
    print("\n" + config.get_config_summary())
