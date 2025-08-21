#!/usr/bin/env python3
"""
Configuration Module
Centralized configuration for the BNB analyzer
"""

from typing import Dict, List


class Config:
    """Configuration settings for the analyzer"""
    
    # API Settings
    BINANCE_BASE_URL = "https://api.binance.com/api/v3"
    DEFAULT_SYMBOL = "BNBUSDT"
    REQUEST_TIMEOUT = 10
    RETRY_ATTEMPTS = 3
    
    # Cache Settings
    CACHE_TTL_KLINES = 30  # seconds
    CACHE_TTL_PRICE = 10   # seconds
    CACHE_TTL_TICKER = 30  # seconds
    
    # Technical Analysis Settings
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD_DEV = 2.0
    
    # Elliott Wave Settings
    ELLIOTT_MIN_DATA_POINTS = 20
    ELLIOTT_PIVOT_LOOKBACK = 2
    
    # Signal Generation Settings
    STRONG_BUY_THRESHOLD = 2  # Bull score must exceed bear score by this much
    STRONG_SELL_THRESHOLD = 2
    MIN_CONFIDENCE = 60  # Minimum confidence for position sizing
    
    # Position Sizing Settings
    MAX_POSITION_SIZE = 50  # Maximum position size percentage
    MAX_LEVERAGE = 2.0
    CONSERVATIVE_SIZE = 10  # Conservative position size for low confidence
    
    # Price Level Settings
    OVERSOLD_RSI = 30
    OVERBOUGHT_RSI = 70
    BULLISH_PRICE_THRESHOLD = 650  # Below this = bullish signal
    BEARISH_PRICE_THRESHOLD = 850  # Above this = bearish signal
    
    # Support/Resistance Levels
    SUPPORT_LEVELS = [732, 680, 644, 600, 550, 519]
    RESISTANCE_LEVELS = [869, 850, 820, 800, 780]
    LEVEL_PROXIMITY_THRESHOLD = 20  # Distance to consider "at level"
    
    # Fibonacci Settings
    FIBONACCI_RATIOS = {
        "0%": 0.0,
        "23.6%": 0.236,
        "38.2%": 0.382,
        "50%": 0.5,
        "61.8%": 0.618,
        "78.6%": 0.786,
        "100%": 1.0,
        "161.8%": 1.618,
        "261.8%": 2.618,
        "423.6%": 4.236
    }
    
    # Timeframe Configuration (Daily+ focus)
    TIMEFRAMES = {
        "1d": {"interval": "1d", "limit": 30, "period": "1 Month"},
        "1w": {"interval": "1w", "limit": 12, "period": "3 Months"}, 
        "1M": {"interval": "1M", "limit": 6, "period": "6 Months"},
        "1M_year": {"interval": "1M", "limit": 12, "period": "1 Year"}  # Fixed: use 1M with 12 months
    }
    
    # Legacy short-term timeframes (for compatibility)
    SHORT_TIMEFRAMES = {
        "1h": {"interval": "1h", "limit": 168, "period": "Week"},
        "4h": {"interval": "4h", "limit": 180, "period": "Month"}
    }
    
    # Display Settings
    USE_COLORS = True
    CHART_WIDTH = 50
    CHART_HEIGHT = 10
    DISPLAY_PRECISION = 2  # Decimal places for prices
    
    # Historical Cycle Data
    CYCLES = [
        {"bottom": 363, "top": 641, "next_bottom": 408, "date": "2024-03"},
        {"bottom": 408, "top": 793, "next_bottom": 519, "date": "2024-08"},
        {"bottom": 519, "top": 869, "next_bottom": None, "date": "2025-02"}
    ]
    
    # Risk Management
    DEFAULT_STOP_LOSS_PERCENT = 5.0  # 5% stop loss
    DEFAULT_TAKE_PROFIT_PERCENT = 8.0  # 8% take profit
    MAX_DAILY_TRADES = 5
    
    # Alerts and Notifications
    ENABLE_ALERTS = False
    ALERT_RSI_OVERSOLD = 25
    ALERT_RSI_OVERBOUGHT = 75
    ALERT_VOLUME_SPIKE_MULTIPLIER = 2.0
    
    @classmethod
    def get_position_size_rules(cls) -> Dict:
        """Get position sizing rules based on confidence"""
        return {
            "high_confidence": {
                "min_confidence": 80,
                "strong_buy_size": "50% with 2x leverage",
                "buy_size": "30% with 1.5x leverage",
                "strong_sell_size": "33% short position",
                "sell_size": "20% short position"
            },
            "medium_confidence": {
                "min_confidence": 70,
                "strong_buy_size": "30% with 1.5x leverage", 
                "buy_size": "20% with 1x leverage",
                "strong_sell_size": "20% short position",
                "sell_size": "10% short position"
            },
            "low_confidence": {
                "min_confidence": 60,
                "size": "10% - Low confidence"
            }
        }
    
    @classmethod
    def get_target_multipliers(cls) -> Dict:
        """Get target price multipliers for different actions"""
        return {
            "STRONG_BUY": {"target": 1.15, "stop": 0.95},
            "BUY": {"target": 1.08, "stop": 0.97},
            "STRONG_SELL": {"target": 0.85, "stop": 1.05},
            "SELL": {"target": 0.92, "stop": 1.03}
        }


# Create global config instance
config = Config()


# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration"""
    CACHE_TTL_KLINES = 60  # Longer cache for development
    RETRY_ATTEMPTS = 1  # Fewer retries for faster feedback


class ProductionConfig(Config):
    """Production environment configuration"""
    ENABLE_ALERTS = True
    MAX_DAILY_TRADES = 10
    RETRY_ATTEMPTS = 5  # More retries for reliability


def get_config(environment: str = "production") -> Config:
    """Get configuration based on environment"""
    configs = {
        "development": DevelopmentConfig(),
        "production": ProductionConfig(),
        "default": Config()
    }
    return configs.get(environment, configs["default"])


# Example usage
if __name__ == "__main__":
    print("Configuration Test:")
    print(f"Default Symbol: {config.DEFAULT_SYMBOL}")
    print(f"RSI Period: {config.RSI_PERIOD}")
    print(f"Position Size Rules: {config.get_position_size_rules()}")
    print(f"Target Multipliers: {config.get_target_multipliers()}")
