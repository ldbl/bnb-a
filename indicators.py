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


# Example usage
if __name__ == "__main__":
    # Test data
    test_prices = [100, 102, 101, 105, 107, 106, 109, 111, 108, 112, 115, 113, 117, 119, 116, 120]
    
    indicators = TechnicalIndicators()
    
    print("Technical Indicators Test:")
    print(f"RSI: {indicators.calculate_rsi(test_prices)}")
    print(f"MACD: {indicators.calculate_macd(test_prices)}")
    print(f"Bollinger: {indicators.calculate_bollinger(test_prices)}")
