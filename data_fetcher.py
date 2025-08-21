#!/usr/bin/env python3
"""
Data Fetcher Module
Handles all Binance API communication and data processing
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time


class BinanceDataFetcher:
    """Class for fetching and processing Binance market data"""
    
    def __init__(self, symbol: str = "BNBUSDT"):
        self.base_url = "https://api.binance.com/api/v3"
        self.symbol = symbol
        self.request_timeout = 10
        self.retry_attempts = 3
    
    def fetch_klines(self, interval: str, limit: int = 100) -> List:
        """Fetch historical candlestick data from Binance"""
        for attempt in range(self.retry_attempts):
            try:
                params = {
                    "symbol": self.symbol,
                    "interval": interval,
                    "limit": min(limit, 1000)  # Binance max limit
                }
                
                response = requests.get(
                    f"{self.base_url}/klines",
                    params=params,
                    timeout=self.request_timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"API Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.retry_attempts - 1:
                    time.sleep(1)  # Wait before retry
        
        return []
    
    def get_current_price(self) -> Optional[float]:
        """Get current market price"""
        try:
            response = requests.get(
                f"{self.base_url}/ticker/price",
                params={"symbol": self.symbol},
                timeout=self.request_timeout
            )
            
            if response.status_code == 200:
                return float(response.json()['price'])
            
        except Exception as e:
            print(f"Error fetching current price: {e}")
        
        return None
    
    def get_24h_ticker(self) -> Optional[Dict]:
        """Get 24h price change statistics"""
        try:
            response = requests.get(
                f"{self.base_url}/ticker/24hr",
                params={"symbol": self.symbol},
                timeout=self.request_timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "price_change": float(data['priceChange']),
                    "price_change_percent": float(data['priceChangePercent']),
                    "high_price": float(data['highPrice']),
                    "low_price": float(data['lowPrice']),
                    "volume": float(data['volume']),
                    "quote_volume": float(data['quoteVolume'])
                }
                
        except Exception as e:
            print(f"Error fetching 24h ticker: {e}")
        
        return None
    
    def process_klines_data(self, klines: List) -> Dict:
        """Process raw klines data into useful format"""
        if not klines:
            return {
                "timestamps": [],
                "opens": [],
                "highs": [],
                "lows": [],
                "closes": [],
                "volumes": []
            }
        
        return {
            "timestamps": [datetime.fromtimestamp(k[0]/1000) for k in klines],
            "opens": [float(k[1]) for k in klines],
            "highs": [float(k[2]) for k in klines],
            "lows": [float(k[3]) for k in klines],
            "closes": [float(k[4]) for k in klines],
            "volumes": [float(k[5]) for k in klines]
        }
    
    def analyze_timeframes(self) -> Dict:
        """Analyze multiple timeframes for trend detection (Daily+ focus)"""
        timeframes_config = {
            "1d": {"interval": "1d", "limit": 30, "period": "1 Month"},
            "1w": {"interval": "1w", "limit": 12, "period": "3 Months"},
            "1M": {"interval": "1M", "limit": 6, "period": "6 Months"},
            "3M": {"interval": "3M", "limit": 4, "period": "1 Year"}
        }
        
        analysis = {}
        
        for tf_name, tf_config in timeframes_config.items():
            klines = self.fetch_klines(tf_config["interval"], tf_config["limit"])
            
            if not klines:
                continue
            
            data = self.process_klines_data(klines)
            closes = data["closes"]
            
            if len(closes) < 20:
                continue
            
            # Calculate basic trend metrics
            current_price = closes[-1]
            price_20_ago = closes[-20] if len(closes) >= 20 else closes[0]
            
            trend = "BULLISH" if current_price > price_20_ago else "BEARISH"
            strength = abs((current_price - price_20_ago) / price_20_ago * 100)
            
            # Calculate RSI for this timeframe
            from indicators import TechnicalIndicators
            rsi = TechnicalIndicators.calculate_rsi(closes)
            
            # Calculate volatility
            recent_highs = data["highs"][-10:]
            recent_lows = data["lows"][-10:]
            volatility = ((max(recent_highs) - min(recent_lows)) / current_price) * 100
            
            analysis[tf_config["period"]] = {
                "timeframe": tf_name,
                "trend": trend,
                "strength": round(strength, 2),
                "rsi": rsi,
                "volatility": round(volatility, 2),
                "current_price": current_price,
                "price_20_ago": price_20_ago,
                "data": data
            }
        
        return analysis
    
    def get_market_summary(self) -> Dict:
        """Get comprehensive market summary"""
        current_price = self.get_current_price()
        ticker_24h = self.get_24h_ticker()
        
        # Get daily data for additional metrics
        daily_klines = self.fetch_klines("1d", 30)
        daily_data = self.process_klines_data(daily_klines)
        
        summary = {
            "symbol": self.symbol,
            "timestamp": datetime.now(),
            "current_price": current_price,
            "24h_change": ticker_24h.get("price_change_percent", 0) if ticker_24h else 0,
            "24h_high": ticker_24h.get("high_price", 0) if ticker_24h else 0,
            "24h_low": ticker_24h.get("low_price", 0) if ticker_24h else 0,
            "24h_volume": ticker_24h.get("volume", 0) if ticker_24h else 0
        }
        
        # Add trend information
        if daily_data["closes"]:
            closes = daily_data["closes"]
            summary.update({
                "7d_change": ((closes[-1] - closes[-7]) / closes[-7] * 100) if len(closes) >= 7 else 0,
                "30d_change": ((closes[-1] - closes[-30]) / closes[-30] * 100) if len(closes) >= 30 else 0,
                "30d_high": max(daily_data["highs"]) if daily_data["highs"] else 0,
                "30d_low": min(daily_data["lows"]) if daily_data["lows"] else 0
            })
        
        return summary
    
    def get_support_resistance_levels(self, lookback_days: int = 90) -> Dict:
        """Calculate support and resistance levels from historical data"""
        daily_klines = self.fetch_klines("1d", lookback_days)
        data = self.process_klines_data(daily_klines)
        
        if not data["highs"] or not data["lows"]:
            return {"support": [], "resistance": []}
        
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]
        current_price = closes[-1] if closes else 0
        
        # Find significant highs and lows
        significant_highs = []
        significant_lows = []
        
        # Look for local extremes
        for i in range(2, len(highs) - 2):
            # Local high
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                significant_highs.append(highs[i])
            
            # Local low
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                significant_lows.append(lows[i])
        
        # Filter levels close to current price
        resistance_levels = [h for h in significant_highs if h > current_price]
        support_levels = [l for l in significant_lows if l < current_price]
        
        # Sort and take most relevant levels
        resistance_levels.sort()
        support_levels.sort(reverse=True)
        
        return {
            "support": support_levels[:5],  # Top 5 support levels
            "resistance": resistance_levels[:5],  # Top 5 resistance levels
            "current_price": current_price
        }


# Example usage
if __name__ == "__main__":
    fetcher = BinanceDataFetcher("BNBUSDT")
    
    print("Testing Binance Data Fetcher...")
    
    # Test current price
    price = fetcher.get_current_price()
    print(f"Current BNB Price: ${price}")
    
    # Test market summary
    summary = fetcher.get_market_summary()
    print(f"24h Change: {summary['24h_change']:.2f}%")
    
    # Test timeframe analysis
    mtf = fetcher.analyze_timeframes()
    for period, data in mtf.items():
        print(f"{period}: {data['trend']} ({data['strength']:.2f}%)")
