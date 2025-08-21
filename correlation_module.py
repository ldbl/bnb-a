#!/usr/bin/env python3
"""
Cryptocurrency Correlation Analysis Module
Analyzes BNB correlation with BTC and ETH for enhanced signal generation
"""

import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import time
from logger import get_logger

class CorrelationAnalyzer:
    """Analyzes correlation between BNB and major cryptocurrencies"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.logger = get_logger(__name__)
        self.request_timeout = 10
        self.symbols = {
            "BNB": "BNBUSDT",
            "BTC": "BTCUSDT", 
            "ETH": "ETHUSDT"
        }
        self.correlation_thresholds = {
            "strong_positive": 0.7,
            "moderate_positive": 0.4,
            "weak": 0.2,
            "moderate_negative": -0.4,
            "strong_negative": -0.7
        }
        
        # Alert thresholds for critical correlation events
        self.alert_thresholds = {
            "correlation_breakdown": 0.3,      # Drop below 0.3 when usually high
            "negative_correlation": -0.5,      # Strong negative correlation 
            "extreme_performance_gap": 5.0,    # >5% performance difference
            "leadership_change": True,         # BNB leading when usually follows
            "correlation_spike": 0.9,          # Unusually high correlation
            "independence_threshold": 3.0      # 3%+ independent movement
        }
    
    def fetch_price_data(self, symbol: str, interval: str = "1h", limit: int = 100) -> List[float]:
        """Fetch price data for correlation analysis with timeout and logging"""
        try:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            
            self.logger.debug(f"Fetching price data for {symbol} ({interval}, {limit} periods)")
            response = requests.get(
                f"{self.base_url}/klines", 
                params=params, 
                timeout=self.request_timeout
            )
            
            if response.status_code == 200:
                klines = response.json()
                prices = [float(k[4]) for k in klines]
                self.logger.debug(f"Successfully fetched {len(prices)} price points for {symbol}")
                return prices
            else:
                self.logger.error(f"API Error for {symbol}: {response.status_code} - {response.text}")
                return []
                
        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout while fetching {symbol} data")
            return []
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error fetching {symbol} data: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error fetching {symbol} data: {e}")
            return []
    
    def calculate_correlation(self, prices1: List[float], prices2: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(prices1) != len(prices2) or len(prices1) < 2:
            return 0.0
        
        # Calculate returns instead of prices for better correlation
        returns1 = [prices1[i] / prices1[i-1] - 1 for i in range(1, len(prices1))]
        returns2 = [prices2[i] / prices2[i-1] - 1 for i in range(1, len(prices2))]
        
        try:
            correlation_matrix = np.corrcoef(returns1, returns2)
            return correlation_matrix[0, 1]
        except:
            return 0.0
    
    def get_correlation_strength(self, correlation: float) -> str:
        """Get correlation strength description"""
        abs_corr = abs(correlation)
        
        if abs_corr >= self.correlation_thresholds["strong_positive"]:
            return "STRONG" + (" POSITIVE" if correlation > 0 else " NEGATIVE")
        elif abs_corr >= self.correlation_thresholds["moderate_positive"]:
            return "MODERATE" + (" POSITIVE" if correlation > 0 else " NEGATIVE")
        elif abs_corr >= self.correlation_thresholds["weak"]:
            return "WEAK" + (" POSITIVE" if correlation > 0 else " NEGATIVE")
        else:
            return "NO CORRELATION"
    
    def analyze_market_leadership(self, bnb_prices: List[float], btc_prices: List[float], 
                                 eth_prices: List[float]) -> Dict:
        """Analyze which market is leading price movements"""
        
        # Calculate recent returns (last 24 hours)
        bnb_return_24h = (bnb_prices[-1] / bnb_prices[-25] - 1) * 100 if len(bnb_prices) >= 25 else 0
        btc_return_24h = (btc_prices[-1] / btc_prices[-25] - 1) * 100 if len(btc_prices) >= 25 else 0
        eth_return_24h = (eth_prices[-1] / eth_prices[-25] - 1) * 100 if len(eth_prices) >= 25 else 0
        
        # Determine market leader
        returns = {"BNB": bnb_return_24h, "BTC": btc_return_24h, "ETH": eth_return_24h}
        leader = max(returns, key=returns.get)
        laggard = min(returns, key=returns.get)
        
        return {
            "leader": leader,
            "laggard": laggard,
            "bnb_performance": bnb_return_24h,
            "btc_performance": btc_return_24h,
            "eth_performance": eth_return_24h,
            "bnb_vs_btc": bnb_return_24h - btc_return_24h,
            "bnb_vs_eth": bnb_return_24h - eth_return_24h
        }
    
    def get_correlation_signals(self, correlation_data: Dict) -> Dict:
        """Generate trading signals based on correlation analysis"""
        
        signals = {
            "correlation_score": 0,
            "strength": "NEUTRAL",
            "recommendation": "WAIT",
            "confidence": 50,
            "reasoning": []
        }
        
        btc_corr = correlation_data["btc_correlation"]
        eth_corr = correlation_data["eth_correlation"]
        leadership = correlation_data["leadership"]
        
        # Correlation-based scoring
        if abs(btc_corr) > 0.8:  # Very high correlation
            if leadership["bnb_vs_btc"] > 2:  # BNB outperforming
                signals["correlation_score"] += 2
                signals["reasoning"].append("BNB outperforming despite high BTC correlation")
            elif leadership["bnb_vs_btc"] < -2:  # BNB underperforming
                signals["correlation_score"] -= 2
                signals["reasoning"].append("BNB underperforming with high BTC correlation")
        
        # Leadership analysis
        if leadership["leader"] == "BNB":
            signals["correlation_score"] += 1
            signals["reasoning"].append("BNB is leading the market")
        elif leadership["laggard"] == "BNB":
            signals["correlation_score"] -= 1
            signals["reasoning"].append("BNB is lagging the market")
        
        # Independent movement detection
        if abs(btc_corr) < 0.3 and abs(eth_corr) < 0.3:
            if leadership["bnb_performance"] > 1:
                signals["correlation_score"] += 2
                signals["reasoning"].append("BNB moving independently upward")
            elif leadership["bnb_performance"] < -1:
                signals["correlation_score"] -= 2
                signals["reasoning"].append("BNB moving independently downward")
        
        # Generate final recommendation
        if signals["correlation_score"] >= 2:
            signals["recommendation"] = "BUY"
            signals["strength"] = "BULLISH"
            signals["confidence"] = min(70 + (signals["correlation_score"] * 5), 85)
        elif signals["correlation_score"] <= -2:
            signals["recommendation"] = "SELL"  
            signals["strength"] = "BEARISH"
            signals["confidence"] = min(70 + (abs(signals["correlation_score"]) * 5), 85)
        else:
            signals["recommendation"] = "WAIT"
            signals["strength"] = "NEUTRAL"
            signals["confidence"] = 50 + abs(signals["correlation_score"]) * 5
        
        return signals
    
    def run_correlation_analysis(self, interval: str = "1d", periods: int = 30) -> Dict:
        """Run complete correlation analysis"""
        
        print(f"\n📊 CORRELATION ANALYSIS")
        print("=" * 50)
        
        # Fetch data for all symbols
        print("🔄 Fetching market data...")
        
        bnb_prices = self.fetch_price_data(self.symbols["BNB"], interval, periods)
        btc_prices = self.fetch_price_data(self.symbols["BTC"], interval, periods)
        eth_prices = self.fetch_price_data(self.symbols["ETH"], interval, periods)
        
        if not all([bnb_prices, btc_prices, eth_prices]):
            print("❌ Error fetching market data")
            return {}
        
        # Calculate correlations
        btc_correlation = self.calculate_correlation(bnb_prices, btc_prices)
        eth_correlation = self.calculate_correlation(bnb_prices, eth_prices)
        btc_eth_correlation = self.calculate_correlation(btc_prices, eth_prices)
        
        # Analyze market leadership
        leadership = self.analyze_market_leadership(bnb_prices, btc_prices, eth_prices)
        
        # Generate correlation data
        correlation_data = {
            "btc_correlation": btc_correlation,
            "eth_correlation": eth_correlation,
            "btc_eth_correlation": btc_eth_correlation,
            "leadership": leadership,
            "analysis_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "data_points": len(bnb_prices),
            "interval": interval
        }
        
        # Get signals
        signals = self.get_correlation_signals(correlation_data)
        correlation_data["signals"] = signals
        
        # Display results
        self.display_correlation_analysis(correlation_data)
        
        return correlation_data
    
    def display_correlation_analysis(self, data: Dict):
        """Display correlation analysis results"""
        
        print(f"\n📈 CORRELATION COEFFICIENTS")
        print("-" * 30)
        print(f"BNB vs BTC: {data['btc_correlation']:.3f} ({self.get_correlation_strength(data['btc_correlation'])})")
        print(f"BNB vs ETH: {data['eth_correlation']:.3f} ({self.get_correlation_strength(data['eth_correlation'])})")
        print(f"BTC vs ETH: {data['btc_eth_correlation']:.3f} ({self.get_correlation_strength(data['btc_eth_correlation'])})")
        
        print(f"\n🏆 MARKET LEADERSHIP (24h)")
        print("-" * 30)
        leadership = data["leadership"]
        print(f"Leader: {leadership['leader']}")
        print(f"BNB Performance: {leadership['bnb_performance']:+.2f}%")
        print(f"BTC Performance: {leadership['btc_performance']:+.2f}%")
        print(f"ETH Performance: {leadership['eth_performance']:+.2f}%")
        print(f"BNB vs BTC: {leadership['bnb_vs_btc']:+.2f}%")
        print(f"BNB vs ETH: {leadership['bnb_vs_eth']:+.2f}%")
        
        signals = data["signals"]
        print(f"\n🎯 CORRELATION SIGNALS")
        print("-" * 30)
        print(f"Recommendation: {signals['recommendation']}")
        print(f"Strength: {signals['strength']}")
        print(f"Confidence: {signals['confidence']}%")
        print(f"Score: {signals['correlation_score']}")
        
        if signals["reasoning"]:
            print(f"\n💭 REASONING:")
            for reason in signals["reasoning"]:
                print(f"   • {reason}")
        
        print(f"\n⏰ Analysis Time: {data['analysis_time']}")
        print(f"📊 Data Points: {data['data_points']} ({data['interval']} intervals)")

    def get_multi_timeframe_correlation(self) -> Dict:
        """Analyze correlation across multiple timeframes"""
        
        timeframes = {
            "1d": {"interval": "1d", "periods": 30, "name": "1 Month"},
            "1w": {"interval": "1w", "periods": 12, "name": "3 Months"}, 
            "1M": {"interval": "1M", "periods": 6, "name": "6 Months"},
            "1M_year": {"interval": "1M", "periods": 12, "name": "1 Year"}  # Fixed: use 1M with 12 periods
        }
        
        results = {}
        
        print(f"\n🔄 MULTI-TIMEFRAME CORRELATION ANALYSIS")
        print("=" * 55)
        
        for tf_key, tf_config in timeframes.items():
            print(f"\n📊 {tf_config['name']} ({tf_config['interval']}):")
            print("-" * 35)
            
            # Fetch data
            bnb_prices = self.fetch_price_data(self.symbols["BNB"], tf_config["interval"], tf_config["periods"])
            btc_prices = self.fetch_price_data(self.symbols["BTC"], tf_config["interval"], tf_config["periods"])
            eth_prices = self.fetch_price_data(self.symbols["ETH"], tf_config["interval"], tf_config["periods"])
            
            if all([bnb_prices, btc_prices, eth_prices]):
                btc_corr = self.calculate_correlation(bnb_prices, btc_prices)
                eth_corr = self.calculate_correlation(bnb_prices, eth_prices)
                
                results[tf_key] = {
                    "btc_correlation": btc_corr,
                    "eth_correlation": eth_corr,
                    "timeframe": tf_config["name"]
                }
                
                print(f"BNB vs BTC: {btc_corr:.3f} ({self.get_correlation_strength(btc_corr)})")
                print(f"BNB vs ETH: {eth_corr:.3f} ({self.get_correlation_strength(eth_corr)})")
            else:
                print("❌ Error fetching data")
                results[tf_key] = None
            
            time.sleep(0.1)  # Rate limiting
        
        return results
    
    def check_critical_correlation_activity(self) -> Dict:
        """Check if there are critical correlation anomalies that should be shown automatically"""
        
        try:
            # Get recent correlation data (daily+ focus)
            correlation_data = self.run_correlation_analysis("1d", 30)
            
            if not correlation_data or "signals" not in correlation_data:
                return {"show_alert": False, "reason": "No correlation data available"}
            
            critical_signals = []
            alert_score = 0
            
            btc_corr = correlation_data.get("btc_correlation", 0)
            eth_corr = correlation_data.get("eth_correlation", 0)
            leadership = correlation_data.get("leadership", {})
            signals = correlation_data.get("signals", {})
            
            # Check for correlation breakdown
            if abs(btc_corr) < self.alert_thresholds["correlation_breakdown"] and abs(eth_corr) < self.alert_thresholds["correlation_breakdown"]:
                critical_signals.append("🔗 Correlation breakdown - BNB moving independently")
                alert_score += 8
            
            # Check for negative correlation (unusual)
            if btc_corr < self.alert_thresholds["negative_correlation"] or eth_corr < self.alert_thresholds["negative_correlation"]:
                critical_signals.append(f"📉 Negative correlation detected (BTC: {btc_corr:.2f}, ETH: {eth_corr:.2f})")
                alert_score += 6
            
            # Check for extreme performance gaps
            bnb_vs_btc = abs(leadership.get("bnb_vs_btc", 0))
            bnb_vs_eth = abs(leadership.get("bnb_vs_eth", 0))
            
            if bnb_vs_btc >= self.alert_thresholds["extreme_performance_gap"]:
                direction = "outperforming" if leadership.get("bnb_vs_btc", 0) > 0 else "underperforming"
                critical_signals.append(f"⚡ BNB {direction} BTC by {bnb_vs_btc:.1f}%")
                alert_score += 5
            
            if bnb_vs_eth >= self.alert_thresholds["extreme_performance_gap"]:
                direction = "outperforming" if leadership.get("bnb_vs_eth", 0) > 0 else "underperforming"
                critical_signals.append(f"⚡ BNB {direction} ETH by {bnb_vs_eth:.1f}%")
                alert_score += 5
            
            # Check for BNB leadership (unusual)
            if leadership.get("leader") == "BNB":
                critical_signals.append("👑 BNB leading the market (unusual)")
                alert_score += 4
            
            # Check for unusually high correlation
            if abs(btc_corr) >= self.alert_thresholds["correlation_spike"] or abs(eth_corr) >= self.alert_thresholds["correlation_spike"]:
                critical_signals.append(f"📈 Unusually high correlation (BTC: {btc_corr:.2f}, ETH: {eth_corr:.2f})")
                alert_score += 3
            
            # Check independence score from signals
            correlation_score = signals.get("correlation_score", 0)
            if abs(correlation_score) >= 2:  # High correlation signal score
                signal_type = "bullish" if correlation_score > 0 else "bearish"
                critical_signals.append(f"🎯 Strong {signal_type} correlation signal (Score: {correlation_score})")
                alert_score += 4
            
            # Determine if alert should be shown
            show_alert = alert_score >= 6  # Lower threshold than whale (correlation is more subtle)
            
            return {
                "show_alert": show_alert,
                "alert_score": alert_score,
                "critical_signals": critical_signals,
                "correlation_data": {
                    "btc_correlation": btc_corr,
                    "eth_correlation": eth_corr,
                    "leadership": leadership.get("leader", "Unknown"),
                    "bnb_performance": leadership.get("bnb_performance", 0),
                    "signal_strength": signals.get("strength", "NEUTRAL")
                }
            }
            
        except Exception as e:
            return {"show_alert": False, "reason": f"Error checking correlation activity: {e}"}
    
    def get_critical_correlation_alert_text(self, alert_data: Dict) -> str:
        """Generate formatted alert text for critical correlation activity"""
        
        if not alert_data.get("show_alert"):
            return ""
        
        signals = alert_data.get("critical_signals", [])
        corr_data = alert_data.get("correlation_data", {})
        
        alert_text = f"\n📊 CRITICAL CORRELATION ANOMALY DETECTED\n"
        alert_text += "=" * 50 + "\n"
        
        for signal in signals:
            alert_text += f"{signal}\n"
        
        alert_text += f"\nBTC Correlation: {corr_data.get('btc_correlation', 0):.3f}"
        alert_text += f"\nETH Correlation: {corr_data.get('eth_correlation', 0):.3f}"
        alert_text += f"\nMarket Leader: {corr_data.get('leadership', 'Unknown')}"
        alert_text += f"\nBNB Performance: {corr_data.get('bnb_performance', 0):+.2f}%"
        alert_text += f"\nSignal Strength: {corr_data.get('signal_strength', 'NEUTRAL')}"
        
        alert_text += f"\n\nAlert Score: {alert_data.get('alert_score', 0)}/20"
        alert_text += "\n💡 Consider: Check correlation analysis for details"
        alert_text += "\n" + "=" * 50
        
        return alert_text

# Example usage
if __name__ == "__main__":
    analyzer = CorrelationAnalyzer()
    
    # Single timeframe analysis
    result = analyzer.run_correlation_analysis("1h", 100)
    
    # Multi-timeframe analysis
    multi_tf_result = analyzer.get_multi_timeframe_correlation()
