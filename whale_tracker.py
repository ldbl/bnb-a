#!/usr/bin/env python3
"""
Whale Tracker Module
Tracks large BNB transactions and whale movements
"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json


class WhaleTracker:
    """Track large BNB transactions and whale activity"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.bscscan_url = "https://api.bscscan.com/api"
        
        # Whale thresholds
        self.whale_thresholds = {
            "mega_whale": 100000,    # 100K+ BNB
            "whale": 50000,          # 50K+ BNB  
            "large_holder": 10000,   # 10K+ BNB
            "medium_holder": 1000    # 1K+ BNB
        }
        
        # Price monitoring
        self.price_change_threshold = 0.02  # 2% price change alert
        self.volume_spike_threshold = 2.0   # 2x average volume
        
        # Known whale addresses (examples - you'd need real ones)
        self.known_whales = {
            "binance_hot": "0x28c6c06298d514db089934071355e5743bf21d60",
            "binance_cold": "0x21a31ee1afc51d94c2efccaa2092ad1028285549",
            # Add more known whale addresses
        }
        
        # Alert thresholds for critical activity
        self.alert_thresholds = {
            "critical_volume_spike": 3.0,      # 3x+ volume spike
            "mega_whale_activity": 50000,      # 50K+ BNB movement
            "multiple_whale_signals": 3,       # 3+ whale signals in period
            "price_impact": 0.03,              # 3%+ price change with volume
            "unusual_activity_score": 8        # High unusual activity score
        }
        
    def get_current_price_and_volume(self) -> Dict:
        """Get current BNB price and volume data"""
        try:
            # 24h ticker
            ticker_response = requests.get(f"{self.base_url}/ticker/24hr", 
                                         params={"symbol": "BNBUSDT"})
            
            if ticker_response.status_code == 200:
                ticker_data = ticker_response.json()
                
                return {
                    "price": float(ticker_data["lastPrice"]),
                    "price_change_24h": float(ticker_data["priceChangePercent"]),
                    "volume_24h": float(ticker_data["volume"]),
                    "volume_usdt_24h": float(ticker_data["quoteVolume"]),
                    "high_24h": float(ticker_data["highPrice"]),
                    "low_24h": float(ticker_data["lowPrice"]),
                    "trades_count": int(ticker_data["count"])
                }
        except Exception as e:
            print(f"Error fetching price data: {e}")
        
        return {}
    
    def get_whale_activity_summary(self, days_back: int = 1) -> Dict:
        """Get whale activity summary using klines data (much more efficient)"""
        try:
            # Determine appropriate interval based on period
            if days_back <= 1:
                interval = "15m"
                limit = min(96, days_back * 96)  # 96 x 15min = 24h
            elif days_back <= 3:
                interval = "1h" 
                limit = min(72, days_back * 24)  # 24h x days
            else:
                interval = "4h"
                limit = min(42, days_back * 6)   # 6 x 4h = 24h x days
            
            # Get klines data
            response = requests.get(f"{self.base_url}/klines", 
                                  params={
                                      "symbol": "BNBUSDT",
                                      "interval": interval,
                                      "limit": limit
                                  })
            
            if response.status_code == 200:
                klines = response.json()
                
                whale_activity = {
                    "period": f"{days_back} days",
                    "interval": interval,
                    "total_candles": len(klines),
                    "high_volume_periods": [],
                    "price_movements": [],
                    "volume_analysis": {},
                    "whale_signals": []
                }
                
                # Analyze each candle for whale activity
                volumes = []
                for i, kline in enumerate(klines):
                    timestamp = datetime.fromtimestamp(kline[0] / 1000)
                    open_price = float(kline[1])
                    high_price = float(kline[2])
                    low_price = float(kline[3])
                    close_price = float(kline[4])
                    volume = float(kline[5])
                    
                    volumes.append(volume)
                    
                    # Calculate price change
                    price_change = ((close_price - open_price) / open_price) * 100
                    
                    # Store period data
                    period_data = {
                        "timestamp": timestamp,
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "volume": volume,
                        "price_change": price_change
                    }
                    
                    whale_activity["price_movements"].append(period_data)
                
                # Calculate volume statistics
                if volumes:
                    avg_volume = sum(volumes) / len(volumes)
                    max_volume = max(volumes)
                    
                    whale_activity["volume_analysis"] = {
                        "average_volume": avg_volume,
                        "max_volume": max_volume,
                        "volume_spike_threshold": avg_volume * 2
                    }
                    
                    # Find high volume periods (potential whale activity)
                    spike_threshold = avg_volume * 2
                    for period in whale_activity["price_movements"]:
                        if period["volume"] > spike_threshold:
                            whale_activity["high_volume_periods"].append({
                                "timestamp": period["timestamp"],
                                "volume": period["volume"],
                                "volume_ratio": period["volume"] / avg_volume,
                                "price_change": period["price_change"],
                                "whale_signal": self.classify_whale_signal(period["volume"], period["price_change"], avg_volume)
                            })
                    
                    # Sort by volume descending
                    whale_activity["high_volume_periods"].sort(key=lambda x: x["volume"], reverse=True)
                
                return whale_activity
                
        except Exception as e:
            print(f"Error fetching whale summary: {e}")
        
        return {}
    
    def classify_whale_signal(self, volume: float, price_change: float, avg_volume: float) -> str:
        """Classify whale signal based on volume and price action"""
        volume_ratio = volume / avg_volume
        
        if volume_ratio >= 5:
            if price_change > 2:
                return "🐋 MEGA WHALE BUY"
            elif price_change < -2:
                return "🐋 MEGA WHALE SELL"
            else:
                return "🐋 MEGA WHALE ACTIVITY"
        elif volume_ratio >= 3:
            if price_change > 1:
                return "🐳 WHALE BUY"
            elif price_change < -1:
                return "🐳 WHALE SELL"
            else:
                return "🐳 WHALE ACTIVITY"
        elif volume_ratio >= 2:
            if price_change > 0.5:
                return "🦈 LARGE BUY"
            elif price_change < -0.5:
                return "🦈 LARGE SELL"
            else:
                return "🦈 LARGE ACTIVITY"
        else:
            return "📊 VOLUME SPIKE"
    
    def categorize_whale(self, quantity: float) -> str:
        """Categorize whale based on transaction size"""
        if quantity >= self.whale_thresholds["mega_whale"]:
            return "🐋 MEGA WHALE"
        elif quantity >= self.whale_thresholds["whale"]:
            return "🐳 WHALE"
        elif quantity >= self.whale_thresholds["large_holder"]:
            return "🦈 LARGE HOLDER"
        elif quantity >= self.whale_thresholds["medium_holder"]:
            return "🐟 MEDIUM HOLDER"
        else:
            return "🐠 SMALL HOLDER"
    
    def analyze_order_book_whales(self) -> Dict:
        """Analyze order book for whale walls"""
        try:
            orderbook_response = requests.get(f"{self.base_url}/depth", 
                                            params={
                                                "symbol": "BNBUSDT",
                                                "limit": 100
                                            })
            
            if orderbook_response.status_code == 200:
                data = orderbook_response.json()
                
                # Analyze bids (buy orders)
                bids = [(float(price), float(qty)) for price, qty in data["bids"]]
                asks = [(float(price), float(qty)) for price, qty in data["asks"]]
                
                # Find whale walls
                whale_bids = [(price, qty) for price, qty in bids 
                             if qty >= self.whale_thresholds["large_holder"]]
                whale_asks = [(price, qty) for price, qty in asks 
                             if qty >= self.whale_thresholds["large_holder"]]
                
                # Calculate total whale support/resistance
                total_whale_support = sum(price * qty for price, qty in whale_bids)
                total_whale_resistance = sum(price * qty for price, qty in whale_asks)
                
                # Find largest walls
                largest_bid_wall = max(whale_bids, key=lambda x: x[1]) if whale_bids else None
                largest_ask_wall = max(whale_asks, key=lambda x: x[1]) if whale_asks else None
                
                return {
                    "whale_bids": whale_bids,
                    "whale_asks": whale_asks,
                    "total_whale_support": total_whale_support,
                    "total_whale_resistance": total_whale_resistance,
                    "largest_bid_wall": largest_bid_wall,
                    "largest_ask_wall": largest_ask_wall,
                    "whale_bid_count": len(whale_bids),
                    "whale_ask_count": len(whale_asks)
                }
                
        except Exception as e:
            print(f"Error analyzing order book: {e}")
        
        return {}
    
    def get_exchange_flows(self) -> Dict:
        """Simulate exchange flow analysis (would need real API)"""
        # This would normally connect to exchange APIs or on-chain data
        # For demo purposes, we'll simulate some data
        
        current_time = datetime.now()
        
        # Simulated large movements
        simulated_flows = {
            "inflows_24h": [
                {"exchange": "Binance", "amount": 45000, "time": current_time - timedelta(hours=2)},
                {"exchange": "Coinbase", "amount": 23000, "time": current_time - timedelta(hours=5)},
                {"exchange": "Kraken", "amount": 15000, "time": current_time - timedelta(hours=8)}
            ],
            "outflows_24h": [
                {"exchange": "Binance", "amount": 67000, "time": current_time - timedelta(hours=1)},
                {"exchange": "Huobi", "amount": 34000, "time": current_time - timedelta(hours=4)},
                {"exchange": "OKEx", "amount": 18000, "time": current_time - timedelta(hours=7)}
            ]
        }
        
        # Calculate net flows
        total_inflows = sum(flow["amount"] for flow in simulated_flows["inflows_24h"])
        total_outflows = sum(flow["amount"] for flow in simulated_flows["outflows_24h"])
        net_flow = total_outflows - total_inflows  # Positive = net outflow (bullish)
        
        return {
            "inflows_24h": simulated_flows["inflows_24h"],
            "outflows_24h": simulated_flows["outflows_24h"],
            "total_inflows": total_inflows,
            "total_outflows": total_outflows,
            "net_flow": net_flow,
            "flow_sentiment": "🟢 BULLISH" if net_flow > 0 else "🔴 BEARISH" if net_flow < -50000 else "🟡 NEUTRAL"
        }
    
    def analyze_whale_sentiment(self, large_trades: List[Dict]) -> Dict:
        """Analyze whale sentiment from recent large trades"""
        if not large_trades:
            return {"sentiment": "UNKNOWN", "confidence": 0}
        
        # Analyze last 50 large trades
        recent_trades = large_trades[:50]
        
        buy_volume = 0
        sell_volume = 0
        buy_count = 0
        sell_count = 0
        
        for trade in recent_trades:
            if trade["is_buyer_maker"]:  # Sell order (buyer is market taker)
                sell_volume += trade["quantity"]
                sell_count += 1
            else:  # Buy order (seller is market taker)
                buy_volume += trade["quantity"]
                buy_count += 1
        
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return {"sentiment": "UNKNOWN", "confidence": 0}
        
        buy_ratio = buy_volume / total_volume
        sell_ratio = sell_volume / total_volume
        
        # Determine sentiment
        if buy_ratio > 0.65:
            sentiment = "🟢 STRONG BULLISH"
            confidence = min(95, 50 + (buy_ratio - 0.5) * 100)
        elif buy_ratio > 0.55:
            sentiment = "🟢 BULLISH"
            confidence = min(80, 40 + (buy_ratio - 0.5) * 80)
        elif buy_ratio < 0.35:
            sentiment = "🔴 STRONG BEARISH"
            confidence = min(95, 50 + (0.5 - buy_ratio) * 100)
        elif buy_ratio < 0.45:
            sentiment = "🔴 BEARISH"
            confidence = min(80, 40 + (0.5 - buy_ratio) * 80)
        else:
            sentiment = "🟡 NEUTRAL"
            confidence = 60
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 1),
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "buy_ratio": round(buy_ratio * 100, 1),
            "sell_ratio": round(sell_ratio * 100, 1),
            "buy_count": buy_count,
            "sell_count": sell_count
        }
    
    def detect_unusual_activity(self, market_data: Dict, high_volume_periods: List[Dict]) -> List[str]:
        """Detect unusual whale activity patterns"""
        alerts = []
        
        # Price movement alerts
        if abs(market_data.get("price_change_24h", 0)) > self.price_change_threshold * 100:
            change = market_data["price_change_24h"]
            alerts.append(f"🚨 Large price movement: {change:+.2f}% in 24h")
        
        # Volume spike detection (simplified)
        volume_24h = market_data.get("volume_24h", 0)
        if volume_24h > 1000000:  # Arbitrary threshold for demo
            alerts.append(f"📊 High volume detected: {volume_24h:,.0f} BNB")
        
        # Whale activity alerts
        if high_volume_periods:
            mega_whale_signals = [p for p in high_volume_periods if "MEGA WHALE" in p["whale_signal"]]
            if mega_whale_signals:
                alerts.append(f"🐋 {len(mega_whale_signals)} mega whale signal(s) detected!")
            
            # Recent whale activity (last 4 hours)
            recent_time = datetime.now() - timedelta(hours=4)
            recent_signals = [p for p in high_volume_periods if p["timestamp"] > recent_time]
            if len(recent_signals) > 2:
                alerts.append(f"⚡ {len(recent_signals)} whale signals in last 4 hours")
            
            # Very high volume ratios
            extreme_signals = [p for p in high_volume_periods if p["volume_ratio"] > 5]
            if extreme_signals:
                alerts.append(f"🔥 {len(extreme_signals)} extreme volume spike(s) detected!")
        
        return alerts
    
    def multi_period_whale_analysis(self):
        """Analyze whale activity across multiple time periods"""
        print("\n📅 MULTI-PERIOD WHALE ANALYSIS")
        print("=" * 60)
        
        periods = [
            (1, "24 hours"),
            (3, "3 days"), 
            (7, "1 week")
        ]
        
        results = {}
        
        for days, period_name in periods:
            print(f"\n📊 {period_name.upper()}:")
            print("-" * 40)
            
            try:
                # Get whale activity summary for this period
                whale_summary = self.get_whale_activity_summary(days_back=days)
                
                if whale_summary and whale_summary.get("high_volume_periods"):
                    high_vol_periods = whale_summary["high_volume_periods"]
                    vol_analysis = whale_summary.get("volume_analysis", {})
                    
                    # Count whale signal types
                    mega_whale_signals = len([p for p in high_vol_periods if "MEGA WHALE" in p["whale_signal"]])
                    whale_signals = len([p for p in high_vol_periods if "🐳 WHALE" in p["whale_signal"] and "MEGA" not in p["whale_signal"]])
                    large_signals = len([p for p in high_vol_periods if "🦈 LARGE" in p["whale_signal"]])
                    
                    # Analyze sentiment
                    buy_signals = len([p for p in high_vol_periods if "BUY" in p["whale_signal"]])
                    sell_signals = len([p for p in high_vol_periods if "SELL" in p["whale_signal"]])
                    total_signals = buy_signals + sell_signals
                    
                    if total_signals > 0:
                        buy_ratio = (buy_signals / total_signals) * 100
                        sentiment_level = "🟢 BULLISH" if buy_ratio > 60 else "🔴 BEARISH" if buy_ratio < 40 else "🟡 NEUTRAL"
                    else:
                        buy_ratio = 50
                        sentiment_level = "🟡 NO SIGNALS"
                    
                    # Calculate total volume from high volume periods
                    total_volume = sum(p["volume"] for p in high_vol_periods)
                    avg_volume = vol_analysis.get("average_volume", 0)
                    
                    # Display summary
                    print(f"   Whale Signals: {len(high_vol_periods)}")
                    print(f"   Mega Whale Signals: {mega_whale_signals}")
                    print(f"   Whale Signals: {whale_signals}")
                    print(f"   Large Signals: {large_signals}")
                    print(f"   Average Volume: {avg_volume:,.0f} BNB")
                    print(f"   Sentiment: {sentiment_level}")
                    print(f"   Buy/Sell Ratio: {buy_ratio:.1f}%/{100-buy_ratio:.1f}%")
                    
                    results[period_name] = {
                        "signals_count": len(high_vol_periods),
                        "mega_whale_signals": mega_whale_signals,
                        "whale_signals": whale_signals,
                        "large_signals": large_signals,
                        "average_volume": avg_volume,
                        "sentiment": sentiment_level,
                        "buy_ratio": buy_ratio,
                        "top_signal": max(high_vol_periods, key=lambda x: x["volume_ratio"]) if high_vol_periods else None
                    }
                    
                    # Show biggest signal
                    if high_vol_periods:
                        biggest = max(high_vol_periods, key=lambda x: x["volume_ratio"])
                        time_str = biggest["timestamp"].strftime("%m/%d %H:%M")
                        print(f"   Biggest Signal: {biggest['whale_signal']}")
                        print(f"     📊 {biggest['volume']:,.0f} BNB ({biggest['volume_ratio']:.1f}x) | {time_str}")
                        
                else:
                    print(f"   ✅ No significant whale activity detected")
                    results[period_name] = {"signals_count": 0}
                    
            except Exception as e:
                print(f"   ❌ Error analyzing {period_name}: {e}")
        
        # Summary comparison
        if len(results) > 1:
            print(f"\n🏆 PERIOD COMPARISON:")
            print("=" * 40)
            
            for period_name, data in results.items():
                if data.get("signals_count", 0) > 0:
                    sentiment = data["sentiment"]
                    emoji = "🟢" if "BULLISH" in sentiment else "🔴" if "BEARISH" in sentiment else "🟡"
                    print(f"   {period_name}: {emoji} {data['signals_count']} signals | {sentiment}")
            
            # Trend analysis
            signal_counts = [data.get("signals_count", 0) for data in results.values()]
            if len(signal_counts) >= 2:
                if signal_counts[0] > signal_counts[1]:
                    trend = "📈 INCREASING"
                elif signal_counts[0] < signal_counts[1]:
                    trend = "📉 DECREASING"
                else:
                    trend = "➡️ STABLE"
                
                print(f"\n   Activity Trend: {trend}")
                
                # Volume trend
                volumes = [data.get("average_volume", 0) for data in results.values()]
                if len(volumes) >= 2 and volumes[0] > 0 and volumes[1] > 0:
                    if volumes[0] > volumes[1] * 1.2:
                        print(f"   Volume Trend: 📈 Increasing whale activity")
                    elif volumes[0] < volumes[1] * 0.8:
                        print(f"   Volume Trend: 📉 Decreasing whale activity")
                    else:
                        print(f"   Volume Trend: ➡️ Stable whale activity")
        
        print("\n" + "=" * 60)
        return results
    
    def display_whale_analysis(self, days_back: int = 1):
        """Display comprehensive whale tracking analysis"""
        print("\n🐋 WHALE TRACKER ANALYSIS")
        print("=" * 60)
        
        period_name = f"{days_back} day{'s' if days_back > 1 else ''}"
        print(f"📅 Analysis Period: Last {period_name}")
        
        # Get market data
        print("📊 Fetching market data...")
        market_data = self.get_current_price_and_volume()
        
        if market_data:
            print(f"💰 Current Price: ${market_data['price']:.2f}")
            print(f"📈 24h Change: {market_data['price_change_24h']:+.2f}%")
            print(f"📊 24h Volume: {market_data['volume_24h']:,.0f} BNB")
            print(f"💵 24h Volume (USDT): ${market_data['volume_usdt_24h']:,.0f}")
        
        print(f"⏰ Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get whale activity summary for the specified period  
        print(f"\n🔍 Analyzing whale activity for last {period_name}...")
        whale_summary = self.get_whale_activity_summary(days_back=days_back)
        
        if whale_summary:
            print(f"✅ Analyzed {whale_summary['total_candles']} {whale_summary['interval']} candles")
            
            # Volume analysis
            vol_analysis = whale_summary.get("volume_analysis", {})
            if vol_analysis:
                print(f"\n📊 VOLUME ANALYSIS:")
                print("-" * 40)
                print(f"   Average Volume: {vol_analysis['average_volume']:,.0f} BNB")
                print(f"   Max Volume: {vol_analysis['max_volume']:,.0f} BNB")
                print(f"   Spike Threshold: {vol_analysis['volume_spike_threshold']:,.0f} BNB")
            
            # High volume periods (whale activity)
            high_vol_periods = whale_summary.get("high_volume_periods", [])
            if high_vol_periods:
                print(f"\n🐋 WHALE ACTIVITY DETECTED:")
                print("-" * 50)
                print(f"   Found {len(high_vol_periods)} high volume periods")
                
                for i, period in enumerate(high_vol_periods[:10]):
                    time_str = period["timestamp"].strftime("%m/%d %H:%M")
                    volume_ratio = period["volume_ratio"]
                    price_change = period["price_change"]
                    signal = period["whale_signal"]
                    
                    print(f"{i+1:2}. {signal}")
                    print(f"    📊 Volume: {period['volume']:,.0f} BNB ({volume_ratio:.1f}x avg)")
                    print(f"    📈 Price Change: {price_change:+.2f}% | ⏰ {time_str}")
                    print()
                
                # Analyze whale sentiment from volume patterns
                buy_signals = len([p for p in high_vol_periods if "BUY" in p["whale_signal"]])
                sell_signals = len([p for p in high_vol_periods if "SELL" in p["whale_signal"]])
                total_signals = buy_signals + sell_signals
                
                if total_signals > 0:
                    buy_ratio = (buy_signals / total_signals) * 100
                    sell_ratio = (sell_signals / total_signals) * 100
                    
                    print(f"\n🧠 WHALE SENTIMENT ANALYSIS:")
                    print("-" * 40)
                    
                    if buy_ratio > 60:
                        sentiment_level = "🟢 BULLISH"
                    elif sell_ratio > 60:
                        sentiment_level = "🔴 BEARISH"
                    else:
                        sentiment_level = "🟡 NEUTRAL"
                    
                    print(f"   Whale Sentiment: {sentiment_level}")
                    print(f"   Buy Signals: {buy_signals} ({buy_ratio:.1f}%)")
                    print(f"   Sell Signals: {sell_signals} ({sell_ratio:.1f}%)")
                    print(f"   Confidence: {min(95, 50 + abs(buy_ratio - 50))}%")
                    
                    sentiment = {
                        "sentiment": sentiment_level,
                        "buy_ratio": buy_ratio,
                        "sell_ratio": sell_ratio,
                        "buy_count": buy_signals,
                        "sell_count": sell_signals,
                        "confidence": min(95, 50 + abs(buy_ratio - 50))
                    }
            else:
                print(f"\n🐋 WHALE ACTIVITY:")
                print("-" * 40)
                print("   ✅ No significant whale activity detected")
                print("   📊 Volume levels remain within normal ranges")
        else:
            print("❌ Could not fetch whale activity data")
            sentiment = {"sentiment": "🟡 NO DATA", "confidence": 0}
        
        # Order book whale walls
        print(f"\n📋 WHALE WALLS ANALYSIS:")
        print("-" * 40)
        orderbook_data = self.analyze_order_book_whales()
        
        if orderbook_data:
            if orderbook_data.get("largest_bid_wall"):
                price, qty = orderbook_data["largest_bid_wall"]
                print(f"🟢 Largest Buy Wall: {qty:,.0f} BNB @ ${price:.2f}")
            
            if orderbook_data.get("largest_ask_wall"):
                price, qty = orderbook_data["largest_ask_wall"]
                print(f"🔴 Largest Sell Wall: {qty:,.0f} BNB @ ${price:.2f}")
            
            print(f"📊 Whale Buy Orders: {orderbook_data.get('whale_bid_count', 0)}")
            print(f"📊 Whale Sell Orders: {orderbook_data.get('whale_ask_count', 0)}")
            
            support = orderbook_data.get("total_whale_support", 0)
            resistance = orderbook_data.get("total_whale_resistance", 0)
            if support > 0 or resistance > 0:
                print(f"🛡️ Total Whale Support: ${support:,.0f}")
                print(f"⚡ Total Whale Resistance: ${resistance:,.0f}")
        
        # Exchange flows
        print(f"\n🏦 EXCHANGE FLOWS (24h):")
        print("-" * 40)
        flows = self.get_exchange_flows()
        
        print(f"📥 Total Inflows: {flows['total_inflows']:,.0f} BNB")
        print(f"📤 Total Outflows: {flows['total_outflows']:,.0f} BNB")
        print(f"⚖️ Net Flow: {flows['net_flow']:+,.0f} BNB")
        print(f"🎯 Flow Sentiment: {flows['flow_sentiment']}")
        
        if flows.get("outflows_24h"):
            print(f"\n📤 Recent Large Outflows:")
            for flow in flows["outflows_24h"][:3]:
                time_str = flow["time"].strftime("%H:%M")
                print(f"   • {flow['exchange']}: {flow['amount']:,.0f} BNB @ {time_str}")
        
        # Unusual activity detection
        print(f"\n🚨 UNUSUAL ACTIVITY ALERTS:")
        print("-" * 40)
        alerts = self.detect_unusual_activity(market_data, whale_summary.get("high_volume_periods", []))
        
        if alerts:
            for alert in alerts:
                print(f"   • {alert}")
        else:
            print("   ✅ No unusual activity detected")
        
        # Trading implications
        print(f"\n💡 WHALE TRADING IMPLICATIONS:")
        print("-" * 40)
        
        if sentiment.get("sentiment"):
            if "STRONG BULLISH" in sentiment["sentiment"]:
                print("   🚀 Whales are heavily accumulating - Strong bullish signal")
                print("   🎯 Consider long positions following whale activity")
                print("   🛡️ Use whale support levels as stop loss")
            elif "BULLISH" in sentiment["sentiment"]:
                print("   📈 Whales showing bullish bias - Moderate buy signal")
                print("   🎯 Look for entry on pullbacks to whale support")
            elif "STRONG BEARISH" in sentiment["sentiment"]:
                print("   📉 Whales are distributing - Strong bearish signal")
                print("   🎯 Consider short positions or avoid longs")
                print("   ⚡ Watch whale resistance levels")
            elif "BEARISH" in sentiment["sentiment"]:
                print("   🔻 Whales showing bearish bias - Caution advised")
                print("   🎯 Wait for whale accumulation before buying")
            else:
                print("   ⚖️ Whales are neutral - Wait for clearer signals")
                print("   🎯 Monitor for changes in whale behavior")
        
        # Net flow implications
        if flows.get("net_flow"):
            if flows["net_flow"] > 50000:
                print("   🏦 Strong exchange outflows - Whales moving to cold storage (Bullish)")
            elif flows["net_flow"] < -50000:
                print("   🏦 Strong exchange inflows - Whales preparing to sell (Bearish)")
        
        print("\n" + "=" * 60)
        
        return {
            "market_data": market_data,
            "whale_summary": whale_summary,
            "sentiment": sentiment,
            "orderbook": orderbook_data,
            "flows": flows,
            "alerts": alerts
        }
    
    def check_critical_whale_activity(self, days_back: int = 1) -> Dict:
        """Check if there's critical whale activity that should be shown automatically"""
        
        try:
            # Get whale activity summary (lightweight version)
            whale_summary = self.get_whale_activity_summary(days_back)
            
            if not whale_summary or "high_volume_periods" not in whale_summary:
                return {"show_alert": False, "reason": "No data available"}
            
            critical_signals = []
            alert_score = 0
            
            # Check volume spikes
            high_vol_periods = whale_summary["high_volume_periods"]
            mega_whale_count = sum(1 for p in high_vol_periods if "MEGA WHALE" in p.get("whale_signal", ""))
            extreme_whale_count = sum(1 for p in high_vol_periods if "EXTREME WHALE" in p.get("whale_signal", ""))
            
            if extreme_whale_count > 0:
                critical_signals.append(f"🚨 {extreme_whale_count} EXTREME WHALE signal(s)")
                alert_score += 10
            
            if mega_whale_count >= 2:
                critical_signals.append(f"🐋 {mega_whale_count} MEGA WHALE signals")
                alert_score += 6
            
            # Check volume spikes
            volume_spikes = [p for p in high_vol_periods if p.get("volume_multiplier", 1) >= self.alert_thresholds["critical_volume_spike"]]
            if len(volume_spikes) > 0:
                max_spike = max(p.get("volume_multiplier", 1) for p in volume_spikes)
                critical_signals.append(f"📊 Volume spike: {max_spike:.1f}x normal")
                alert_score += 5
            
            # Check price impact
            significant_moves = [p for p in high_vol_periods if abs(p.get("price_change_pct", 0)) >= self.alert_thresholds["price_impact"] * 100]
            if len(significant_moves) > 0:
                max_move = max(abs(p.get("price_change_pct", 0)) for p in significant_moves)
                critical_signals.append(f"💥 Price impact: {max_move:.1f}%")
                alert_score += 4
            
            # Check multiple signals
            total_whale_signals = mega_whale_count + extreme_whale_count
            if total_whale_signals >= self.alert_thresholds["multiple_whale_signals"]:
                critical_signals.append(f"⚡ Multiple whale signals: {total_whale_signals}")
                alert_score += 3
            
            # Determine if alert should be shown
            show_alert = alert_score >= 8  # Threshold for showing alert
            
            return {
                "show_alert": show_alert,
                "alert_score": alert_score,
                "critical_signals": critical_signals,
                "period": f"{days_back} day{'s' if days_back > 1 else ''}",
                "summary": {
                    "mega_whale_count": mega_whale_count,
                    "extreme_whale_count": extreme_whale_count,
                    "volume_spikes": len(volume_spikes),
                    "price_impacts": len(significant_moves)
                }
            }
            
        except Exception as e:
            return {"show_alert": False, "reason": f"Error checking whale activity: {e}"}
    
    def get_critical_whale_alert_text(self, alert_data: Dict) -> str:
        """Generate formatted alert text for critical whale activity"""
        
        if not alert_data.get("show_alert"):
            return ""
        
        signals = alert_data.get("critical_signals", [])
        period = alert_data.get("period", "recent")
        
        alert_text = f"\n🚨 CRITICAL WHALE ACTIVITY DETECTED ({period.upper()})\n"
        alert_text += "=" * 55 + "\n"
        
        for signal in signals:
            alert_text += f"{signal}\n"
        
        alert_text += f"\nAlert Score: {alert_data.get('alert_score', 0)}/20"
        alert_text += "\n💡 Consider: Check whale tracking analysis for details"
        alert_text += "\n" + "=" * 55
        
        return alert_text


# Example usage
if __name__ == "__main__":
    tracker = WhaleTracker()
    
    print("🐋 WHALE TRACKER - PERIOD SELECTION")
    print("=" * 50)
    print("1. Last 24 hours")
    print("2. Last 3 days") 
    print("3. Last week")
    print("4. Multi-period analysis")
    
    try:
        choice = input("\nSelect period (1-4): ").strip()
        
        if choice == "1":
            results = tracker.display_whale_analysis(days_back=1)
        elif choice == "2":
            results = tracker.display_whale_analysis(days_back=3)
        elif choice == "3":
            results = tracker.display_whale_analysis(days_back=7)
        elif choice == "4":
            results = tracker.multi_period_whale_analysis()
        else:
            print("Invalid choice, using default (24 hours)")
            results = tracker.display_whale_analysis(days_back=1)
            
    except KeyboardInterrupt:
        print("\n\n👋 Analysis stopped by user.")
