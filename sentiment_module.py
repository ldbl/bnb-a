#!/usr/bin/env python3
"""
Sentiment Analysis Module
Combines Fear & Greed Index with social media sentiment analysis
"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import re


class SentimentAnalyzer:
    """Analyze market sentiment using multiple sources"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        
        # Fear & Greed thresholds
        self.fear_greed_levels = {
            "extreme_fear": (0, 20),
            "fear": (21, 40),
            "neutral": (41, 60),
            "greed": (61, 80),
            "extreme_greed": (81, 100)
        }
        
        # Social sentiment keywords
        self.bullish_keywords = [
            "moon", "bullish", "pump", "rocket", "hodl", "buy", "long",
            "bull", "green", "gains", "profit", "lambo", "diamond hands",
            "to the moon", "ath", "breakout", "rally", "surge"
        ]
        
        self.bearish_keywords = [
            "dump", "crash", "bear", "red", "sell", "short", "drop",
            "fall", "dip", "correction", "loss", "panic", "fear",
            "bearish", "decline", "plummet", "tank", "blood"
        ]
        
        # News sentiment indicators
        self.positive_news_indicators = [
            "partnership", "adoption", "upgrade", "launch", "announcement",
            "integration", "expansion", "growth", "investment", "backing"
        ]
        
        self.negative_news_indicators = [
            "regulation", "ban", "hack", "exploit", "investigation",
            "lawsuit", "warning", "restriction", "concern", "volatility"
        ]
    
    def get_fear_greed_index(self) -> Dict:
        """Get Fear & Greed Index (simulated - would use real API)"""
        # This would normally call the actual Fear & Greed API
        # For demo purposes, we'll simulate based on current market conditions
        
        try:
            # Get market data to simulate Fear & Greed
            response = requests.get(f"{self.base_url}/ticker/24hr", 
                                  params={"symbol": "BNBUSDT"})
            
            if response.status_code == 200:
                data = response.json()
                price_change = float(data["priceChangePercent"])
                volume = float(data["volume"])
                
                # Simulate Fear & Greed based on price action and volume
                base_score = 50  # Neutral baseline
                
                # Price influence (±30 points)
                if price_change > 5:
                    price_factor = min(30, price_change * 3)
                elif price_change < -5:
                    price_factor = max(-30, price_change * 3)
                else:
                    price_factor = price_change * 2
                
                # Volume influence (±10 points)
                volume_factor = min(10, (volume - 400000) / 50000) if volume > 400000 else max(-10, (volume - 400000) / 50000)
                
                # Calculate final score
                fear_greed_score = int(base_score + price_factor + volume_factor)
                fear_greed_score = max(0, min(100, fear_greed_score))  # Clamp to 0-100
                
                # Determine level
                level = self.get_fear_greed_level(fear_greed_score)
                
                return {
                    "score": fear_greed_score,
                    "level": level,
                    "timestamp": datetime.now(),
                    "previous_score": fear_greed_score - 3,  # Simulated previous
                    "change": 3,
                    "factors": {
                        "price_change": price_change,
                        "volume": volume,
                        "price_factor": round(price_factor, 1),
                        "volume_factor": round(volume_factor, 1)
                    }
                }
        except Exception as e:
            print(f"Error getting Fear & Greed Index: {e}")
        
        return {
            "score": 50,
            "level": "neutral",
            "timestamp": datetime.now(),
            "error": "Could not fetch data"
        }
    
    def get_fear_greed_level(self, score: int) -> str:
        """Convert Fear & Greed score to level"""
        for level, (min_score, max_score) in self.fear_greed_levels.items():
            if min_score <= score <= max_score:
                return level
        return "neutral"
    
    def analyze_social_sentiment(self) -> Dict:
        """Simulate social media sentiment analysis"""
        # This would normally integrate with Twitter API, Reddit API, etc.
        # For demo purposes, we'll simulate sentiment based on market conditions
        
        current_time = datetime.now()
        
        # Simulated social media mentions
        simulated_mentions = {
            "twitter": {
                "total_mentions": 1247,
                "bullish_mentions": 456,
                "bearish_mentions": 321,
                "neutral_mentions": 470,
                "top_hashtags": ["#BNB", "#Binance", "#crypto", "#HODL", "#altcoin"],
                "influencer_sentiment": "neutral"
            },
            "reddit": {
                "total_posts": 89,
                "upvoted_posts": 52,
                "downvoted_posts": 23,
                "neutral_posts": 14,
                "subreddits": ["r/cryptocurrency", "r/binance", "r/altcoin"],
                "hot_discussions": 34
            },
            "telegram": {
                "group_mentions": 234,
                "positive_reactions": 156,
                "negative_reactions": 78,
                "sentiment_score": 65
            }
        }
        
        # Calculate overall social sentiment
        twitter_sentiment = (simulated_mentions["twitter"]["bullish_mentions"] - 
                           simulated_mentions["twitter"]["bearish_mentions"]) / simulated_mentions["twitter"]["total_mentions"]
        
        reddit_sentiment = (simulated_mentions["reddit"]["upvoted_posts"] - 
                          simulated_mentions["reddit"]["downvoted_posts"]) / simulated_mentions["reddit"]["total_posts"]
        
        telegram_sentiment = (simulated_mentions["telegram"]["positive_reactions"] - 
                             simulated_mentions["telegram"]["negative_reactions"]) / simulated_mentions["telegram"]["group_mentions"]
        
        # Weighted average
        overall_sentiment = (twitter_sentiment * 0.5 + reddit_sentiment * 0.3 + telegram_sentiment * 0.2)
        sentiment_score = int(50 + overall_sentiment * 50)  # Convert to 0-100 scale
        
        # Determine sentiment level
        if sentiment_score >= 70:
            sentiment_level = "🟢 VERY POSITIVE"
        elif sentiment_score >= 60:
            sentiment_level = "🟢 POSITIVE"
        elif sentiment_score >= 40:
            sentiment_level = "🟡 NEUTRAL"
        elif sentiment_score >= 30:
            sentiment_level = "🔴 NEGATIVE"
        else:
            sentiment_level = "🔴 VERY NEGATIVE"
        
        return {
            "overall_sentiment": sentiment_level,
            "sentiment_score": sentiment_score,
            "platforms": simulated_mentions,
            "individual_sentiments": {
                "twitter": twitter_sentiment,
                "reddit": reddit_sentiment,
                "telegram": telegram_sentiment
            },
            "timestamp": current_time
        }
    
    def analyze_news_sentiment(self) -> Dict:
        """Simulate news sentiment analysis"""
        # This would normally use news APIs like NewsAPI, CoinAPI, etc.
        # For demo purposes, we'll simulate recent news sentiment
        
        simulated_news = [
            {
                "title": "Binance Expands Services in European Markets",
                "sentiment": "positive",
                "relevance": 0.8,
                "timestamp": datetime.now() - timedelta(hours=2),
                "source": "CoinDesk"
            },
            {
                "title": "BNB Shows Strong Technical Indicators Amid Market Volatility",
                "sentiment": "neutral",
                "relevance": 0.9,
                "timestamp": datetime.now() - timedelta(hours=5),
                "source": "CoinTelegraph"
            },
            {
                "title": "Regulatory Concerns Continue to Impact Crypto Markets",
                "sentiment": "negative",
                "relevance": 0.6,
                "timestamp": datetime.now() - timedelta(hours=8),
                "source": "Reuters"
            },
            {
                "title": "DeFi Growth Drives Increased Demand for BNB",
                "sentiment": "positive",
                "relevance": 0.7,
                "timestamp": datetime.now() - timedelta(hours=12),
                "source": "Decrypt"
            }
        ]
        
        # Calculate weighted news sentiment
        total_weight = 0
        weighted_sentiment = 0
        
        for news in simulated_news:
            weight = news["relevance"]
            if news["sentiment"] == "positive":
                sentiment_value = 1
            elif news["sentiment"] == "negative":
                sentiment_value = -1
            else:
                sentiment_value = 0
            
            weighted_sentiment += sentiment_value * weight
            total_weight += weight
        
        if total_weight > 0:
            avg_sentiment = weighted_sentiment / total_weight
        else:
            avg_sentiment = 0
        
        # Convert to score
        news_score = int(50 + avg_sentiment * 30)
        
        # Determine level
        if news_score >= 65:
            news_level = "🟢 POSITIVE NEWS"
        elif news_score >= 45:
            news_level = "🟡 NEUTRAL NEWS"
        else:
            news_level = "🔴 NEGATIVE NEWS"
        
        return {
            "news_sentiment": news_level,
            "news_score": news_score,
            "recent_news": simulated_news,
            "positive_count": len([n for n in simulated_news if n["sentiment"] == "positive"]),
            "negative_count": len([n for n in simulated_news if n["sentiment"] == "negative"]),
            "neutral_count": len([n for n in simulated_news if n["sentiment"] == "neutral"])
        }
    
    def get_market_momentum_indicators(self) -> Dict:
        """Get momentum indicators that affect sentiment"""
        try:
            # Get multiple timeframe data
            intervals = ["1h", "4h", "1d"]
            momentum_data = {}
            
            for interval in intervals:
                response = requests.get(f"{self.base_url}/klines", 
                                      params={
                                          "symbol": "BNBUSDT",
                                          "interval": interval,
                                          "limit": 24
                                      })
                
                if response.status_code == 200:
                    klines = response.json()
                    closes = [float(k[4]) for k in klines]
                    
                    if len(closes) >= 2:
                        price_change = (closes[-1] - closes[0]) / closes[0] * 100
                        momentum_data[interval] = {
                            "price_change": round(price_change, 2),
                            "trend": "🟢 UP" if price_change > 0 else "🔴 DOWN" if price_change < 0 else "🟡 FLAT"
                        }
            
            # Calculate overall momentum score
            momentum_score = 50  # Neutral baseline
            
            for interval, data in momentum_data.items():
                change = data["price_change"]
                if interval == "1h":
                    weight = 0.2
                elif interval == "4h":
                    weight = 0.3
                else:  # 1d
                    weight = 0.5
                
                momentum_score += change * weight * 2
            
            momentum_score = max(0, min(100, int(momentum_score)))
            
            return {
                "momentum_score": momentum_score,
                "timeframe_data": momentum_data,
                "overall_momentum": "🟢 BULLISH" if momentum_score > 60 else "🔴 BEARISH" if momentum_score < 40 else "🟡 NEUTRAL"
            }
            
        except Exception as e:
            print(f"Error getting momentum indicators: {e}")
            return {"momentum_score": 50, "error": str(e)}
    
    def calculate_composite_sentiment(self, fear_greed: Dict, social: Dict, 
                                    news: Dict, momentum: Dict) -> Dict:
        """Calculate composite sentiment score from all sources"""
        
        # Weights for different sentiment sources
        weights = {
            "fear_greed": 0.3,
            "social": 0.25,
            "news": 0.25,
            "momentum": 0.2
        }
        
        # Get individual scores
        fg_score = fear_greed.get("score", 50)
        social_score = social.get("sentiment_score", 50)
        news_score = news.get("news_score", 50)
        momentum_score = momentum.get("momentum_score", 50)
        
        # Calculate weighted average
        composite_score = (
            fg_score * weights["fear_greed"] +
            social_score * weights["social"] +
            news_score * weights["news"] +
            momentum_score * weights["momentum"]
        )
        
        composite_score = int(composite_score)
        
        # Determine overall sentiment
        if composite_score >= 75:
            overall_sentiment = "🟢 EXTREMELY BULLISH"
            action = "STRONG_BUY"
        elif composite_score >= 65:
            overall_sentiment = "🟢 BULLISH"
            action = "BUY"
        elif composite_score >= 55:
            overall_sentiment = "🟢 SLIGHTLY BULLISH"
            action = "WEAK_BUY"
        elif composite_score >= 45:
            overall_sentiment = "🟡 NEUTRAL"
            action = "WAIT"
        elif composite_score >= 35:
            overall_sentiment = "🔴 SLIGHTLY BEARISH"
            action = "WEAK_SELL"
        elif composite_score >= 25:
            overall_sentiment = "🔴 BEARISH"
            action = "SELL"
        else:
            overall_sentiment = "🔴 EXTREMELY BEARISH"
            action = "STRONG_SELL"
        
        return {
            "composite_score": composite_score,
            "overall_sentiment": overall_sentiment,
            "action": action,
            "individual_scores": {
                "fear_greed": fg_score,
                "social_media": social_score,
                "news": news_score,
                "momentum": momentum_score
            },
            "weights": weights,
            "confidence": min(95, 60 + abs(composite_score - 50))
        }
    
    def get_sentiment_trading_signals(self, composite: Dict) -> Dict:
        """Generate trading signals based on sentiment analysis"""
        
        score = composite["composite_score"]
        action = composite["action"]
        
        signals = {
            "primary_signal": action,
            "strength": composite["confidence"],
            "entry_zones": [],
            "targets": [],
            "stop_loss": None,
            "position_size": "0%",
            "time_horizon": "short_term",
            "risk_level": "medium"
        }
        
        # Get current price for calculations
        try:
            response = requests.get(f"{self.base_url}/ticker/price", 
                                  params={"symbol": "BNBUSDT"})
            if response.status_code == 200:
                current_price = float(response.json()["price"])
                
                if action in ["STRONG_BUY", "BUY"]:
                    signals["entry_zones"] = [
                        f"Current: ${current_price:.2f}",
                        f"Pullback: ${current_price * 0.98:.2f}",
                        f"Strong dip: ${current_price * 0.95:.2f}"
                    ]
                    signals["targets"] = [
                        f"Target 1: ${current_price * 1.03:.2f}",
                        f"Target 2: ${current_price * 1.05:.2f}",
                        f"Target 3: ${current_price * 1.08:.2f}"
                    ]
                    signals["stop_loss"] = f"${current_price * 0.93:.2f}"
                    signals["position_size"] = "25%" if action == "BUY" else "40%"
                    
                elif action in ["STRONG_SELL", "SELL"]:
                    signals["entry_zones"] = [
                        f"Current: ${current_price:.2f}",
                        f"Bounce: ${current_price * 1.02:.2f}",
                        f"Strong bounce: ${current_price * 1.05:.2f}"
                    ]
                    signals["targets"] = [
                        f"Target 1: ${current_price * 0.97:.2f}",
                        f"Target 2: ${current_price * 0.95:.2f}",
                        f"Target 3: ${current_price * 0.92:.2f}"
                    ]
                    signals["stop_loss"] = f"${current_price * 1.07:.2f}"
                    signals["position_size"] = "20%" if action == "SELL" else "35%"
                
                # Adjust risk level based on sentiment strength
                if score > 75 or score < 25:
                    signals["risk_level"] = "high"
                elif 40 <= score <= 60:
                    signals["risk_level"] = "low"
        
        except Exception as e:
            print(f"Error generating trading signals: {e}")
        
        return signals
    
    def display_sentiment_analysis(self):
        """Display comprehensive sentiment analysis"""
        print("\n🎭 SENTIMENT ANALYSIS")
        print("=" * 60)
        
        print("📊 Gathering sentiment data from multiple sources...")
        print(f"⏰ Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get all sentiment data
        print("\n📈 Fetching Fear & Greed Index...")
        fear_greed = self.get_fear_greed_index()
        
        print("📱 Analyzing social media sentiment...")
        social = self.analyze_social_sentiment()
        
        print("📰 Processing news sentiment...")
        news = self.analyze_news_sentiment()
        
        print("📊 Calculating momentum indicators...")
        momentum = self.get_market_momentum_indicators()
        
        print("🧮 Computing composite sentiment...")
        composite = self.calculate_composite_sentiment(fear_greed, social, news, momentum)
        
        # Display Fear & Greed Index
        print(f"\n😨 FEAR & GREED INDEX:")
        print("-" * 40)
        
        if "error" not in fear_greed:
            score = fear_greed["score"]
            level = fear_greed["level"].replace("_", " ").title()
            change = fear_greed.get("change", 0)
            change_icon = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            
            print(f"   Score: {score}/100")
            print(f"   Level: {level}")
            print(f"   Change: {change_icon} {change:+d} points")
            
            if score <= 20:
                print(f"   💡 Extreme Fear - Often good buying opportunity")
            elif score >= 80:
                print(f"   ⚠️ Extreme Greed - Consider taking profits")
            else:
                print(f"   🎯 {level} sentiment in the market")
        
        # Display Social Media Sentiment
        print(f"\n📱 SOCIAL MEDIA SENTIMENT:")
        print("-" * 40)
        
        print(f"   Overall: {social['overall_sentiment']}")
        print(f"   Score: {social['sentiment_score']}/100")
        
        twitter_data = social['platforms']['twitter']
        print(f"   🐦 Twitter: {twitter_data['total_mentions']} mentions")
        print(f"     • Bullish: {twitter_data['bullish_mentions']} | Bearish: {twitter_data['bearish_mentions']}")
        
        reddit_data = social['platforms']['reddit']
        print(f"   📺 Reddit: {reddit_data['total_posts']} posts")
        print(f"     • Upvoted: {reddit_data['upvoted_posts']} | Downvoted: {reddit_data['downvoted_posts']}")
        
        telegram_data = social['platforms']['telegram']
        print(f"   💬 Telegram: {telegram_data['group_mentions']} mentions")
        print(f"     • Positive: {telegram_data['positive_reactions']} | Negative: {telegram_data['negative_reactions']}")
        
        # Display News Sentiment
        print(f"\n📰 NEWS SENTIMENT:")
        print("-" * 40)
        
        print(f"   Overall: {news['news_sentiment']}")
        print(f"   Score: {news['news_score']}/100")
        print(f"   Positive Articles: {news['positive_count']}")
        print(f"   Negative Articles: {news['negative_count']}")
        print(f"   Neutral Articles: {news['neutral_count']}")
        
        print(f"\n   📑 Recent Headlines:")
        for i, article in enumerate(news['recent_news'][:3], 1):
            sentiment_icon = "🟢" if article['sentiment'] == 'positive' else "🔴" if article['sentiment'] == 'negative' else "🟡"
            time_str = article['timestamp'].strftime("%H:%M")
            print(f"     {i}. {sentiment_icon} {article['title'][:50]}...")
            print(f"        {article['source']} | {time_str} | Relevance: {article['relevance']:.1f}")
        
        # Display Momentum Indicators
        print(f"\n📊 MOMENTUM INDICATORS:")
        print("-" * 40)
        
        if "error" not in momentum:
            print(f"   Overall Momentum: {momentum['overall_momentum']}")
            print(f"   Momentum Score: {momentum['momentum_score']}/100")
            
            print(f"   Timeframe Analysis:")
            for timeframe, data in momentum['timeframe_data'].items():
                print(f"     {timeframe}: {data['trend']} ({data['price_change']:+.2f}%)")
        
        # Display Composite Sentiment
        print(f"\n🎯 COMPOSITE SENTIMENT ANALYSIS:")
        print("=" * 40)
        
        print(f"   Overall Sentiment: {composite['overall_sentiment']}")
        print(f"   Composite Score: {composite['composite_score']}/100")
        print(f"   Confidence: {composite['confidence']}%")
        print(f"   Action: {composite['action']}")
        
        print(f"\n   📊 Component Breakdown:")
        scores = composite['individual_scores']
        weights = composite['weights']
        
        print(f"     Fear & Greed: {scores['fear_greed']}/100 (Weight: {weights['fear_greed']:.0%})")
        print(f"     Social Media: {scores['social_media']}/100 (Weight: {weights['social']:.0%})")
        print(f"     News: {scores['news']}/100 (Weight: {weights['news']:.0%})")
        print(f"     Momentum: {scores['momentum']}/100 (Weight: {weights['momentum']:.0%})")
        
        # Generate and display trading signals
        print(f"\n💡 SENTIMENT-BASED TRADING SIGNALS:")
        print("-" * 40)
        
        signals = self.get_sentiment_trading_signals(composite)
        
        print(f"   Primary Signal: {signals['primary_signal']}")
        print(f"   Signal Strength: {signals['strength']}%")
        print(f"   Risk Level: {signals['risk_level'].title()}")
        print(f"   Position Size: {signals['position_size']}")
        
        if signals['entry_zones']:
            print(f"   Entry Zones:")
            for zone in signals['entry_zones']:
                print(f"     🎯 {zone}")
        
        if signals['targets']:
            print(f"   Targets:")
            for target in signals['targets']:
                print(f"     🏆 {target}")
        
        if signals['stop_loss']:
            print(f"   Stop Loss: 🛑 {signals['stop_loss']}")
        
        # Market interpretation
        print(f"\n🧠 SENTIMENT INTERPRETATION:")
        print("-" * 40)
        
        score = composite['composite_score']
        
        if score >= 70:
            print("   📈 Strong bullish sentiment across all indicators")
            print("   💡 Market euphoria - Great for riding trends, watch for tops")
            print("   🎯 Consider taking profits if you're already long")
        elif score >= 60:
            print("   🟢 Moderately bullish sentiment")
            print("   💡 Good environment for long positions")
            print("   🎯 Look for pullbacks as entry opportunities")
        elif score >= 40:
            print("   🟡 Mixed/Neutral sentiment")
            print("   💡 Market is undecided - wait for clearer signals")
            print("   🎯 Focus on technical analysis for direction")
        elif score >= 30:
            print("   🔴 Moderately bearish sentiment")
            print("   💡 Caution advised - consider defensive positioning")
            print("   🎯 Short-term rallies may be selling opportunities")
        else:
            print("   📉 Strong bearish sentiment")
            print("   💡 High fear in market - potential contrarian opportunity")
            print("   🎯 Wait for capitulation before considering longs")
        
        print("\n" + "=" * 60)
        
        return {
            "fear_greed": fear_greed,
            "social": social,
            "news": news,
            "momentum": momentum,
            "composite": composite,
            "signals": signals
        }


# Example usage
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    results = analyzer.display_sentiment_analysis()
