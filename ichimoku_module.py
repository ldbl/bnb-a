#!/usr/bin/env python3
"""
Ichimoku Cloud Analysis Module
Complete Ichimoku Kinko Hyo (Equilibrium Chart) implementation
"""

import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import time


class IchimokuAnalyzer:
    """Ichimoku Cloud technical analysis"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        
        # Ichimoku parameters (standard settings)
        self.tenkan_period = 9    # Conversion Line
        self.kijun_period = 26    # Base Line  
        self.senkou_span_b_period = 52  # Leading Span B
        self.chikou_span_offset = 26    # Lagging Span offset
        self.senkou_span_offset = 26    # Cloud offset
    
    def fetch_ichimoku_data(self, interval: str = "1d", limit: int = 100):
        """Fetch data for Ichimoku analysis"""
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
            print(f"Error fetching data: {e}")
            return []
    
    def process_klines_data(self, klines: List) -> Dict:
        """Process klines into OHLC data"""
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
    
    def calculate_tenkan_sen(self, highs: List[float], lows: List[float]) -> List[Optional[float]]:
        """Calculate Tenkan Sen (Conversion Line) - (Highest High + Lowest Low) / 2 over 9 periods"""
        tenkan_values = []
        
        for i in range(len(highs)):
            if i < self.tenkan_period - 1:
                tenkan_values.append(None)
            else:
                period_highs = highs[i - self.tenkan_period + 1:i + 1]
                period_lows = lows[i - self.tenkan_period + 1:i + 1]
                
                highest_high = max(period_highs)
                lowest_low = min(period_lows)
                
                tenkan_value = (highest_high + lowest_low) / 2
                tenkan_values.append(tenkan_value)
        
        return tenkan_values
    
    def calculate_kijun_sen(self, highs: List[float], lows: List[float]) -> List[Optional[float]]:
        """Calculate Kijun Sen (Base Line) - (Highest High + Lowest Low) / 2 over 26 periods"""
        kijun_values = []
        
        for i in range(len(highs)):
            if i < self.kijun_period - 1:
                kijun_values.append(None)
            else:
                period_highs = highs[i - self.kijun_period + 1:i + 1]
                period_lows = lows[i - self.kijun_period + 1:i + 1]
                
                highest_high = max(period_highs)
                lowest_low = min(period_lows)
                
                kijun_value = (highest_high + lowest_low) / 2
                kijun_values.append(kijun_value)
        
        return kijun_values
    
    def calculate_senkou_span_a(self, tenkan_values: List[Optional[float]], 
                              kijun_values: List[Optional[float]]) -> List[Optional[float]]:
        """Calculate Senkou Span A (Leading Span A) - (Tenkan + Kijun) / 2, projected 26 periods ahead"""
        senkou_a_values = []
        
        for i in range(len(tenkan_values)):
            if tenkan_values[i] is None or kijun_values[i] is None:
                senkou_a_values.append(None)
            else:
                senkou_a = (tenkan_values[i] + kijun_values[i]) / 2
                senkou_a_values.append(senkou_a)
        
        return senkou_a_values
    
    def calculate_senkou_span_b(self, highs: List[float], lows: List[float]) -> List[Optional[float]]:
        """Calculate Senkou Span B (Leading Span B) - (Highest High + Lowest Low) / 2 over 52 periods"""
        senkou_b_values = []
        
        for i in range(len(highs)):
            if i < self.senkou_span_b_period - 1:
                senkou_b_values.append(None)
            else:
                period_highs = highs[i - self.senkou_span_b_period + 1:i + 1]
                period_lows = lows[i - self.senkou_span_b_period + 1:i + 1]
                
                highest_high = max(period_highs)
                lowest_low = min(period_lows)
                
                senkou_b = (highest_high + lowest_low) / 2
                senkou_b_values.append(senkou_b)
        
        return senkou_b_values
    
    def calculate_chikou_span(self, closes: List[float]) -> List[Optional[float]]:
        """Calculate Chikou Span (Lagging Span) - Current close projected 26 periods back"""
        chikou_values = [None] * len(closes)
        
        for i in range(self.chikou_span_offset, len(closes)):
            chikou_values[i - self.chikou_span_offset] = closes[i]
        
        return chikou_values
    
    def calculate_all_ichimoku_lines(self, data: Dict) -> Dict:
        """Calculate all Ichimoku lines"""
        highs = data["highs"]
        lows = data["lows"]
        closes = data["closes"]
        
        # Calculate all lines
        tenkan_sen = self.calculate_tenkan_sen(highs, lows)
        kijun_sen = self.calculate_kijun_sen(highs, lows)
        senkou_span_a = self.calculate_senkou_span_a(tenkan_sen, kijun_sen)
        senkou_span_b = self.calculate_senkou_span_b(highs, lows)
        chikou_span = self.calculate_chikou_span(closes)
        
        return {
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_span_a": senkou_span_a,
            "senkou_span_b": senkou_span_b,
            "chikou_span": chikou_span,
            "timestamps": data["timestamps"],
            "closes": closes,
            "highs": highs,
            "lows": lows
        }
    
    def analyze_ichimoku_signals(self, ichimoku_data: Dict) -> Dict:
        """Analyze Ichimoku signals and generate trading recommendations"""
        
        # Get current values (last index)
        current_idx = len(ichimoku_data["closes"]) - 1
        current_price = ichimoku_data["closes"][current_idx]
        
        # Current line values
        tenkan_current = ichimoku_data["tenkan_sen"][current_idx]
        kijun_current = ichimoku_data["kijun_sen"][current_idx]
        
        # Cloud values (projected forward, so we look at current position)
        cloud_idx = max(0, current_idx - self.senkou_span_offset)
        senkou_a_current = ichimoku_data["senkou_span_a"][cloud_idx] if cloud_idx < len(ichimoku_data["senkou_span_a"]) else None
        senkou_b_current = ichimoku_data["senkou_span_b"][cloud_idx] if cloud_idx < len(ichimoku_data["senkou_span_b"]) else None
        
        # Chikou Span
        chikou_current = ichimoku_data["chikou_span"][current_idx] if current_idx < len(ichimoku_data["chikou_span"]) else None
        
        signals = {
            "current_price": current_price,
            "tenkan_sen": tenkan_current,
            "kijun_sen": kijun_current,
            "senkou_span_a": senkou_a_current,
            "senkou_span_b": senkou_b_current,
            "chikou_span": chikou_current,
            "signals": [],
            "cloud_status": "NEUTRAL",
            "overall_trend": "NEUTRAL",
            "strength": 0,
            "action": "WAIT"
        }
        
        # Skip analysis if we don't have enough data
        if not all([tenkan_current, kijun_current, senkou_a_current, senkou_b_current]):
            signals["signals"].append("Insufficient data for complete Ichimoku analysis")
            return signals
        
        # 1. Tenkan/Kijun Cross (TK Cross)
        if len(ichimoku_data["tenkan_sen"]) >= 2:
            prev_tenkan = ichimoku_data["tenkan_sen"][current_idx - 1]
            prev_kijun = ichimoku_data["kijun_sen"][current_idx - 1]
            
            if prev_tenkan and prev_kijun:
                # Bullish TK Cross
                if prev_tenkan <= prev_kijun and tenkan_current > kijun_current:
                    signals["signals"].append("🔴→🟢 Bullish TK Cross - Tenkan crossed above Kijun")
                    signals["strength"] += 2
                
                # Bearish TK Cross
                elif prev_tenkan >= prev_kijun and tenkan_current < kijun_current:
                    signals["signals"].append("🟢→🔴 Bearish TK Cross - Tenkan crossed below Kijun")
                    signals["strength"] -= 2
        
        # 2. Price vs Kijun Sen
        if current_price > kijun_current:
            signals["signals"].append("💚 Price above Kijun Sen - Bullish momentum")
            signals["strength"] += 1
        else:
            signals["signals"].append("❤️ Price below Kijun Sen - Bearish momentum")
            signals["strength"] -= 1
        
        # 3. Cloud analysis
        cloud_top = max(senkou_a_current, senkou_b_current)
        cloud_bottom = min(senkou_a_current, senkou_b_current)
        
        if current_price > cloud_top:
            signals["cloud_status"] = "ABOVE_CLOUD"
            signals["signals"].append("☁️⬆️ Price above Cloud - Strong bullish trend")
            signals["strength"] += 3
        elif current_price < cloud_bottom:
            signals["cloud_status"] = "BELOW_CLOUD"
            signals["signals"].append("☁️⬇️ Price below Cloud - Strong bearish trend")
            signals["strength"] -= 3
        else:
            signals["cloud_status"] = "IN_CLOUD"
            signals["signals"].append("☁️🔄 Price in Cloud - Consolidation/uncertainty")
        
        # 4. Cloud color (Span A vs Span B)
        if senkou_a_current > senkou_b_current:
            signals["signals"].append("🟢☁️ Green Cloud - Bullish cloud formation")
            signals["strength"] += 1
        else:
            signals["signals"].append("🔴☁️ Red Cloud - Bearish cloud formation")
            signals["strength"] -= 1
        
        # 5. Chikou Span analysis
        if chikou_current:
            chikou_reference_idx = current_idx - self.chikou_span_offset
            if chikou_reference_idx >= 0:
                reference_price = ichimoku_data["closes"][chikou_reference_idx]
                
                if chikou_current > reference_price:
                    signals["signals"].append("📈 Chikou Span above past price - Confirms bullish momentum")
                    signals["strength"] += 1
                else:
                    signals["signals"].append("📉 Chikou Span below past price - Confirms bearish momentum")
                    signals["strength"] -= 1
        
        # 6. Overall trend determination
        if signals["strength"] >= 4:
            signals["overall_trend"] = "STRONG_BULLISH"
            signals["action"] = "STRONG_BUY"
        elif signals["strength"] >= 2:
            signals["overall_trend"] = "BULLISH"
            signals["action"] = "BUY"
        elif signals["strength"] <= -4:
            signals["overall_trend"] = "STRONG_BEARISH"
            signals["action"] = "STRONG_SELL"
        elif signals["strength"] <= -2:
            signals["overall_trend"] = "BEARISH"
            signals["action"] = "SELL"
        else:
            signals["overall_trend"] = "NEUTRAL"
            signals["action"] = "WAIT"
        
        # 7. Support/Resistance levels
        signals["support_levels"] = []
        signals["resistance_levels"] = []
        
        if signals["cloud_status"] == "ABOVE_CLOUD":
            signals["support_levels"].append(f"Cloud Top: ${cloud_top:.2f}")
            signals["support_levels"].append(f"Kijun Sen: ${kijun_current:.2f}")
        elif signals["cloud_status"] == "BELOW_CLOUD":
            signals["resistance_levels"].append(f"Cloud Bottom: ${cloud_bottom:.2f}")
            signals["resistance_levels"].append(f"Kijun Sen: ${kijun_current:.2f}")
        else:
            signals["support_levels"].append(f"Cloud Bottom: ${cloud_bottom:.2f}")
            signals["resistance_levels"].append(f"Cloud Top: ${cloud_top:.2f}")
        
        return signals
    
    def get_current_price(self) -> Optional[float]:
        """Get current BNB price"""
        try:
            response = requests.get(f"{self.base_url}/ticker/price", 
                                  params={"symbol": "BNBUSDT"})
            if response.status_code == 200:
                return float(response.json()['price'])
        except:
            pass
        return None
    
    def display_ichimoku_analysis(self, interval: str = "1d", limit: int = 100):
        """Display complete Ichimoku analysis"""
        print("\n☁️ ICHIMOKU CLOUD ANALYSIS")
        print("=" * 60)
        
        # Get current price
        current_price = self.get_current_price()
        if current_price:
            print(f"💰 Current BNB Price: ${current_price:.2f}")
        
        print(f"📊 Timeframe: {interval} | Periods: {limit}")
        print(f"⏰ Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Fetch and process data
        print(f"\n📡 Fetching {interval} data...")
        klines = self.fetch_ichimoku_data(interval, limit)
        
        if not klines:
            print("❌ Failed to fetch data")
            return None
        
        data = self.process_klines_data(klines)
        print(f"✅ Processed {len(data['closes'])} candles")
        
        # Calculate Ichimoku
        print("🧮 Calculating Ichimoku lines...")
        ichimoku_data = self.calculate_all_ichimoku_lines(data)
        
        # Analyze signals
        print("🔍 Analyzing Ichimoku signals...")
        signals = self.analyze_ichimoku_signals(ichimoku_data)
        
        # Display results
        print(f"\n☁️ ICHIMOKU INDICATORS:")
        print(f"   Tenkan Sen (9): ${signals['tenkan_sen']:.2f}" if signals['tenkan_sen'] else "   Tenkan Sen: N/A")
        print(f"   Kijun Sen (26): ${signals['kijun_sen']:.2f}" if signals['kijun_sen'] else "   Kijun Sen: N/A")
        print(f"   Senkou Span A: ${signals['senkou_span_a']:.2f}" if signals['senkou_span_a'] else "   Senkou Span A: N/A")
        print(f"   Senkou Span B: ${signals['senkou_span_b']:.2f}" if signals['senkou_span_b'] else "   Senkou Span B: N/A")
        print(f"   Chikou Span: ${signals['chikou_span']:.2f}" if signals['chikou_span'] else "   Chikou Span: N/A")
        
        print(f"\n🎯 ICHIMOKU ANALYSIS:")
        print(f"   Cloud Status: {signals['cloud_status']}")
        print(f"   Overall Trend: {signals['overall_trend']}")
        print(f"   Signal Strength: {signals['strength']}")
        print(f"   Action: {signals['action']}")
        
        print(f"\n📋 DETAILED SIGNALS:")
        for signal in signals['signals']:
            print(f"   • {signal}")
        
        if signals['support_levels']:
            print(f"\n🛡️ SUPPORT LEVELS:")
            for level in signals['support_levels']:
                print(f"   📉 {level}")
        
        if signals['resistance_levels']:
            print(f"\n⚡ RESISTANCE LEVELS:")
            for level in signals['resistance_levels']:
                print(f"   📈 {level}")
        
        # Trading implications
        print(f"\n💡 TRADING IMPLICATIONS:")
        
        if signals['action'] == "STRONG_BUY":
            print(f"   🚀 Strong bullish setup - Consider aggressive long positions")
            print(f"   🎯 Entry: Current levels or pullback to cloud")
            print(f"   🛑 Stop: Below cloud bottom")
        
        elif signals['action'] == "BUY":
            print(f"   📈 Bullish setup - Consider long positions")
            print(f"   🎯 Entry: Pullback to Kijun Sen or cloud top")
            print(f"   🛑 Stop: Below Kijun Sen")
        
        elif signals['action'] == "STRONG_SELL":
            print(f"   📉 Strong bearish setup - Consider short positions")
            print(f"   🎯 Entry: Current levels or bounce to cloud")
            print(f"   🛑 Stop: Above cloud top")
        
        elif signals['action'] == "SELL":
            print(f"   🔻 Bearish setup - Consider short positions")
            print(f"   🎯 Entry: Bounce to Kijun Sen or cloud bottom")
            print(f"   🛑 Stop: Above Kijun Sen")
        
        else:
            print(f"   ⚖️ Neutral/Consolidation - Wait for clearer signals")
            print(f"   🎯 Watch for breakout from cloud or TK cross")
        
        print("\n" + "=" * 60)
        
        return signals
    
    def multi_period_ichimoku_analysis(self):
        """Analyze Ichimoku across different time periods - 3, 6, 12 months"""
        print("\n📅 MULTI-PERIOD ICHIMOKU ANALYSIS")
        print("=" * 60)
        
        # Get current price
        current_price = self.get_current_price()
        if current_price:
            print(f"💰 Current BNB Price: ${current_price:.2f}")
        
        print(f"⏰ Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Define periods
        periods = [
            ("3 месеца", "1d", 90),    # 3 months daily
            ("6 месеца", "1d", 180),   # 6 months daily  
            ("1 година", "1w", 52)     # 1 year weekly
        ]
        
        results = {}
        
        for period_name, interval, limit in periods:
            print(f"\n📊 {period_name} ({interval}):")
            print("-" * 40)
            
            # Fetch data
            klines = self.fetch_ichimoku_data(interval, limit)
            if not klines:
                print(f"❌ Failed to fetch {interval} data")
                continue
            
            # Process and analyze
            data = self.process_klines_data(klines)
            ichimoku_data = self.calculate_all_ichimoku_lines(data)
            signals = self.analyze_ichimoku_signals(ichimoku_data)
            
            results[period_name] = signals
            
            # Display key metrics
            print(f"   Cloud Status: {signals['cloud_status']}")
            print(f"   Overall Trend: {signals['overall_trend']}")
            print(f"   Action: {signals['action']}")
            print(f"   Signal Strength: {signals['strength']}")
            
            # Show key levels
            if signals['tenkan_sen'] and signals['kijun_sen']:
                tk_diff = signals['tenkan_sen'] - signals['kijun_sen']
                tk_status = "🟢 Bullish" if tk_diff > 0 else "🔴 Bearish" if tk_diff < 0 else "🟡 Neutral"
                print(f"   TK Cross: {tk_status} (${tk_diff:.2f})")
            
            # Cloud thickness (strength indicator)
            if signals['senkou_span_a'] and signals['senkou_span_b']:
                cloud_thickness = abs(signals['senkou_span_a'] - signals['senkou_span_b'])
                cloud_strength = "Thick" if cloud_thickness > current_price * 0.02 else "Thin"
                cloud_color = "Green" if signals['senkou_span_a'] > signals['senkou_span_b'] else "Red"
                print(f"   Cloud: {cloud_color} & {cloud_strength} (${cloud_thickness:.2f})")
            
            # Key signals
            key_signals = [s for s in signals['signals'] if any(keyword in s for keyword in ['Cross', 'Cloud', 'Strong'])]
            if key_signals:
                print(f"   Key Signals:")
                for signal in key_signals[:2]:  # Show top 2 signals
                    clean_signal = signal.replace('🔴→🟢', '').replace('🟢→🔴', '').replace('☁️⬆️', '').replace('☁️⬇️', '').replace('🟢☁️', '').replace('🔴☁️', '').strip()
                    print(f"     • {clean_signal}")
        
        # Multi-period summary
        if results:
            print(f"\n🏆 MULTI-PERIOD SUMMARY:")
            print("=" * 40)
            
            actions = [r['action'] for r in results.values()]
            cloud_statuses = [r['cloud_status'] for r in results.values()]
            
            # Count actions
            bullish_count = sum(1 for action in actions if 'BUY' in action)
            bearish_count = sum(1 for action in actions if 'SELL' in action)
            neutral_count = len(actions) - bullish_count - bearish_count
            
            # Count cloud positions
            above_cloud = sum(1 for status in cloud_statuses if status == 'ABOVE_CLOUD')
            below_cloud = sum(1 for status in cloud_statuses if status == 'BELOW_CLOUD')
            in_cloud = sum(1 for status in cloud_statuses if status == 'IN_CLOUD')
            
            # Overall assessment
            if bullish_count >= 2:
                overall_bias = "🟢 BULLISH"
            elif bearish_count >= 2:
                overall_bias = "🔴 BEARISH"
            else:
                overall_bias = "🟡 MIXED"
            
            print(f"   Overall Bias: {overall_bias}")
            print(f"   Bullish Periods: {bullish_count}/{len(results)}")
            print(f"   Bearish Periods: {bearish_count}/{len(results)}")
            print(f"   Neutral Periods: {neutral_count}/{len(results)}")
            print(f"   Above Cloud: {above_cloud}/{len(results)}")
            print(f"   Below Cloud: {below_cloud}/{len(results)}")
            print(f"   In Cloud: {in_cloud}/{len(results)}")
            
            # Trading recommendation
            print(f"\n💡 TRADING RECOMMENDATION:")
            if bullish_count >= 2 and above_cloud >= 2:
                print(f"   📈 Strong multi-period bullish alignment")
                print(f"   🎯 Consider long positions on pullbacks")
                print(f"   🛡️ Use cloud as dynamic support")
            elif bearish_count >= 2 and below_cloud >= 2:
                print(f"   📉 Strong multi-period bearish alignment")
                print(f"   🎯 Consider short positions on bounces")
                print(f"   ⚡ Use cloud as dynamic resistance")
            elif in_cloud >= 2:
                print(f"   🔄 Multi-period consolidation phase")
                print(f"   ⚖️ Wait for clear breakout direction")
                print(f"   🎯 Watch for cloud exits as entry signals")
            else:
                print(f"   🔀 Mixed signals across timeframes")
                print(f"   ⚠️ Exercise caution - conflicting trends")
                print(f"   🎯 Focus on shorter timeframes for entries")
        
        print("\n" + "=" * 60)
        return results
    
    def multi_timeframe_ichimoku(self):
        """Analyze Ichimoku across multiple timeframes"""
        print("\n⏰ MULTI-TIMEFRAME ICHIMOKU ANALYSIS")
        print("=" * 60)
        
        timeframes = [
            ("1d", 30, "Short-term (1 Month)"),
            ("1w", 12, "Medium-term (3 Months)"),
            ("1M", 6, "Long-term (6 Months)")
        ]
        
        results = {}
        
        for interval, limit, description in timeframes:
            print(f"\n📊 {description} ({interval}):")
            print("-" * 30)
            
            # Get data
            klines = self.fetch_ichimoku_data(interval, limit)
            if not klines:
                print(f"❌ Failed to fetch {interval} data")
                continue
            
            # Process and analyze
            data = self.process_klines_data(klines)
            ichimoku_data = self.calculate_all_ichimoku_lines(data)
            signals = self.analyze_ichimoku_signals(ichimoku_data)
            
            results[description] = signals
            
            print(f"   Cloud Status: {signals['cloud_status']}")
            print(f"   Trend: {signals['overall_trend']}")
            print(f"   Action: {signals['action']}")
            print(f"   Strength: {signals['strength']}")
        
        # Summary
        if results:
            print(f"\n🏆 MULTI-TIMEFRAME SUMMARY:")
            print("-" * 30)
            
            actions = [r['action'] for r in results.values()]
            
            bullish_count = sum(1 for action in actions if 'BUY' in action)
            bearish_count = sum(1 for action in actions if 'SELL' in action)
            
            if bullish_count > bearish_count:
                overall = "BULLISH"
                emoji = "🟢"
            elif bearish_count > bullish_count:
                overall = "BEARISH"
                emoji = "🔴"
            else:
                overall = "NEUTRAL"
                emoji = "🟡"
            
            print(f"   Overall Bias: {emoji} {overall}")
            print(f"   Bullish TFs: {bullish_count}/{len(results)}")
            print(f"   Bearish TFs: {bearish_count}/{len(results)}")
        
        return results


# Example usage
if __name__ == "__main__":
    analyzer = IchimokuAnalyzer()
    
    # Single timeframe analysis
    signals = analyzer.display_ichimoku_analysis("1d", 100)
    
    # Multi-timeframe analysis
    mtf_results = analyzer.multi_timeframe_ichimoku()
