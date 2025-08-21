#!/usr/bin/env python3
"""
Historical Analysis of BNB 90-day Bearish Patterns
Find when similar conditions led to major drops
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from logger import get_logger

class BNBHistoricalAnalysis:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.base_url = "https://api.binance.com/api/v3"
        
    def fetch_historical_data(self, months_back=24):
        """Fetch historical BNB data for analysis"""
        
        print(f"📊 Fetching {months_back} months of BNB historical data...")
        
        # Get data in chunks (Binance has limit of 1000 candles)
        all_data = []
        end_time = int(datetime.now().timestamp() * 1000)
        
        # Calculate chunks needed
        days_back = months_back * 30
        chunks_needed = days_back // 1000 + 1
        
        for chunk in range(chunks_needed):
            params = {
                "symbol": "BNBUSDT",
                "interval": "1d",
                "limit": 1000,
                "endTime": end_time
            }
            
            try:
                response = requests.get(f"{self.base_url}/klines", params=params, timeout=10)
                if response.status_code == 200:
                    klines = response.json()
                    all_data.extend(klines)
                    
                    # Update end_time for next chunk
                    if klines:
                        end_time = int(klines[0][0]) - 1
                        
                    print(f"   ✅ Chunk {chunk + 1}/{chunks_needed}: {len(klines)} candles")
                else:
                    print(f"   ❌ Failed chunk {chunk + 1}: {response.status_code}")
                    break
                    
            except Exception as e:
                print(f"   ❌ Error in chunk {chunk + 1}: {e}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert to numeric
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)  # Sort chronologically
        
        print(f"📈 Total historical data: {len(df)} days ({df.index.min().date()} to {df.index.max().date()})")
        return df
    
    def find_90day_drops(self, df, min_drop_percent=15):
        """Find periods where BNB dropped significantly over 90 days"""
        
        print(f"\n🔍 SEARCHING FOR 90-DAY DROPS (>{min_drop_percent}%)")
        print("=" * 60)
        
        major_drops = []
        
        # Check every possible 90-day period
        for i in range(len(df) - 90):
            start_date = df.index[i]
            end_date = df.index[i + 90] if i + 90 < len(df) else df.index[-1]
            
            start_price = df['close'].iloc[i]
            end_price = df['close'].iloc[i + 90] if i + 90 < len(df) else df['close'].iloc[-1]
            
            # Calculate 90-day return
            return_90d = (end_price - start_price) / start_price * 100
            
            # Find peak in the period for max drawdown
            period_data = df.iloc[i:i+91] if i + 90 < len(df) else df.iloc[i:]
            peak_price = period_data['high'].max()
            trough_price = period_data['low'].min()
            
            max_drawdown = (trough_price - peak_price) / peak_price * 100
            
            # Check if it's a significant drop
            if return_90d <= -min_drop_percent or max_drawdown <= -min_drop_percent:
                
                # Find the exact peak and trough dates
                peak_date = period_data[period_data['high'] == peak_price].index[0]
                trough_date = period_data[period_data['low'] == trough_price].index[0]
                
                major_drops.append({
                    'period_start': start_date,
                    'period_end': end_date,
                    'start_price': start_price,
                    'end_price': end_price,
                    'return_90d': return_90d,
                    'peak_date': peak_date,
                    'peak_price': peak_price,
                    'trough_date': trough_date,
                    'trough_price': trough_price,
                    'max_drawdown': max_drawdown
                })
        
        # Remove overlapping periods (keep the worst ones)
        filtered_drops = self._filter_overlapping_drops(major_drops)
        
        # Sort by severity
        filtered_drops.sort(key=lambda x: x['max_drawdown'])
        
        print(f"📉 Found {len(filtered_drops)} major 90-day bearish periods:")
        print()
        
        for i, drop in enumerate(filtered_drops[:10], 1):  # Show top 10
            print(f"🔴 #{i}: {drop['period_start'].strftime('%Y-%m-%d')} → {drop['period_end'].strftime('%Y-%m-%d')}")
            print(f"   📊 90-day return: {drop['return_90d']:+.1f}%")
            print(f"   📉 Max drawdown: {drop['max_drawdown']:.1f}%")
            print(f"   💰 Peak: ${drop['peak_price']:.2f} ({drop['peak_date'].strftime('%Y-%m-%d')})")
            print(f"   💰 Trough: ${drop['trough_price']:.2f} ({drop['trough_date'].strftime('%Y-%m-%d')})")
            print()
        
        return filtered_drops
    
    def _filter_overlapping_drops(self, drops, min_gap_days=45):
        """Filter overlapping periods, keeping the most severe ones"""
        
        if not drops:
            return []
        
        # Sort by start date
        drops.sort(key=lambda x: x['period_start'])
        
        filtered = [drops[0]]
        
        for drop in drops[1:]:
            # Check if this drop overlaps with the last filtered one
            last_drop = filtered[-1]
            gap = (drop['period_start'] - last_drop['period_end']).days
            
            if gap >= min_gap_days:
                # No overlap, add it
                filtered.append(drop)
            else:
                # Overlap - keep the worse one
                if drop['max_drawdown'] < last_drop['max_drawdown']:
                    filtered[-1] = drop
        
        return filtered
    
    def analyze_bearish_patterns(self, df, drops):
        """Analyze what conditions preceded major drops"""
        
        print("🔍 ANALYZING CONDITIONS BEFORE MAJOR DROPS")
        print("=" * 60)
        
        patterns = {
            'high_rsi_before_drop': 0,
            'multiple_green_days': 0,
            'high_volume_spikes': 0,
            'price_near_resistance': 0,
            'seasonal_patterns': {},
            'avg_pre_drop_gain': []
        }
        
        for drop in drops[:10]:  # Analyze top 10 drops
            start_date = drop['period_start']
            
            # Look at 30 days before the drop period
            pre_period_start = start_date - timedelta(days=30)
            pre_period_data = df[pre_period_start:start_date]
            
            if len(pre_period_data) < 20:  # Need enough data
                continue
            
            print(f"📅 Pre-drop analysis: {pre_period_start.strftime('%Y-%m-%d')} → {start_date.strftime('%Y-%m-%d')}")
            
            # 1. RSI before drop
            rsi = self._calculate_rsi(pre_period_data['close'], 14)
            avg_rsi = rsi.iloc[-10:].mean() if len(rsi) >= 10 else rsi.mean()
            if avg_rsi > 65:
                patterns['high_rsi_before_drop'] += 1
                print(f"   🔴 High RSI detected: {avg_rsi:.1f}")
            
            # 2. Consecutive green days
            daily_changes = pre_period_data['close'].pct_change()
            recent_changes = daily_changes.iloc[-10:]
            green_days = (recent_changes > 0).sum()
            if green_days >= 7:
                patterns['multiple_green_days'] += 1
                print(f"   🟢 Multiple green days: {green_days}/10")
            
            # 3. Volume spikes
            volume_ma = pre_period_data['volume'].rolling(20).mean()
            volume_spikes = (pre_period_data['volume'] > volume_ma * 2).sum()
            if volume_spikes >= 3:
                patterns['high_volume_spikes'] += 1
                print(f"   📈 Volume spikes detected: {volume_spikes}")
            
            # 4. Pre-drop gain
            period_start_price = pre_period_data['close'].iloc[0]
            period_end_price = pre_period_data['close'].iloc[-1]
            pre_drop_gain = (period_end_price - period_start_price) / period_start_price * 100
            patterns['avg_pre_drop_gain'].append(pre_drop_gain)
            print(f"   📊 30d gain before drop: {pre_drop_gain:+.1f}%")
            
            # 5. Seasonal analysis
            month = start_date.month
            if month not in patterns['seasonal_patterns']:
                patterns['seasonal_patterns'][month] = 0
            patterns['seasonal_patterns'][month] += 1
            
            print()
        
        # Summary
        total_drops = len([d for d in drops[:10] if len(df[d['period_start'] - timedelta(days=30):d['period_start']]) >= 20])
        
        print("📊 PATTERN SUMMARY:")
        print("-" * 30)
        print(f"🔴 High RSI before drop: {patterns['high_rsi_before_drop']}/{total_drops} ({patterns['high_rsi_before_drop']/total_drops*100:.1f}%)")
        print(f"🟢 Multiple green days: {patterns['multiple_green_days']}/{total_drops} ({patterns['multiple_green_days']/total_drops*100:.1f}%)")
        print(f"📈 High volume periods: {patterns['high_volume_spikes']}/{total_drops} ({patterns['high_volume_spikes']/total_drops*100:.1f}%)")
        
        if patterns['avg_pre_drop_gain']:
            avg_gain = np.mean(patterns['avg_pre_drop_gain'])
            print(f"📊 Avg 30d gain before drop: {avg_gain:+.1f}%")
        
        # Seasonal patterns
        if patterns['seasonal_patterns']:
            print(f"\n📅 SEASONAL PATTERNS:")
            months = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for month, count in sorted(patterns['seasonal_patterns'].items()):
                print(f"   {months[month]}: {count} drops")
        
        return patterns
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def compare_current_conditions(self, df, patterns):
        """Compare current conditions with historical pre-drop patterns"""
        
        print("\n🎯 CURRENT CONDITIONS VS HISTORICAL PATTERNS")
        print("=" * 60)
        
        # Get last 30 days
        current_data = df.iloc[-30:]
        current_price = df['close'].iloc[-1]
        
        print(f"💰 Current BNB Price: ${current_price:.2f}")
        print(f"📅 Analysis Date: {df.index[-1].strftime('%Y-%m-%d')}")
        print()
        
        # Check current conditions
        print("🔍 CURRENT RISK FACTORS:")
        risk_score = 0
        
        # 1. RSI check
        current_rsi = self._calculate_rsi(current_data['close'], 14).iloc[-1]
        print(f"📊 Current RSI(14): {current_rsi:.1f}")
        if current_rsi > 65:
            risk_score += 1
            print("   🔴 WARNING: High RSI (historically preceded drops)")
        else:
            print("   🟢 RSI in normal range")
        
        # 2. Recent performance
        month_ago_price = df['close'].iloc[-30]
        recent_gain = (current_price - month_ago_price) / month_ago_price * 100
        avg_pre_drop_gain = np.mean(patterns['avg_pre_drop_gain']) if patterns['avg_pre_drop_gain'] else 0
        
        print(f"📈 30-day gain: {recent_gain:+.1f}% (historical avg before drops: {avg_pre_drop_gain:+.1f}%)")
        if recent_gain > avg_pre_drop_gain + 5:  # 5% buffer
            risk_score += 1
            print("   🔴 WARNING: Rapid gains (similar to pre-drop periods)")
        else:
            print("   🟢 Gain rate in normal range")
        
        # 3. Green days
        daily_changes = current_data['close'].pct_change()
        recent_green_days = (daily_changes.iloc[-10:] > 0).sum()
        print(f"🟢 Green days (last 10): {recent_green_days}")
        if recent_green_days >= 7:
            risk_score += 1
            print("   🔴 WARNING: Too many consecutive green days")
        else:
            print("   🟢 Normal distribution of green/red days")
        
        # 4. Volume analysis
        current_volume = current_data['volume'].iloc[-1]
        avg_volume = current_data['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume
        print(f"📈 Volume vs 20d avg: {volume_ratio:.2f}x")
        
        # 5. Seasonal risk
        current_month = df.index[-1].month
        seasonal_risk = patterns['seasonal_patterns'].get(current_month, 0)
        months = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        print(f"📅 Current month ({months[current_month]}): {seasonal_risk} historical drops")
        
        print(f"\n⚠️ TOTAL RISK SCORE: {risk_score}/3")
        
        if risk_score >= 2:
            print("🔴 HIGH RISK: Current conditions similar to historical pre-drop periods!")
            recommendation = "Consider reducing BNB position"
        elif risk_score == 1:
            print("🟡 MODERATE RISK: Some warning signs present")
            recommendation = "Monitor closely, consider partial profit-taking"
        else:
            print("🟢 LOW RISK: Current conditions don't match typical pre-drop patterns")
            recommendation = "Normal position management"
        
        print(f"💡 RECOMMENDATION: {recommendation}")
        
        return risk_score

def main():
    analyzer = BNBHistoricalAnalysis()
    
    # Fetch historical data
    df = analyzer.fetch_historical_data(months_back=24)
    
    # Find major 90-day drops
    major_drops = analyzer.find_90day_drops(df, min_drop_percent=15)
    
    # Analyze patterns
    patterns = analyzer.analyze_bearish_patterns(df, major_drops)
    
    # Compare with current conditions
    risk_score = analyzer.compare_current_conditions(df, patterns)
    
    print("\n" + "=" * 60)
    print("✅ HISTORICAL BEARISH ANALYSIS COMPLETED!")
    print("💡 This analysis helps explain why the 90-day ML prediction is bearish")
    print("=" * 60)

if __name__ == "__main__":
    main()
