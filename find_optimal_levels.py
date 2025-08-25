#!/usr/bin/env python3
"""
Find Optimal BNB Trading Levels
Find the best entry price (LONG) and exit price (SHORT) based on most touches
"""

import pandas as pd
import numpy as np
from collections import defaultdict

def find_optimal_levels():
    """Find optimal entry and exit prices based on most touches"""
    
    print("🎯 BNB Optimal Trading Levels Analysis")
    print("=" * 60)
    
    # Load monthly data
    data = pd.read_csv('real_bnb_data/bnb_monthly_real.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Use all data for complete analysis
    data_analysis = data.copy()
    
    print(f"📊 Total monthly records: {len(data)}")
    print(f"📊 Records for analysis (all months): {len(data_analysis)}")
    print(f"📅 Analysis range: {data_analysis['Date'].min()} to {data_analysis['Date'].max()}")
    print()
    
    # Price range analysis (all months)
    min_price = data_analysis['Low'].min()
    max_price = data_analysis['High'].max()
    print(f"💰 Price range (analysis): ${min_price:.2f} - ${max_price:.2f}")
    print()
    
    # Create price levels every $25 (extended to cover new highs)
    # Also add levels around current price for better analysis
    base_levels = list(np.arange(200, 1500, 25))
    current_levels = list(np.arange(800, 1000, 25))  # Add levels around current price
    price_levels = sorted(list(set(base_levels + current_levels)))
    level_touches = defaultdict(int)
    
    print("🔍 Analyzing price level touches...")
    print(f"🔍 Debug: Price levels range: ${min(price_levels):.0f} - ${max(price_levels):.0f}")
    print(f"🔍 Debug: Total price levels: {len(price_levels)}")
    
    # Count touches for each price level (all months)
    for _, row in data_analysis.iterrows():
        low = row['Low']
        high = row['High']
        
        for level in price_levels:
            if low <= level <= high:
                level_touches[level] += 1
    
    # Debug: Show some high levels and their touches
    print(f"🔍 Debug: Sample high levels and touches:")
    high_levels = [level for level in price_levels if level >= 800]
    for level in high_levels[:10]:
        touches = level_touches.get(level, 0)
        print(f"      ${level:.0f}: {touches} touches")
    print()
    
    # Convert to sorted list
    sorted_levels = sorted(level_touches.items(), key=lambda x: x[1], reverse=True)
    
    print("📊 Top 10 Most Touched Price Levels:")
    print("-" * 50)
    
    for i, (price, touches) in enumerate(sorted_levels[:10]):
        print(f"{i+1:2d}. ${price:6.0f}: {touches:2d} touches")
    
    print()
    
    # Find optimal LONG entry (support level with most touches)
    print("🟢 Optimal LONG Entry Analysis:")
    print("-" * 40)
    
    # Filter for support levels (below current price)
    current_price = data_analysis['Close'].iloc[-1]
    support_levels = [(price, touches) for price, touches in sorted_levels if price < current_price]
    
    if support_levels:
        best_support = support_levels[0]
        print(f"🏆 Best Support Level: ${best_support[0]:.0f}")
        print(f"   Touches: {best_support[1]} months")
        print(f"   Distance from current: ${current_price - best_support[0]:.0f}")
        print(f"   Entry zone: ${best_support[0]-25:.0f} - ${best_support[0]+25:.0f}")
        
        # Show all good support levels
        print("\n📈 All Good Support Levels (>3 touches):")
        for price, touches in support_levels[:5]:
            if touches >= 3:
                print(f"   ${price:6.0f}: {touches:2d} touches")
    else:
        print("❌ No support levels found below current price")
    
    print()
    
    # Find optimal SHORT exit (resistance level with most touches)
    print("🔴 Optimal SHORT Exit Analysis:")
    print("-" * 40)
    
    # Filter for resistance levels (above current price)
    # If no levels above current, use highest levels with touches
    resistance_levels = [(price, touches) for price, touches in sorted_levels if price > current_price]
    
    # Debug: Show current price and available levels
    print(f"🔍 Debug: Current price: ${current_price:.2f}")
    print(f"🔍 Debug: Available levels above current: {len(resistance_levels)}")
    print(f"🔍 Debug: Highest price level: ${max(price_levels):.0f}")
    
    # If no resistance levels above current, use highest levels with touches
    if not resistance_levels:
        print(f"🔍 Debug: No levels above current, using highest levels with touches:")
        resistance_levels = [(price, touches) for price, touches in sorted_levels if touches > 0 and price > 600]
        resistance_levels.sort(key=lambda x: x[0], reverse=True)  # Sort by price descending
        print(f"🔍 Debug: Found {len(resistance_levels)} levels with touches above $600")
    
    print(f"🔍 Debug: Top 5 levels above current:")
    levels_above = [(price, touches) for price, touches in sorted_levels if price > current_price]
    for i, (price, touches) in enumerate(levels_above[:5]):
        print(f"      ${price:.0f}: {touches} touches")
    if resistance_levels:
        print(f"🔍 Debug: Top 3 resistance levels: {resistance_levels[:3]}")
    print()
    
    if resistance_levels:
        best_resistance = resistance_levels[0]
        print(f"🏆 Best Resistance Level: ${best_resistance[0]:.0f}")
        print(f"   Touches: {best_resistance[1]} months")
        print(f"   Distance from current: ${best_resistance[0] - current_price:.0f}")
        print(f"   Exit zone: ${best_resistance[0]-25:.0f} - ${best_resistance[0]+25:.0f}")
        
        # Show all good resistance levels
        print("\n📉 All Good Resistance Levels (>3 touches):")
        for price, touches in resistance_levels[:5]:
            if touches >= 3:
                print(f"   ${price:6.0f}: {touches:2d} touches")
    else:
        print("❌ No resistance levels found above current price")
    
    print()
    
    # ХАЙДУШКИ КОДЕКС Analysis
    print("🥋 ХАЙДУШКИ КОДЕКС Trading Plan:")
    print("-" * 40)
    
    if support_levels and resistance_levels:
        best_support = support_levels[0]
        best_resistance = resistance_levels[0]
        
        # Calculate averaged support level from top 3
        top_3_support = support_levels[:3]
        avg_support_price = sum(level[0] for level in top_3_support) / len(top_3_support)
        avg_support_touches = sum(level[1] for level in top_3_support) / len(top_3_support)
        
        print(f"📊 Top 3 Support Levels Analysis:")
        print(f"   ${top_3_support[0][0]:.0f}: {top_3_support[0][1]} touches")
        print(f"   ${top_3_support[1][0]:.0f}: {top_3_support[1][1]} touches")
        print(f"   ${top_3_support[2][0]:.0f}: {top_3_support[2][1]} touches")
        print(f"   🎯 Averaged Support: ${avg_support_price:.0f} ({avg_support_touches:.1f} avg touches)")
        print()
        
        print(f"📈 LONG Strategy (Individual Levels):")
        print(f"   Entry: ${best_support[0]:.0f} (котва level)")
        print(f"   Stop Loss: ${best_support[0] - 50:.0f}")
        print(f"   Target: ${best_resistance[0]:.0f}")
        
        risk = best_support[0] - (best_support[0] - 50)
        reward = best_resistance[0] - best_support[0]
        risk_reward = reward / risk
        
        print(f"   Risk/Reward: 1:{risk_reward:.1f}")
        
        print(f"\n📈 LONG Strategy (Averaged Level):")
        print(f"   Entry: ${avg_support_price:.0f} (averaged котва)")
        print(f"   Stop Loss: ${avg_support_price - 50:.0f}")
        print(f"   Target: ${best_resistance[0]:.0f}")
        
        risk = avg_support_price - (avg_support_price - 50)
        reward = best_resistance[0] - avg_support_price
        risk_reward = reward / risk
        
        print(f"   Risk/Reward: 1:{risk_reward:.1f}")
        
        print(f"\n📉 SHORT Strategy:")
        print(f"   Entry: ${best_resistance[0]:.0f} (resistance)")
        print(f"   Stop Loss: ${best_resistance[0] + 50:.0f}")
        print(f"   Target: ${avg_support_price:.0f}")
        
        risk = (best_resistance[0] + 50) - best_resistance[0]
        reward = best_resistance[0] - avg_support_price
        risk_reward = reward / risk
        
        print(f"   Risk/Reward: 1:{risk_reward:.1f}")
        
        print(f"\n🎯 Summary:")
        print(f"   Best Individual Entry: ${best_support[0]:.0f} ({best_support[1]} touches)")
        print(f"   Averaged Entry: ${avg_support_price:.0f} ({avg_support_touches:.1f} avg touches)")
        print(f"   Best Exit: ${best_resistance[0]:.0f} ({best_resistance[1]} touches)")
        print(f"   Trading Range: ${best_resistance[0] - avg_support_price:.0f} points")
        
        print(f"\n💡 Recommendation:")
        if avg_support_touches >= 10:
            print(f"   ✅ Use averaged level ${avg_support_price:.0f} (high reliability)")
        else:
            print(f"   ⚠️ Use individual level ${best_support[0]:.0f} (better touches)")
    
    # Detailed monthly analysis
    print()
    print("📅 Monthly Touch Analysis:")
    print("-" * 40)
    
    # Show months with most level touches (without last 3 months)
    month_touches = defaultdict(int)
    for _, row in data_analysis.iterrows():
        month = row['Date'].strftime('%Y-%m')
        low = row['Low']
        high = row['High']
        
        touches_this_month = 0
        for level in price_levels:
            if low <= level <= high:
                touches_this_month += 1
        
        month_touches[month] = touches_this_month
    
    # Sort months by touches
    sorted_months = sorted(month_touches.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 5 months with most level touches:")
    for i, (month, touches) in enumerate(sorted_months[:5]):
        print(f"   {month}: {touches} level touches")

if __name__ == "__main__":
    find_optimal_levels()
