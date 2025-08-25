#!/usr/bin/env python3
"""
Analyze BNB Price Level Touches
Check how many times $550 and $700 levels are reached in monthly data
"""

import pandas as pd
import numpy as np

def analyze_price_levels():
    """Analyze $550 and $700 price level touches"""
    
    print("🎯 BNB Price Level Analysis - $550 and $700")
    print("=" * 60)
    
    # Load monthly data
    data = pd.read_csv('real_bnb_data/bnb_monthly_real.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    
    print(f"📊 Total monthly records: {len(data)}")
    print(f"📅 Date range: {data['Date'].min()} to {data['Date'].max()}")
    print()
    
    # Analyze $550 level
    print("🔴 $550 Level Analysis:")
    print("-" * 30)
    
    # Find months where $550 level is touched
    touches_550 = data[(data['Low'] <= 550) & (data['High'] >= 550)]
    
    print(f"Total touches: {len(touches_550)}")
    print()
    
    if len(touches_550) > 0:
        print("Touch details:")
        for idx, row in touches_550.iterrows():
            print(f"  {row['Date'].strftime('%Y-%m')}: Low=${row['Low']:.2f}, High=${row['High']:.2f}")
            
            # Calculate how many times in the month
            if row['Low'] <= 550 <= row['High']:
                print(f"    → $550 level crossed during this month")
    else:
        print("  No touches found")
    
    print()
    
    # Analyze $700 level
    print("🟢 $700 Level Analysis:")
    print("-" * 30)
    
    # Find months where $700 level is touched
    touches_700 = data[(data['Low'] <= 700) & (data['High'] >= 700)]
    
    print(f"Total touches: {len(touches_700)}")
    print()
    
    if len(touches_700) > 0:
        print("Touch details:")
        for idx, row in touches_700.iterrows():
            print(f"  {row['Date'].strftime('%Y-%m')}: Low=${row['Low']:.2f}, High=${row['High']:.2f}")
            
            # Calculate how many times in the month
            if row['Low'] <= 700 <= row['High']:
                print(f"    → $700 level crossed during this month")
    else:
        print("  No touches found")
    
    print()
    
    # Summary statistics
    print("📈 Summary:")
    print("-" * 30)
    print(f"$550 level touched in {len(touches_550)} months")
    print(f"$700 level touched in {len(touches_700)} months")
    
    # Calculate frequency
    total_months = len(data)
    freq_550 = (len(touches_550) / total_months) * 100
    freq_700 = (len(touches_700) / total_months) * 100
    
    print(f"$550 frequency: {freq_550:.1f}% of months")
    print(f"$700 frequency: {freq_700:.1f}% of months")
    
    # ХАЙДУШКИ КОДЕКС analysis
    print()
    print("🥋 ХАЙДУШКИ КОДЕКС Analysis:")
    print("-" * 30)
    
    if len(touches_550) > 0:
        print("✅ $550 level - Good support level for LONG entries")
        print("   Rule #1: Котва levels - Working")
        print("   Rule #7: Stop loss - Protected")
    else:
        print("⚠️ $550 level - Not tested recently")
    
    if len(touches_700) > 0:
        print("✅ $700 level - Good resistance level for SHORT entries")
        print("   Rule #5: Take profit targets - Achievable")
        print("   Rule #6: One battle - Clear levels")
    else:
        print("⚠️ $700 level - Not tested recently")

if __name__ == "__main__":
    analyze_price_levels()
