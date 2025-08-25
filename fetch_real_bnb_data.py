#!/usr/bin/env python3
"""
Fetch Real BNB Data - No Modifications
Downloads real BNB historical data from Yahoo Finance
"""

import pandas as pd
import yfinance as yf
from datetime import datetime
import os

def fetch_real_bnb_data():
    """Fetch real BNB data without any modifications"""
    
    print("🎯 Fetching Real BNB Data (No Modifications)")
    print("=" * 50)
    
    # Download real BNB data
    print("📥 Downloading BNB-USD data from Yahoo Finance...")
    bnb = yf.download('BNB-USD', 
                      start='2023-01-01', 
                      end='2025-08-25', 
                      interval='1d',
                      progress=False)
    
    # Fix column names (remove MultiIndex)
    bnb.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    print(f"✅ Downloaded {len(bnb)} records")
    print(f"📅 Date range: {bnb.index.min()} to {bnb.index.max()}")
    print(f"💰 Price range: ${float(bnb['Close'].min()):.2f} - ${float(bnb['Close'].max()):.2f}")
    
    # Create output directory
    os.makedirs('real_bnb_data', exist_ok=True)
    
    # Save daily data (no modifications)
    daily_file = 'real_bnb_data/bnb_daily_real.csv'
    bnb.to_csv(daily_file)
    print(f"💾 Daily data saved: {daily_file}")
    
    # Create weekly data (OHLCV aggregation)
    print("\n📅 Creating weekly data...")
    weekly = bnb.resample('W').agg({
        'Open': 'first',
        'High': 'max', 
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    
    weekly_file = 'real_bnb_data/bnb_weekly_real.csv'
    weekly.to_csv(weekly_file)
    print(f"💾 Weekly data saved: {weekly_file}")
    
    # Create monthly data (OHLCV aggregation)
    print("\n📆 Creating monthly data...")
    monthly = bnb.resample('M').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min', 
        'Close': 'last',
        'Volume': 'sum'
    })
    
    monthly_file = 'real_bnb_data/bnb_monthly_real.csv'
    monthly.to_csv(monthly_file)
    print(f"💾 Monthly data saved: {monthly_file}")
    
    # Print sample data
    print("\n📊 Sample Daily Data (Last 5 days):")
    print(bnb.tail()[['Open', 'High', 'Low', 'Close', 'Volume']])
    
    print("\n📊 Sample Weekly Data (Last 5 weeks):")
    print(weekly.tail()[['Open', 'High', 'Low', 'Close', 'Volume']])
    
    print("\n📊 Sample Monthly Data (Last 5 months):")
    print(monthly.tail()[['Open', 'High', 'Low', 'Close', 'Volume']])
    
    print("\n🎉 Real BNB data fetched successfully!")
    print("📁 All files saved in 'real_bnb_data/' directory")
    
    return bnb, weekly, monthly

if __name__ == "__main__":
    fetch_real_bnb_data()
