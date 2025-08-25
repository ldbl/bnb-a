#!/usr/bin/env python3
"""
BNB Data Generator - Realistic Historical Data for 2 Years
Generates daily, weekly, and monthly OHLCV data for BNB trading analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple
import random

class BNBDataGenerator:
    """
    Generates realistic BNB historical data for trading analysis
    
    Features:
    - 2 years of historical data (2023-2025)
    - Daily, weekly, and monthly intervals
    - Realistic price movements and volatility
    - Volume patterns and market cycles
    - BNB-specific price ranges and trends
    """
    
    def __init__(self):
        """Initialize BNB Data Generator"""
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2025, 8, 25)
        
        # BNB price ranges for different periods
        self.bnb_price_ranges = {
            '2023': {'min': 200, 'max': 400, 'trend': 'bearish'},
            '2024': {'min': 400, 'max': 800, 'trend': 'bullish'},
            '2025': {'min': 600, 'max': 900, 'trend': 'volatile'}
        }
        
        # Market cycles and events
        self.market_events = {
            '2023-03': {'event': 'Bear Market Bottom', 'impact': -0.15},
            '2023-06': {'event': 'Summer Recovery', 'impact': 0.10},
            '2023-09': {'event': 'Q3 Correction', 'impact': -0.08},
            '2023-12': {'event': 'Year End Rally', 'impact': 0.12},
            '2024-03': {'event': 'Bull Market Start', 'impact': 0.25},
            '2024-06': {'event': 'Q2 Continuation', 'impact': 0.20},
            '2024-09': {'event': 'Q3 Peak', 'impact': 0.15},
            '2024-12': {'event': 'Year End Peak', 'impact': 0.10},
            '2025-03': {'event': 'Q1 Correction', 'impact': -0.12},
            '2025-06': {'event': 'Q2 Recovery', 'impact': 0.08},
            '2025-08': {'event': 'Current Period', 'impact': 0.05}
        }
        
        # Seasonal patterns
        self.seasonal_patterns = {
            'Q1': {'volatility': 1.2, 'trend': 'mixed'},
            'Q2': {'volatility': 1.0, 'trend': 'bullish'},
            'Q3': {'volatility': 1.3, 'trend': 'volatile'},
            'Q4': {'volatility': 1.1, 'trend': 'bullish'}
        }
        
        # Random seed for reproducibility
        np.random.seed(42)
        random.seed(42)
    
    def generate_daily_data(self) -> pd.DataFrame:
        """Generate daily OHLCV data for 2 years"""
        print("📊 Generating daily BNB data (2023-2025)...")
        
        # Create date range
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        # Initialize data
        data = []
        current_price = 300  # Starting price in 2023
        
        for date in date_range:
            # Get year and quarter for context
            year = date.year
            quarter = f"Q{(date.month - 1) // 3 + 1}"
            
            # Apply market events
            event_key = f"{year}-{date.month:02d}"
            event_impact = 0
            if event_key in self.market_events:
                event_impact = self.market_events[event_key]['impact']
            
            # Apply seasonal patterns
            seasonal_volatility = self.seasonal_patterns[quarter]['volatility']
            
            # Generate price movement
            daily_return = self._generate_daily_return(year, quarter, event_impact, seasonal_volatility)
            
            # Calculate OHLC
            open_price = current_price
            close_price = open_price * (1 + daily_return)
            
            # Generate high and low with realistic spreads
            daily_volatility = abs(daily_return) * 2
            high_price = max(open_price, close_price) * (1 + np.random.uniform(0, daily_volatility))
            low_price = min(open_price, close_price) * (1 - np.random.uniform(0, daily_volatility))
            
            # Generate volume
            volume = self._generate_volume(open_price, daily_volatility, date)
            
            # Store data
            data.append({
                'date': date,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': int(volume)
            })
            
            # Update current price for next day
            current_price = close_price
        
        df = pd.DataFrame(data)
        print(f"✅ Generated {len(df)} daily records")
        return df
    
    def generate_weekly_data(self) -> pd.DataFrame:
        """Generate weekly OHLCV data for 2 years"""
        print("📅 Generating weekly BNB data (2023-2025)...")
        
        # Get daily data first
        daily_data = self.generate_daily_data()
        
        # Resample to weekly
        weekly_data = []
        
        for year in range(2023, 2026):
            for week in range(1, 53):
                # Get week start and end dates
                week_start = datetime(year, 1, 1) + timedelta(weeks=week-1)
                week_end = week_start + timedelta(days=6)
                
                # Filter daily data for this week
                week_mask = (daily_data['date'] >= week_start) & (daily_data['date'] <= week_end)
                week_daily = daily_data[week_mask]
                
                if len(week_daily) > 0:
                    # Calculate weekly OHLCV
                    open_price = week_daily.iloc[0]['open']
                    high_price = week_daily['high'].max()
                    low_price = week_daily['low'].min()
                    close_price = week_daily.iloc[-1]['close']
                    volume = week_daily['volume'].sum()
                    
                    weekly_data.append({
                        'date': week_start,
                        'open': round(open_price, 2),
                        'high': round(high_price, 2),
                        'low': round(low_price, 2),
                        'close': round(close_price, 2),
                        'volume': int(volume)
                    })
        
        df = pd.DataFrame(weekly_data)
        print(f"✅ Generated {len(df)} weekly records")
        return df
    
    def generate_monthly_data(self) -> pd.DataFrame:
        """Generate monthly OHLCV data for 2 years"""
        print("📆 Generating monthly BNB data (2023-2025)...")
        
        # Get daily data first
        daily_data = self.generate_daily_data()
        
        # Resample to monthly
        monthly_data = []
        
        for year in range(2023, 2026):
            for month in range(1, 13):
                # Get month start and end dates
                month_start = datetime(year, month, 1)
                if month == 12:
                    month_end = datetime(year + 1, 1, 1) - timedelta(days=1)
                else:
                    month_end = datetime(year, month + 1, 1) - timedelta(days=1)
                
                # Filter daily data for this month
                month_mask = (daily_data['date'] >= month_start) & (daily_data['date'] <= month_end)
                month_daily = daily_data[month_mask]
                
                if len(month_daily) > 0:
                    # Calculate monthly OHLCV
                    open_price = month_daily.iloc[0]['open']
                    high_price = month_daily['high'].max()
                    low_price = month_daily['low'].min()
                    close_price = month_daily.iloc[-1]['close']
                    volume = month_daily['volume'].sum()
                    
                    monthly_data.append({
                        'date': month_start,
                        'open': round(open_price, 2),
                        'high': round(high_price, 2),
                        'low': round(low_price, 2),
                        'close': round(close_price, 2),
                        'volume': int(volume)
                    })
        
        df = pd.DataFrame(monthly_data)
        print(f"✅ Generated {len(df)} monthly records")
        return df
    
    def _generate_daily_return(self, year: int, quarter: str, event_impact: float, 
                              seasonal_volatility: float) -> float:
        """Generate realistic daily return based on market context"""
        
        # Base volatility by year
        if year == 2023:
            base_volatility = 0.03  # Bear market - higher volatility
        elif year == 2024:
            base_volatility = 0.025  # Bull market - moderate volatility
        else:  # 2025
            base_volatility = 0.035  # Volatile market - higher volatility
        
        # Apply seasonal and event adjustments
        adjusted_volatility = base_volatility * seasonal_volatility
        
        # Generate random return
        daily_return = np.random.normal(0, adjusted_volatility)
        
        # Apply event impact
        daily_return += event_impact * 0.1  # Scale event impact
        
        # Apply trend bias
        if year == 2023:
            daily_return -= 0.001  # Slight bearish bias
        elif year == 2024:
            daily_return += 0.001  # Slight bullish bias
        # 2025 is neutral
        
        # Limit extreme moves
        daily_return = np.clip(daily_return, -0.15, 0.15)
        
        return daily_return
    
    def _generate_volume(self, price: float, volatility: float, date: datetime) -> int:
        """Generate realistic volume based on price and volatility"""
        
        # Base volume (in BNB)
        base_volume = 1000000  # 1M BNB base volume
        
        # Volume increases with volatility
        volatility_multiplier = 1 + (volatility * 10)
        
        # Weekend effect (lower volume)
        if date.weekday() >= 5:  # Saturday/Sunday
            volume_multiplier = 0.7
        else:
            volume_multiplier = 1.0
        
        # Month-end effect (higher volume)
        if date.day >= 25:
            volume_multiplier *= 1.2
        
        # Calculate final volume
        volume = base_volume * volatility_multiplier * volume_multiplier
        
        # Add some randomness
        volume *= np.random.uniform(0.8, 1.2)
        
        return int(volume)
    
    def save_data(self, data: pd.DataFrame, filename: str, format: str = 'csv'):
        """Save data to file"""
        try:
            if format.lower() == 'csv':
                data.to_csv(filename, index=False)
                print(f"💾 Saved {filename} ({len(data)} records)")
            elif format.lower() == 'json':
                # Convert dates to strings for JSON
                data_copy = data.copy()
                data_copy['date'] = data_copy['date'].dt.strftime('%Y-%m-%d')
                data_copy.to_json(filename, orient='records', indent=2)
                print(f"💾 Saved {filename} ({len(data)} records)")
            else:
                print(f"❌ Unsupported format: {format}")
                
        except Exception as e:
            print(f"❌ Error saving {filename}: {e}")
    
    def generate_all_data(self, output_dir: str = 'generated_data'):
        """Generate all data types and save to files"""
        print("🚀 Generating complete BNB dataset (2023-2025)...")
        print("=" * 60)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate daily data
        daily_data = self.generate_daily_data()
        self.save_data(daily_data, f"{output_dir}/bnb_daily_2023_2025.csv")
        
        # Generate weekly data
        weekly_data = self.generate_weekly_data()
        self.save_data(weekly_data, f"{output_dir}/bnb_weekly_2023_2025.csv")
        
        # Generate monthly data
        monthly_data = self.generate_monthly_data()
        self.save_data(monthly_data, f"{output_dir}/bnb_monthly_2023_2025.csv")
        
        # Generate summary statistics
        self._generate_summary_stats(daily_data, weekly_data, monthly_data, output_dir)
        
        print("=" * 60)
        print("🎉 All BNB data generated successfully!")
        print(f"📁 Output directory: {output_dir}")
    
    def _generate_summary_stats(self, daily_data: pd.DataFrame, weekly_data: pd.DataFrame, 
                               monthly_data: pd.DataFrame, output_dir: str):
        """Generate summary statistics for the data"""
        print("📊 Generating summary statistics...")
        
        stats = {
            'dataset_info': {
                'daily_records': len(daily_data),
                'weekly_records': len(weekly_data),
                'monthly_records': len(monthly_data),
                'date_range': f"{daily_data['date'].min().strftime('%Y-%m-%d')} to {daily_data['date'].max().strftime('%Y-%m-%d')}"
            },
            'price_statistics': {
                'daily': {
                    'min_price': float(daily_data['low'].min()),
                    'max_price': float(daily_data['high'].max()),
                    'avg_price': float(daily_data['close'].mean()),
                    'price_volatility': float(daily_data['close'].pct_change().std())
                },
                'weekly': {
                    'min_price': float(weekly_data['low'].min()),
                    'max_price': float(weekly_data['high'].max()),
                    'avg_price': float(weekly_data['close'].mean()),
                    'price_volatility': float(weekly_data['close'].pct_change().std())
                },
                'monthly': {
                    'min_price': float(monthly_data['low'].min()),
                    'max_price': float(monthly_data['high'].max()),
                    'avg_price': float(monthly_data['close'].mean()),
                    'price_volatility': float(monthly_data['close'].pct_change().std())
                }
            },
            'volume_statistics': {
                'daily_avg_volume': int(daily_data['volume'].mean()),
                'weekly_avg_volume': int(weekly_data['volume'].mean()),
                'monthly_avg_volume': int(monthly_data['volume'].mean())
            },
            'market_cycles': {
                '2023': 'Bear market with high volatility',
                '2024': 'Bull market with moderate volatility',
                '2025': 'Volatile market with mixed trends'
            }
        }
        
        # Save statistics
        stats_file = f"{output_dir}/data_summary.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"📊 Summary statistics saved to {stats_file}")
        
        # Print key statistics
        print("\n📈 Key Statistics:")
        print(f"   Daily Records: {stats['dataset_info']['daily_records']}")
        print(f"   Weekly Records: {stats['dataset_info']['weekly_records']}")
        print(f"   Monthly Records: {stats['dataset_info']['monthly_records']}")
        print(f"   Price Range: ${stats['price_statistics']['daily']['min_price']:.2f} - ${stats['price_statistics']['daily']['max_price']:.2f}")
        print(f"   Average Daily Volume: {stats['volume_statistics']['daily_avg_volume']:,} BNB")


def main():
    """Main function to generate BNB data"""
    print("🎯 BNB Data Generator - Realistic Historical Data (2023-2025)")
    print("=" * 70)
    
    # Initialize generator
    generator = BNBDataGenerator()
    
    # Generate all data
    generator.generate_all_data()
    
    print("\n💡 Usage Examples:")
    print("   # Load daily data")
    print("   daily_data = pd.read_csv('generated_data/bnb_daily_2023_2025.csv')")
    print("   daily_data['date'] = pd.to_datetime(daily_data['date'])")
    print("   ")
    print("   # Load weekly data")
    print("   weekly_data = pd.read_csv('generated_data/bnb_weekly_2023_2025.csv')")
    print("   ")
    print("   # Load monthly data")
    print("   monthly_data = pd.read_csv('generated_data/bnb_monthly_2023_2025.csv')")
    
    print("\n🚀 Data generation complete! Ready for trading analysis.")


if __name__ == "__main__":
    main()
