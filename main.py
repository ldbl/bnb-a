#!/usr/bin/env python3
"""
BNB Advanced Trading Analyzer - Refactored Modular Version
Main application entry point using modular components
"""

import time
from typing import Dict
from data_fetcher import BinanceDataFetcher
from signal_generator import TradingSignalGenerator
from display import TradingDisplay
from fib import FibonacciAnalyzer


class BNBAdvancedAnalyzer:
    """Main BNB trading analyzer using modular components"""
    
    def __init__(self, symbol: str = "BNBUSDT"):
        self.symbol = symbol
        self.data_fetcher = BinanceDataFetcher(symbol)
        self.signal_generator = TradingSignalGenerator()
        self.display = TradingDisplay()
        self.fibonacci_analyzer = FibonacciAnalyzer()
        
        print(f"🚀 BNB Advanced Analyzer initialized for {symbol}")
        print("📦 All modules loaded successfully!")
    
    def get_market_data(self) -> Dict:
        """Fetch current market data and analysis"""
        try:
            # Get current price
            current_price = self.data_fetcher.get_current_price()
            if not current_price:
                return {"error": "Unable to fetch current price"}
            
            # Get historical data for analysis
            daily_klines = self.data_fetcher.fetch_klines("1d", 100)
            if not daily_klines:
                return {"error": "Unable to fetch historical data"}
            
            # Process the data
            processed_data = self.data_fetcher.process_klines_data(daily_klines)
            
            # Get additional market info
            market_summary = self.data_fetcher.get_market_summary()
            
            return {
                "current_price": current_price,
                "prices": processed_data["closes"],
                "volumes": processed_data["volumes"],
                "market_summary": market_summary,
                "processed_data": processed_data
            }
            
        except Exception as e:
            return {"error": f"Error fetching market data: {e}"}
    
    def analyze_market(self) -> Dict:
        """Perform complete market analysis"""
        # Get market data
        market_data = self.get_market_data()
        
        if "error" in market_data:
            return market_data
        
        try:
            # Get multi-timeframe analysis
            mtf_analysis = self.data_fetcher.analyze_timeframes()
            
            # Generate comprehensive trading signal
            signal = self.signal_generator.generate_comprehensive_signal(
                current_price=market_data["current_price"],
                prices=market_data["prices"],
                volumes=market_data["volumes"],
                mtf_analysis=mtf_analysis
            )
            
            # Add market summary data to signal
            signal["market_data"] = market_data["market_summary"]
            
            return signal
            
        except Exception as e:
            return {"error": f"Error during analysis: {e}"}
    
    def display_analysis(self):
        """Display the complete trading analysis"""
        signal = self.analyze_market()
        
        if "error" in signal:
            print(f"❌ Error: {signal['error']}")
            return False
        
        # Display using the display module
        self.display.display_full_analysis(signal)
        return True
    
    def show_fibonacci_analysis(self):
        """Show detailed Fibonacci analysis"""
        print("\n" + "="*60)
        print("📐 DETAILED FIBONACCI ANALYSIS")
        print("="*60)
        
        try:
            self.fibonacci_analyzer.display_analysis()
        except Exception as e:
            print(f"❌ Error in Fibonacci analysis: {e}")
    
    def show_market_summary(self):
        """Show detailed market summary"""
        market_data = self.get_market_data()
        
        if "error" in market_data:
            print(f"❌ Error: {market_data['error']}")
            return
        
        self.display.print_header("DETAILED MARKET SUMMARY")
        
        summary = market_data["market_summary"]
        
        # Current price info
        print(f"\n📊 PRICE INFORMATION")
        print(f"Current Price: {self.display.format_price(summary['current_price'])}")
        print(f"24h Change: {self.display.format_percentage(summary.get('24h_change', 0))}")
        print(f"24h High: {self.display.format_price(summary.get('24h_high', 0))}")
        print(f"24h Low: {self.display.format_price(summary.get('24h_low', 0))}")
        
        # Volume info
        volume = summary.get('24h_volume', 0)
        print(f"24h Volume: {volume:,.0f} BNB")
        
        # Historical changes
        print(f"\n📈 HISTORICAL PERFORMANCE")
        print(f"7-day Change: {self.display.format_percentage(summary.get('7d_change', 0))}")
        print(f"30-day Change: {self.display.format_percentage(summary.get('30d_change', 0))}")
        print(f"30-day High: {self.display.format_price(summary.get('30d_high', 0))}")
        print(f"30-day Low: {self.display.format_price(summary.get('30d_low', 0))}")
        
        # Support and resistance levels
        print(f"\n🎯 KEY LEVELS")
        sr_levels = self.data_fetcher.get_support_resistance_levels()
        
        print("Support Levels:")
        for i, level in enumerate(sr_levels['support'][:3], 1):
            print(f"  S{i}: {self.display.format_price(level)}")
        
        print("Resistance Levels:")
        for i, level in enumerate(sr_levels['resistance'][:3], 1):
            print(f"  R{i}: {self.display.format_price(level)}")
        
        print("\n" + "="*60)
    
    def run(self):
        """Main application loop"""
        print(f"\n🎯 Starting BNB Advanced Trading Analyzer")
        print(f"💡 Analyzing {self.symbol} with advanced technical indicators\n")
        
        while True:
            try:
                # Display main analysis
                success = self.display_analysis()
                
                if not success:
                    print("❌ Failed to get analysis. Retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                
                # Show menu and get user choice
                choice = self.display.display_menu()
                
                if choice == "1":
                    print("\n🔄 Refreshing analysis...")
                    time.sleep(1)
                    
                elif choice == "2":
                    self.show_fibonacci_analysis()
                    input(f"\n{self.display.colorize('Press Enter to continue...', 'cyan')}")
                    
                elif choice == "3":
                    self.show_market_summary()
                    input(f"\n{self.display.colorize('Press Enter to continue...', 'cyan')}")
                    
                elif choice == "4":
                    self.display.toggle_colors()
                    time.sleep(1)
                    
                elif choice == "5":
                    print(f"\n{self.display.colorize('👋 Thank you for using BNB Advanced Analyzer!', 'green')}")
                    break
                    
                else:
                    print(f"\n{self.display.colorize('❌ Invalid choice. Please select 1-5.', 'red')}")
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print(f"\n\n{self.display.colorize('👋 Analysis stopped by user. Goodbye!', 'yellow')}")
                break
                
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("🔄 Restarting in 3 seconds...")
                time.sleep(3)


def main():
    """Main entry point"""
    try:
        # Create and run the analyzer
        analyzer = BNBAdvancedAnalyzer("BNBUSDT")
        analyzer.run()
        
    except Exception as e:
        print(f"❌ Failed to start analyzer: {e}")
        print("🔧 Please check your internet connection and try again.")


if __name__ == "__main__":
    main()