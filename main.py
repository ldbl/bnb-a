#!/usr/bin/env python3
"""
BNB Advanced Trading Analyzer - Clean Version with Single ML System
Main application entry point with BNB Enhanced ML only
"""

import time
from typing import Dict
from data_fetcher import BinanceDataFetcher
from signal_generator import TradingSignalGenerator
from display import TradingDisplay
from fib import FibonacciAnalyzer
from elliott_wave import ElliottWaveAnalyzer
from ichimoku_module import IchimokuAnalyzer
from whale_tracker import WhaleTracker
from sentiment_module import SentimentAnalyzer
from correlation_module import CorrelationAnalyzer
from bnb_enhanced_ml import BNBEnhancedML
from trend_reversal import TrendReversalDetector


class BNBAdvancedAnalyzer:
    """Main BNB trading analyzer with single enhanced ML system"""
    
    def __init__(self, symbol: str = "BNBUSDT"):
        self.symbol = symbol
        self.data_fetcher = BinanceDataFetcher(symbol)
        self.signal_generator = TradingSignalGenerator()
        self.display = TradingDisplay()
        self.fibonacci_analyzer = FibonacciAnalyzer()
        self.elliott_analyzer = ElliottWaveAnalyzer()
        self.ichimoku_analyzer = IchimokuAnalyzer()
        self.whale_tracker = WhaleTracker()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.ml_system = BNBEnhancedML()
        self.reversal_detector = TrendReversalDetector()
        
        print(f"🚀 BNB Advanced Analyzer initialized for {symbol}")
        print("📦 All modules loaded successfully!")
        print("🧠 ML System: BNB Enhanced with Multi-Crypto Intelligence")
    
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
            
            # Get enhanced information for main screen
            signal["enhanced_fibonacci"] = self.signal_generator.get_enhanced_fibonacci_info(market_data["current_price"])
            signal["multi_period_elliott"] = self.signal_generator.get_multi_period_elliott_waves()
            
            # Check for critical alerts
            signal["alerts"] = self.check_critical_alerts()
            
            return signal
            
        except Exception as e:
            return {"error": f"Error during analysis: {e}"}
    
    def check_critical_alerts(self) -> Dict:
        """Check for critical alerts from all analysis modules"""
        alerts = {
            "whale_alerts": [],
            "correlation_alerts": [],
            "fibonacci_alerts": [],
            "indicator_alerts": [],
            "reversal_alerts": [],
            "show_any": False
        }
        
        try:
            # Get market data for Fibonacci and indicators alerts
            market_data = self.get_market_data()
            if "error" not in market_data:
                current_price = market_data["current_price"]
                prices = market_data["prices"]
                volumes = market_data["volumes"]
                
                # Check Fibonacci alerts
                fib_alert = self.fibonacci_analyzer.check_critical_fibonacci_alerts(current_price)
                if fib_alert.get("show_alert"):
                    alerts["fibonacci_alerts"].append({
                        "type": "fibonacci_levels",
                        "data": fib_alert
                    })
                    alerts["show_any"] = True
                
                # Check technical indicator alerts
                from indicators import TechnicalIndicators
                indicator_alert = TechnicalIndicators.check_critical_indicator_alerts(prices, volumes)
                if indicator_alert.get("show_alert"):
                    alerts["indicator_alerts"].append({
                        "type": "technical_indicators",
                        "data": indicator_alert
                    })
                    alerts["show_any"] = True
            
            # Check whale activity alerts (24h)
            whale_alert = self.whale_tracker.check_critical_whale_activity(days_back=1)
            if whale_alert.get("show_alert"):
                alerts["whale_alerts"].append({
                    "period": "24h",
                    "data": whale_alert
                })
                alerts["show_any"] = True
            
            # Check correlation alerts
            corr_alert = self.correlation_analyzer.check_critical_correlation_activity()
            if corr_alert.get("show_alert"):
                alerts["correlation_alerts"].append({
                    "type": "market_correlation",
                    "data": corr_alert
                })
                alerts["show_any"] = True
            
            # Check trend reversal alerts
            reversal_alert = self.reversal_detector.check_critical_reversal_alerts()
            if reversal_alert.get("show_alert"):
                alerts["reversal_alerts"].append({
                    "type": "trend_reversal",
                    "data": reversal_alert
                })
                alerts["show_any"] = True
                
        except Exception as e:
            # Don't let alert checking break main analysis
            alerts["error"] = f"Error checking alerts: {e}"
        
        return alerts
    
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
    
    def show_elliott_wave_analysis(self):
        """Show comprehensive Elliott Wave analysis"""
        print("\n" + "="*60)
        print("🌊 COMPREHENSIVE ELLIOTT WAVE ANALYSIS")
        print("="*60)
        
        try:
            self.elliott_analyzer.run_unified_analysis()
        except Exception as e:
            print(f"❌ Error in Elliott Wave analysis: {e}")
    
    def show_ichimoku_analysis(self):
        """Show comprehensive Ichimoku Cloud analysis"""
        print("\n" + "="*60)
        print("☁️ COMPREHENSIVE ICHIMOKU CLOUD ANALYSIS")
        print("="*60)
        
        try:
            # Multi-period analysis (3, 6, 12 months)
            period_results = self.ichimoku_analyzer.multi_period_ichimoku_analysis()
            
            # Multi-timeframe analysis (4h, 1d, 1w)
            print("\n" + "="*60)
            mtf_results = self.ichimoku_analyzer.multi_timeframe_ichimoku()
            
        except Exception as e:
            print(f"❌ Error in Ichimoku analysis: {e}")
    
    def show_whale_tracking(self):
        """Show comprehensive whale tracking analysis"""
        print("\n" + "="*60)
        print("🐋 COMPREHENSIVE WHALE TRACKING ANALYSIS")
        print("="*60)
        
        try:
            print("📅 SELECT ANALYSIS PERIOD:")
            print("1. Last 24 hours")
            print("2. Last 3 days")
            print("3. Last week") 
            print("4. Multi-period comparison")
            
            choice = input(f"\n{self.display.colorize('Select period (1-4): ', 'cyan')}").strip()
            
            if choice == "1":
                results = self.whale_tracker.display_whale_analysis(days_back=1)
            elif choice == "2":
                results = self.whale_tracker.display_whale_analysis(days_back=3)
            elif choice == "3":
                results = self.whale_tracker.display_whale_analysis(days_back=7)
            elif choice == "4":
                results = self.whale_tracker.multi_period_whale_analysis()
            else:
                print("Invalid choice, using default (24 hours)")
                results = self.whale_tracker.display_whale_analysis(days_back=1)
                
        except Exception as e:
            print(f"❌ Error in whale tracking: {e}")
    
    def show_sentiment_analysis(self):
        """Show comprehensive sentiment analysis"""
        print("\n" + "="*60)
        print("🎭 COMPREHENSIVE SENTIMENT ANALYSIS")
        print("="*60)
        
        try:
            results = self.sentiment_analyzer.display_sentiment_analysis()
        except Exception as e:
            print(f"❌ Error in sentiment analysis: {e}")
    
    def show_correlation_analysis(self):
        """Show comprehensive correlation analysis"""
        print("\n" + "="*60)
        print("📊 BNB CORRELATION ANALYSIS WITH BTC/ETH")
        print("="*60)
        
        try:
            print("📅 SELECT ANALYSIS TYPE:")
            print("1. Real-time correlation (1 hour intervals)")
            print("2. Multi-timeframe correlation analysis")
            print("3. Both analyses")
            
            choice = input(f"\n{self.display.colorize('Select analysis (1-3): ', 'cyan')}").strip()
            
            if choice == "1":
                results = self.correlation_analyzer.run_correlation_analysis("1d", 30)
            elif choice == "2":
                results = self.correlation_analyzer.get_multi_timeframe_correlation()
            elif choice == "3":
                print("🔄 Running comprehensive correlation analysis...")
                results1 = self.correlation_analyzer.run_correlation_analysis("1d", 30)
                results2 = self.correlation_analyzer.get_multi_timeframe_correlation()
            else:
                print("Invalid choice, using default (real-time analysis)")
                results = self.correlation_analyzer.run_correlation_analysis("1d", 30)
                
        except Exception as e:
            print(f"❌ Error in correlation analysis: {e}")
    
    def show_ml_analysis(self):
        """Show BNB ML analysis enhanced with multi-crypto intelligence"""
        print("\n" + "="*60)
        print("🎯 BNB ENHANCED ML ANALYSIS")
        print("🧠 Enhanced with Top 10 Cryptocurrency Intelligence")
        print("="*60)
        
        try:
            print("📅 SELECT PREDICTION HORIZON:")
            print("1. 7 days ahead (1 week)")
            print("2. 30 days ahead (1 month)") 
            print("3. 90 days ahead (3 months)")
            print("4. Train enhanced models first")
            print("5. Show discovered patterns")
            
            choice = input(f"\n{self.display.colorize('Select option (1-5): ', 'cyan')}").strip()
            
            if choice == "1":
                periods = 7
            elif choice == "2":
                periods = 30
            elif choice == "3":
                periods = 90
            elif choice == "4":
                print("🧠 Training BNB enhanced models with multi-crypto intelligence...")
                result = self.ml_system.train_bnb_enhanced_model(30)
                if "success" in result:
                    print(f"✅ Training completed: {result['models_trained']} enhanced models")
                    print(f"📊 Learning from: {result['learning_cryptos']} cryptocurrencies")
                    print(f"🧠 Universal insights: {result['universal_insights']}")
                    print(f"🔧 Enhanced features: {result['enhanced_features']}")
                    periods = 30
                else:
                    print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
                    return
            elif choice == "5":
                # Show pattern analysis
                print("🧠 PATTERN DISCOVERY ANALYSIS:")
                print("=" * 40)
                print("This feature analyzes what patterns the ML discovered")
                print("from the top 10 cryptocurrencies and applies to BNB.")
                print()
                print("💡 Train models first to see discovered patterns!")
                return
            else:
                periods = 30
                print("Invalid choice, using 30 days (1 month)")
            
            # Make enhanced prediction
            print(f"\n🔮 Making BNB enhanced prediction ({periods} days ahead)...")
            print("🧠 Using patterns learned from top 10 cryptocurrencies...")
            
            prediction = self.ml_system.predict_bnb_enhanced(periods)
            
            if "error" in prediction:
                print(f"❌ Enhanced prediction error: {prediction['error']}")
                print("💡 Try training enhanced models first (option 4)")
                return
            
            # Display enhanced results
            current_price = prediction["current_bnb_price"]
            pred_label = prediction["prediction_label"]
            confidence = prediction["confidence"]
            ensemble_pred = prediction["ensemble_prediction"]
            
            print(f"\n🎯 BNB ENHANCED PREDICTION RESULTS")
            print("=" * 50)
            print(f"💰 Current BNB Price: ${current_price:.2f}")
            print(f"⏰ Prediction Horizon: {periods} hours ahead")
            print(f"🧠 Enhanced Analysis: Multi-Crypto Intelligence")
            
            # Color coding based on prediction
            if ensemble_pred == 1:  # Bullish reversal
                icon = "🟢📈"
                advice = "BULLISH REVERSAL - Consider LONG position"
                color = "green"
                signal = "BUY SIGNAL 🚀"
            elif ensemble_pred == 2:  # Bearish reversal
                icon = "🔴📉"
                advice = "BEARISH REVERSAL - Consider SHORT position"
                color = "red"
                signal = "SELL SIGNAL 💥"
            else:  # No reversal
                icon = "🟡➡️"
                advice = "NO REVERSAL - Current trend continues"
                color = "yellow"
                signal = "HOLD ⏸️"
            
            print(f"{icon} Prediction: {self.display.colorize(pred_label, color)}")
            print(f"🎲 Confidence: {confidence:.1%}")
            print(f"📊 Signal: {signal}")
            print(f"💡 Advice: {advice}")
            
            # Enhanced risk assessment
            print(f"\n⚠️ ENHANCED RISK ASSESSMENT:")
            print("-" * 30)
            
            if confidence >= 0.8:
                risk_level = "🟢 LOW RISK"
                position_size = "Standard position (3-5%)"
                recommendation = "High confidence with multi-crypto validation"
            elif confidence >= 0.6:
                risk_level = "🟡 MEDIUM RISK"
                position_size = "Reduced position (1-3%)"
                recommendation = "Moderate confidence from enhanced analysis"
            else:
                risk_level = "🔴 HIGH RISK"
                position_size = "Minimal position (<1%)"
                recommendation = "Low confidence - wait for better signal"
            
            print(f"Risk Level: {risk_level}")
            print(f"Position Sizing: {position_size}")
            print(f"Recommendation: {recommendation}")
            
            # Show universal insights if available
            if "universal_insights" in prediction:
                insights = prediction["universal_insights"]
                if insights and len(insights) > 0:
                    print(f"\n🧠 UNIVERSAL PATTERNS APPLIED:")
                    print("-" * 30)
                    for i, insight in enumerate(insights[:3], 1):  # Show top 3
                        print(f"{i}. {insight}")
                    if len(insights) > 3:
                        print(f"   ... and {len(insights) - 3} more patterns")
                    print("💡 These patterns were learned from all cryptocurrencies!")
            
            # Enhancement details
            print(f"\n🌐 ENHANCEMENT DETAILS:")
            print("-" * 25)
            print("✅ Fibonacci Levels: Enhanced with cross-crypto effectiveness")
            print("✅ Volume Spikes: Validated across multiple assets")
            print("✅ Candlestick Patterns: Cross-validated effectiveness weights")
            print("✅ Market Correlation: BTC leadership intelligence")
            print("✅ Pattern Library: Universal crypto market patterns")
            
            # Model breakdown
            if confidence > 0:
                print(f"\n📊 ENHANCED MODEL BREAKDOWN:")
                print("-" * 30)
                
                for model_name, model_pred in prediction["individual_predictions"].items():
                    pred_num = model_pred["prediction"]
                    pred_text = ["No Reversal", "Bullish", "Bearish"][pred_num]
                    proba = model_pred.get("probability", [])
                    
                    if proba:
                        max_proba = max(proba)
                        icon = "🎯" if max_proba > 0.8 else "🎲" if max_proba > 0.6 else "❓"
                        print(f"{icon} {model_name}: {pred_text} ({max_proba:.1%})")
                    else:
                        print(f"🤖 {model_name}: {pred_text}")
            
            print(f"\n🎯 BNB Enhanced Analysis - Powered by Multi-Crypto Intelligence!")
            
        except Exception as e:
            print(f"❌ Error in BNB enhanced analysis: {e}")
    
    def show_reversal_analysis(self):
        """Show comprehensive trend reversal analysis"""
        print("\n" + "="*60)
        print("🔄 COMPREHENSIVE TREND REVERSAL ANALYSIS")
        print("="*60)
        
        try:
            # Multi-timeframe analysis
            results = self.reversal_detector.multi_timeframe_reversal_analysis()
            
            if "error" in results:
                print(f"❌ Error: {results['error']}")
            else:
                print(f"\n✅ Analysis completed successfully!")
                
        except Exception as e:
            print(f"❌ Error in reversal analysis: {e}")
    
    def show_market_summary(self):
        """Show market summary and overview"""
        print("\n" + "="*60)
        print("📊 MARKET SUMMARY & OVERVIEW")
        print("="*60)
        
        signal = self.analyze_market()
        if "error" in signal:
            print(f"❌ Error: {signal['error']}")
            return
        
        # Display key metrics
        self.display.display_summary_only(signal)
    
    def display_main_menu(self) -> str:
        """Display the main menu and get user choice"""
        print("\n" + "="*60)
        print("🎯 BNB ADVANCED ANALYZER - MAIN MENU")
        print("🧠 Enhanced with Multi-Crypto Intelligence")
        print("="*60)
        print("1. Show complete market analysis")
        print("2. Show Fibonacci analysis")
        print("3. Show Elliott Wave analysis")
        print("4. Show Ichimoku Cloud analysis")
        print("5. Show Whale Tracking analysis")
        print("6. Show Sentiment Analysis")
        print("7. Show Correlation Analysis (BTC/ETH)")
        print("8. Show ML Analysis (Enhanced with Multi-Crypto Intelligence)")
        print("9. Show Trend Reversal Analysis")
        print("10. Show market summary")
        print("11. Toggle colors")
        print("12. Exit")
        
        return input(f"\n{self.display.colorize('Choice (1-12): ', 'cyan')}")

    def run(self):
        """Main application loop"""
        print(f"\n{'='*60}")
        print("🎯 BNB ADVANCED TRADING ANALYZER")
        print("🧠 Enhanced with Multi-Crypto Intelligence")
        print(f"{'='*60}")
        print(f"📊 Analyzing: {self.symbol}")
        print("🤖 ML System: BNB Enhanced (Learning from Top 10 Cryptos)")
        print("💡 Tip: Train ML models first (Option 8 → 4) for best results")
        
        while True:
            try:
                choice = self.display_main_menu()
                
                if choice == "1":
                    self.display_analysis()
                    input(f"\n{self.display.colorize('Press Enter to continue...', 'cyan')}")
                    
                elif choice == "2":
                    self.show_fibonacci_analysis()
                    input(f"\n{self.display.colorize('Press Enter to continue...', 'cyan')}")
                    
                elif choice == "3":
                    self.show_elliott_wave_analysis()
                    input(f"\n{self.display.colorize('Press Enter to continue...', 'cyan')}")
                    
                elif choice == "4":
                    self.show_ichimoku_analysis()
                    input(f"\n{self.display.colorize('Press Enter to continue...', 'cyan')}")
                    
                elif choice == "5":
                    self.show_whale_tracking()
                    input(f"\n{self.display.colorize('Press Enter to continue...', 'cyan')}")
                    
                elif choice == "6":
                    self.show_sentiment_analysis()
                    input(f"\n{self.display.colorize('Press Enter to continue...', 'cyan')}")
                    
                elif choice == "7":
                    self.show_correlation_analysis()
                    input(f"\n{self.display.colorize('Press Enter to continue...', 'cyan')}")
                    
                elif choice == "8":
                    self.show_ml_analysis()
                    input(f"\n{self.display.colorize('Press Enter to continue...', 'cyan')}")
                    
                elif choice == "9":
                    self.show_reversal_analysis()
                    input(f"\n{self.display.colorize('Press Enter to continue...', 'cyan')}")
                    
                elif choice == "10":
                    self.show_market_summary()
                    input(f"\n{self.display.colorize('Press Enter to continue...', 'cyan')}")
                    
                elif choice == "11":
                    self.display.toggle_colors()
                    time.sleep(1)
                    
                elif choice == "12":
                    print(f"\n{self.display.colorize('👋 Thank you for using BNB Advanced Analyzer!', 'green')}")
                    break
                    
                else:
                    print(f"\n{self.display.colorize('❌ Invalid choice. Please select 1-12.', 'red')}")
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print(f"\n\n{self.display.colorize('👋 Analysis stopped by user. Goodbye!', 'yellow')}")
                break
            except Exception as e:
                print(f"\n{self.display.colorize(f'❌ Unexpected error: {e}', 'red')}")
                time.sleep(2)


def main():
    """Main entry point"""
    try:
        analyzer = BNBAdvancedAnalyzer()
        analyzer.run()
    except KeyboardInterrupt:
        print(f"\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")


if __name__ == "__main__":
    main()
