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
            signal["current_price"] = market_data["current_price"]  # Ensure current_price is in signal
            
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
            
            # Check correlation alerts (use daily data for main analysis)
            try:
                corr_alert = self.correlation_analyzer.check_critical_correlation_activity()
                if corr_alert.get("show_alert"):
                    alerts["correlation_alerts"].append({
                        "type": "market_correlation",
                        "data": corr_alert
                    })
                    alerts["show_any"] = True
            except:
                # Skip correlation alerts if there's an error
                pass
            
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
        """Display the complete trading analysis with key insights from all modules"""
        print("\n" + "="*80)
        print("🎯 COMPLETE MARKET ANALYSIS - ALL MODULES SUMMARY")
        print("="*80)
        
        # Get market data and analysis
        signal = self.analyze_market()
        
        if "error" in signal:
            print(f"❌ Error: {signal['error']}")
            return False
        
        try:
            # 1. MARKET OVERVIEW
            print(f"\n📊 MARKET OVERVIEW:")
            current_price = signal.get('current_price', 'N/A')
            if isinstance(current_price, (int, float)):
                print(f"   💰 Current Price: ${current_price:,.4f}")
            else:
                print(f"   💰 Current Price: {current_price}")
            print(f"   📈 Signal: {signal.get('signal', 'N/A')}")
            print(f"   🎯 Confidence: {signal.get('confidence', 'N/A')}")
            
            # 2. FIBONACCI KEY LEVELS
            print(f"\n📐 FIBONACCI KEY LEVELS:")
            try:
                 # Get Fibonacci data directly from analyzer
                 current_price = signal.get('current_price', 0)
                 if current_price and current_price != 'N/A':
                     fib_data = self.fibonacci_analyzer.get_fibonacci_signals(current_price)
                     if fib_data and 'fibonacci_levels' in fib_data:
                         levels = fib_data['fibonacci_levels']
                         
                         # Show current position
                         print(f"   💰 Current Price: ${current_price:,.2f}")
                         
                         # Find closest level
                         closest_level = None
                         min_distance = float('inf')
                         for level_name, level_price in levels.items():
                             if isinstance(level_price, (int, float)) and level_price > 0:
                                 distance = abs(current_price - level_price) / current_price * 100
                                 if distance < min_distance:
                                     min_distance = distance
                                     closest_level = (level_name, level_price, distance)
                         
                         if closest_level:
                             level_name, level_price, distance = closest_level
                             print(f"   🎯 Closest Level: {level_name} at ${level_price:,.2f} ({distance:.1f}% away)")
                         
                         # Show all levels within 15%
                         print(f"   📊 Key Levels:")
                         for level_name, level_price in levels.items():
                             if isinstance(level_price, (int, float)) and level_price > 0:
                                 distance = abs(current_price - level_price) / current_price * 100
                                 if distance < 15:  # Show levels within 15%
                                     print(f"      {level_name}: ${level_price:,.2f} ({distance:.1f}% away)")
                     else:
                         print("   💡 Fibonacci analysis in progress...")
                 else:
                     print("   💡 Current price not available for Fibonacci analysis")
            except Exception as e:
                print(f"   💡 Fibonacci analysis: {str(e)[:50]}")
            
            # 3. ELLIOTT WAVE STATUS
            print(f"\n🌊 ELLIOTT WAVE STATUS:")
            try:
                # Get Elliott Wave data directly from analyzer
                current_price = signal.get('current_price', 0)
                if current_price and current_price != 'N/A':
                    # Get recent prices for Elliott Wave analysis
                    market_data = self.get_market_data()
                    if "error" not in market_data:
                        prices = market_data["prices"]
                        elliott_data = self.elliott_analyzer.detect_elliott_wave(prices)
                        if elliott_data:
                            wave = elliott_data.get('wave', 'Unknown')
                            confidence = elliott_data.get('confidence', 0)
                            print(f"   🌊 Current Wave: {wave}")
                            print(f"   🎯 Confidence: {confidence}%")
                            print(f"   📊 Trend: {elliott_data.get('description', 'Unknown')}")
                            
                            # Show wave degree and timeframes
                            degree = elliott_data.get('degree', 'Unknown')
                            print(f"   ⏰ Timeframe: {degree}")
                            
                            # Show projections if available
                            projections = elliott_data.get('projections', {})
                            if projections:
                                print(f"   🎯 Next Targets:")
                                for level, price in list(projections.items())[:3]:  # Show first 3
                                    if isinstance(price, (int, float)) and price > 0:
                                        distance = abs(current_price - price) / current_price * 100
                                        print(f"      {level}: ${price:,.2f} ({distance:.1f}% away)")
                        else:
                            print("   💡 Elliott Wave analysis in progress...")
                    else:
                        print("   💡 Market data not available for Elliott Wave analysis")
                else:
                    print("   💡 Current price not available for Elliott Wave analysis")
            except Exception as e:
                print(f"   💡 Elliott Wave analysis: {str(e)[:50]}")
            
            # 4. TECHNICAL INDICATORS
            print(f"\n📊 TECHNICAL INDICATORS:")
            try:
                # Get technical indicators directly from analyzer
                current_price = signal.get('current_price', 0)
                if current_price and current_price != 'N/A':
                    market_data = self.get_market_data()
                    if "error" not in market_data:
                        prices = market_data["prices"]
                        volumes = market_data["volumes"]
                        
                        # Calculate RSI
                        from indicators import TechnicalIndicators
                        tech_indicators = TechnicalIndicators()
                        rsi = tech_indicators.calculate_rsi(prices)
                        macd = tech_indicators.calculate_macd(prices)
                        bollinger = tech_indicators.calculate_bollinger(prices)
                        
                        if rsi is not None:
                            print(f"   📈 RSI: {rsi:.2f}")
                        if macd and 'macd' in macd:
                            print(f"   📊 MACD: {macd['macd']:.2f}")
                        if bollinger and 'position' in bollinger:
                            print(f"   📉 Bollinger: {bollinger['position']}")
                    else:
                        print("   💡 Market data not available for technical indicators")
                else:
                    print("   💡 Current price not available for technical indicators")
            except Exception as e:
                print(f"   💡 Technical indicators: {str(e)[:50]}")
            
            # 5. WHALE ACTIVITY ALERTS
            print(f"\n🐋 WHALE ACTIVITY:")
            whale_alerts = signal.get('alerts', {}).get('whale_alerts', [])
            if whale_alerts:
                for alert in whale_alerts:
                    data = alert.get('data', {})
                    if data.get('show_alert'):
                        print(f"   🚨 {data.get('message', 'Whale activity detected')}")
            else:
                print("   ✅ No significant whale activity")
            
            # 6. TREND REVERSAL SIGNALS
            print(f"\n🔄 TREND REVERSAL:")
            reversal_alerts = signal.get('alerts', {}).get('reversal_alerts', [])
            if reversal_alerts:
                for alert in reversal_alerts:
                    data = alert.get('data', {})
                    if data.get('show_alert'):
                        print(f"   🚨 {data.get('message', 'Reversal signal detected')}")
            else:
                print("   ✅ No reversal signals")
            
            # 7. ML PREDICTIONS (if available)
            print(f"\n🤖 ML PREDICTIONS:")
            
            # Enhanced ML
            try:
                ml_result = self.ml_system.predict_enhanced(7)  # 7 days ahead
                if 'error' not in ml_result:
                    pred = ml_result.get('prediction_label', 'Unknown')
                    conf = ml_result.get('confidence', 0)
                    print(f"   📊 Enhanced ML: {pred} ({conf:.1%} confidence)")
                else:
                    print("   💡 Enhanced ML: Train models first (Option 8 → 4)")
            except:
                print("   💡 Enhanced ML: Train models first (Option 8 → 4)")
            
            # Revolutionary Models
            print(f"   🚀 REVOLUTIONARY 2025 MODELS:")
            print(f"      📊 Status: Models initialized, ready for training")
            
            # Helformer
            try:
                helformer_result = self.ml_system.predict_helformer(1)  # Use 1 period as trained
                if 'error' not in helformer_result:
                    pred = helformer_result.get('prediction_label', 'Unknown')
                    conf = helformer_result.get('confidence', 0)
                    print(f"      🔥 Helformer: {pred} ({conf:.1%} confidence)")
                else:
                    print(f"      💡 Helformer: Ready to train (Option 8 → 5)")
            except:
                print(f"      💡 Helformer: Ready to train (Option 8 → 5)")
            
            # TFT
            try:
                tft_result = self.ml_system.predict_tft(24)
                if 'error' not in tft_result:
                    pred = tft_result.get('prediction_label', 'Unknown')
                    conf = tft_result.get('confidence', 0)
                    print(f"      🚀 TFT: {pred} ({conf:.1%} confidence)")
                else:
                    print(f"      💡 TFT: Ready to train (Option 8 → 6)")
            except:
                print(f"      💡 TFT: Ready to train (Option 8 → 6)")
            
            # Performer
            try:
                performer_result = self.ml_system.predict_performer(7)
                if 'error' not in performer_result:
                    pred = performer_result.get('prediction_label', 'Unknown')
                    conf = performer_result.get('confidence', 0)
                    print(f"      ⚡ Performer: {pred} ({conf:.1%} confidence)")
                else:
                    print(f"      💡 Performer: Ready to train (Option 8 → 7)")
            except:
                print(f"      💡 Performer: Ready to train (Option 8 → 7)")
            
            # 8. CRITICAL ALERTS SUMMARY
            print(f"\n🚨 CRITICAL ALERTS:")
            alerts = signal.get('alerts', {})
            if alerts.get('show_any'):
                total_alerts = len(alerts.get('whale_alerts', [])) + len(alerts.get('reversal_alerts', []))
                print(f"   ⚠️  {total_alerts} active alerts - Check individual modules for details")
            else:
                print("   ✅ No critical alerts")
            
            # 9. TRADING RECOMMENDATION
            print(f"\n💡 TRADING RECOMMENDATION:")
            signal_strength = signal.get('signal_strength', 'neutral')
            if signal_strength == 'strong_buy':
                print("   🚀 STRONG BUY SIGNAL - Consider entering long position")
            elif signal_strength == 'buy':
                print("   📈 BUY SIGNAL - Monitor for entry opportunities")
            elif signal_strength == 'strong_sell':
                print("   🔴 STRONG SELL SIGNAL - Consider exiting or shorting")
            elif signal_strength == 'sell':
                print("   📉 SELL SIGNAL - Monitor for exit opportunities")
            else:
                print("   ⏸️  NEUTRAL - Wait for clearer signals")
            
            print(f"\n{'='*80}")
            print("💡 For detailed analysis of any module, select individual options (2-9)")
            print("🤖 For ML predictions, select Option 8")
            
        except Exception as e:
            print(f"❌ Error displaying analysis: {e}")
            return False
        
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
            print("📅 SELECT MODEL & PREDICTION HORIZON:")
            print("🧠 ENHANCED ML MODELS:")
            print("1. 7 days ahead (1 week) - Enhanced ML")
            print("2. 30 days ahead (1 month) - Enhanced ML") 
            print("3. 90 days ahead (3 months) - Enhanced ML")
            print("4. Train enhanced models first")
            print()
            print("🚀 REVOLUTIONARY 2025 MODELS:")
            print("5. Helformer prediction (925% return potential)")
            print("6. TFT prediction (Multi-horizon forecasting)")
            print("7. Performer prediction (Linear complexity)")
            print("8. Ensemble prediction (All models combined)")
            print()
            print("9. Show discovered patterns")
            
            choice = input(f"\n{self.display.colorize('Select option (1-9): ', 'cyan')}").strip()
            
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
                # Helformer prediction
                print("🔥 HELFORMER PREDICTION (925% Excess Return Potential)")
                print("=" * 60)
                helformer_result = self.ml_system.predict_helformer(1, 24)  # Use 1 period as trained
                if "error" not in helformer_result:
                    self._display_revolutionary_prediction(helformer_result, "Helformer")
                else:
                    print(f"❌ Helformer error: {helformer_result['error']}")
                    print("💡 Train Helformer model first with: python3 train_revolutionary_models.py --models helformer --periods 10")
                return
                
            elif choice == "6":
                # TFT prediction
                print("🚀 TFT PREDICTION (Multi-Horizon Forecasting)")
                print("=" * 55)
                tft_result = self.ml_system.predict_tft(24, [1, 6, 12, 24, 48])
                if "error" not in tft_result:
                    self._display_revolutionary_prediction(tft_result, "TFT")
                else:
                    print(f"❌ TFT error: {tft_result['error']}")
                    print("💡 Train TFT model first with: python3 train_revolutionary_models.py --models tft --periods 24")
                return
                
            elif choice == "7":
                # Performer prediction
                print("⚡ PERFORMER PREDICTION (Linear Complexity O(N))")
                print("=" * 55)
                performer_result = self.ml_system.predict_performer(168, True)
                if "error" not in performer_result:
                    self._display_revolutionary_prediction(performer_result, "Performer")
                else:
                    print(f"❌ Performer error: {performer_result['error']}")
                    print("💡 Train Performer model first with: python3 train_revolutionary_models.py --models performer --periods 7")
                return
                
            elif choice == "8":
                # Ensemble prediction
                print("🎭 ENSEMBLE PREDICTION (All Revolutionary Models)")
                print("=" * 55)
                print("💡 Running all available revolutionary models...")
                self._run_ensemble_prediction()
                return
                
            elif choice == "9":
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
            print(f"⏰ Prediction Horizon: {periods} days ahead")
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
    
    def _display_revolutionary_prediction(self, result, model_name):
        """Display revolutionary model prediction results"""
        print(f"\n🎯 {model_name.upper()} PREDICTION RESULTS")
        print("=" * 50)
        
        if model_name == "Helformer":
            current_price = result.get('current_price', 0)
            predicted_price = result.get('predicted_price', 0)
            direction = result.get('predicted_direction', 'Unknown')
            confidence = result.get('confidence', 0)
            
            print(f"💰 Current Price: ${current_price:.2f}")
            print(f"🎯 Predicted Price: ${predicted_price:.2f}")
            print(f"📊 Direction: {direction}")
            print(f"🎲 Confidence: {confidence:.1%}")
            print(f"🔥 Model: Helformer (925% excess return potential)")
            
        elif model_name == "TFT":
            current_price = result.get('current_bnb_price', 0)
            primary = result.get('primary_prediction', {})
            
            print(f"💰 Current Price: ${current_price:.2f}")
            print(f"🎯 Primary Prediction: {primary}")
            print(f"🚀 Multi-horizon forecasting with uncertainty quantification")
            
            # Show multi-horizon if available
            multi_horizon = result.get('multi_horizon_forecasts', {})
            if multi_horizon:
                print(f"📊 Multi-Horizon Forecasts:")
                for horizon, forecast in multi_horizon.items():
                    print(f"   {horizon}: {forecast}")
                    
        elif model_name == "Performer":
            current_price = result.get('current_price', 0)
            primary = result.get('primary_prediction', {})
            
            print(f"💰 Current Price: ${current_price:.2f}")
            print(f"🎯 Primary Prediction: {primary}")
            print(f"⚡ Linear Complexity: O(N) vs O(N²) standard Transformers")
            print(f"🚀 FAVOR+ attention mechanism")
            
            # Show multi-horizon if available
            multi_horizon = result.get('multi_horizon_forecasts', {})
            if multi_horizon:
                print(f"📊 Multi-Horizon Forecasts:")
                for horizon, forecast in multi_horizon.items():
                    print(f"   {horizon}: ${forecast.get('price', 0):.2f} ({forecast.get('percentage_change', 0):+.1f}%)")
        
        print(f"\n🚀 {model_name} analysis completed!")
        input("\nPress Enter to continue...")
    
    def _run_ensemble_prediction(self):
        """Run ensemble prediction combining multiple revolutionary models"""
        print("🎭 Running ensemble prediction with available models...")
        
        models_results = {}
        
        # Try Enhanced ML first
        try:
            enhanced_result = self.ml_system.predict_bnb_enhanced(30)
            if "error" not in enhanced_result:
                models_results["Enhanced"] = enhanced_result
                print("✅ Enhanced ML prediction successful")
            else:
                print("❌ Enhanced ML not available")
        except:
            print("❌ Enhanced ML not available")
        
        # Try Revolutionary models (will show errors if not trained)
        revolutionary_models = [
                            ("Helformer", lambda: self.ml_system.predict_helformer(1, 24)),  # Use 1 period as trained
            ("TFT", lambda: self.ml_system.predict_tft(24, [1, 6, 12, 24])),
            ("Performer", lambda: self.ml_system.predict_performer(168, True))
        ]
        
        for model_name, predict_func in revolutionary_models:
            try:
                result = predict_func()
                if "error" not in result:
                    models_results[model_name] = result
                    print(f"✅ {model_name} prediction successful")
                else:
                    print(f"❌ {model_name} not available: {result['error']}")
            except Exception as e:
                print(f"❌ {model_name} not available: {e}")
        
        if models_results:
            print(f"\n🎭 ENSEMBLE RESULTS ({len(models_results)} models)")
            print("=" * 50)
            
            for model_name, result in models_results.items():
                print(f"\n🤖 {model_name.upper()}:")
                if model_name == "Enhanced":
                    pred = result.get('prediction_label', 'Unknown')
                    conf = result.get('confidence', 0)
                    print(f"   📊 Prediction: {pred}")
                    print(f"   🎲 Confidence: {conf:.1%}")
                else:
                    # For revolutionary models, show basic info
                    print(f"   📊 Model available with predictions")
            
            print(f"\n💡 For detailed analysis, select individual models (options 5-7)")
        else:
            print("❌ No models available for ensemble prediction")
            print("💡 Train models first with:")
            print("   python3 train_revolutionary_models.py --models all --periods 7")
        
        input("\nPress Enter to continue...")
    
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
