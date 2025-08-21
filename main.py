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
from elliott_wave import ElliottWaveAnalyzer
from ichimoku_module import IchimokuAnalyzer
from whale_tracker import WhaleTracker
from sentiment_module import SentimentAnalyzer
from correlation_module import CorrelationAnalyzer
from ml_predictor import MLPredictor
from ml_enhanced import TrendReversalML
from multi_crypto_ml import MultiCryptoML
from bnb_enhanced_ml import BNBEnhancedML
from trend_reversal import TrendReversalDetector


class BNBAdvancedAnalyzer:
    """Main BNB trading analyzer using modular components"""
    
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
        self.ml_predictor = MLPredictor()
        self.ml_reversal = TrendReversalML()
        self.multi_crypto_ml = MultiCryptoML()
        self.bnb_enhanced_ml = BNBEnhancedML()
        self.reversal_detector = TrendReversalDetector()
        
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
            "ml_alerts": [],
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
            
            # Check ML prediction alerts
            ml_alert = self.ml_predictor.check_critical_ml_alerts("1d")
            if ml_alert.get("show_alert"):
                alerts["ml_alerts"].append({
                    "type": "ml_predictions",
                    "data": ml_alert
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
    
    def show_ml_predictions(self):
        """Show comprehensive ML predictions analysis"""
        print("\n" + "="*60)
        print("🤖 COMPREHENSIVE ML PREDICTIONS ANALYSIS")
        print("="*60)
        
        try:
            print("📅 SELECT ANALYSIS TYPE:")
            print("1. Daily predictions (1d, 1w)")
            print("2. Strategic long-term analysis (1m, 6m, 1y)") 
            print("3. Train models and predict")
            print("4. Full ML analysis (all timeframes)")
            
            choice = input(f"\n{self.display.colorize('Select analysis (1-4): ', 'cyan')}").strip()
            
            if choice == "1":
                print("🔍 Select daily timeframe:")
                print("  a) 1 day predictions")
                print("  b) 1 week predictions")
                sub_choice = input("Choice (a/b): ").strip().lower()
                if sub_choice == "a":
                    self.ml_predictor.display_ml_analysis("1d")
                else:
                    self.ml_predictor.display_ml_analysis("1w")
            elif choice == "2":
                print("📊 Running strategic long-term analysis...")
                self.ml_predictor.display_strategic_analysis()
            elif choice == "3":
                print("🤖 Training models...")
                horizon = input("Enter horizon (1d/1w/1m): ").strip() or "1d"
                training_result = self.ml_predictor.train_models_for_horizon(horizon)
                if "error" not in training_result:
                    self.ml_predictor.display_ml_analysis(horizon)
                else:
                    print(f"❌ Training failed: {training_result['error']}")
            elif choice == "4":
                self.ml_predictor.run_full_ml_analysis()
            else:
                print("Invalid choice, using strategic analysis")
                self.ml_predictor.display_strategic_analysis()
                
        except Exception as e:
            print(f"❌ Error in ML analysis: {e}")
    
    def show_ml_reversal_analysis(self):
        """Show ML-based trend reversal analysis"""
        print("\n" + "="*60)
        print("🤖 ML TREND REVERSAL DETECTION")
        print("="*60)
        
        try:
            print("📅 SELECT PREDICTION HORIZON:")
            print("1. 5 hours ahead")
            print("2. 10 hours ahead") 
            print("3. 20 hours ahead")
            print("4. Train new models")
            
            choice = input(f"\n{self.display.colorize('Select option (1-4): ', 'cyan')}").strip()
            
            if choice == "1":
                periods = 5
            elif choice == "2":
                periods = 10
            elif choice == "3":
                periods = 20
            elif choice == "4":
                print("🎓 Training new models...")
                # Get fresh data for training
                training_data = self.ml_predictor.fetch_training_data("1h", 2000)
                if training_data is not None:
                    result = self.ml_reversal.train_reversal_models(training_data, 10)
                    if "success" in result:
                        print(f"✅ Training completed: {result['models_trained']} models")
                        periods = 10
                    else:
                        print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
                        return
                else:
                    print("❌ Could not fetch training data")
                    return
            else:
                print("Invalid choice, using default (10 hours)")
                periods = 10
            
            # Get recent data for prediction
            recent_data = self.ml_predictor.fetch_training_data("1h", 200)
            if recent_data is None:
                print("❌ Could not fetch recent market data")
                return
            
            # Make prediction
            print(f"\n🔮 Making ML trend reversal prediction for {periods} hours ahead...")
            prediction = self.ml_reversal.predict_reversal(recent_data, periods)
            
            if "error" in prediction:
                print(f"❌ Prediction error: {prediction['error']}")
                print("💡 Try training models first (option 4)")
                return
            
            # Display results
            current_price = recent_data['close'].iloc[-1]
            pred_label = prediction['prediction_label']
            confidence = prediction['confidence']
            ensemble_pred = prediction['ensemble_prediction']
            
            print(f"\n🎯 ML REVERSAL PREDICTION RESULTS")
            print("=" * 40)
            print(f"💰 Current Price: ${current_price:.2f}")
            print(f"⏰ Time Horizon: {periods} hours ahead")
            
            # Color coding based on prediction
            if ensemble_pred == 1:  # Bullish reversal
                icon = "🟢📈"
                advice = "BULLISH REVERSAL - Consider LONG position"
                color = "green"
            elif ensemble_pred == 2:  # Bearish reversal
                icon = "🔴📉"
                advice = "BEARISH REVERSAL - Consider SHORT position"
                color = "red"
            else:  # No reversal
                icon = "🟡➡️"
                advice = "NO REVERSAL - Current trend continues"
                color = "yellow"
            
            print(f"{icon} Prediction: {self.display.colorize(pred_label, color)}")
            print(f"🎲 Confidence: {confidence:.1%}")
            print(f"💡 Advice: {advice}")
            
            # Show individual model predictions if confidence is decent
            if confidence > 0:
                print(f"\n📊 INDIVIDUAL MODEL PREDICTIONS:")
                print("-" * 30)
                
                for model_name, model_pred in prediction['individual_predictions'].items():
                    pred_num = model_pred['prediction']
                    pred_text = ["No Reversal", "Bullish Reversal", "Bearish Reversal"][pred_num]
                    proba = model_pred.get('probability', [])
                    
                    if proba:
                        max_proba = max(proba)
                        print(f"🤖 {model_name}: {pred_text} ({max_proba:.1%})")
                    else:
                        print(f"🤖 {model_name}: {pred_text}")
            
            print(f"\n⚠️ This is ML prediction, not financial advice!")
            
        except Exception as e:
            print(f"❌ Error in ML reversal analysis: {e}")
    
    def show_multi_crypto_analysis(self):
        """Show multi-cryptocurrency ML analysis"""
        print("\n" + "="*60)
        print("🌐 MULTI-CRYPTOCURRENCY ML ANALYSIS")
        print("="*60)
        
        try:
            print("🎯 SELECT TARGET ASSET:")
            print("1. BNB (Binance Coin)")
            print("2. ETH (Ethereum)")
            print("3. SOL (Solana)")
            print("4. ADA (Cardano)")
            print("5. Custom asset")
            
            choice = input(f"\n{self.display.colorize('Select asset (1-5): ', 'cyan')}").strip()
            
            asset_map = {
                "1": "BNB",
                "2": "ETH", 
                "3": "SOL",
                "4": "ADA"
            }
            
            if choice in asset_map:
                target_asset = asset_map[choice]
            elif choice == "5":
                target_asset = input("Enter asset symbol (e.g., AVAX): ").strip().upper()
            else:
                target_asset = "BNB"
                print("Invalid choice, using BNB")
            
            print(f"\n📅 SELECT PREDICTION HORIZON:")
            print("1. 5 hours ahead")
            print("2. 10 hours ahead")
            print("3. 20 hours ahead")
            print("4. Train models first")
            
            horizon_choice = input(f"\n{self.display.colorize('Select horizon (1-4): ', 'cyan')}").strip()
            
            if horizon_choice == "1":
                periods = 5
            elif horizon_choice == "2":
                periods = 10
            elif horizon_choice == "3":
                periods = 20
            elif horizon_choice == "4":
                print("🎓 Training multi-crypto models...")
                result = self.multi_crypto_ml.train_multi_crypto_models(target_asset, 10)
                if "success" in result:
                    print(f"✅ Training completed: {result['models_trained']} models")
                    print(f"📊 Cross-asset features: {result['feature_count']}")
                    periods = 10
                else:
                    print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
                    return
            else:
                periods = 10
                print("Invalid choice, using 10 hours")
            
            # Make multi-crypto prediction
            print(f"\n🔮 Making multi-crypto prediction for {target_asset} ({periods}h ahead)...")
            prediction = self.multi_crypto_ml.predict_multi_crypto_reversal(target_asset, periods)
            
            if "error" in prediction:
                print(f"❌ Prediction error: {prediction['error']}")
                print("💡 Try training models first (option 4)")
                return
            
            # Display results
            current_price = prediction["current_price"]
            pred_label = prediction["prediction_label"]
            confidence = prediction["confidence"]
            ensemble_pred = prediction["ensemble_prediction"]
            
            print(f"\n🎯 MULTI-CRYPTO PREDICTION RESULTS")
            print("=" * 45)
            print(f"💰 Current {target_asset} Price: ${current_price:.2f}")
            print(f"⏰ Prediction Horizon: {periods} hours ahead")
            
            # Color coding based on prediction
            if ensemble_pred == 1:  # Bullish reversal
                icon = "🟢📈"
                advice = "BULLISH REVERSAL - Consider LONG position"
                color = "green"
            elif ensemble_pred == 2:  # Bearish reversal
                icon = "🔴📉"
                advice = "BEARISH REVERSAL - Consider SHORT position"
                color = "red"
            else:  # No reversal
                icon = "🟡➡️"
                advice = "NO REVERSAL - Current trend continues"
                color = "yellow"
            
            print(f"{icon} Prediction: {self.display.colorize(pred_label, color)}")
            print(f"🎲 Confidence: {confidence:.1%}")
            print(f"💡 Advice: {advice}")
            
            # Market context
            if "market_context" in prediction:
                context = prediction["market_context"]
                print(f"\n🌐 MARKET CONTEXT:")
                print("-" * 20)
                
                if "btc_price" in context:
                    btc_price = context["btc_price"]
                    print(f"₿ Bitcoin: ${btc_price:.2f}")
                
                if "btc_correlation" in context:
                    btc_corr = context["btc_correlation"]
                    corr_strength = "Strong" if abs(btc_corr) > 0.7 else "Moderate" if abs(btc_corr) > 0.4 else "Weak"
                    print(f"🔗 BTC Correlation: {btc_corr:.3f} ({corr_strength})")
                
                if "btc_dominance" in context:
                    dominance = context["btc_dominance"]
                    print(f"👑 BTC Dominance: {dominance:.1%}")
            
            # Show model agreement if confidence is decent
            if confidence > 0:
                print(f"\n📊 MODEL BREAKDOWN:")
                print("-" * 20)
                
                for model_name, model_pred in prediction["individual_predictions"].items():
                    pred_num = model_pred["prediction"]
                    pred_text = ["No Reversal", "Bullish", "Bearish"][pred_num]
                    proba = model_pred.get("probability", [])
                    
                    if proba:
                        max_proba = max(proba)
                        icon = "🎯" if max_proba > 0.8 else "🎲"
                        print(f"{icon} {model_name}: {pred_text} ({max_proba:.1%})")
                    else:
                        print(f"🤖 {model_name}: {pred_text}")
            
            # Market overview
            print(f"\n📊 TOP CRYPTO SNAPSHOT:")
            print("-" * 25)
            
            overview = self.multi_crypto_ml.get_market_overview()
            if "error" not in overview:
                count = 0
                for symbol, data in overview["market_summary"].items():
                    if count >= 5:  # Show top 5
                        break
                    change = data["daily_change_pct"]
                    crypto_name = data["crypto_name"]
                    price = data["price"]
                    emoji = "🟢" if change > 0 else "🔴" if change < 0 else "🟡"
                    print(f"{emoji} {crypto_name}: ${price:.2f} ({change:+.1f}%)")
                    count += 1
            
            print(f"\n⚠️ Multi-crypto ML analysis - Enhanced with cross-asset intelligence!")
            
        except Exception as e:
            print(f"❌ Error in multi-crypto analysis: {e}")
    
    def show_bnb_enhanced_analysis(self):
        """Show BNB analysis enhanced with multi-crypto intelligence"""
        print("\n" + "="*60)
        print("🎯 BNB ENHANCED ML ANALYSIS")
        print("🧠 Enhanced with Top 10 Cryptocurrency Intelligence")
        print("="*60)
        
        try:
            print("📅 SELECT PREDICTION HORIZON:")
            print("1. 5 hours ahead")
            print("2. 10 hours ahead") 
            print("3. 20 hours ahead")
            print("4. Train enhanced models first")
            print("5. Show discovered patterns")
            
            choice = input(f"\n{self.display.colorize('Select option (1-5): ', 'cyan')}").strip()
            
            if choice == "1":
                periods = 5
            elif choice == "2":
                periods = 10
            elif choice == "3":
                periods = 20
            elif choice == "4":
                print("🧠 Training BNB enhanced models with multi-crypto intelligence...")
                result = self.bnb_enhanced_ml.train_bnb_enhanced_model(10)
                if "success" in result:
                    print(f"✅ Training completed: {result['models_trained']} enhanced models")
                    print(f"📊 Learning from: {result['learning_cryptos']} cryptocurrencies")
                    print(f"🧠 Universal insights: {result['universal_insights']}")
                    print(f"🔧 Enhanced features: {result['enhanced_features']}")
                    periods = 10
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
                periods = 10
                print("Invalid choice, using 10 hours")
            
            # Make enhanced prediction
            print(f"\n🔮 Making BNB enhanced prediction ({periods}h ahead)...")
            print("🧠 Using patterns learned from top 10 cryptocurrencies...")
            
            prediction = self.bnb_enhanced_ml.predict_bnb_enhanced(periods)
            
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
            
            # Trading suggestions
            print(f"\n💼 ENHANCED TRADING SUGGESTIONS:")
            print("-" * 35)
            
            if ensemble_pred == 1 and confidence > 0.7:  # Strong bullish
                print("🟢 ENHANCED BULLISH SETUP:")
                print(f"   📈 Entry: ${current_price:.2f}")
                print(f"   ⏰ Timeframe: {periods}h enhanced prediction")
                print(f"   🛡️ Stop Loss: Enhanced risk management (3-5% below)")
                print(f"   🎯 Target: Multi-crypto validated levels")
                print(f"   🧠 Intelligence: Cross-crypto pattern confirmation")
                
            elif ensemble_pred == 2 and confidence > 0.7:  # Strong bearish
                print("🔴 ENHANCED BEARISH SETUP:")
                print(f"   📉 Entry: ${current_price:.2f}")
                print(f"   ⏰ Timeframe: {periods}h enhanced prediction")
                print(f"   🛡️ Stop Loss: Enhanced risk management (3-5% above)")
                print(f"   🎯 Target: Multi-crypto validated levels")
                print(f"   🧠 Intelligence: Cross-crypto pattern confirmation")
                
        else:
                print("🟡 ENHANCED ANALYSIS SUGGESTS:")
                print(f"   📊 Current Signal: {signal}")
                print(f"   💡 Recommendation: Wait for higher confidence")
                print(f"   🧠 Enhanced Intelligence: Suggests patience")
                print(f"   🌐 Monitor: Cross-crypto market movements")
            
            print(f"\n🎯 BNB Enhanced Analysis - Powered by Multi-Crypto Intelligence!")
            
        except Exception as e:
            print(f"❌ Error in BNB enhanced analysis: {e}")
    
    def show_reversal_analysis(self):
        """Show comprehensive trend reversal analysis"""
        print("\n" + "="*60)
        print("🔄 COMPREHENSIVE TREND REVERSAL ANALYSIS")
        print("="*60)
        
        try:
            results = self.reversal_detector.multi_timeframe_reversal_analysis()
            
            if "error" in results:
                print(f"❌ Error: {results['error']}")
            else:
                print(f"\n✅ Analysis completed successfully!")
                
                # Summary
                overall = results.get("overall_signals", {})
                high_conviction = results.get("high_conviction_signals", [])
                
                if high_conviction:
                    print(f"\n🚨 HIGH CONVICTION SIGNALS:")
                    for signal in high_conviction:
                        print(f"   • {signal}")
                
                if overall.get("bullish", 0) >= 3:
                    print(f"\n🟢 TREND OUTLOOK: BULLISH REVERSAL EXPECTED")
                elif overall.get("bearish", 0) >= 3:
                    print(f"\n🔴 TREND OUTLOOK: BEARISH REVERSAL EXPECTED")
                else:
                    print(f"\n🟡 TREND OUTLOOK: MIXED/UNCLEAR SIGNALS")
                    
        except Exception as e:
            print(f"❌ Error in reversal analysis: {e}")
    
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
                choice = self.display_enhanced_menu()
                
                if choice == "1":
                    print("\n🔄 Refreshing analysis...")
                    time.sleep(1)
                    
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
                    self.show_ml_predictions()
                    input(f"\n{self.display.colorize('Press Enter to continue...', 'cyan')}")
                    
                elif choice == "9":
                    self.show_ml_reversal_analysis()
                    input(f"\n{self.display.colorize('Press Enter to continue...', 'cyan')}")
                    
                elif choice == "10":
                    self.show_multi_crypto_analysis()
                    input(f"\n{self.display.colorize('Press Enter to continue...', 'cyan')}")
                    
                elif choice == "11":
                    self.show_bnb_enhanced_analysis()
                    input(f"\n{self.display.colorize('Press Enter to continue...', 'cyan')}")
                    
                elif choice == "12":
                    self.show_reversal_analysis()
                    input(f"\n{self.display.colorize('Press Enter to continue...', 'cyan')}")
                    
                elif choice == "13":
                    self.show_market_summary()
                    input(f"\n{self.display.colorize('Press Enter to continue...', 'cyan')}")
                    
                elif choice == "14":
                    self.display.toggle_colors()
                    time.sleep(1)
                    
                elif choice == "15":
                    print(f"\n{self.display.colorize('👋 Thank you for using BNB Advanced Analyzer!', 'green')}")
                    break
                    
                else:
                    print(f"\n{self.display.colorize('❌ Invalid choice. Please select 1-15.', 'red')}")
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print(f"\n\n{self.display.colorize('👋 Analysis stopped by user. Goodbye!', 'yellow')}")
                break
                
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("🔄 Restarting in 3 seconds...")
                time.sleep(3)

    def display_enhanced_menu(self) -> str:
        """Display enhanced menu with all analysis options"""
        print(f"\n{self.display.colorize('Options:', 'bold')}")
        print("1. Refresh analysis")
        print("2. Show detailed Fibonacci analysis")
        print("3. Show Elliott Wave analysis")
        print("4. Show Ichimoku Cloud analysis")
        print("5. Show Whale Tracking analysis")
        print("6. Show Sentiment Analysis")
        print("7. Show Correlation Analysis (BTC/ETH)")
        print("8. Show ML Predictions (AI Price Forecasts)")
        print("9. Show ML Trend Reversal Detection")
        print("10. Show Multi-Crypto ML Analysis")
        print("11. Show BNB Enhanced ML (Multi-Crypto Intelligence)")
        print("12. Show Trend Reversal Analysis")
        print("13. Show market summary")
        print("14. Toggle colors")
        print("15. Exit")
        
        return input(f"\n{self.display.colorize('Choice (1-15): ', 'cyan')}")


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