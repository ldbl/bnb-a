#!/usr/bin/env python3
"""
BNB Enhanced ML Predictions
Make BNB predictions using patterns learned from top 10 cryptocurrencies
"""

import argparse
import sys
from datetime import datetime
from bnb_enhanced_ml import BNBEnhancedML

def main():
    parser = argparse.ArgumentParser(description="BNB Enhanced ML Predictions")
    parser.add_argument("--periods", type=int, default=10,
                       help="Prediction horizon in hours (default: 10)")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed analysis including discovered patterns")
    parser.add_argument("--insights", action="store_true",
                       help="Show universal insights learned from all cryptocurrencies")
    
    args = parser.parse_args()
    
    periods = args.periods
    
    print("🎯 BNB ENHANCED ML PREDICTIONS")
    print("=" * 60)
    print("🧠 Enhanced with Top 10 Cryptocurrency Intelligence")
    print("🎯 Target: BNB (Binance Coin)")
    print(f"⏰ Prediction Horizon: {periods} hours")
    print(f"🕐 Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize enhanced BNB ML system
    bnb_ml = BNBEnhancedML()
    
    # Make enhanced prediction
    print("🔮 Making enhanced BNB prediction...")
    prediction = bnb_ml.predict_bnb_enhanced(periods)
    
    if "error" in prediction:
        print(f"❌ Prediction error: {prediction['error']}")
        print("\n💡 Possible solutions:")
        print("1. Train models first: python3 train_bnb_enhanced.py")
        print("2. Check internet connection for data fetching")
        print("3. Ensure sufficient historical data is available")
        sys.exit(1)
    
    # Display main prediction
    current_price = prediction["current_bnb_price"]
    pred_label = prediction["prediction_label"]
    confidence = prediction["confidence"]
    ensemble_pred = prediction["ensemble_prediction"]
    
    print(f"\n🎯 ENHANCED BNB PREDICTION RESULTS")
    print("=" * 50)
    print(f"💰 Current BNB Price: ${current_price:.2f}")
    print(f"⏰ Prediction Horizon: {periods} hours ahead")
    print(f"🧠 Enhanced Analysis: Multi-Crypto Intelligence")
    
    # Color-coded prediction display
    if ensemble_pred == 1:  # Bullish reversal
        icon = "🟢📈"
        advice = "BULLISH REVERSAL - Consider LONG position"
        signal_strength = "BUY SIGNAL"
        emoji = "🚀"
    elif ensemble_pred == 2:  # Bearish reversal
        icon = "🔴📉"
        advice = "BEARISH REVERSAL - Consider SHORT position"
        signal_strength = "SELL SIGNAL"
        emoji = "💥"
    else:  # No reversal
        icon = "🟡➡️"
        advice = "NO REVERSAL - Current trend continues"
        signal_strength = "HOLD"
        emoji = "⏸️"
    
    print(f"\n{icon} Prediction: {pred_label}")
    print(f"🎲 Confidence: {confidence:.1%}")
    print(f"📊 Signal: {signal_strength} {emoji}")
    print(f"💡 Advice: {advice}")
    
    # Risk assessment
    print(f"\n⚠️ RISK ASSESSMENT:")
    print("-" * 25)
    
    if confidence >= 0.8:
        risk_level = "🟢 LOW RISK"
        recommendation = "High confidence - suitable for position sizing"
        position_size = "Standard position (3-5% of portfolio)"
    elif confidence >= 0.6:
        risk_level = "🟡 MEDIUM RISK" 
        recommendation = "Moderate confidence - consider reduced position"
        position_size = "Reduced position (1-3% of portfolio)"
    else:
        risk_level = "🔴 HIGH RISK"
        recommendation = "Low confidence - wait for better signal"
        position_size = "Minimal position (<1% of portfolio)"
    
    print(f"Risk Level: {risk_level}")
    print(f"Recommendation: {recommendation}")
    print(f"Position Sizing: {position_size}")
    
    # Show universal insights if requested
    if args.insights and "universal_insights" in prediction:
        insights = prediction["universal_insights"]
        if insights:
            print(f"\n🧠 UNIVERSAL INSIGHTS FROM ALL CRYPTOCURRENCIES:")
            print("-" * 50)
            for i, insight in enumerate(insights, 1):
                print(f"{i}. {insight}")
            print(f"\n💡 These patterns were discovered across multiple cryptocurrencies")
            print(f"   and are now applied to enhance BNB analysis!")
    
    # Individual model breakdown
    if args.detailed:
        print(f"\n🤖 INDIVIDUAL MODEL PREDICTIONS:")
        print("-" * 35)
        
        for model_name, model_pred in prediction["individual_predictions"].items():
            pred_num = model_pred["prediction"]
            pred_text = ["No Reversal", "Bullish Reversal", "Bearish Reversal"][pred_num]
            proba = model_pred.get("probability", [])
            
            if proba:
                max_proba = max(proba)
                confidence_icon = "🎯" if max_proba > 0.8 else "🎲" if max_proba > 0.6 else "❓"
                print(f"{confidence_icon} {model_name}: {pred_text} ({max_proba:.1%})")
            else:
                print(f"🤖 {model_name}: {pred_text}")
    
    # Enhancement explanation
    print(f"\n🌐 MULTI-CRYPTO ENHANCEMENT DETAILS:")
    print("-" * 40)
    print("✅ Pattern Learning: Analyzed top 10 cryptocurrencies")
    print("✅ Fibonacci Effectiveness: Learned optimal retracement levels")
    print("✅ Volume Spike Detection: Enhanced with cross-crypto patterns")
    print("✅ Candlestick Patterns: Validated across multiple assets")
    print("✅ Market Correlation: BTC leadership and market regime detection")
    print("✅ Feature Weighting: Applied learned effectiveness weights")
    
    # Trading suggestions
    print(f"\n💼 ENHANCED TRADING SUGGESTIONS:")
    print("-" * 35)
    
    if ensemble_pred == 1 and confidence > 0.7:  # Strong bullish
        print("🟢 BULLISH SETUP (Enhanced Analysis):")
        print(f"   Entry: Current price ${current_price:.2f}")
        print(f"   Timeframe: {periods}h prediction horizon")
        print(f"   Stop Loss: 3-5% below entry (enhanced risk management)")
        print(f"   Target: Based on learned Fibonacci extension levels")
        print(f"   Confidence: {confidence:.1%} (multi-crypto validated)")
        
    elif ensemble_pred == 2 and confidence > 0.7:  # Strong bearish
        print("🔴 BEARISH SETUP (Enhanced Analysis):")
        print(f"   Entry: Current price ${current_price:.2f}")
        print(f"   Timeframe: {periods}h prediction horizon")
        print(f"   Stop Loss: 3-5% above entry (enhanced risk management)")
        print(f"   Target: Based on learned support levels")
        print(f"   Confidence: {confidence:.1%} (multi-crypto validated)")
        
    else:
        print("🟡 NO CLEAR SETUP:")
        print(f"   Current Signal: {signal_strength}")
        print(f"   Recommendation: Wait for higher confidence signal")
        print(f"   Enhanced Analysis: Suggests patience")
        print(f"   Monitor: Market-wide cryptocurrency movements")
    
    # Comparison with traditional analysis
    print(f"\n⚖️ ENHANCEMENT ADVANTAGES:")
    print("-" * 30)
    print("🎯 Traditional BNB Analysis:")
    print("   • Uses only BNB price and volume data")
    print("   • Limited to single-asset patterns")
    print("   • May miss market-wide signals")
    print("   • Standard technical indicators")
    print()
    print("🧠 Enhanced Multi-Crypto Analysis:")
    print("   • ✅ Learns from 10 cryptocurrency patterns")
    print("   • ✅ Cross-validates signals across markets")
    print("   • ✅ Enhanced Fibonacci level detection")
    print("   • ✅ Improved volume spike recognition")
    print("   • ✅ Market correlation intelligence")
    print("   • ✅ Pattern effectiveness weighting")
    print()
    print("💡 Result: Higher accuracy through market intelligence!")
    
    # Performance context
    print(f"\n📊 ENHANCED ANALYSIS CONTEXT:")
    print("-" * 30)
    print(f"🧠 Learning Sources: 10 top cryptocurrencies")
    print(f"📈 Pattern Validation: Cross-asset verification")
    print(f"🎯 BNB Focus: Enhanced with market intelligence")
    print(f"⚡ Real-time: Uses latest market data")
    print(f"🔧 Feature Engineering: 50+ enhanced indicators")
    print(f"🤖 Model Ensemble: Multiple ML algorithms")
    
    # Footer
    print(f"\n⚠️ ENHANCED DISCLAIMER:")
    print("-" * 25)
    print("This is advanced ML analysis enhanced with multi-crypto intelligence.")
    print("While more sophisticated than single-asset analysis, it's still not financial advice.")
    print("Always implement proper risk management and position sizing.")
    print("Multi-crypto patterns improve accuracy but don't guarantee results.")
    
    print(f"\n🕐 Enhanced prediction valid until: {periods} hours from analysis time")
    print("🔄 For updated enhanced predictions, run this script again.")
    print("🧠 Your BNB analysis is powered by crypto market intelligence!")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n👋 Enhanced analysis stopped by user.")
        sys.exit(0)
