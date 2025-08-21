#!/usr/bin/env python3
"""
Multi-Cryptocurrency ML Prediction Script
Make real-time predictions using trained cross-asset models
"""

import argparse
import sys
from datetime import datetime
from multi_crypto_ml import MultiCryptoML

def main():
    parser = argparse.ArgumentParser(description="Multi-Cryptocurrency ML Predictions")
    parser.add_argument("--asset", type=str, default="BNB",
                       help="Target asset for prediction (default: BNB)")
    parser.add_argument("--periods", type=int, default=10,
                       help="Prediction horizon in hours (default: 10)")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed cross-asset analysis")
    parser.add_argument("--compare", action="store_true",
                       help="Compare with traditional single-asset analysis")
    
    args = parser.parse_args()
    
    asset = args.asset.upper()
    periods = args.periods
    
    print("🌐 MULTI-CRYPTOCURRENCY ML PREDICTIONS")
    print("=" * 60)
    print(f"🎯 Target Asset: {asset}")
    print(f"⏰ Prediction Horizon: {periods} hours")
    print(f"🕐 Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize multi-crypto ML system
    multi_ml = MultiCryptoML()
    
    # Make prediction
    print("🔮 Making multi-crypto prediction...")
    prediction = multi_ml.predict_multi_crypto_reversal(asset, periods)
    
    if "error" in prediction:
        print(f"❌ Prediction error: {prediction['error']}")
        print("\n💡 Possible solutions:")
        print("1. Train models first: python3 train_multi_crypto.py")
        print("2. Check if asset is supported")
        print("3. Ensure internet connection for data fetching")
        sys.exit(1)
    
    # Display main prediction
    current_price = prediction["current_price"]
    pred_label = prediction["prediction_label"]
    confidence = prediction["confidence"]
    ensemble_pred = prediction["ensemble_prediction"]
    
    print(f"\n🎯 MULTI-CRYPTO PREDICTION RESULTS")
    print("=" * 50)
    print(f"💰 Current {asset} Price: ${current_price:.2f}")
    print(f"⏰ Prediction Horizon: {periods} hours ahead")
    
    # Color-coded prediction display
    if ensemble_pred == 1:  # Bullish reversal
        icon = "🟢📈"
        advice = "BULLISH REVERSAL - Consider LONG position"
        signal_strength = "BUY SIGNAL"
    elif ensemble_pred == 2:  # Bearish reversal
        icon = "🔴📉"
        advice = "BEARISH REVERSAL - Consider SHORT position"
        signal_strength = "SELL SIGNAL"
    else:  # No reversal
        icon = "🟡➡️"
        advice = "NO REVERSAL - Current trend continues"
        signal_strength = "HOLD"
    
    print(f"\n{icon} Prediction: {pred_label}")
    print(f"🎲 Confidence: {confidence:.1%}")
    print(f"📊 Signal: {signal_strength}")
    print(f"💡 Advice: {advice}")
    
    # Risk assessment
    print(f"\n⚠️ RISK ASSESSMENT:")
    print("-" * 25)
    
    if confidence >= 0.8:
        risk_level = "🟢 LOW RISK"
        recommendation = "High confidence - suitable for position sizing"
    elif confidence >= 0.6:
        risk_level = "🟡 MEDIUM RISK" 
        recommendation = "Moderate confidence - consider reduced position"
    else:
        risk_level = "🔴 HIGH RISK"
        recommendation = "Low confidence - wait for better signal"
    
    print(f"Risk Level: {risk_level}")
    print(f"Recommendation: {recommendation}")
    
    # Cross-asset market context
    if "market_context" in prediction and args.detailed:
        context = prediction["market_context"]
        print(f"\n🌐 CROSS-ASSET MARKET CONTEXT:")
        print("-" * 35)
        
        if "btc_price" in context:
            btc_price = context["btc_price"]
            print(f"₿ Bitcoin Price: ${btc_price:.2f}")
        
        if "btc_correlation" in context:
            btc_corr = context["btc_correlation"]
            corr_strength = "Strong" if abs(btc_corr) > 0.7 else "Moderate" if abs(btc_corr) > 0.4 else "Weak"
            corr_direction = "Positive" if btc_corr > 0 else "Negative"
            print(f"🔗 BTC Correlation: {btc_corr:.3f} ({corr_strength} {corr_direction})")
        
        if "btc_dominance" in context:
            dominance = context["btc_dominance"]
            print(f"👑 BTC Dominance: {dominance:.1%}")
            
            if dominance > 0.5:
                print("   📊 Market favoring Bitcoin (risk-off sentiment)")
            else:
                print("   📊 Market favoring altcoins (risk-on sentiment)")
    
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
    
    # Market overview
    print(f"\n📊 TOP 10 CRYPTO MARKET SNAPSHOT:")
    print("-" * 40)
    
    overview = multi_ml.get_market_overview()
    if "error" not in overview:
        # Sort by daily change for interesting insights
        market_data = overview["market_summary"]
        sorted_cryptos = sorted(market_data.items(), 
                              key=lambda x: x[1]["daily_change_pct"], 
                              reverse=True)
        
        print("Top performers (24h):")
        for i, (symbol, data) in enumerate(sorted_cryptos[:3]):
            change = data["daily_change_pct"]
            crypto_name = data["crypto_name"]
            price = data["price"]
            print(f"  🚀 {crypto_name}: ${price:.2f} (+{change:.1f}%)")
        
        print("\nWeakest performers (24h):")
        for i, (symbol, data) in enumerate(sorted_cryptos[-3:]):
            change = data["daily_change_pct"]
            crypto_name = data["crypto_name"]
            price = data["price"]
            print(f"  💥 {crypto_name}: ${price:.2f} ({change:.1f}%)")
    
    # Trading suggestions based on prediction
    print(f"\n💼 TRADING SUGGESTIONS:")
    print("-" * 25)
    
    if ensemble_pred == 1 and confidence > 0.7:  # Strong bullish
        print("🟢 BULLISH SETUP:")
        print(f"   Entry: Current price ${current_price:.2f}")
        print(f"   Target: {periods}h timeframe")
        print(f"   Stop Loss: Consider 3-5% below entry")
        print(f"   Position Size: Based on {risk_level.split()[1]} risk")
        
    elif ensemble_pred == 2 and confidence > 0.7:  # Strong bearish
        print("🔴 BEARISH SETUP:")
        print(f"   Entry: Current price ${current_price:.2f}")
        print(f"   Target: {periods}h timeframe")
        print(f"   Stop Loss: Consider 3-5% above entry")
        print(f"   Position Size: Based on {risk_level.split()[1]} risk")
        
    else:
        print("🟡 NO CLEAR SETUP:")
        print(f"   Current Signal: {signal_strength}")
        print(f"   Recommendation: Wait for higher confidence signal")
        print(f"   Monitor: Market context and cross-asset movements")
    
    # Comparison with single-asset if requested
    if args.compare:
        print(f"\n⚖️ COMPARISON: Multi-Asset vs Single-Asset")
        print("-" * 45)
        print("🌐 Multi-Asset Analysis:")
        print(f"   ✓ Uses {len(multi_ml.crypto_symbols)} cryptocurrencies")
        print(f"   ✓ Cross-asset correlations and divergences")
        print(f"   ✓ Market dominance and sector rotation")
        print(f"   ✓ Enhanced confidence through confirmation")
        print()
        print("🎯 Single-Asset Analysis:")
        print(f"   • Uses only {asset} price and volume data")
        print(f"   • Traditional technical indicators")
        print(f"   • May miss market-wide signals")
        print()
        print("💡 Multi-asset provides more context and potentially higher accuracy!")
    
    # Footer
    print(f"\n⚠️ DISCLAIMER:")
    print("-" * 15)
    print("This is ML-based analysis, not financial advice.")
    print("Always do your own research and risk management.")
    print("Past performance doesn't guarantee future results.")
    
    print(f"\n🕐 Prediction valid until: {periods} hours from analysis time")
    print("🔄 For updated predictions, run this script again.")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n👋 Analysis stopped by user.")
        sys.exit(0)
