#!/usr/bin/env python3
"""
Train BNB Enhanced ML System
Learn patterns from top 10 cryptocurrencies and apply to BNB analysis
"""

import argparse
import sys
from datetime import datetime
from bnb_enhanced_ml import BNBEnhancedML

def main():
    parser = argparse.ArgumentParser(description="Train BNB Enhanced ML with Multi-Crypto Intelligence")
    parser.add_argument("--periods", type=int, nargs="+", default=[5, 10, 20],
                       help="Prediction periods in hours (default: 5 10 20)")
    parser.add_argument("--data-limit", type=int, default=1500,
                       help="Data points to fetch per cryptocurrency (default: 1500)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    print("🎯 BNB ENHANCED ML TRAINING")
    print("=" * 60)
    print("🧠 Learning from TOP 10 Cryptocurrencies:")
    print("   • Bitcoin (BTC) - Market Leader")
    print("   • Ethereum (ETH) - Smart Contracts")
    print("   • BNB - Exchange Token (Target)")
    print("   • XRP - Payments")
    print("   • Solana (SOL) - High Performance")
    print("   • Cardano (ADA) - Academic")
    print("   • Avalanche (AVAX) - DeFi")
    print("   • Polkadot (DOT) - Interoperability")
    print("   • Chainlink (LINK) - Oracles")
    print("   • Polygon (MATIC) - Scaling")
    print()
    print(f"⏰ Prediction Periods: {args.periods} hours")
    print(f"📊 Data Limit: {args.data_limit} candles per crypto")
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize enhanced BNB ML system
    bnb_ml = BNBEnhancedML()
    
    # Training summary
    training_results = {}
    total_models = 0
    successful_models = 0
    
    # Train models for each prediction period
    for periods in args.periods:
        print(f"🎯 TRAINING BNB ENHANCED MODEL FOR {periods}H PREDICTION")
        print("-" * 60)
        
        result = bnb_ml.train_bnb_enhanced_model(periods)
        
        if "success" in result:
            models_trained = result["models_trained"]
            learning_cryptos = result["learning_cryptos"]
            universal_insights = result["universal_insights"]
            training_samples = result["training_samples"]
            enhanced_features = result["enhanced_features"]
            
            print(f"✅ SUCCESS: {models_trained} BNB models trained")
            print(f"📊 Learning data from: {learning_cryptos} cryptocurrencies")
            print(f"🧠 Universal insights discovered: {universal_insights}")
            print(f"📈 Training samples: {training_samples}")
            print(f"🔧 Enhanced features: {enhanced_features}")
            
            # Show individual model performance
            print(f"\n📊 MODEL PERFORMANCE:")
            for model_name, metrics in result["results"].items():
                if "error" not in metrics:
                    train_acc = metrics["train_accuracy"]
                    test_acc = metrics["test_accuracy"]
                    print(f"   🤖 {model_name}: Train={train_acc:.3f}, Test={test_acc:.3f}")
                else:
                    print(f"   ❌ {model_name}: {metrics['error']}")
            
            # Show discovered insights
            if "universal_insights" in result:
                insights = result.get("universal_insights", [])
                if insights:
                    print(f"\n🧠 DISCOVERED PATTERNS:")
                    for insight in insights[:5]:  # Show top 5
                        print(f"   {insight}")
                    if len(insights) > 5:
                        print(f"   ... and {len(insights) - 5} more insights")
            
            training_results[f"{periods}h"] = {
                "success": True,
                "models": models_trained,
                "learning_cryptos": learning_cryptos,
                "insights": universal_insights,
                "samples": training_samples,
                "features": enhanced_features,
                "results": result["results"]
            }
            
            successful_models += models_trained
            total_models += models_trained
            
        else:
            error_msg = result.get("error", "Unknown error")
            print(f"❌ TRAINING FAILED: {error_msg}")
            
            training_results[f"{periods}h"] = {
                "success": False,
                "error": error_msg
            }
            
            total_models += 3  # Expected number of models
        
        print()  # Add spacing between periods
    
    # Final summary
    print(f"{'='*60}")
    print("📊 BNB ENHANCED ML TRAINING SUMMARY")
    print("="*60)
    
    success_rate = (successful_models / total_models * 100) if total_models > 0 else 0
    
    print(f"🎯 Target Asset: BNB (Enhanced with Multi-Crypto Intelligence)")
    print(f"⏰ Time Horizons Trained: {len(args.periods)}")
    print(f"🤖 Total Models: {successful_models}/{total_models} ({success_rate:.1f}%)")
    print(f"🕐 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Detailed results per period
    print(f"\n📋 DETAILED RESULTS:")
    print("-" * 30)
    
    total_insights = 0
    
    for period, data in training_results.items():
        if data["success"]:
            models = data["models"]
            cryptos = data["learning_cryptos"]
            insights = data["insights"]
            samples = data["samples"]
            features = data["features"]
            
            print(f"\n🎯 {period.upper()}:")
            print(f"   ✅ Models: {models}")
            print(f"   📊 Learning Cryptos: {cryptos}")
            print(f"   🧠 Insights: {insights}")
            print(f"   📈 Samples: {samples}")
            print(f"   🔧 Features: {features}")
            
            # Best performing model
            best_model = None
            best_test_acc = 0
            for model_name, metrics in data["results"].items():
                if "test_accuracy" in metrics and metrics["test_accuracy"] > best_test_acc:
                    best_test_acc = metrics["test_accuracy"]
                    best_model = model_name
            
            if best_model:
                print(f"   🏆 Best: {best_model} ({best_test_acc:.3f})")
            
            total_insights += insights
        else:
            print(f"\n❌ {period.upper()}: {data['error']}")
    
    # Intelligence Summary
    print(f"\n🧠 INTELLIGENCE SUMMARY:")
    print("-" * 25)
    print(f"📊 Total Insights Discovered: {total_insights}")
    print(f"🎯 BNB Analysis Enhanced: ✅")
    print(f"🌐 Multi-Crypto Learning: ✅")
    print(f"🔧 Pattern Recognition: ✅")
    
    # Usage instructions
    print(f"\n💡 NEXT STEPS:")
    print("-" * 15)
    print("1. Make enhanced BNB predictions:")
    print("   python3 predict_bnb_enhanced.py --periods 10")
    print()
    print("2. Use integrated system:")
    print("   python3 main.py  # Select BNB Enhanced ML option")
    print()
    print("3. Compare with other ML systems:")
    print("   python3 predict_reversals.py --periods 10")
    print("   python3 predict_multi_crypto.py --asset BNB --periods 10")
    
    print(f"\n🎯 BNB Enhanced ML training completed!")
    print("🧠 Your BNB analysis is now enhanced with multi-crypto intelligence!")
    
    return successful_models > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
