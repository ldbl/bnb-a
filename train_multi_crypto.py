#!/usr/bin/env python3
"""
Multi-Cryptocurrency ML Training Script
Train ML models using cross-asset analysis for enhanced prediction accuracy
"""

import argparse
import sys
from datetime import datetime
from multi_crypto_ml import MultiCryptoML

def main():
    parser = argparse.ArgumentParser(description="Train Multi-Cryptocurrency ML Models")
    parser.add_argument("--assets", type=str, default="BNB,ETH,SOL,ADA", 
                       help="Comma-separated list of assets to train (default: BNB,ETH,SOL,ADA)")
    parser.add_argument("--periods", type=int, nargs="+", default=[5, 10, 20],
                       help="Prediction periods in hours (default: 5 10 20)")
    parser.add_argument("--data-limit", type=int, default=2000,
                       help="Number of data points to fetch for training (default: 2000)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Parse assets
    target_assets = [asset.strip().upper() for asset in args.assets.split(",")]
    
    print("🌐 MULTI-CRYPTOCURRENCY ML TRAINING")
    print("=" * 60)
    print(f"🎯 Target Assets: {', '.join(target_assets)}")
    print(f"⏰ Prediction Periods: {args.periods} hours")
    print(f"📊 Training Data Limit: {args.data_limit} candles")
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize multi-crypto ML system
    multi_ml = MultiCryptoML()
    
    # Fetch data once for all training
    print("📥 Fetching multi-cryptocurrency data...")
    multi_data = multi_ml.fetch_multi_crypto_data("1h", args.data_limit)
    
    if not multi_data:
        print("❌ Failed to fetch cryptocurrency data")
        sys.exit(1)
    
    print(f"✅ Fetched data for {len(multi_data)} cryptocurrencies")
    
    # Training summary
    training_results = {}
    total_models = 0
    successful_models = 0
    
    # Train models for each asset and period combination
    for asset in target_assets:
        print(f"\n🎯 TRAINING MODELS FOR {asset}")
        print("-" * 40)
        
        asset_results = {}
        
        for periods in args.periods:
            print(f"\n📅 Training {asset} for {periods}h prediction horizon...")
            
            result = multi_ml.train_multi_crypto_models(asset, periods)
            
            if "success" in result:
                models_trained = result["models_trained"]
                training_samples = result["training_samples"]
                feature_count = result["feature_count"]
                
                print(f"✅ Success: {models_trained} models trained")
                print(f"📊 Training samples: {training_samples}")
                print(f"🎯 Cross-asset features: {feature_count}")
                
                # Show individual model performance
                for model_name, metrics in result["results"].items():
                    if "error" not in metrics:
                        train_acc = metrics["train_accuracy"]
                        test_acc = metrics["test_accuracy"]
                        print(f"   🤖 {model_name}: Train={train_acc:.3f}, Test={test_acc:.3f}")
                
                asset_results[f"{periods}h"] = {
                    "success": True,
                    "models": models_trained,
                    "samples": training_samples,
                    "features": feature_count,
                    "results": result["results"]
                }
                
                successful_models += models_trained
                total_models += models_trained
                
            else:
                error_msg = result.get("error", "Unknown error")
                print(f"❌ Training failed: {error_msg}")
                
                asset_results[f"{periods}h"] = {
                    "success": False,
                    "error": error_msg
                }
                
                total_models += 3  # Expected number of models
        
        training_results[asset] = asset_results
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 TRAINING SUMMARY")
    print("="*60)
    
    success_rate = (successful_models / total_models * 100) if total_models > 0 else 0
    
    print(f"🎯 Assets Trained: {len(target_assets)}")
    print(f"⏰ Time Horizons: {len(args.periods)}")
    print(f"🤖 Total Models: {successful_models}/{total_models} ({success_rate:.1f}%)")
    print(f"🕐 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Detailed results per asset
    print(f"\n📋 DETAILED RESULTS:")
    print("-" * 30)
    
    for asset, results in training_results.items():
        print(f"\n🎯 {asset}:")
        for period, data in results.items():
            if data["success"]:
                models = data["models"]
                samples = data["samples"]
                features = data["features"]
                print(f"   ✅ {period}: {models} models, {samples} samples, {features} features")
                
                # Best performing model
                best_model = None
                best_test_acc = 0
                for model_name, metrics in data["results"].items():
                    if "test_accuracy" in metrics and metrics["test_accuracy"] > best_test_acc:
                        best_test_acc = metrics["test_accuracy"]
                        best_model = model_name
                
                if best_model:
                    print(f"      🏆 Best: {best_model} ({best_test_acc:.3f})")
            else:
                print(f"   ❌ {period}: {data['error']}")
    
    # Show market overview
    print(f"\n📊 CURRENT MARKET OVERVIEW:")
    print("-" * 30)
    
    overview = multi_ml.get_market_overview()
    if "error" not in overview:
        for symbol, data in overview["market_summary"].items():
            change = data["daily_change_pct"]
            emoji = "🟢" if change > 0 else "🔴" if change < 0 else "🟡"
            crypto_name = data["crypto_name"]
            price = data["price"]
            print(f"{emoji} {crypto_name}: ${price:.2f} ({change:+.1f}%)")
    
    # Usage instructions
    print(f"\n💡 NEXT STEPS:")
    print("-" * 15)
    print("1. Test predictions:")
    print("   python3 predict_multi_crypto.py --asset BNB --periods 10")
    print()
    print("2. Integrate with main system:")
    print("   python3 main.py  # Select Multi-Crypto ML option")
    print()
    print("3. Compare with single-asset ML:")
    print("   python3 predict_reversals.py --periods 10")
    
    print(f"\n🚀 Multi-crypto ML training completed!")
    
    return successful_models > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
