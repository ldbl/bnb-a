#!/usr/bin/env python3
"""
ML Model Training Script
Easy-to-use script for training trend reversal detection models
"""

import argparse
from datetime import datetime
from ml_enhanced import TrendReversalML
from ml_predictor import MLPredictor


def train_models(data_limit: int = 2000, periods: list = None, save_models: bool = True):
    """Train trend reversal models with specified parameters"""
    
    if periods is None:
        periods = [5, 10, 20]  # Default prediction horizons
    
    print("🤖 BNB TREND REVERSAL ML TRAINING")
    print("=" * 50)
    print(f"📊 Data limit: {data_limit} samples")
    print(f"🎯 Prediction periods: {periods}")
    print(f"💾 Save models: {save_models}")
    print()
    
    # Initialize systems
    ml_system = TrendReversalML()
    base_predictor = MLPredictor()
    
    # Fetch training data
    print("📥 Fetching training data from Binance...")
    training_data = base_predictor.fetch_training_data("1h", data_limit)
    
    if training_data is None:
        print("❌ Failed to fetch training data")
        return False
    
    print(f"✅ Fetched {len(training_data)} training samples")
    print(f"📅 Date range: {training_data.index[0]} to {training_data.index[-1]}")
    
    # Train models for each period
    results = {}
    total_models = 0
    
    for period in periods:
        print(f"\n🧠 Training models for {period} periods ahead...")
        print("-" * 30)
        
        result = ml_system.train_reversal_models(training_data, period)
        results[period] = result
        
        if "success" in result:
            models_trained = result['models_trained']
            total_models += models_trained
            
            print(f"✅ Success: {models_trained} models trained")
            print(f"📊 Training samples: {result['training_samples']}")
            print(f"🎯 Features used: {result['feature_count']}")
            
            # Show individual model performance
            for model_name, model_result in result['results'].items():
                if 'test_accuracy' in model_result:
                    accuracy = model_result['test_accuracy']
                    print(f"   📈 {model_name}: {accuracy:.1%} accuracy")
                else:
                    print(f"   ❌ {model_name}: {model_result.get('error', 'Failed')}")
        else:
            print(f"❌ Training failed: {result.get('error', 'Unknown error')}")
    
    # Final summary
    print("\n" + "=" * 50)
    print("📋 TRAINING SUMMARY")
    print("=" * 50)
    
    status = ml_system.get_model_status()
    print(f"🎯 Total models trained: {total_models}")
    print(f"💾 Models saved to disk: {len(status['saved_models'])}")
    print(f"📁 Model directory: {status['model_directory']}")
    
    if status['saved_models']:
        print(f"\n💾 Saved models:")
        for model_name in status['saved_models']:
            print(f"   📦 {model_name}")
    
    # Test prediction (if models were trained)
    if total_models > 0:
        print(f"\n🧪 Testing prediction on latest data...")
        try:
            # Use last 100 samples for prediction test
            test_data = training_data.tail(100)
            prediction = ml_system.predict_reversal(test_data, periods[0])
            
            if "error" not in prediction:
                pred_label = prediction['prediction_label']
                confidence = prediction['confidence']
                print(f"✅ Prediction test successful!")
                print(f"   🎯 Prediction: {pred_label}")
                print(f"   🎲 Confidence: {confidence:.1%}")
            else:
                print(f"❌ Prediction test failed: {prediction['error']}")
                
        except Exception as e:
            print(f"❌ Prediction test error: {e}")
    
    print(f"\n🎉 Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return True


def main():
    """Main function with command line argument parsing"""
    
    parser = argparse.ArgumentParser(description='Train BNB Trend Reversal ML Models')
    parser.add_argument('--data-limit', type=int, default=2000, 
                        help='Number of data samples to fetch (default: 2000)')
    parser.add_argument('--periods', type=int, nargs='+', default=[5, 10, 20],
                        help='Prediction periods in hours (default: 5 10 20)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save models to disk')
    
    args = parser.parse_args()
    
    # Train models
    success = train_models(
        data_limit=args.data_limit,
        periods=args.periods,
        save_models=not args.no_save
    )
    
    if success:
        print("\n🚀 Training completed successfully!")
        print("💡 Models are ready for trend reversal prediction")
    else:
        print("\n❌ Training failed")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
