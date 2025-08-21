#!/usr/bin/env python3
"""
Trend Reversal Prediction Script
Use trained ML models to predict trend reversals
"""

import argparse
from datetime import datetime
from ml_enhanced import TrendReversalML
from ml_predictor import MLPredictor


def predict_reversals(periods: int = 10, show_details: bool = False):
    """Make trend reversal predictions using trained models"""
    
    print("🔮 BNB TREND REVERSAL PREDICTION")
    print("=" * 50)
    print(f"🎯 Prediction horizon: {periods} periods ahead")
    print(f"📊 Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize systems
    ml_system = TrendReversalML()
    base_predictor = MLPredictor()
    
    # Check if models exist
    status = ml_system.get_model_status()
    if not status['saved_models']:
        print("❌ No trained models found!")
        print("💡 Run: python3 train_ml_models.py")
        return False
    
    print(f"✅ Found {len(status['saved_models'])} saved models")
    
    # Load models for the specified period
    model_key = f"reversal_{periods}"
    available_models = [name for name in status['saved_models'] 
                       if f"reversal_{periods}d" in name]
    
    if not available_models:
        print(f"❌ No models trained for {periods} periods ahead")
        print(f"💡 Available periods: {[name.split('_')[2].replace('d', '') for name in status['saved_models']]}")
        return False
    
    print(f"📦 Loading models for {periods} periods ahead:")
    for model_name in available_models:
        print(f"   🤖 {model_name}")
    
    # Fetch recent data for prediction
    print(f"\n📥 Fetching recent market data...")
    recent_data = base_predictor.fetch_training_data("1h", 200)
    
    if recent_data is None:
        print("❌ Failed to fetch market data")
        return False
    
    current_price = recent_data['close'].iloc[-1]
    print(f"💰 Current BNB price: ${current_price:.2f}")
    
    # Load models and make prediction
    try:
        # Load all models for this period
        models_loaded = {}
        for model_file in available_models:
            model_name = model_file.replace(f"_reversal_{periods}d", "")
            loaded_model = ml_system.load_model(model_file)
            if loaded_model:
                models_loaded[model_name] = loaded_model
        
        if not models_loaded:
            print("❌ Failed to load any models")
            return False
        
        print(f"✅ Loaded {len(models_loaded)} models")
        
        # Update the ML system with loaded models
        ml_system.models[model_key] = {name: data['model'] for name, data in models_loaded.items()}
        
        # Update scalers (we need to rebuild this for the prediction to work)
        for model_name, model_data in models_loaded.items():
            if 'scaler' in model_data:
                scaler_key = f"{model_name}_{periods}"
                ml_system.scalers[scaler_key] = model_data['scaler']
        
        # Update feature columns
        if models_loaded:
            sample_metadata = list(models_loaded.values())[0]['metadata']
            ml_system.feature_columns = sample_metadata.get('feature_columns', [])
        
        # Make prediction
        print(f"\n🔮 Making trend reversal prediction...")
        prediction = ml_system.predict_reversal(recent_data, periods)
        
        if "error" in prediction:
            print(f"❌ Prediction failed: {prediction['error']}")
            return False
        
        # Display results
        print("\n" + "🎯 PREDICTION RESULTS")
        print("=" * 30)
        
        pred_label = prediction['prediction_label']
        confidence = prediction['confidence']
        ensemble_pred = prediction['ensemble_prediction']
        
        # Color coding
        if ensemble_pred == 1:  # Bullish reversal
            icon = "🟢📈"
            advice = "Consider LONG position"
        elif ensemble_pred == 2:  # Bearish reversal
            icon = "🔴📉"
            advice = "Consider SHORT position"
        else:  # No reversal
            icon = "🟡➡️"
            advice = "HOLD current position"
        
        print(f"{icon} Prediction: {pred_label}")
        print(f"🎲 Confidence: {confidence:.1%}")
        print(f"💡 Advice: {advice}")
        print(f"⏰ Time horizon: {periods} hours ahead")
        
        # Show individual model predictions if requested
        if show_details:
            print(f"\n📊 INDIVIDUAL MODEL PREDICTIONS:")
            print("-" * 40)
            
            for model_name, model_pred in prediction['individual_predictions'].items():
                pred_num = model_pred['prediction']
                pred_text = ["No Reversal", "Bullish Reversal", "Bearish Reversal"][pred_num]
                proba = model_pred.get('probability', [])
                
                if proba:
                    max_proba = max(proba)
                    print(f"🤖 {model_name}: {pred_text} ({max_proba:.1%})")
                else:
                    print(f"🤖 {model_name}: {pred_text}")
        
        # Risk assessment
        print(f"\n⚠️ RISK ASSESSMENT:")
        print("-" * 20)
        
        if confidence >= 0.8:
            risk_level = "🟢 LOW"
        elif confidence >= 0.6:
            risk_level = "🟡 MEDIUM"
        else:
            risk_level = "🔴 HIGH"
        
        print(f"Risk Level: {risk_level}")
        print(f"Model Agreement: {confidence:.1%}")
        
        if confidence < 0.6:
            print("💡 Low confidence - consider additional analysis")
        
        # Trading suggestion
        if ensemble_pred != 0:  # Some reversal predicted
            print(f"\n💼 TRADING SUGGESTION:")
            print("-" * 25)
            
            if ensemble_pred == 1:  # Bullish
                target_price = current_price * 1.05  # 5% target
                stop_loss = current_price * 0.98    # 2% stop loss
                print(f"🎯 Target: ${target_price:.2f} (+5%)")
                print(f"🛑 Stop Loss: ${stop_loss:.2f} (-2%)")
            else:  # Bearish
                target_price = current_price * 0.95  # 5% target down
                stop_loss = current_price * 1.02    # 2% stop loss up
                print(f"🎯 Target: ${target_price:.2f} (-5%)")
                print(f"🛑 Stop Loss: ${stop_loss:.2f} (+2%)")
            
            print(f"⚡ Entry: Around ${current_price:.2f}")
        
        print(f"\n🕐 Prediction valid until: {(datetime.now()).strftime('%Y-%m-%d %H:%M')} + {periods}h")
        print("⚠️ This is not financial advice - trade at your own risk!")
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False
    
    return True


def main():
    """Main function with command line arguments"""
    
    parser = argparse.ArgumentParser(description='Predict BNB Trend Reversals')
    parser.add_argument('--periods', type=int, default=10,
                        help='Prediction horizon in hours (default: 10)')
    parser.add_argument('--details', action='store_true',
                        help='Show detailed individual model predictions')
    
    args = parser.parse_args()
    
    success = predict_reversals(args.periods, args.details)
    
    if success:
        print("\n✅ Prediction completed successfully!")
    else:
        print("\n❌ Prediction failed")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
