#!/usr/bin/env python3
"""
Automatic ML Model Retraining Scheduler
Monitors model performance and retrains when needed
"""

import os
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

from bnb_enhanced_ml import BNBEnhancedML
from logger import get_logger

class MLRetrainScheduler:
    def __init__(self):
        self.logger = get_logger(__name__)
        self.bnb_ml = BNBEnhancedML()
        self.model_dir = Path("ml_models_bnb_enhanced")
        self.schedule_file = Path("ml_retrain_schedule.json")
        
        # Retraining configuration
        self.retrain_config = {
            "max_model_age_days": 30,  # Retrain if model is older than 30 days
            "min_accuracy_threshold": 0.85,  # Retrain if accuracy drops below 85%
            "check_interval_hours": 24,  # Check every 24 hours
            "prediction_periods": [7, 30, 90],  # Standard periods
            "auto_retrain": True,  # Enable automatic retraining
            "backup_old_models": True  # Keep backups
        }
        
        self.load_schedule()
    
    def load_schedule(self):
        """Load existing retrain schedule"""
        if self.schedule_file.exists():
            with open(self.schedule_file, 'r') as f:
                self.schedule = json.load(f)
        else:
            self.schedule = {
                "last_check": None,
                "last_retrain": None,
                "model_performance": {},
                "retrain_history": []
            }
    
    def save_schedule(self):
        """Save retrain schedule"""
        with open(self.schedule_file, 'w') as f:
            json.dump(self.schedule, f, indent=2, default=str)
    
    def check_retrain_needed(self):
        """Check if models need retraining"""
        
        print("🔍 CHECKING ML MODEL HEALTH")
        print("=" * 50)
        
        now = datetime.now()
        retrain_needed = {}
        
        for period in self.retrain_config["prediction_periods"]:
            print(f"\n📊 Checking {period}-day model...")
            
            model_path = self.model_dir / f"bnb_enhanced_{period}.pkl"
            metadata_path = self.model_dir / f"bnb_enhanced_{period}_metadata.json"
            
            needs_retrain = False
            reason = []
            
            # Check 1: Model exists
            if not model_path.exists():
                needs_retrain = True
                reason.append("Model file missing")
                print(f"   ❌ Model file not found")
            else:
                print(f"   ✅ Model file exists")
                
                # Check 2: Model age
                model_age = self._get_model_age(metadata_path)
                if model_age:
                    if model_age > self.retrain_config["max_model_age_days"]:
                        needs_retrain = True
                        reason.append(f"Model too old ({model_age} days)")
                        print(f"   ⏰ Model age: {model_age} days (❌ > {self.retrain_config['max_model_age_days']})")
                    else:
                        print(f"   ⏰ Model age: {model_age} days (✅)")
                
                # Check 3: Performance degradation
                performance_ok = self._check_model_performance(period)
                if not performance_ok:
                    needs_retrain = True
                    reason.append("Performance degraded")
                    print(f"   📉 Performance check: ❌ Below threshold")
                else:
                    print(f"   📈 Performance check: ✅")
            
            retrain_needed[period] = {
                "needed": needs_retrain,
                "reasons": reason
            }
        
        # Update schedule
        self.schedule["last_check"] = now.isoformat()
        self.save_schedule()
        
        return retrain_needed
    
    def _get_model_age(self, metadata_path):
        """Get model age in days"""
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            created_date = datetime.fromisoformat(metadata.get("created_at", ""))
            age = (datetime.now() - created_date).days
            return age
        except:
            return None
    
    def _check_model_performance(self, period):
        """Check if model performance is still acceptable"""
        
        # Simple performance check - in a real system you'd validate against recent data
        # For now, we'll check if we can make predictions without errors
        try:
            prediction = self.bnb_ml.predict_bnb_enhanced(period)
            if "error" in prediction:
                return False
            
            # Check confidence - if too low, might need retraining
            confidence = prediction.get("confidence", 0)
            if confidence < self.retrain_config["min_accuracy_threshold"]:
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Performance check failed for {period}d model: {e}")
            return False
    
    def auto_retrain_models(self, periods_to_retrain):
        """Automatically retrain models that need it"""
        
        if not self.retrain_config["auto_retrain"]:
            print("🔒 Auto-retrain disabled. Manual intervention required.")
            return
        
        successful_retrains = 0
        failed_retrains = 0
        
        for period, info in periods_to_retrain.items():
            if not info["needed"]:
                continue
            
            print(f"\n🔄 RETRAINING {period}-DAY MODEL")
            print("-" * 40)
            print(f"📝 Reasons: {', '.join(info['reasons'])}")
            
            try:
                # Backup old model
                if self.retrain_config["backup_old_models"]:
                    self._backup_model(period)
                
                # Retrain
                print(f"🎯 Starting retraining for {period} days...")
                result = self.bnb_ml.train_bnb_enhanced_model(period)
                
                if "success" in result:
                    successful_retrains += 1
                    print(f"✅ Successfully retrained {period}-day model")
                    
                    # Log retraining
                    self.schedule["retrain_history"].append({
                        "date": datetime.now().isoformat(),
                        "period": period,
                        "reasons": info["reasons"],
                        "success": True
                    })
                else:
                    failed_retrains += 1
                    error = result.get("error", "Unknown error")
                    print(f"❌ Failed to retrain {period}-day model: {error}")
                    
                    self.schedule["retrain_history"].append({
                        "date": datetime.now().isoformat(),
                        "period": period,
                        "reasons": info["reasons"],
                        "success": False,
                        "error": error
                    })
                
            except Exception as e:
                failed_retrains += 1
                print(f"❌ Exception during {period}-day retraining: {e}")
                self.logger.error(f"Retraining failed for {period}d: {e}")
        
        # Update schedule
        if successful_retrains > 0:
            self.schedule["last_retrain"] = datetime.now().isoformat()
        
        self.save_schedule()
        
        print(f"\n📊 RETRAINING SUMMARY:")
        print(f"✅ Successful: {successful_retrains}")
        print(f"❌ Failed: {failed_retrains}")
        
        return successful_retrains, failed_retrains
    
    def _backup_model(self, period):
        """Backup existing model before retraining"""
        
        model_path = self.model_dir / f"bnb_enhanced_{period}.pkl"
        metadata_path = self.model_dir / f"bnb_enhanced_{period}_metadata.json"
        
        if model_path.exists():
            backup_dir = self.model_dir / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_model = backup_dir / f"bnb_enhanced_{period}_{timestamp}.pkl"
            backup_metadata = backup_dir / f"bnb_enhanced_{period}_metadata_{timestamp}.json"
            
            # Copy files
            import shutil
            shutil.copy2(model_path, backup_model)
            if metadata_path.exists():
                shutil.copy2(metadata_path, backup_metadata)
            
            print(f"💾 Backed up old model to: {backup_model}")
    
    def get_retrain_recommendations(self):
        """Get recommendations for retraining schedule"""
        
        print("💡 ML RETRAINING RECOMMENDATIONS")
        print("=" * 50)
        
        print("📅 RECOMMENDED SCHEDULE:")
        print("   • Daily health checks (automated)")
        print("   • Weekly performance validation")
        print("   • Monthly full retraining (unless performance is good)")
        print("   • Immediate retrain after major market events")
        print()
        
        print("🎯 PERFORMANCE THRESHOLDS:")
        print(f"   • Model age: max {self.retrain_config['max_model_age_days']} days")
        print(f"   • Accuracy threshold: {self.retrain_config['min_accuracy_threshold']*100:.1f}%")
        print("   • Prediction errors: immediate retrain")
        print()
        
        print("⚡ TRIGGER CONDITIONS:")
        print("   • Model file missing or corrupted")
        print("   • Accuracy drops below threshold")
        print("   • Major market regime change detected")
        print("   • New cryptocurrency enters top 10")
        print()
        
        print("🔧 SETUP AUTOMATION:")
        print("   1. Add to crontab for daily checks:")
        print("      0 2 * * * cd /path/to/bnb-a && python3 auto_retrain_scheduler.py")
        print("   2. Monitor logs for retraining events")
        print("   3. Set up alerts for failed retrains")
    
    def manual_retrain_all(self):
        """Manually retrain all models"""
        
        print("🔄 MANUAL FULL RETRAINING")
        print("=" * 40)
        
        periods = self.retrain_config["prediction_periods"]
        successful = 0
        
        for period in periods:
            try:
                print(f"\n🎯 Retraining {period}-day model...")
                
                # Backup first
                if self.retrain_config["backup_old_models"]:
                    self._backup_model(period)
                
                result = self.bnb_ml.train_bnb_enhanced_model(period)
                
                if "success" in result:
                    successful += 1
                    print(f"✅ {period}-day model: SUCCESS")
                else:
                    print(f"❌ {period}-day model: FAILED - {result.get('error')}")
                    
            except Exception as e:
                print(f"❌ {period}-day model: EXCEPTION - {e}")
        
        print(f"\n📊 Manual retraining completed: {successful}/{len(periods)} successful")
        
        # Update schedule
        self.schedule["last_retrain"] = datetime.now().isoformat()
        self.schedule["retrain_history"].append({
            "date": datetime.now().isoformat(),
            "type": "manual_full",
            "successful_models": successful,
            "total_models": len(periods)
        })
        self.save_schedule()

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Model Retraining Scheduler")
    parser.add_argument("--check", action="store_true", help="Check if retraining is needed")
    parser.add_argument("--auto-retrain", action="store_true", help="Auto retrain models that need it")
    parser.add_argument("--manual-retrain", action="store_true", help="Manually retrain all models")
    parser.add_argument("--recommendations", action="store_true", help="Show retraining recommendations")
    parser.add_argument("--status", action="store_true", help="Show current model status")
    
    args = parser.parse_args()
    
    scheduler = MLRetrainScheduler()
    
    if args.recommendations:
        scheduler.get_retrain_recommendations()
    elif args.status:
        retrain_needed = scheduler.check_retrain_needed()
        print(f"\n📋 RETRAIN STATUS:")
        for period, info in retrain_needed.items():
            status = "❌ NEEDED" if info["needed"] else "✅ OK"
            print(f"   {period}-day model: {status}")
            if info["reasons"]:
                print(f"      Reasons: {', '.join(info['reasons'])}")
    elif args.manual_retrain:
        scheduler.manual_retrain_all()
    elif args.check and args.auto_retrain:
        retrain_needed = scheduler.check_retrain_needed()
        if any(info["needed"] for info in retrain_needed.values()):
            print("\n🔄 Auto-retraining needed models...")
            scheduler.auto_retrain_models(retrain_needed)
        else:
            print("\n✅ All models are healthy, no retraining needed")
    elif args.check:
        retrain_needed = scheduler.check_retrain_needed()
        needs_retrain = [period for period, info in retrain_needed.items() if info["needed"]]
        if needs_retrain:
            print(f"\n⚠️ Models need retraining: {needs_retrain}")
            print("💡 Run with --auto-retrain to automatically retrain them")
        else:
            print("\n✅ All models are healthy")
    else:
        print("🤖 ML Model Health Checker")
        print("Usage: python3 auto_retrain_scheduler.py [--check] [--auto-retrain] [--manual-retrain] [--recommendations] [--status]")

if __name__ == "__main__":
    main()
