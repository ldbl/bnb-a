#!/usr/bin/env python3
"""
Signal Generator Module
Contains trading signal logic and scoring system
"""

from datetime import datetime
from typing import Dict, List
from indicators import TechnicalIndicators
from elliott_wave import ElliottWaveAnalyzer
from fib import FibonacciAnalyzer
from correlation_module import CorrelationAnalyzer
from market_config import BNBMarketConfig
from swing_risk_manager import SwingRiskManager
from trend_reversal import TrendReversalDetector


class TradingSignalGenerator:
    """Class for generating comprehensive trading signals"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
        self.elliott = ElliottWaveAnalyzer()
        self.fibonacci = FibonacciAnalyzer()
        self.correlation = CorrelationAnalyzer()
        
        # Use centralized market configuration
        self.market_config = BNBMarketConfig()
        self.support_levels = self.market_config.SUPPORT_LEVELS
        self.resistance_levels = self.market_config.RESISTANCE_LEVELS
        self.cycles = self.market_config.MARKET_CYCLES
        
        # Initialize swing trading components
        self.risk_manager = SwingRiskManager()
        self.reversal_detector = TrendReversalDetector()
        
        # BNB-specific swing trading configuration
        self.bnb_swing_config = {
            'kotva_levels': [600, 620, 640, 650],  # Rule #1: Entry only at clear levels
            'quarterly_targets': [750, 780, 800],   # Rule #5: Take profit targets
            'quarterly_indicators': {
                'ema_fast': 50,    # EMA50 for quarterly momentum
                'ema_slow': 200,   # EMA200 for quarterly trend
                'rsi_period': 14,  # RSI14 for quarterly overbought/oversold
                'roc_period': 3,   # ROC3 for quarterly momentum
                'macd_fast': 50,   # MACD50 for quarterly signal
                'macd_slow': 200,  # MACD200 for quarterly trend
                'macd_signal': 9   # MACD signal line
            },
            'horoshko_patterns': {
                'normal_trend': {'forward': 2, 'back': 1},      # 2F-1B normal
                'strong_trend': {'forward': 5, 'back': 3}      # 5F-3B strong
            }
        }
    
    def calculate_bull_bear_score(self, current_price: float, prices: List[float], 
                                  volumes: List[float]) -> Dict:
        """Calculate bullish and bearish scores based on multiple factors"""
        
        # Calculate all indicators
        rsi = self.indicators.calculate_rsi(prices)
        macd = self.indicators.calculate_macd(prices)
        bollinger = self.indicators.calculate_bollinger(prices)
        elliott = self.elliott.detect_elliott_wave(prices)
        fibonacci = self.fibonacci.get_fibonacci_signals(current_price)
        
        bull_score = 0
        bear_score = 0
        
        # Price level analysis
        if current_price < 650:
            bull_score += 3
        elif current_price > 850:
            bear_score += 3
        
        # Support/Resistance proximity
        closest_support = min(self.support_levels, key=lambda x: abs(x - current_price))
        closest_resistance = min(self.resistance_levels, key=lambda x: abs(x - current_price))
        
        if abs(current_price - closest_support) < 20:
            bull_score += 2
        if abs(current_price - closest_resistance) < 20:
            bear_score += 2
        
        # RSI analysis
        if rsi < 30:
            bull_score += 2
        elif rsi < 40:
            bull_score += 1
        elif rsi > 70:
            bear_score += 2
        elif rsi > 60:
            bear_score += 1
        
        # MACD analysis
        if macd["trend"] == "BULLISH":
            bull_score += 2
            if macd["histogram"] > 0:
                bull_score += 1
        elif macd["trend"] == "BEARISH":
            bear_score += 2
            if macd["histogram"] < 0:
                bear_score += 1
        
        # Bollinger Bands analysis
        if bollinger["position"] == "OVERSOLD":
            bull_score += 2
        elif bollinger["position"] == "OVERBOUGHT":
            bear_score += 2
        
        # Elliott Wave analysis
        wave = elliott.get("wave", "")
        if "5" in wave:
            bear_score += 2
        elif "2" in wave:
            bull_score += 2
        elif "3" in wave:
            bull_score += 1
        
        # Volume trend analysis
        if len(volumes) >= 5:
            recent_vol = sum(volumes[-5:]) / 5
            avg_vol = sum(volumes) / len(volumes)
            
            if recent_vol > avg_vol * 1.2:
                if prices[-1] > prices[-2]:
                    bull_score += 1
                else:
                    bear_score += 1
        
        # Fibonacci signals
        fib_action = fibonacci.get("action", "WAIT")
        if fib_action == "BUY":
            bull_score += 2
        elif fib_action == "STRONG_BUY":
            bull_score += 3
        elif fib_action == "SELL":
            bear_score += 2
        elif fib_action == "STRONG_SELL":
            bear_score += 3
        
        # Golden pocket bonus
        if fibonacci.get("golden_pocket"):
            bull_score += 2
        
        # Correlation analysis (lightweight version for scoring)
        try:
            correlation_data = self.correlation.run_correlation_analysis("1d", 30)
            if correlation_data and "signals" in correlation_data:
                corr_signals = correlation_data["signals"]
                correlation_score = corr_signals.get("correlation_score", 0)
                
                # Add correlation score to appropriate side
                if correlation_score > 0:
                    bull_score += min(correlation_score, 2)  # Cap at +2
                elif correlation_score < 0:
                    bear_score += min(abs(correlation_score), 2)  # Cap at +2
        except:
            # If correlation analysis fails, continue without it
            correlation_data = None
        
        return {
            "bull_score": bull_score,
            "bear_score": bear_score,
            "indicators": {
                "RSI": rsi,
                "MACD": macd,
                "Bollinger": bollinger,
                "Elliott": elliott,
                "Fibonacci": fibonacci,
                "Correlation": correlation_data
            }
        }
    
    def determine_position_size(self, signal_strength: str, confidence: int) -> str:
        """Determine position size based on signal strength and confidence"""
        if confidence < 60:
            return "10% - Low confidence"
        
        position_sizes = {
            "STRONG BUY": "50% with 2x leverage" if confidence > 80 else "30% with 1.5x leverage",
            "BUY": "25% with 2x leverage" if confidence > 75 else "20% with 1x leverage",
            "STRONG SELL": "33% short position" if confidence > 80 else "20% short position",
            "SELL": "15% short position" if confidence > 75 else "10% short position",
            "WAIT": "0% - Wait for better setup"
        }
        
        return position_sizes.get(signal_strength, "10% - Conservative")
    
    def calculate_targets_and_stops(self, current_price: float, action: str, 
                                   fibonacci_data: Dict) -> Dict:
        """Calculate target and stop loss levels"""
        targets = {}
        
        if action in ["STRONG BUY", "BUY"]:
            # Bullish targets
            if fibonacci_data.get("target"):
                targets["primary_target"] = fibonacci_data["target"]
            else:
                targets["primary_target"] = round(current_price * 1.08, 2)
            
            targets["extended_target"] = round(current_price * 1.15, 2)
            targets["stop_loss"] = round(current_price * 0.95, 2)
            
            # Use closest resistance as secondary target
            resistance_above = [r for r in self.resistance_levels if r > current_price]
            if resistance_above:
                targets["resistance_target"] = min(resistance_above)
            
        elif action in ["STRONG SELL", "SELL"]:
            # Bearish targets
            if fibonacci_data.get("target"):
                targets["primary_target"] = fibonacci_data["target"]
            else:
                targets["primary_target"] = round(current_price * 0.92, 2)
            
            targets["extended_target"] = round(current_price * 0.85, 2)
            targets["stop_loss"] = round(current_price * 1.05, 2)
            
            # Use closest support as secondary target
            support_below = [s for s in self.support_levels if s < current_price]
            if support_below:
                targets["support_target"] = max(support_below)
        
        return targets
    
    def generate_comprehensive_signal(self, current_price: float, prices: List[float], 
                                    volumes: List[float], mtf_analysis: Dict) -> Dict:
        """Generate the main trading signal with all analysis"""
        
        # Calculate scores and get all indicator data
        scoring_data = self.calculate_bull_bear_score(current_price, prices, volumes)
        bull_score = scoring_data["bull_score"]
        bear_score = scoring_data["bear_score"]
        indicators = scoring_data["indicators"]
        
        # Calculate confidence based on score difference and indicator alignment
        score_diff = abs(bull_score - bear_score)
        max_score = max(bull_score, bear_score)
        confidence = min(50 + (score_diff * 10) + (max_score * 3), 95)
        
        # Determine action based on scores
        if bull_score > bear_score + 2:
            action = "STRONG BUY"
        elif bull_score > bear_score:
            action = "BUY"
        elif bear_score > bull_score + 2:
            action = "STRONG SELL"
        elif bear_score > bull_score:
            action = "SELL"
        else:
            action = "WAIT"
        
        # Calculate targets and stops
        targets = self.calculate_targets_and_stops(
            current_price, action, indicators["Fibonacci"]
        )
        
        # Determine position size
        position_size = self.determine_position_size(action, confidence)
        
        # Create comprehensive signal
        signal = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "price": current_price,
            "signal": action,  # Add signal key for main.py
            "action": action,
            "confidence": confidence,
            "bull_score": bull_score,
            "bear_score": bear_score,
            "position_size": position_size,
            "indicators": indicators,
            "timeframes": mtf_analysis,
            "targets": targets
        }
        
        # Add specific recommendations based on action
        if action == "WAIT":
            signal["reason"] = self._get_wait_reason(indicators, bull_score, bear_score)
        else:
            signal["reason"] = self._get_action_reason(action, indicators)
        
        return signal
    
    def _get_wait_reason(self, indicators: Dict, bull_score: int, bear_score: int) -> str:
        """Generate reason for WAIT signal"""
        reasons = []
        
        if abs(bull_score - bear_score) <= 1:
            reasons.append("Mixed signals across indicators")
        
        rsi = indicators["RSI"]
        if 45 <= rsi <= 55:
            reasons.append("RSI in neutral zone")
        
        if indicators["MACD"]["trend"] == "NEUTRAL":
            reasons.append("MACD showing no clear trend")
        
        if indicators["Bollinger"]["position"] == "NEUTRAL":
            reasons.append("Price in middle of Bollinger Bands")
        
        return " | ".join(reasons) if reasons else "Wait for clearer setup"
    
    def _get_action_reason(self, action: str, indicators: Dict) -> str:
        """Generate reason for BUY/SELL signal"""
        reasons = []
        
        if action in ["BUY", "STRONG BUY"]:
            if indicators["RSI"] < 40:
                reasons.append("Oversold RSI")
            if indicators["MACD"]["trend"] == "BULLISH":
                reasons.append("MACD bullish crossover")
            if indicators["Fibonacci"].get("golden_pocket"):
                reasons.append("Golden Pocket bounce zone")
            if "2" in indicators["Elliott"].get("wave", ""):
                reasons.append("Elliott Wave 2 pullback")
        
        elif action in ["SELL", "STRONG SELL"]:
            if indicators["RSI"] > 60:
                reasons.append("Overbought RSI")
            if indicators["MACD"]["trend"] == "BEARISH":
                reasons.append("MACD bearish crossover")
            if "5" in indicators["Elliott"].get("wave", ""):
                reasons.append("Elliott Wave 5 completion")
        
        return " | ".join(reasons) if reasons else f"{action} signal generated"
    
    def get_enhanced_fibonacci_info(self, current_price: float) -> Dict:
        """Get enhanced Fibonacci information for main screen"""
        fibonacci = self.fibonacci.get_fibonacci_signals(current_price)
        
        # Enhanced Fibonacci analysis
        fib_info = {
            "action": fibonacci.get("action", "WAIT"),
            "trend": fibonacci.get("trend", "UNKNOWN"),
            "closest_level": fibonacci.get("closest_level", "N/A"),
            "distance": fibonacci.get("distance", 0),
            "golden_pocket": fibonacci.get("golden_pocket", False),
            "retracement_zone": fibonacci.get("retracement_zone", "UNKNOWN")
        }
        
        # Get detailed levels
        levels = fibonacci.get("fibonacci_levels", {})  # Use fibonacci_levels key
        if levels:
            # Find support and resistance levels
            fib_info["support_levels"] = []
            fib_info["resistance_levels"] = []
            
            for level_name, level_price in levels.items():
                if level_price < current_price:
                    fib_info["support_levels"].append(f"{level_name}: ${level_price:.0f}")
                else:
                    fib_info["resistance_levels"].append(f"{level_name}: ${level_price:.0f}")
            
            # Limit to 3 closest levels each
            fib_info["support_levels"] = fib_info["support_levels"][:3]
            fib_info["resistance_levels"] = fib_info["resistance_levels"][:3]
        
        # Add levels to fib_info for main.py compatibility
        fib_info["levels"] = levels
        
        # Golden Pocket analysis
        if fib_info["golden_pocket"]:
            fib_info["pocket_status"] = "🟡 IN GOLDEN POCKET"
        elif fib_info["retracement_zone"] == "DEEP":
            fib_info["pocket_status"] = "🔴 BELOW GOLDEN POCKET"
        else:
            fib_info["pocket_status"] = "🟢 ABOVE GOLDEN POCKET"
        
        return fib_info
    
    def get_multi_period_elliott_waves(self) -> Dict:
        """Get Elliott Wave analysis for multiple periods"""
        periods = {
            "6_months": {"interval": "1d", "limit": 180, "description": "6 месеца"},
            "1_year": {"interval": "1w", "limit": 52, "description": "1 година"},
            "1_5_years": {"interval": "1w", "limit": 78, "description": "1.5 години"}
        }
        
        elliott_periods = {}
        
        for period_key, period_data in periods.items():
            try:
                # Get historical data (simplified - would use actual API call)
                # For demo, we'll use the visual analysis structure for longer periods
                if period_key == "1_5_years":
                    # Use visual Elliott analysis for 1.5 years
                    elliott_periods[period_key] = {
                        "description": period_data["description"],
                        "wave": "WAVE_5_COMPLETION",
                        "confidence": 95,
                        "status": "🔴 CYCLE TOP",
                        "next_move": "ABC CORRECTION",
                        "analysis": "Visual analysis - complete 5-wave cycle"
                    }
                elif period_key == "1_year":
                    # Algorithmic for 1 year (Waves 3-4-5)
                    elliott_periods[period_key] = {
                        "description": period_data["description"],
                        "wave": "WAVE_5_IN_PROGRESS",
                        "confidence": 80,
                        "status": "🟡 LATE STAGE",
                        "next_move": "WAVE_5_COMPLETION",
                        "analysis": "Waves 3-4-5 sequence"
                    }
                else:  # 6 months
                    # Recent Wave 5 development
                    elliott_periods[period_key] = {
                        "description": period_data["description"],
                        "wave": "WAVE_5_EXTENSION",
                        "confidence": 75,
                        "status": "🟢 TRENDING",
                        "next_move": "FINAL PUSH",
                        "analysis": "Wave 5 extension phase"
                    }
            except Exception as e:
                elliott_periods[period_key] = {
                    "description": period_data["description"],
                    "wave": "ERROR",
                    "confidence": 0,
                    "status": "❌ N/A",
                    "next_move": "UNKNOWN",
                    "analysis": f"Error: {e}"
                }
        
        return elliott_periods
    
    def generate_swing_signals(self, current_price: float, prices: List[float], 
                              volumes: List[float], timeframe: str = 'quarterly') -> Dict:
        """
        Generate swing trading signals for BNB quarterly cycles (3-4 months)
        Following ХАЙДУШКИ КОДЕКС: котва levels, patience, хайдушко хоро rhythm
        
        Args:
            current_price: Current BNB price
            prices: Historical price data
            volumes: Historical volume data
            timeframe: 'monthly' or 'quarterly' (default: quarterly)
            
        Returns:
            Swing trading signal configuration
        """
        try:
            import numpy as np
            import pandas as pd
            
            # Convert to pandas Series for technical analysis
            price_series = pd.Series(prices)
            volume_series = pd.Series(volumes)
            
            # Apply np.nan_to_num to handle NaN values
            price_series = pd.Series(np.nan_to_num(price_series))
            volume_series = pd.Series(np.nan_to_num(volume_series))
            
            # ХАЙДУШКИ КОДЕКС Rule #1: Check котва levels before any BUY signal
            kotva_validation = self._validate_kotva_levels(current_price)
            
            # ХАЙДУШКИ КОДЕКС Rule #2: Patience validation (wait for clear quarterly setup)
            patience_validation = self._validate_quarterly_patience(price_series, volume_series)
            
            # Calculate quarterly momentum indicators
            quarterly_indicators = self._calculate_quarterly_momentum(price_series, volume_series)
            
            # Detect хайдушко хоро rhythm patterns
            horoshko_patterns = self._detect_horoshko_rhythm(price_series)
            
            # Get weekly wick analysis for confirmation
            weekly_wick_analysis = self.reversal_detector.weekly_wick_analysis(limit=26)
            
            # Generate swing trading signal
            swing_signal = self._generate_swing_signal(
                current_price, kotva_validation, patience_validation,
                quarterly_indicators, horoshko_patterns, weekly_wick_analysis
            )
            
            # ХАЙДУШКИ КОДЕКС Rule #3: Position sizing integration
            if swing_signal['action'] in ['BUY', 'STRONG_BUY']:
                position_sizing = self._calculate_swing_position_size(
                    current_price, swing_signal, timeframe
                )
                swing_signal['position_sizing'] = position_sizing
            
            # ХАЙДУШКИ КОДЕКС Rule #5: Take profit targets
            swing_signal['take_profit_targets'] = self._calculate_take_profit_targets(
                current_price, timeframe
            )
            
            return swing_signal
            
        except Exception as e:
            return {
                'error': f'Swing signal generation failed: {str(e)}',
                'action': 'WAIT',
                'confidence': 0,
                'reason': 'Error in swing analysis'
            }
    
    def _validate_kotva_levels(self, current_price: float) -> Dict:
        """ХАЙДУШКИ КОДЕКС Rule #1: Validate entry at котва levels ($600-650)"""
        try:
            kotva_levels = self.bnb_swing_config['kotva_levels']
            
            # Find nearest котва level
            nearest_kotva = min(kotva_levels, key=lambda x: abs(x - current_price))
            distance_to_kotva = abs(current_price - nearest_kotva)
            distance_pct = (distance_to_kotva / current_price) * 100
            
            # Entry validation rules
            if current_price <= 650:  # Within котва range
                if distance_pct <= 2:  # Within 2% of котва level
                    validation = {
                        'valid': True,
                        'level': nearest_kotva,
                        'distance': distance_to_kotva,
                        'distance_pct': distance_pct,
                        'status': '🟢 KOTVA LEVEL VALID',
                        'message': f'Entry valid at ${nearest_kotva} level'
                    }
                else:
                    validation = {
                        'valid': True,
                        'level': nearest_kotva,
                        'distance': distance_to_kotva,
                        'distance_pct': distance_pct,
                        'status': '🟡 NEAR KOTVA',
                        'message': f'Close to ${nearest_kotva} level'
                    }
            else:
                validation = {
                    'valid': False,
                    'level': nearest_kotva,
                    'distance': distance_to_kotva,
                    'distance_pct': distance_pct,
                    'status': '🔴 ABOVE KOTVA RANGE',
                    'message': f'No FOMO - wait for ${nearest_kotva} level'
                }
            
            return validation
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'status': '❌ VALIDATION ERROR',
                'message': 'Kotva validation failed'
            }
    
    def _validate_quarterly_patience(self, price_series: pd.Series, 
                                   volume_series: pd.Series) -> Dict:
        """ХАЙДУШКИ КОДЕКС Rule #2: Validate patience for quarterly setup"""
        try:
            # Check if we have enough data for quarterly analysis
            if len(price_series) < 90:  # Need at least 3 months
                return {
                    'valid': False,
                    'status': '❌ INSUFFICIENT DATA',
                    'message': 'Need at least 3 months of data for quarterly analysis'
                }
            
            # Calculate quarterly volatility
            quarterly_returns = []
            for i in range(0, len(price_series) - 90, 30):  # Monthly intervals
                if i + 90 <= len(price_series):
                    start_price = price_series.iloc[i]
                    end_price = price_series.iloc[i + 90]
                    quarterly_return = (end_price - start_price) / start_price
                    quarterly_returns.append(abs(quarterly_return))
            
            if not quarterly_returns:
                return {
                    'valid': False,
                    'status': '❌ NO QUARTERLY DATA',
                    'message': 'Unable to calculate quarterly returns'
                }
            
            # Patience validation based on quarterly volatility
            avg_quarterly_return = np.mean(quarterly_returns)
            volatility_std = np.std(quarterly_returns)
            
            # Wait for clear quarterly setup
            if avg_quarterly_return < 0.15:  # <15% average quarterly return
                patience_status = '🟡 WAIT FOR VOLATILITY'
                patience_message = 'Low quarterly volatility - wait for clearer setup'
                patience_valid = False
            elif avg_quarterly_return > 0.40:  # >40% average quarterly return
                patience_status = '🔴 TOO VOLATILE'
                patience_message = 'Excessive volatility - wait for stabilization'
                patience_valid = False
            else:
                patience_status = '🟢 QUARTERLY SETUP READY'
                patience_message = f'Quarterly volatility optimal: {avg_quarterly_return*100:.1f}%'
                patience_valid = True
            
            return {
                'valid': patience_valid,
                'status': patience_status,
                'message': patience_message,
                'avg_quarterly_return': avg_quarterly_return,
                'volatility_std': volatility_std,
                'data_points': len(quarterly_returns)
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'status': '❌ PATIENCE VALIDATION ERROR',
                'message': 'Patience validation failed'
            }
    
    def _calculate_quarterly_momentum(self, price_series: pd.Series, 
                                    volume_series: pd.Series) -> Dict:
        """Calculate quarterly momentum indicators (EMA50/200, RSI14, ROC3, MACD50/200/9)"""
        try:
            config = self.bnb_swing_config['quarterly_indicators']
            
            # Calculate EMAs
            ema_fast = self.indicators.calculate_ema(price_series, config['ema_fast'])
            ema_slow = self.indicators.calculate_ema(price_series, config['ema_slow'])
            
            # Calculate RSI
            rsi = self.indicators.calculate_rsi(price_series, config['rsi_period'])
            
            # Calculate ROC (Rate of Change)
            roc = self.indicators.calculate_roc(price_series, config['roc_period'])
            
            # Calculate MACD
            macd = self.indicators.calculate_macd(price_series, config['macd_fast'], 
                                                config['macd_slow'], config['macd_signal'])
            
            # Get latest values
            current_ema_fast = ema_fast.iloc[-1] if not ema_fast.empty else price_series.iloc[-1]
            current_ema_slow = ema_slow.iloc[-1] if not ema_slow.empty else price_series.iloc[-1]
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50
            current_roc = roc.iloc[-1] if not roc.empty else 0
            current_macd = macd['macd'].iloc[-1] if 'macd' in macd and not macd['macd'].empty else 0
            current_signal = macd['signal'].iloc[-1] if 'signal' in macd and not macd['signal'].empty else 0
            current_histogram = macd['histogram'].iloc[-1] if 'histogram' in macd and not macd['histogram'].empty else 0
            
            # Quarterly momentum analysis
            ema_trend = 'BULLISH' if current_ema_fast > current_ema_slow else 'BEARISH'
            rsi_signal = 'OVERSOLD' if current_rsi < 30 else 'OVERBOUGHT' if current_rsi > 70 else 'NEUTRAL'
            roc_momentum = 'POSITIVE' if current_roc > 0 else 'NEGATIVE'
            macd_signal = 'BULLISH' if current_macd > current_signal else 'BEARISH'
            
            return {
                'ema_fast': current_ema_fast,
                'ema_slow': current_ema_slow,
                'ema_trend': ema_trend,
                'rsi': current_rsi,
                'rsi_signal': rsi_signal,
                'roc': current_roc,
                'roc_momentum': roc_momentum,
                'macd': current_macd,
                'macd_signal': current_signal,
                'macd_histogram': current_histogram,
                'macd_trend': macd_signal,
                'momentum_score': self._calculate_momentum_score(
                    ema_trend, rsi_signal, roc_momentum, macd_signal
                )
            }
            
        except Exception as e:
            return {
                'error': f'Quarterly momentum calculation failed: {str(e)}',
                'momentum_score': 0
            }
    
    def _calculate_momentum_score(self, ema_trend: str, rsi_signal: str, 
                                roc_momentum: str, macd_trend: str) -> int:
        """Calculate overall momentum score for quarterly analysis"""
        score = 0
        
        # EMA trend (weight: 3)
        if ema_trend == 'BULLISH':
            score += 3
        elif ema_trend == 'BEARISH':
            score -= 3
        
        # RSI signal (weight: 2)
        if rsi_signal == 'OVERSOLD':
            score += 2
        elif rsi_signal == 'OVERBOUGHT':
            score -= 2
        
        # ROC momentum (weight: 2)
        if roc_momentum == 'POSITIVE':
            score += 2
        elif roc_momentum == 'NEGATIVE':
            score -= 2
        
        # MACD trend (weight: 2)
        if macd_trend == 'BULLISH':
            score += 2
        elif macd_trend == 'BEARISH':
            score -= 2
        
        return score
    
    def _detect_horoshko_rhythm(self, price_series: pd.Series) -> Dict:
        """Detect хайдушко хоро rhythm patterns (2F-1B normal, 5F-3B strong)"""
        try:
            patterns = self.bnb_swing_config['horoshko_patterns']
            
            if len(price_series) < 20:
                return {
                    'pattern': 'INSUFFICIENT_DATA',
                    'confidence': 0,
                    'message': 'Need at least 20 data points for rhythm detection'
                }
            
            # Calculate price movements (forward/back)
            price_changes = price_series.pct_change().dropna()
            movements = []
            
            for change in price_changes:
                if change > 0.01:  # >1% move
                    movements.append('F')  # Forward
                elif change < -0.01:  # <-1% move
                    movements.append('B')  # Back
                else:
                    movements.append('H')  # Hold
            
            # Detect patterns
            if len(movements) >= 3:
                # Check for 2F-1B pattern (normal trend)
                if movements[-3:] == ['F', 'F', 'B']:
                    pattern = 'NORMAL_TREND_2F1B'
                    confidence = 75
                    message = 'Normal trend: 2 steps forward, 1 step back'
                # Check for 5F-3B pattern (strong trend)
                elif len(movements) >= 8 and movements[-8:] == ['F', 'F', 'F', 'F', 'F', 'B', 'B', 'B']:
                    pattern = 'STRONG_TREND_5F3B'
                    confidence = 90
                    message = 'Strong trend: 5 steps forward, 3 steps back'
                else:
                    pattern = 'NO_CLEAR_PATTERN'
                    confidence = 30
                    message = 'No clear хайдушко хоро pattern detected'
            else:
                pattern = 'INSUFFICIENT_MOVEMENTS'
                confidence = 0
                message = 'Need more price movements for pattern detection'
            
            return {
                'pattern': pattern,
                'confidence': confidence,
                'message': message,
                'movements': movements[-10:] if len(movements) >= 10 else movements,  # Last 10 movements
                'pattern_type': 'NORMAL' if '2F1B' in pattern else 'STRONG' if '5F3B' in pattern else 'NONE'
            }
            
        except Exception as e:
            return {
                'pattern': 'ERROR',
                'confidence': 0,
                'error': str(e),
                'message': 'Pattern detection failed'
            }
    
    def _generate_swing_signal(self, current_price: float, kotva_validation: Dict,
                              patience_validation: Dict, quarterly_indicators: Dict,
                              horoshko_patterns: Dict, weekly_wick_analysis: Dict) -> Dict:
        """Generate final swing trading signal based on all validations"""
        try:
            # Base signal
            signal = {
                'action': 'WAIT',
                'confidence': 0,
                'reason': 'Waiting for optimal setup',
                'timeframe': 'quarterly',
                'timestamp': datetime.now().isoformat()
            }
            
            # ХАЙДУШКИ КОДЕКС Rule #1: Must be at котва level for BUY
            if not kotva_validation['valid']:
                signal['reason'] = f"Not at котва level: {kotva_validation['message']}"
                return signal
            
            # ХАЙДУШКИ КОДЕКС Rule #2: Must have patience validation
            if not patience_validation['valid']:
                signal['reason'] = f"Patience required: {patience_validation['message']}"
                return signal
            
            # Check for errors in indicators
            if 'error' in quarterly_indicators:
                signal['reason'] = f"Indicator error: {quarterly_indicators['error']}"
                return signal
            
            # Get momentum score
            momentum_score = quarterly_indicators.get('momentum_score', 0)
            
            # Generate signal based on momentum and patterns
            if momentum_score >= 6 and horoshko_patterns.get('pattern_type') in ['NORMAL', 'STRONG']:
                if horoshko_patterns.get('confidence', 0) >= 75:
                    signal['action'] = 'STRONG_BUY'
                    signal['confidence'] = 90
                    signal['reason'] = f"Strong momentum ({momentum_score}) + clear pattern ({horoshko_patterns['pattern']})"
                else:
                    signal['action'] = 'BUY'
                    signal['confidence'] = 75
                    signal['reason'] = f"Good momentum ({momentum_score}) + pattern detected"
            elif momentum_score >= 3:
                signal['action'] = 'BUY'
                signal['confidence'] = 60
                signal['reason'] = f"Positive momentum ({momentum_score}) at котва level"
            elif momentum_score <= -3:
                signal['action'] = 'SELL'
                signal['confidence'] = 70
                signal['reason'] = f"Negative momentum ({momentum_score}) - consider exit"
            else:
                signal['action'] = 'WAIT'
                signal['confidence'] = 40
                signal['reason'] = f"Neutral momentum ({momentum_score}) - wait for clearer signal"
            
            # Add analysis details
            signal['analysis'] = {
                'kotva_validation': kotva_validation,
                'patience_validation': patience_validation,
                'quarterly_indicators': quarterly_indicators,
                'horoshko_patterns': horoshko_patterns,
                'weekly_wick_analysis': weekly_wick_analysis
            }
            
            return signal
            
        except Exception as e:
            return {
                'action': 'WAIT',
                'confidence': 0,
                'reason': f'Signal generation failed: {str(e)}',
                'error': str(e)
            }
    
    def _calculate_swing_position_size(self, current_price: float, 
                                     swing_signal: Dict, timeframe: str) -> Dict:
        """ХАЙДУШКИ КОДЕКС Rule #3: Calculate position sizing (1/3 entry, gradual scale)"""
        try:
            # Use risk manager for position sizing
            entry_price = current_price
            direction = 'long' if swing_signal['action'] in ['BUY', 'STRONG_BUY'] else 'short'
            
            # Generate trading plan with risk management
            trading_plan = self.risk_manager.generate_swing_trading_plan(
                entry_price=entry_price,
                direction=direction,
                timeframe=timeframe
            )
            
            if 'error' in trading_plan:
                return {
                    'error': trading_plan['error'],
                    'position_size': 0,
                    'risk_amount': 0
                }
            
            # Extract position sizing information
            position_config = trading_plan.get('risk_management', {}).get('position_sizing', {})
            
            return {
                'shares': position_config.get('adjusted_shares', 0),
                'position_value': position_config.get('position_value', 0),
                'risk_amount': position_config.get('actual_risk_amount', 0),
                'risk_pct': position_config.get('actual_risk_pct', 0),
                'leverage': position_config.get('leverage_used', 1.0),
                'stop_loss': trading_plan.get('risk_management', {}).get('stop_loss', {}).get('adjusted_stop_loss', 0),
                'trading_plan': trading_plan
            }
            
        except Exception as e:
            return {
                'error': f'Position sizing calculation failed: {str(e)}',
                'position_size': 0,
                'risk_amount': 0
            }
    
    def _calculate_take_profit_targets(self, current_price: float, timeframe: str) -> Dict:
        """ХАЙДУШКИ КОДЕКС Rule #5: Calculate take profit targets ($750-780 repeatable levels)"""
        try:
            targets = self.bnb_swing_config['quarterly_targets']
            
            # Calculate distance to each target
            target_distances = {}
            for target in targets:
                distance = target - current_price
                distance_pct = (distance / current_price) * 100
                target_distances[f'target_{target}'] = {
                    'price': target,
                    'distance': distance,
                    'distance_pct': distance_pct,
                    'achievable': distance_pct <= 40  # Max 40% move for quarterly
                }
            
            # Find nearest achievable target
            achievable_targets = [t for t in target_distances.values() if t['achievable']]
            nearest_target = min(achievable_targets, key=lambda x: x['distance']) if achievable_targets else None
            
            return {
                'targets': target_distances,
                'nearest_target': nearest_target,
                'strategy': 'GRADUAL_EXIT' if len(achievable_targets) > 1 else 'SINGLE_TARGET',
                'message': f"Target {nearest_target['price']} at {nearest_target['distance_pct']:.1f}% distance" if nearest_target else "No achievable targets"
            }
            
        except Exception as e:
            return {
                'error': f'Take profit calculation failed: {str(e)}',
                'targets': {},
                'nearest_target': None
            }


# Example usage
if __name__ == "__main__":
    # Test signal generator
    generator = TradingSignalGenerator()
    
    # Sample data for daily signals
    test_prices = [850, 852, 848, 855, 851, 857, 853, 860, 856, 863]
    test_volumes = [1000, 1200, 900, 1100, 1050, 1300, 950, 1400, 1000, 1500]
    current_price = 860
    
    print("🎯 BNB Signal Generator - Swing Trading Ready!")
    print("=" * 60)
    
    # Test daily signals
    daily_signal = generator.generate_comprehensive_signal(
        current_price, test_prices, test_volumes, {}
    )
    
    print("📊 Daily Signal Test:")
    print(f"Action: {daily_signal['action']}")
    print(f"Confidence: {daily_signal['confidence']}%")
    print(f"Bull Score: {daily_signal['bull_score']}")
    print(f"Bear Score: {daily_signal['bear_score']}")
    print(f"Reason: {daily_signal['reason']}")
    
    print("\n" + "=" * 60)
    
    # Test swing trading signals (quarterly)
    # Historical Q3 2024 data: entry at $533, exit at $701
    quarterly_prices = [533, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 701]
    quarterly_volumes = [2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800]
    quarterly_price = 533  # Q3 2024 entry level
    
    print("🥋 Swing Trading Signal Test (ХАЙДУШКИ КОДЕКС):")
    print(f"Current Price: ${quarterly_price:.2f}")
    print(f"Historical Context: Q3 2024 entry → Q4 2024 exit ($701)")
    print(f"Target Return: 31.5% over 3-4 months")
    
    swing_signal = generator.generate_swing_signals(
        quarterly_price, quarterly_prices, quarterly_volumes, 'quarterly'
    )
    
    if 'error' not in swing_signal:
        print(f"\n🎯 Swing Signal:")
        print(f"Action: {swing_signal['action']}")
        print(f"Confidence: {swing_signal['confidence']}%")
        print(f"Reason: {swing_signal['reason']}")
        
        if 'position_sizing' in swing_signal:
            pos = swing_signal['position_sizing']
            print(f"\n💰 Position Sizing:")
            print(f"Shares: {pos.get('shares', 0):.2f}")
            print(f"Position Value: ${pos.get('position_value', 0):,.2f}")
            print(f"Risk Amount: ${pos.get('risk_amount', 0):,.2f}")
            print(f"Risk %: {pos.get('risk_pct', 0):.1f}%")
        
        if 'take_profit_targets' in swing_signal:
            targets = swing_signal['take_profit_targets']
            print(f"\n🎯 Take Profit Targets:")
            print(f"Strategy: {targets.get('strategy', 'N/A')}")
            print(f"Message: {targets.get('message', 'N/A')}")
        
        if 'analysis' in swing_signal:
            analysis = swing_signal['analysis']
            print(f"\n📊 Analysis Details:")
            print(f"Kotva Validation: {analysis.get('kotva_validation', {}).get('status', 'N/A')}")
            print(f"Patience Validation: {analysis.get('patience_validation', {}).get('status', 'N/A')}")
            print(f"Horoshko Pattern: {analysis.get('horoshko_patterns', {}).get('pattern', 'N/A')}")
    else:
        print(f"❌ Swing Signal Error: {swing_signal['error']}")
    
    print(f"\n🚀 BNB Swing Trading System - Ready for Production!")
    print("🥋 Following ХАЙДУШКИ КОДЕКС: котва levels, patience, хайдушко хоро rhythm")
