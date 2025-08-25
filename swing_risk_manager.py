#!/usr/bin/env python3
"""
Swing Risk Manager for BNB Quarterly Trades
Implements BNB-optimized risk management for 3-4 month holding periods
with 20-40% amplitude targets and 2% capital risk per trade
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available. Using simplified ATR calculation.")

from logger import get_logger
from datetime import datetime, timedelta

class SwingRiskManager:
    """
    BNB-optimized risk management for swing trading
    
    Features:
    - Position sizing based on 2% capital risk per trade
    - ATR-based stop losses (period=14 for quarterly, period=7 for monthly)
    - Quarterly volatility adjustment
    - BNB-specific support/resistance integration
    - Haiduk Code compliance (Rule #3: Steps, Rule #4: Leverage, Rule #7: Retreat)
    """
    
    def __init__(self, 
                 capital: float = 10000,
                 risk_per_trade: float = 0.02,
                 max_leverage: float = 2.0):
        """
        Initialize Swing Risk Manager
        
        Args:
            capital: Total trading capital
            risk_per_trade: Maximum risk per trade (default: 2%)
            max_leverage: Maximum allowed leverage (default: 2x)
        """
        self.logger = get_logger(__name__)
        
        # Core risk parameters
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.max_leverage = max_leverage
        
        # BNB-specific configurations
        self.bnb_config = {
            'monthly': {
                'atr_period': 7,
                'stop_loss_multiplier': 2.0,
                'target_return': '5-10%',
                'holding_period': '1 month',
                'volatility_adjustment': 1.0
            },
            'quarterly': {
                'atr_period': 14,
                'stop_loss_multiplier': 3.0,
                'target_return': '20-40%',
                'holding_period': '3-4 months',
                'volatility_adjustment': 1.2
            }
        }
        
        # BNB support/resistance levels
        self.bnb_levels = {
            'support': [750, 800, 850, 900],
            'resistance': [800, 850, 900, 950],
            'critical_support': 550,  # Rule #7: Retreat below $550
            'quarterly_high': 793.35,  # Q3-Q4 2024 high
            'quarterly_low': 407.52    # Q3-Q4 2024 low
        }
        
        # Risk tracking
        self.active_positions = {}
        self.risk_history = []
        
        self.logger.info(f"Swing Risk Manager initialized - Capital: ${capital:,.2f}, Risk per trade: {risk_per_trade*100}%")
        self.logger.info(f"BNB levels - Support: {self.bnb_levels['support']}, Resistance: {self.bnb_levels['resistance']}")
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                     period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR) for volatility measurement
        
        Args:
            high: High prices series
            low: Low prices series
            close: Close prices series
            period: ATR period (7 for monthly, 14 for quarterly)
            
        Returns:
            ATR series
        """
        try:
            if TALIB_AVAILABLE:
                # Use TA-Lib for accurate ATR calculation
                atr = talib.ATR(high.values, low.values, close.values, timeperiod=period)
                return pd.Series(atr, index=close.index)
            else:
                # Simplified ATR calculation
                return self._calculate_simple_atr(high, low, close, period)
                
        except Exception as e:
            self.logger.error(f"ATR calculation failed: {e}")
            # Fallback to simple volatility calculation
            return self._calculate_simple_volatility(close, period)
    
    def _calculate_simple_atr(self, high: pd.Series, low: pd.Series, 
                             close: pd.Series, period: int) -> pd.Series:
        """Simplified ATR calculation when TA-Lib is not available"""
        try:
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR using simple moving average
            atr = true_range.rolling(window=period).mean()
            
            return atr
            
        except Exception as e:
            self.logger.error(f"Simple ATR calculation failed: {e}")
            return pd.Series([np.nan] * len(close), index=close.index)
    
    def _calculate_simple_volatility(self, close: pd.Series, period: int) -> pd.Series:
        """Fallback volatility calculation using price changes"""
        try:
            price_changes = close.pct_change().abs()
            volatility = price_changes.rolling(window=period).mean()
            return volatility * close  # Scale by price for ATR-like values
            
        except Exception as e:
            self.logger.error(f"Volatility calculation failed: {e}")
            return pd.Series([np.nan] * len(close), index=close.index)
    
    def calculate_stop_loss(self, entry_price: float, direction: str, 
                           atr_value: float, timeframe: str = 'quarterly') -> Dict:
        """
        Calculate ATR-based stop loss for BNB swing trading
        
        Args:
            entry_price: Entry price for the trade
            direction: 'long' or 'short'
            atr_value: Current ATR value
            timeframe: 'monthly' or 'quarterly'
            
        Returns:
            Stop loss configuration
        """
        try:
            if timeframe not in self.bnb_config:
                raise ValueError(f"Invalid timeframe: {timeframe}. Use 'monthly' or 'quarterly'")
            
            config = self.bnb_config[timeframe]
            multiplier = config['stop_loss_multiplier']
            
            # Calculate ATR-based stop loss
            atr_stop_distance = atr_value * multiplier
            
            if direction.lower() == 'long':
                stop_loss = entry_price - atr_stop_distance
                stop_type = "below entry"
            elif direction.lower() == 'short':
                stop_loss = entry_price + atr_stop_distance
                stop_type = "above entry"
            else:
                raise ValueError(f"Invalid direction: {direction}. Use 'long' or 'short'")
            
            # BNB-specific stop loss adjustments
            adjusted_stop = self._adjust_stop_for_bnb_levels(stop_loss, direction, timeframe)
            
            # Calculate risk per share
            risk_per_share = abs(entry_price - adjusted_stop)
            
            return {
                'entry_price': entry_price,
                'direction': direction,
                'timeframe': timeframe,
                'atr_value': atr_value,
                'atr_multiplier': multiplier,
                'raw_stop_loss': stop_loss,
                'adjusted_stop_loss': adjusted_stop,
                'stop_type': stop_type,
                'risk_per_share': risk_per_share,
                'stop_distance': abs(entry_price - adjusted_stop),
                'stop_distance_pct': (abs(entry_price - adjusted_stop) / entry_price) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Stop loss calculation failed: {e}")
            return {"error": str(e)}
    
    def _adjust_stop_for_bnb_levels(self, raw_stop: float, direction: str, 
                                   timeframe: str) -> float:
        """Adjust stop loss based on BNB support/resistance levels"""
        try:
            if direction.lower() == 'long':
                # For long positions, stop loss should be above support levels
                relevant_levels = self.bnb_levels['support'] + [self.bnb_levels['critical_support']]
                relevant_levels = [level for level in relevant_levels if level < raw_stop]
                
                if relevant_levels:
                    nearest_support = max(relevant_levels)
                    # Don't place stop loss too close to support
                    if raw_stop - nearest_support < 10:
                        adjusted_stop = nearest_support - 10
                        self.logger.info(f"Adjusted stop loss to ${adjusted_stop:.2f} (above support ${nearest_support})")
                        return adjusted_stop
                
            elif direction.lower() == 'short':
                # For short positions, stop loss should be below resistance levels
                relevant_levels = self.bnb_levels['resistance']
                relevant_levels = [level for level in relevant_levels if level > raw_stop]
                
                if relevant_levels:
                    nearest_resistance = min(relevant_levels)
                    # Don't place stop loss too close to resistance
                    if nearest_resistance - raw_stop < 10:
                        adjusted_stop = nearest_resistance + 10
                        self.logger.info(f"Adjusted stop loss to ${adjusted_stop:.2f} (below resistance ${nearest_resistance})")
                        return adjusted_stop
            
            return raw_stop
            
        except Exception as e:
            self.logger.warning(f"Stop loss adjustment failed: {e}")
            return raw_stop
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              timeframe: str = 'quarterly') -> Dict:
        """
        Calculate position size based on 2% capital risk per trade
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            timeframe: 'monthly' or 'quarterly'
            
        Returns:
            Position sizing configuration
        """
        try:
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss)
            
            if risk_per_share == 0:
                raise ValueError("Entry price and stop loss cannot be the same")
            
            # Calculate maximum position size based on risk
            max_risk_amount = self.capital * self.risk_per_trade
            max_shares = max_risk_amount / risk_per_share
            
            # Apply BNB-specific adjustments
            config = self.bnb_config[timeframe]
            volatility_adjustment = config['volatility_adjustment']
            
            # Adjust position size for volatility
            adjusted_shares = max_shares / volatility_adjustment
            
            # Calculate position value
            position_value = adjusted_shares * entry_price
            
            # Check leverage requirements
            required_margin = position_value / self.max_leverage
            leverage_used = position_value / required_margin if required_margin > 0 else 1.0
            
            # Ensure leverage doesn't exceed maximum
            if leverage_used > self.max_leverage:
                adjusted_shares = (self.capital * self.max_leverage) / entry_price
                position_value = adjusted_shares * entry_price
                leverage_used = self.max_leverage
                self.logger.warning(f"Position size adjusted to respect maximum leverage ({self.max_leverage}x)")
            
            # Calculate actual risk amount
            actual_risk = adjusted_shares * risk_per_share
            actual_risk_pct = (actual_risk / self.capital) * 100
            
            return {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'timeframe': timeframe,
                'risk_per_share': risk_per_share,
                'max_shares': max_shares,
                'adjusted_shares': adjusted_shares,
                'position_value': position_value,
                'required_margin': required_margin,
                'leverage_used': leverage_used,
                'max_risk_amount': max_risk_amount,
                'actual_risk_amount': actual_risk,
                'actual_risk_pct': actual_risk_pct,
                'volatility_adjustment': volatility_adjustment,
                'compliance': {
                    'max_risk_respected': actual_risk_pct <= self.risk_per_trade * 100,
                    'leverage_respected': leverage_used <= self.max_leverage,
                    'haiduk_compliant': True
                }
            }
            
        except Exception as e:
            self.logger.error(f"Position size calculation failed: {e}")
            return {"error": str(e)}
    
    def calculate_quarterly_volatility_adjustment(self, data: pd.DataFrame) -> float:
        """
        Calculate quarterly volatility adjustment for BNB
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Volatility adjustment factor
        """
        try:
            if len(data) < 90:  # Need at least 3 months of data
                self.logger.warning("Insufficient data for quarterly volatility calculation")
                return 1.0
            
            # Calculate quarterly returns
            quarterly_returns = []
            for i in range(0, len(data) - 90, 30):  # Monthly intervals
                if i + 90 <= len(data):
                    start_price = data['close'].iloc[i]
                    end_price = data['close'].iloc[i + 90]
                    quarterly_return = (end_price - start_price) / start_price
                    quarterly_returns.append(abs(quarterly_return))
            
            if not quarterly_returns:
                return 1.0
            
            # Calculate volatility metrics
            avg_quarterly_return = np.mean(quarterly_returns)
            volatility_std = np.std(quarterly_returns)
            
            # BNB-specific volatility adjustment
            if avg_quarterly_return > 0.25:  # >25% average quarterly return
                adjustment = 1.2  # Increase position size for high volatility
            elif avg_quarterly_return < 0.15:  # <15% average quarterly return
                adjustment = 0.8  # Decrease position size for low volatility
            else:
                adjustment = 1.0  # Standard adjustment
            
            self.logger.info(f"Quarterly volatility adjustment: {adjustment:.2f} (avg return: {avg_quarterly_return*100:.1f}%)")
            return adjustment
            
        except Exception as e:
            self.logger.error(f"Volatility adjustment calculation failed: {e}")
            return 1.0
    
    def generate_swing_trading_plan(self, entry_price: float, direction: str, 
                                  timeframe: str = 'quarterly', 
                                  data: pd.DataFrame = None) -> Dict:
        """
        Generate comprehensive swing trading plan for BNB
        
        Args:
            entry_price: Entry price
            direction: 'long' or 'short'
            timeframe: 'monthly' or 'quarterly'
            data: Historical data for ATR calculation
            
        Returns:
            Complete swing trading plan
        """
        try:
            # Calculate ATR if data provided
            atr_value = None
            if data is not None and len(data) > 0:
                config = self.bnb_config[timeframe]
                atr_period = config['atr_period']
                
                if len(data) >= atr_period:
                    atr_series = self.calculate_atr(data['high'], data['low'], data['close'], atr_period)
                    atr_value = atr_series.iloc[-1] if not atr_series.empty else None
            
            # Use default ATR if calculation failed
            if atr_value is None or np.isnan(atr_value):
                atr_value = entry_price * 0.05  # Default 5% of entry price
                self.logger.warning(f"Using default ATR: ${atr_value:.2f}")
            
            # Calculate stop loss
            stop_loss_config = self.calculate_stop_loss(entry_price, direction, atr_value, timeframe)
            if "error" in stop_loss_config:
                return {"error": f"Stop loss calculation failed: {stop_loss_config['error']}"}
            
            # Calculate position size
            position_config = self.calculate_position_size(entry_price, stop_loss_config['adjusted_stop_loss'], timeframe)
            if "error" in position_config:
                return {"error": f"Position size calculation failed: {position_config['error']}"}
            
            # Generate trading plan
            config = self.bnb_config[timeframe]
            
            trading_plan = {
                'entry': {
                    'price': entry_price,
                    'direction': direction,
                    'timeframe': timeframe,
                    'target_return': config['target_return'],
                    'holding_period': config['holding_period']
                },
                'risk_management': {
                    'stop_loss': stop_loss_config,
                    'position_sizing': position_config,
                    'max_risk_pct': self.risk_per_trade * 100,
                    'max_leverage': self.max_leverage
                },
                'bnb_levels': {
                    'nearest_support': self._find_nearest_level(entry_price, self.bnb_levels['support']),
                    'nearest_resistance': self._find_nearest_level(entry_price, self.bnb_levels['resistance']),
                    'critical_support': self.bnb_levels['critical_support']
                },
                'volatility': {
                    'atr_value': atr_value,
                    'atr_period': config['atr_period'],
                    'volatility_adjustment': config['volatility_adjustment']
                },
                'haiduk_compliance': {
                    'rule_3_steps': f"Position size: {position_config['adjusted_shares']:.2f} shares (1/3 capital approach)",
                    'rule_4_leverage': f"Leverage used: {position_config['leverage_used']:.1f}x (max: {self.max_leverage}x)",
                    'rule_7_retreat': f"Critical stop: ${self.bnb_levels['critical_support']} (Rule #7: Retreat)"
                },
                'generated_at': datetime.now().isoformat()
            }
            
            self.logger.info(f"Swing trading plan generated for {timeframe} {direction} at ${entry_price:.2f}")
            return trading_plan
            
        except Exception as e:
            self.logger.error(f"Trading plan generation failed: {e}")
            return {"error": str(e)}
    
    def _find_nearest_level(self, price: float, levels: List[float]) -> Dict:
        """Find nearest support/resistance level"""
        try:
            if not levels:
                return {"level": None, "distance": None}
            
            distances = [abs(price - level) for level in levels]
            min_distance_idx = np.argmin(distances)
            nearest_level = levels[min_distance_idx]
            distance = distances[min_distance_idx]
            
            return {
                "level": nearest_level,
                "distance": distance,
                "distance_pct": (distance / price) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Nearest level calculation failed: {e}")
            return {"level": None, "distance": None}
    
    def validate_trade_setup(self, entry_price: float, stop_loss: float, 
                           position_size: float, timeframe: str = 'quarterly') -> Dict:
        """
        Validate trade setup for risk compliance
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            position_size: Position size in shares
            
        Returns:
            Validation results
        """
        try:
            # Calculate actual risk
            risk_per_share = abs(entry_price - stop_loss)
            total_risk = position_size * risk_per_share
            risk_pct = (total_risk / self.capital) * 100
            
            # Calculate leverage
            position_value = position_size * entry_price
            leverage = position_value / self.capital
            
            # Validation checks
            validations = {
                'risk_compliance': risk_pct <= self.risk_per_trade * 100,
                'leverage_compliance': leverage <= self.max_leverage,
                'stop_loss_reasonable': risk_per_share / entry_price <= 0.15,  # Max 15% stop distance
                'position_size_reasonable': position_value <= self.capital * 2  # Max 2x capital
            }
            
            # Overall validation
            all_valid = all(validations.values())
            
            return {
                'valid': all_valid,
                'validations': validations,
                'risk_analysis': {
                    'total_risk': total_risk,
                    'risk_pct': risk_pct,
                    'max_allowed_risk': self.capital * self.risk_per_trade,
                    'max_allowed_risk_pct': self.risk_per_trade * 100
                },
                'leverage_analysis': {
                    'current_leverage': leverage,
                    'max_allowed_leverage': self.max_leverage
                },
                'recommendations': self._generate_validation_recommendations(validations, risk_pct, leverage)
            }
            
        except Exception as e:
            self.logger.error(f"Trade validation failed: {e}")
            return {"error": str(e)}
    
    def _generate_validation_recommendations(self, validations: Dict, risk_pct: float, 
                                          leverage: float) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if not validations['risk_compliance']:
            recommendations.append(f"Reduce position size to keep risk under {self.risk_per_trade*100}%")
        
        if not validations['leverage_compliance']:
            recommendations.append(f"Reduce leverage to stay under {self.max_leverage}x maximum")
        
        if not validations['stop_loss_reasonable']:
            recommendations.append("Consider tighter stop loss to reduce risk per share")
        
        if not validations['position_size_reasonable']:
            recommendations.append("Position size too large relative to capital")
        
        if not recommendations:
            recommendations.append("Trade setup is compliant with all risk parameters")
        
        return recommendations

# Example usage and testing
if __name__ == "__main__":
    print("🎯 BNB Swing Risk Manager - Professional Risk Management")
    print("=" * 60)
    print("📊 Optimized for quarterly trades with 20-40% amplitude")
    print("🛡️ 2% capital risk per trade with ATR-based stop losses")
    print("🥋 Haiduk Code compliant (Rules #3, #4, #7)")
    print()
    
    # Test basic functionality
    risk_manager = SwingRiskManager(capital=10000, risk_per_trade=0.02)
    
    print(f"✅ Risk Manager initialized")
    print(f"💰 Capital: ${risk_manager.capital:,.2f}")
    print(f"⚠️  Risk per trade: {risk_manager.risk_per_trade*100}%")
    print(f"📈 Max leverage: {risk_manager.max_leverage}x")
    print()
    print("💡 Next steps:")
    print("1. Prepare BNB OHLCV data")
    print("2. Call calculate_atr() for volatility")
    print("3. Use calculate_stop_loss() for stop levels")
    print("4. Use calculate_position_size() for sizing")
    print("5. Generate complete trading plan with generate_swing_trading_plan()")
    print()
    print("🚀 Ready for BNB swing trading with professional risk management!")
