#!/usr/bin/env python3
"""
Unit Tests for Technical Indicators
Tests RSI, MACD, Bollinger Bands, and EMA calculations
"""

import unittest
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators import TechnicalIndicators


class TestTechnicalIndicators(unittest.TestCase):
    """Test cases for technical indicators"""
    
    def setUp(self):
        """Set up test data"""
        # Test price data (trending upward)
        self.test_prices = [
            100.0, 102.0, 101.0, 105.0, 107.0, 106.0, 109.0, 111.0, 
            108.0, 112.0, 115.0, 113.0, 117.0, 119.0, 116.0, 120.0,
            122.0, 118.0, 125.0, 127.0, 124.0, 130.0, 132.0, 129.0,
            135.0, 137.0, 134.0, 140.0, 142.0, 139.0
        ]
        
        # Oscillating price data
        self.oscillating_prices = [
            100, 110, 95, 105, 90, 115, 85, 120, 80, 125,
            75, 130, 70, 135, 65, 140, 60, 145, 55, 150
        ]
        
        # Stable price data
        self.stable_prices = [100] * 20
    
    def test_calculate_ema(self):
        """Test EMA calculation"""
        # Test basic EMA
        ema = TechnicalIndicators.calculate_ema(self.test_prices, 10)
        self.assertIsInstance(ema, float)
        self.assertGreater(ema, 0)
        
        # EMA should be between min and max of recent prices
        recent_prices = self.test_prices[-10:]
        self.assertGreaterEqual(ema, min(recent_prices))
        self.assertLessEqual(ema, max(recent_prices))
        
        # Test with insufficient data
        short_prices = [100, 102, 104]
        ema_short = TechnicalIndicators.calculate_ema(short_prices, 10)
        self.assertEqual(ema_short, 104)  # Should return last price
        
        # Test with empty data
        ema_empty = TechnicalIndicators.calculate_ema([], 10)
        self.assertEqual(ema_empty, 0)
    
    def test_calculate_rsi(self):
        """Test RSI calculation"""
        # Test basic RSI
        rsi = TechnicalIndicators.calculate_rsi(self.test_prices)
        self.assertIsInstance(rsi, float)
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)
        
        # Trending up prices should have RSI > 50
        self.assertGreater(rsi, 50)
        
        # Test with oscillating data
        rsi_osc = TechnicalIndicators.calculate_rsi(self.oscillating_prices)
        self.assertGreaterEqual(rsi_osc, 0)
        self.assertLessEqual(rsi_osc, 100)
        
        # Test with insufficient data
        short_prices = [100, 102]
        rsi_short = TechnicalIndicators.calculate_rsi(short_prices)
        self.assertEqual(rsi_short, 50.0)  # Default neutral RSI
        
        # Test with stable prices (should be neutral, but algorithm may vary)
        rsi_stable = TechnicalIndicators.calculate_rsi(self.stable_prices)
        self.assertGreaterEqual(rsi_stable, 0)
        self.assertLessEqual(rsi_stable, 100)
    
    def test_calculate_macd(self):
        """Test MACD calculation"""
        # Test basic MACD
        macd = TechnicalIndicators.calculate_macd(self.test_prices)
        self.assertIsInstance(macd, dict)
        
        # Check required keys
        required_keys = ['macd', 'signal', 'histogram', 'trend']
        for key in required_keys:
            self.assertIn(key, macd)
        
        # Check data types (could be int if 0)
        self.assertIsInstance(macd['macd'], (int, float))
        self.assertIsInstance(macd['signal'], (int, float))
        self.assertIsInstance(macd['histogram'], (int, float))
        self.assertIn(macd['trend'], ['BULLISH', 'BEARISH', 'NEUTRAL'])
        
        # Histogram should equal MACD - Signal
        expected_histogram = round(macd['macd'] - macd['signal'], 2)
        self.assertEqual(macd['histogram'], expected_histogram)
        
        # Test trend logic
        if macd['macd'] > macd['signal']:
            self.assertEqual(macd['trend'], 'BULLISH')
        elif macd['macd'] < macd['signal']:
            self.assertEqual(macd['trend'], 'BEARISH')
        else:
            self.assertEqual(macd['trend'], 'NEUTRAL')
        
        # Test with insufficient data
        short_prices = [100, 102, 104]
        macd_short = TechnicalIndicators.calculate_macd(short_prices)
        expected_default = {"macd": 0, "signal": 0, "histogram": 0, "trend": "NEUTRAL"}
        self.assertEqual(macd_short, expected_default)
    
    def test_calculate_bollinger(self):
        """Test Bollinger Bands calculation"""
        # Test basic Bollinger Bands
        bb = TechnicalIndicators.calculate_bollinger(self.test_prices)
        self.assertIsInstance(bb, dict)
        
        # Check required keys
        required_keys = ['upper', 'middle', 'lower', 'position']
        for key in required_keys:
            self.assertIn(key, bb)
        
        # Check data types
        self.assertIsInstance(bb['upper'], float)
        self.assertIsInstance(bb['middle'], float)
        self.assertIsInstance(bb['lower'], float)
        self.assertIn(bb['position'], ['OVERBOUGHT', 'OVERSOLD', 'NEUTRAL'])
        
        # Upper should be > Middle > Lower
        self.assertGreater(bb['upper'], bb['middle'])
        self.assertGreater(bb['middle'], bb['lower'])
        
        # Current price should determine position
        current_price = self.test_prices[-1]
        if current_price > bb['upper']:
            self.assertEqual(bb['position'], 'OVERBOUGHT')
        elif current_price < bb['lower']:
            self.assertEqual(bb['position'], 'OVERSOLD')
        else:
            self.assertEqual(bb['position'], 'NEUTRAL')
        
        # Test with insufficient data
        short_prices = [100, 102]
        bb_short = TechnicalIndicators.calculate_bollinger(short_prices)
        expected_default = {"upper": 0, "middle": 0, "lower": 0, "position": "NEUTRAL"}
        self.assertEqual(bb_short, expected_default)
        
        # Test with stable prices (bands should be tight)
        bb_stable = TechnicalIndicators.calculate_bollinger(self.stable_prices)
        band_width = bb_stable['upper'] - bb_stable['lower']
        self.assertLess(band_width, 1.0)  # Very tight bands for stable prices
    
    def test_calculate_all_indicators(self):
        """Test calculating all indicators at once"""
        all_indicators = TechnicalIndicators.calculate_all_indicators(self.test_prices)
        self.assertIsInstance(all_indicators, dict)
        
        # Check that all indicators are present
        required_keys = ['rsi', 'macd', 'bollinger']
        for key in required_keys:
            self.assertIn(key, all_indicators)
        
        # Verify each indicator has correct structure
        self.assertIsInstance(all_indicators['rsi'], float)
        self.assertIsInstance(all_indicators['macd'], dict)
        self.assertIsInstance(all_indicators['bollinger'], dict)
    
    def test_check_critical_indicator_alerts(self):
        """Test critical indicator alerts"""
        # Test with normal data
        alerts = TechnicalIndicators.check_critical_indicator_alerts(self.test_prices)
        self.assertIsInstance(alerts, dict)
        self.assertIn('show_alert', alerts)
        self.assertIsInstance(alerts['show_alert'], bool)
        
        # Test with insufficient data
        short_alerts = TechnicalIndicators.check_critical_indicator_alerts([100, 102])
        self.assertFalse(short_alerts['show_alert'])
        self.assertIn('reason', short_alerts)
        
        # Test extreme RSI conditions
        # Create extremely overbought scenario
        extreme_high_prices = list(range(100, 200, 2))  # Consistently increasing
        high_alerts = TechnicalIndicators.check_critical_indicator_alerts(extreme_high_prices)
        
        if high_alerts.get('show_alert'):
            self.assertIn('critical_signals', high_alerts)
            self.assertIsInstance(high_alerts['critical_signals'], list)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        # Empty price list
        empty_rsi = TechnicalIndicators.calculate_rsi([])
        self.assertEqual(empty_rsi, 50.0)
        
        # Single price
        single_ema = TechnicalIndicators.calculate_ema([100], 5)
        self.assertEqual(single_ema, 100)
        
        # Negative prices (should still work)
        negative_prices = [-10, -5, -15, -3, -8]
        negative_rsi = TechnicalIndicators.calculate_rsi(negative_prices)
        self.assertGreaterEqual(negative_rsi, 0)
        self.assertLessEqual(negative_rsi, 100)
        
        # Very large numbers
        large_prices = [1e6, 1.1e6, 0.9e6, 1.2e6]
        large_macd = TechnicalIndicators.calculate_macd(large_prices)
        self.assertIsInstance(large_macd, dict)
    
    def test_rsi_boundary_conditions(self):
        """Test RSI boundary conditions"""
        # Extremely bullish scenario (all gains)
        bullish_prices = [100 + i*5 for i in range(20)]  # Steady 5-point increases
        bullish_rsi = TechnicalIndicators.calculate_rsi(bullish_prices)
        self.assertGreater(bullish_rsi, 70)  # Should be overbought
        
        # Extremely bearish scenario (all losses)
        bearish_prices = [100 - i*5 for i in range(20)]  # Steady 5-point decreases
        bearish_rsi = TechnicalIndicators.calculate_rsi(bearish_prices)
        self.assertLess(bearish_rsi, 30)  # Should be oversold
    
    def test_macd_signal_line_correctness(self):
        """Test that MACD signal line is calculated correctly"""
        # Use sufficient data for proper MACD calculation
        long_prices = [100 + i + (i % 5) for i in range(50)]  # 50 data points with variation
        macd_result = TechnicalIndicators.calculate_macd(long_prices, fast=12, slow=26, signal=9)
        
        # Signal line should be different from MACD line for varying data
        self.assertNotEqual(macd_result['macd'], macd_result['signal'])
        
        # Histogram should be the difference
        expected_histogram = round(macd_result['macd'] - macd_result['signal'], 2)
        self.assertEqual(macd_result['histogram'], expected_histogram)


class TestIndicatorMathematics(unittest.TestCase):
    """Test mathematical correctness of indicators"""
    
    def test_ema_mathematical_properties(self):
        """Test EMA mathematical properties"""
        prices = [100, 110, 120, 130, 140]
        
        # EMA should be closer to recent prices
        ema_short = TechnicalIndicators.calculate_ema(prices, 2)
        ema_long = TechnicalIndicators.calculate_ema(prices, 4)
        
        # Shorter period EMA should be closer to last price
        last_price = prices[-1]
        self.assertLess(abs(ema_short - last_price), abs(ema_long - last_price))
    
    def test_rsi_mathematical_properties(self):
        """Test RSI mathematical properties"""
        # RSI should be inversely related to recent price movement direction
        increasing_prices = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170]
        decreasing_prices = [170, 165, 160, 155, 150, 145, 140, 135, 130, 125, 120, 115, 110, 105, 100]
        
        rsi_up = TechnicalIndicators.calculate_rsi(increasing_prices)
        rsi_down = TechnicalIndicators.calculate_rsi(decreasing_prices)
        
        # Increasing prices should have higher RSI than decreasing prices
        self.assertGreater(rsi_up, rsi_down)


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestTechnicalIndicators)
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIndicatorMathematics))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
