#!/usr/bin/env python3
"""
Unit Tests for EmailReporter
Tests dependency injection, logging, and report generation
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from email_reporter import EmailReporter, MockAnalyzer


class TestEmailReporter(unittest.TestCase):
    """Test cases for EmailReporter class"""
    
    def setUp(self):
        """Set up test environment"""
        # Mock environment variables
        self.env_vars = {
            'SENDER_EMAIL': 'test@example.com',
            'SENDER_PASSWORD': 'test_password',
            'RECIPIENT_EMAIL': 'recipient@example.com',
            'SMTP_SERVER': 'smtp.test.com',
            'SMTP_PORT': '587'
        }
        
        # Mock analyzer data
        self.mock_data = {
            'market_analysis': {
                'market_data': {'current_price': 850.0},
                'signals': {
                    'action': 'BUY',
                    'confidence': 75,
                    'bull_score': 6,
                    'bear_score': 2,
                    'reasoning': ['RSI oversold', 'Support level bounce']
                }
            },
            'alerts': {
                'show_any': True,
                'whale_alerts': ['Large whale transaction detected'],
                'fibonacci_alerts': ['Price near Golden Pocket']
            },
            'market_data': {'current_price': 850.0}
        }
    
    @patch.dict(os.environ, {
        'SENDER_EMAIL': 'test@example.com',
        'SENDER_PASSWORD': 'test_password', 
        'RECIPIENT_EMAIL': 'recipient@example.com'
    })
    def test_email_reporter_initialization(self):
        """Test EmailReporter initialization with environment variables"""
        mock_analyzer = MockAnalyzer(self.mock_data)
        reporter = EmailReporter(analyzer=mock_analyzer)
        
        self.assertEqual(reporter.sender_email, 'test@example.com')
        self.assertEqual(reporter.recipient_email, 'recipient@example.com')
        self.assertEqual(reporter._analyzer, mock_analyzer)
    
    def test_email_reporter_missing_env_vars(self):
        """Test EmailReporter raises error when env vars missing"""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                EmailReporter()
            
            self.assertIn("Missing required email configuration", str(context.exception))
    
    @patch.dict(os.environ, {
        'SENDER_EMAIL': 'test@example.com',
        'SENDER_PASSWORD': 'test_password',
        'RECIPIENT_EMAIL': 'recipient@example.com'
    })
    def test_dependency_injection(self):
        """Test dependency injection works correctly"""
        mock_analyzer = MockAnalyzer(self.mock_data)
        reporter = EmailReporter(analyzer=mock_analyzer)
        
        # Analyzer should be injected, not lazy-loaded
        self.assertEqual(reporter.analyzer, mock_analyzer)
    
    @patch.dict(os.environ, {
        'SENDER_EMAIL': 'test@example.com',
        'SENDER_PASSWORD': 'test_password',
        'RECIPIENT_EMAIL': 'recipient@example.com'
    })
    @patch('main.BNBAdvancedAnalyzer')
    def test_lazy_loading_analyzer(self, mock_analyzer_class):
        """Test lazy loading when no analyzer injected"""
        mock_analyzer_instance = Mock()
        mock_analyzer_class.return_value = mock_analyzer_instance
        
        reporter = EmailReporter()  # No analyzer injected
        
        # Access analyzer property to trigger lazy loading
        analyzer = reporter.analyzer
        
        # Should create BNBAdvancedAnalyzer
        mock_analyzer_class.assert_called_once()
        self.assertEqual(analyzer, mock_analyzer_instance)
    
    @patch.dict(os.environ, {
        'SENDER_EMAIL': 'test@example.com',
        'SENDER_PASSWORD': 'test_password',
        'RECIPIENT_EMAIL': 'recipient@example.com'
    })
    def test_generate_daily_report(self):
        """Test daily report generation"""
        mock_analyzer = MockAnalyzer(self.mock_data)
        reporter = EmailReporter(analyzer=mock_analyzer)
        
        report = reporter.generate_daily_report()
        
        # Check report contains expected sections
        self.assertIn("BNB ADVANCED DAILY REPORT", report)
        self.assertIn("Current Price: $850.00", report)
        self.assertIn("CRITICAL ALERTS DETECTED", report)
        self.assertIn("STRATEGIC TRADING SIGNALS", report)
        self.assertIn("Action: 🟢 BUY", report)
        self.assertIn("Confidence: 75%", report)
    
    @patch.dict(os.environ, {
        'SENDER_EMAIL': 'test@example.com',
        'SENDER_PASSWORD': 'test_password',
        'RECIPIENT_EMAIL': 'recipient@example.com'
    })
    def test_generate_report_with_error(self):
        """Test report generation handles analyzer errors"""
        mock_analyzer = Mock()
        mock_analyzer.analyze_market.return_value = {"error": "API failure"}
        mock_analyzer.check_critical_alerts.return_value = {}
        
        reporter = EmailReporter(analyzer=mock_analyzer)
        report = reporter.generate_daily_report()
        
        self.assertIn("❌ Error in market analysis: API failure", report)
    
    @patch.dict(os.environ, {
        'SENDER_EMAIL': 'test@example.com',
        'SENDER_PASSWORD': 'test_password',
        'RECIPIENT_EMAIL': 'recipient@example.com'
    })
    def test_format_alerts_summary(self):
        """Test alert summary formatting"""
        mock_analyzer = MockAnalyzer(self.mock_data)
        reporter = EmailReporter(analyzer=mock_analyzer)
        
        alerts = {
            'show_any': True,
            'whale_alerts': ['Alert 1', 'Alert 2'],
            'fibonacci_alerts': ['Fib alert'],
            'indicator_alerts': []
        }
        
        summary = reporter.format_alerts_summary(alerts)
        
        self.assertIn("CRITICAL ALERTS DETECTED", summary)
        self.assertIn("🐋 Whale Activity: 2 alert(s)", summary)
        self.assertIn("📐 Fibonacci Signals: 1 alert(s)", summary)
        self.assertIn("TOTAL: 3 critical alert(s)", summary)
    
    @patch.dict(os.environ, {
        'SENDER_EMAIL': 'test@example.com',
        'SENDER_PASSWORD': 'test_password',
        'RECIPIENT_EMAIL': 'recipient@example.com'
    })
    def test_format_strategic_summary(self):
        """Test strategic summary formatting"""
        mock_analyzer = MockAnalyzer(self.mock_data)
        reporter = EmailReporter(analyzer=mock_analyzer)
        
        signals = {
            'action': 'STRONG_BUY',
            'confidence': 85,
            'bull_score': 8,
            'bear_score': 1,
            'reasoning': ['Strong momentum', 'Volume confirmation', 'Technical breakout']
        }
        
        summary = reporter.format_strategic_summary(signals)
        
        self.assertIn("STRATEGIC TRADING SIGNALS", summary)
        self.assertIn("Action: 🟢🚀 STRONG_BUY", summary)
        self.assertIn("Confidence: 85%", summary)
        self.assertIn("Bull Score: 8 | Bear Score: 1", summary)
        self.assertIn("Strong momentum", summary)
        self.assertIn("Volume confirmation", summary)
        self.assertIn("Technical breakout", summary)
    
    @patch.dict(os.environ, {
        'SENDER_EMAIL': 'test@example.com',
        'SENDER_PASSWORD': 'test_password',
        'RECIPIENT_EMAIL': 'recipient@example.com'
    })
    @patch('smtplib.SMTP')
    def test_send_email_success(self, mock_smtp):
        """Test successful email sending"""
        # Mock SMTP server
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        mock_analyzer = MockAnalyzer(self.mock_data)
        reporter = EmailReporter(analyzer=mock_analyzer)
        
        success = reporter.send_email("Test Subject", "Test Body")
        
        self.assertTrue(success)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with('test@example.com', 'test_password')
        mock_server.sendmail.assert_called_once()
    
    @patch.dict(os.environ, {
        'SENDER_EMAIL': 'test@example.com',
        'SENDER_PASSWORD': 'test_password',
        'RECIPIENT_EMAIL': 'recipient@example.com'
    })
    @patch('smtplib.SMTP')
    def test_send_email_failure(self, mock_smtp):
        """Test email sending failure handling"""
        # Mock SMTP server to raise exception
        mock_smtp.side_effect = Exception("SMTP connection failed")
        
        mock_analyzer = MockAnalyzer(self.mock_data)
        reporter = EmailReporter(analyzer=mock_analyzer)
        
        success = reporter.send_email("Test Subject", "Test Body")
        
        self.assertFalse(success)
    
    @patch.dict(os.environ, {
        'SENDER_EMAIL': 'test@example.com',
        'SENDER_PASSWORD': 'test_password',
        'RECIPIENT_EMAIL': 'recipient@example.com'
    })
    @patch.object(EmailReporter, 'send_email')
    def test_send_daily_report(self, mock_send_email):
        """Test daily report sending process"""
        mock_send_email.return_value = True
        
        mock_analyzer = MockAnalyzer(self.mock_data)
        reporter = EmailReporter(analyzer=mock_analyzer)
        
        success = reporter.send_daily_report()
        
        self.assertTrue(success)
        mock_send_email.assert_called_once()
        
        # Check that subject contains expected format
        call_args = mock_send_email.call_args
        subject = call_args[0][0]  # First argument
        self.assertIn("🚀 BNB Daily Report", subject)
        self.assertIn("$850.00", subject)


class TestMockAnalyzer(unittest.TestCase):
    """Test cases for MockAnalyzer"""
    
    def test_mock_analyzer_default_data(self):
        """Test MockAnalyzer with default data"""
        mock_analyzer = MockAnalyzer()
        
        market_analysis = mock_analyzer.analyze_market()
        alerts = mock_analyzer.check_critical_alerts()
        market_data = mock_analyzer.get_market_data()
        
        self.assertIn('market_data', market_analysis)
        self.assertIn('signals', market_analysis)
        self.assertIn('show_any', alerts)
        self.assertIn('current_price', market_data)
    
    def test_mock_analyzer_custom_data(self):
        """Test MockAnalyzer with custom data"""
        custom_data = {
            'market_analysis': {
                'market_data': {'current_price': 1000.0},
                'signals': {'action': 'SELL', 'confidence': 90}
            }
        }
        
        mock_analyzer = MockAnalyzer(custom_data)
        market_analysis = mock_analyzer.analyze_market()
        
        self.assertEqual(market_analysis['market_data']['current_price'], 1000.0)
        self.assertEqual(market_analysis['signals']['action'], 'SELL')


class TestEmailReporterIntegration(unittest.TestCase):
    """Integration tests for EmailReporter"""
    
    @patch.dict(os.environ, {
        'SENDER_EMAIL': 'test@example.com',
        'SENDER_PASSWORD': 'test_password',
        'RECIPIENT_EMAIL': 'recipient@example.com'
    })
    @patch('smtplib.SMTP')
    def test_full_email_workflow(self, mock_smtp):
        """Test complete email workflow from generation to sending"""
        # Mock SMTP server
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # Create mock analyzer with ML predictor and reversal detector
        mock_analyzer = MockAnalyzer({
            'market_analysis': {
                'market_data': {'current_price': 875.50},
                'signals': {
                    'action': 'WAIT',
                    'confidence': 45,
                    'bull_score': 3,
                    'bear_score': 4,
                    'reasoning': ['Market uncertainty', 'Mixed signals']
                }
            },
            'alerts': {
                'show_any': False
            },
            'market_data': {'current_price': 875.50}
        })
        
        # Add ML predictor and reversal detector
        mock_ml = Mock()
        mock_ml.analyze_long_term_trends.return_value = {
            'cycle_position': {'phase': 'LATE_BULL_MARKET', 'risk_level': 'HIGH'},
            'trend_analysis': {'trend_direction': 'BULLISH', 'monthly_performance': 5.2},
            'long_term_targets': {
                '1_month': {'price': 920.0, 'change_pct': 5.1},
                '6_months': {'price': 1200.0, 'change_pct': 37.1}
            }
        }
        
        mock_reversal = Mock()
        mock_reversal.check_critical_reversal_alerts.return_value = {
            'show_alert': True,
            'reversal_data': {
                'direction': 'BEARISH',
                'conviction': 'MEDIUM',
                'total_score': 15
            }
        }
        
        mock_analyzer.ml_predictor = mock_ml
        mock_analyzer.reversal_detector = mock_reversal
        
        # Test EmailReporter
        reporter = EmailReporter(analyzer=mock_analyzer)
        
        # Generate report
        report = reporter.generate_daily_report()
        
        # Verify report content
        self.assertIn("Current Price: $875.50", report)
        self.assertIn("Action: 🟡 WAIT", report)
        self.assertIn("No critical alerts today", report)
        self.assertIn("AI STRATEGIC OUTLOOK", report)
        self.assertIn("Market Cycle: LATE_BULL_MARKET", report)
        self.assertIn("TREND REVERSAL ALERT", report)
        self.assertIn("Direction: BEARISH", report)
        
        # Send report
        success = reporter.send_daily_report()
        
        # Verify sending
        self.assertTrue(success)
        mock_server.sendmail.assert_called_once()


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)
