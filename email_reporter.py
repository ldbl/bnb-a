#!/usr/bin/env python3
"""
Email Reporter Module
Generates and sends daily BNB analysis reports via email
Refactored with dependency injection and centralized logging
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
from typing import Dict, Optional, Any, Protocol
from abc import ABC, abstractmethod

from logger import get_logger


class AnalyzerProtocol(Protocol):
    """Protocol for analyzer dependency - defines required interface"""
    
    def analyze_market(self) -> Dict[str, Any]:
        """Analyze current market conditions"""
        ...
    
    def check_critical_alerts(self) -> Dict[str, Any]:
        """Check for critical alerts across all modules"""
        ...
    
    def get_market_data(self) -> Dict[str, Any]:
        """Get current market data"""
        ...


class MLPredictorProtocol(Protocol):
    """Protocol for ML predictor dependency"""
    
    def analyze_long_term_trends(self) -> Dict[str, Any]:
        """Analyze long-term market trends"""
        ...


class ReversalDetectorProtocol(Protocol):
    """Protocol for trend reversal detector dependency"""
    
    def check_critical_reversal_alerts(self) -> Dict[str, Any]:
        """Check for critical trend reversal alerts"""
        ...


class EmailReporter:
    """Handles daily email report generation and sending with dependency injection"""
    
    def __init__(self, analyzer: Optional[AnalyzerProtocol] = None, 
                 smtp_server: str = None, smtp_port: int = None):
        """
        Initialize EmailReporter with dependency injection
        
        Args:
            analyzer: Analyzer instance (injected dependency)
            smtp_server: SMTP server (defaults to Gmail)
            smtp_port: SMTP port (defaults to 587)
        """
        self.logger = get_logger(__name__)
        
        # Dependency injection - analyzer can be provided or will be lazy-loaded
        self._analyzer = analyzer
        self._ml_predictor = None
        self._reversal_detector = None
        
        # Email configuration from environment variables or parameters
        self.smtp_server = smtp_server or os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = smtp_port or int(os.getenv('SMTP_PORT', '587'))
        self.sender_email = os.getenv('SENDER_EMAIL')
        self.sender_password = os.getenv('SENDER_PASSWORD')
        self.recipient_email = os.getenv('RECIPIENT_EMAIL')
        
        # Validate required environment variables
        if not all([self.sender_email, self.sender_password, self.recipient_email]):
            error_msg = "Missing required email configuration. Set SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info(f"EmailReporter initialized for {self.recipient_email}")
    
    @property
    def analyzer(self) -> AnalyzerProtocol:
        """Lazy-load analyzer if not injected"""
        if self._analyzer is None:
            self.logger.info("Lazy-loading BNBAdvancedAnalyzer...")
            # Import here to avoid circular dependency
            from main import BNBAdvancedAnalyzer
            self._analyzer = BNBAdvancedAnalyzer()
        return self._analyzer
    
    @property 
    def ml_predictor(self) -> Optional[MLPredictorProtocol]:
        """Get ML predictor from analyzer"""
        if self._ml_predictor is None and hasattr(self.analyzer, 'ml_predictor'):
            self._ml_predictor = self.analyzer.ml_predictor
        return self._ml_predictor
    
    @property
    def reversal_detector(self) -> Optional[ReversalDetectorProtocol]:
        """Get reversal detector from analyzer"""
        if self._reversal_detector is None and hasattr(self.analyzer, 'reversal_detector'):
            self._reversal_detector = self.analyzer.reversal_detector
        return self._reversal_detector
    
    def generate_daily_report(self) -> str:
        """Generate comprehensive daily analysis report"""
        
        self.logger.info("Generating daily BNB analysis report...")
        
        try:
            # Get current analysis
            analysis = self.analyzer.analyze_market()
            alerts = self.analyzer.check_critical_alerts()
            
            if "error" in analysis:
                error_msg = f"Error in market analysis: {analysis['error']}"
                self.logger.error(error_msg)
                return f"❌ {error_msg}"
            
            self.logger.debug("Market analysis completed successfully")
            
            # Build email report
            report_parts = []
            
            # Header
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
            current_price = analysis.get('market_data', {}).get('current_price', 0)
            
            report_parts.append(f"""
🚀 BNB ADVANCED DAILY REPORT
Date: {current_time}
Current Price: ${current_price:.2f}
{'='*50}
""")
            
            # Critical Alerts Summary
            if alerts.get("show_any", False):
                report_parts.append(self.format_alerts_summary(alerts))
                self.logger.info(f"Report includes {len(alerts)} critical alerts")
            else:
                report_parts.append("📊 No critical alerts today - market in normal conditions\n")
                self.logger.debug("No critical alerts detected")
            
            # Strategic Analysis Summary
            signals = analysis.get('signals', {})
            report_parts.append(self.format_strategic_summary(signals))
            
            # ML Strategic Outlook
            try:
                if self.ml_predictor:
                    ml_analysis = self.ml_predictor.analyze_long_term_trends()
                    if "error" not in ml_analysis:
                        report_parts.append(self.format_ml_outlook(ml_analysis))
                        self.logger.debug("ML analysis included in report")
            except Exception as e:
                self.logger.warning(f"Failed to include ML analysis: {e}")
            
            # Trend Reversal Check
            try:
                if self.reversal_detector:
                    reversal_check = self.reversal_detector.check_critical_reversal_alerts()
                    if reversal_check.get("show_alert", False):
                        report_parts.append(self.format_reversal_alert(reversal_check))
                        self.logger.info("Trend reversal alert included in report")
            except Exception as e:
                self.logger.warning(f"Failed to include reversal analysis: {e}")
            
            # Footer
            report_parts.append(f"""
{'='*50}
📧 Automated Daily Report from BNB Advanced Analyzer
🤖 Generated by AI-powered trading analysis system
⚠️  This is for informational purposes only, not financial advice
""")
            
            full_report = "\n".join(report_parts)
            self.logger.info(f"Daily report generated successfully ({len(full_report)} characters)")
            return full_report
            
        except Exception as e:
            error_msg = f"Error generating daily report: {e}"
            self.logger.error(error_msg)
            return f"❌ {error_msg}"
    
    def format_alerts_summary(self, alerts: Dict) -> str:
        """Format critical alerts for email"""
        
        alert_lines = ["🚨 CRITICAL ALERTS DETECTED:"]
        
        # Count alerts by type
        alert_types = {
            'whale_alerts': '🐋 Whale Activity',
            'correlation_alerts': '📊 Correlation Anomalies', 
            'fibonacci_alerts': '📐 Fibonacci Signals',
            'indicator_alerts': '📈 Technical Indicators',
            'ml_alerts': '🤖 ML Predictions',
            'reversal_alerts': '🔄 Trend Reversals'
        }
        
        total_alerts = 0
        for alert_type, alert_name in alert_types.items():
            count = len(alerts.get(alert_type, []))
            if count > 0:
                alert_lines.append(f"   {alert_name}: {count} alert(s)")
                total_alerts += count
        
        alert_lines.append(f"   📍 TOTAL: {total_alerts} critical alert(s)")
        alert_lines.append("   💡 Check detailed analysis for context\n")
        
        return "\n".join(alert_lines)
    
    def format_strategic_summary(self, signals: Dict) -> str:
        """Format strategic trading signals for email"""
        
        action = signals.get('action', 'WAIT')
        confidence = signals.get('confidence', 0)
        bull_score = signals.get('bull_score', 0)
        bear_score = signals.get('bear_score', 0)
        
        # Action emoji
        action_emoji = {
            'STRONG_BUY': '🟢🚀',
            'BUY': '🟢', 
            'WAIT': '🟡',
            'SELL': '🔴',
            'STRONG_SELL': '🔴💥'
        }.get(action, '❓')
        
        summary_lines = [
            "📊 STRATEGIC TRADING SIGNALS:",
            f"   Action: {action_emoji} {action}",
            f"   Confidence: {confidence}%",
            f"   Bull Score: {bull_score} | Bear Score: {bear_score}",
        ]
        
        # Add reasoning if available
        reasoning = signals.get('reasoning', [])
        if reasoning:
            summary_lines.append("   💭 Key Factors:")
            for reason in reasoning[:3]:  # Top 3 reasons
                summary_lines.append(f"      • {reason}")
        
        summary_lines.append("")
        return "\n".join(summary_lines)
    
    def format_ml_outlook(self, ml_analysis: Dict) -> str:
        """Format ML strategic outlook for email"""
        
        cycle = ml_analysis.get('cycle_position', {})
        targets = ml_analysis.get('long_term_targets', {})
        trend = ml_analysis.get('trend_analysis', {})
        
        outlook_lines = [
            "🤖 AI STRATEGIC OUTLOOK:",
            f"   Market Cycle: {cycle.get('phase', 'Unknown')}",
            f"   Risk Level: {cycle.get('risk_level', 'Unknown')}",
            f"   Trend Direction: {trend.get('trend_direction', 'Unknown')}",
            f"   Monthly Performance: {trend.get('monthly_performance', 0):+.1f}%",
            ""
        ]
        
        # Price targets
        if targets:
            outlook_lines.append("🎯 PRICE TARGETS:")
            for period, target in targets.items():
                price = target.get('price', 0)
                change = target.get('change_pct', 0)
                emoji = '🚀' if change > 0 else '💥'
                period_name = period.replace('_', ' ').title()
                outlook_lines.append(f"   {period_name}: {emoji} ${price:.2f} ({change:+.1f}%)")
        
        outlook_lines.append("")
        return "\n".join(outlook_lines)
    
    def format_reversal_alert(self, reversal_data: Dict) -> str:
        """Format trend reversal alert for email"""
        
        reversal_info = reversal_data.get('reversal_data', {})
        direction = reversal_info.get('direction', 'UNKNOWN')
        conviction = reversal_info.get('conviction', 'LOW')
        score = reversal_info.get('total_score', 0)
        
        reversal_lines = [
            "🔄 TREND REVERSAL ALERT:",
            f"   Direction: {direction}",
            f"   Conviction: {conviction}",
            f"   Reversal Score: {score}/25",
            "   📈 Consider position adjustment based on signals",
            ""
        ]
        
        return "\n".join(reversal_lines)
    
    def send_email(self, subject: str, body: str) -> bool:
        """Send email with the daily report"""
        
        self.logger.info(f"Attempting to send email to {self.recipient_email}")
        
        try:
            # Create message
            message = MIMEMultipart()
            message["From"] = self.sender_email
            message["To"] = self.recipient_email
            message["Subject"] = subject
            
            # Add body to email
            message.attach(MIMEText(body, "plain"))
            
            # Create secure connection and send email
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                
                text = message.as_string()
                server.sendmail(self.sender_email, self.recipient_email, text)
            
            self.logger.info(f"✅ Email sent successfully to {self.recipient_email}")
            return True
            
        except smtplib.SMTPException as e:
            self.logger.error(f"SMTP error while sending email: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error while sending email: {e}")
            return False
    
    def send_daily_report(self) -> bool:
        """Generate and send the daily BNB analysis report"""
        
        self.logger.info("Starting daily report generation and sending process")
        
        # Generate report content
        report_content = self.generate_daily_report()
        
        # Create subject with current date and price
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            market_data = self.analyzer.get_market_data()
            current_price = market_data.get('current_price', 0)
            subject = f"🚀 BNB Daily Report - {current_date} | ${current_price:.2f}"
            self.logger.debug(f"Email subject: {subject}")
        except Exception as e:
            subject = f"🚀 BNB Daily Report - {current_date}"
            self.logger.warning(f"Could not get current price for subject: {e}")
        
        # Send email
        success = self.send_email(subject, report_content)
        
        if success:
            self.logger.info("📧 Daily report sent successfully!")
        else:
            self.logger.error("❌ Failed to send daily report")
        
        return success


class MockAnalyzer:
    """Mock analyzer for testing purposes"""
    
    def __init__(self, mock_data: Dict = None):
        self.mock_data = mock_data or {}
        self.ml_predictor = None
        self.reversal_detector = None
    
    def analyze_market(self) -> Dict[str, Any]:
        return self.mock_data.get('market_analysis', {
            'market_data': {'current_price': 850.0},
            'signals': {
                'action': 'BUY',
                'confidence': 75,
                'bull_score': 6,
                'bear_score': 2,
                'reasoning': ['RSI oversold', 'Support level bounce']
            }
        })
    
    def check_critical_alerts(self) -> Dict[str, Any]:
        return self.mock_data.get('alerts', {
            'show_any': True,
            'whale_alerts': ['Large whale transaction detected'],
            'fibonacci_alerts': ['Price near Golden Pocket']
        })
    
    def get_market_data(self) -> Dict[str, Any]:
        return self.mock_data.get('market_data', {'current_price': 850.0})


def main():
    """Main function for running daily email report"""
    
    logger = get_logger(__name__)
    
    try:
        # Check if running in test mode
        test_mode = os.getenv('TEST_MODE', 'false').lower() == 'true'
        
        if test_mode:
            logger.info("🧪 Running in TEST MODE - checking email configuration...")
            
            # Check environment variables
            required_vars = ['SENDER_EMAIL', 'SENDER_PASSWORD', 'RECIPIENT_EMAIL']
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                logger.error(f"❌ Missing environment variables: {missing_vars}")
                return False
            
            logger.info("✅ Email configuration looks good!")
            
            # Generate and print report (don't send) with mock data
            mock_analyzer = MockAnalyzer()
            reporter = EmailReporter(analyzer=mock_analyzer)
            report = reporter.generate_daily_report()
            
            print("\n📧 GENERATED REPORT:")
            print("-" * 50)
            print(report)
            print("-" * 50)
            
            logger.info("Test mode completed successfully")
            return True
        
        else:
            # Normal mode - send actual email
            logger.info("Running in PRODUCTION MODE - sending actual email")
            reporter = EmailReporter()  # Will lazy-load real analyzer
            return reporter.send_daily_report()
            
    except Exception as e:
        logger.error(f"❌ Error in email reporter: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)