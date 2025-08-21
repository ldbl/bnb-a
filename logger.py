#!/usr/bin/env python3
"""
Logging Module
Centralized logging system for the BNB analyzer
"""

import logging
import os
from datetime import datetime
from typing import Optional


class TradingLogger:
    """Custom logger for trading analysis"""
    
    def __init__(self, name: str = "BNBAnalyzer", log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup console and file handlers"""
        
        # Create logs directory if it doesn't exist
        if not os.path.exists("logs"):
            os.makedirs("logs")
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        
        # File handler
        log_filename = f"logs/bnb_analyzer_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, **kwargs)
    
    def log_api_call(self, endpoint: str, symbol: str, interval: Optional[str] = None):
        """Log API call"""
        msg = f"API Call: {endpoint} for {symbol}"
        if interval:
            msg += f" ({interval})"
        self.debug(msg)
    
    def log_signal(self, signal: dict):
        """Log trading signal"""
        self.info(
            f"Signal Generated: {signal.get('action', 'UNKNOWN')} "
            f"at ${signal.get('price', 0)} "
            f"(Confidence: {signal.get('confidence', 0)}%)"
        )
    
    def log_analysis_start(self, symbol: str):
        """Log analysis start"""
        self.info(f"Starting analysis for {symbol}")
    
    def log_analysis_complete(self, symbol: str, duration: float):
        """Log analysis completion"""
        self.info(f"Analysis complete for {symbol} in {duration:.2f}s")
    
    def log_error_with_context(self, error: Exception, context: str):
        """Log error with context"""
        self.error(f"Error in {context}: {str(error)}")
    
    def log_cache_hit(self, key: str):
        """Log cache hit"""
        self.debug(f"Cache HIT for key: {key}")
    
    def log_cache_miss(self, key: str):
        """Log cache miss"""
        self.debug(f"Cache MISS for key: {key}")


# Global logger instance
logger = TradingLogger()


# Decorator for logging function calls
def log_function_call(func):
    """Decorator to log function calls"""
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling function: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Function {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Function {func.__name__} failed: {str(e)}")
            raise
    return wrapper


def log_performance(func):
    """Decorator to log function performance"""
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"Function {func.__name__} took {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {duration:.3f}s: {str(e)}")
            raise
    return wrapper


# Example usage
if __name__ == "__main__":
    # Test logging
    test_logger = TradingLogger("TestLogger")
    
    test_logger.info("This is an info message")
    test_logger.debug("This is a debug message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    
    # Test signal logging
    test_signal = {
        "action": "BUY",
        "price": 850.50,
        "confidence": 75
    }
    test_logger.log_signal(test_signal)
    
    print("Check logs/ directory for log files")
