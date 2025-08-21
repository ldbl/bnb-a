#!/usr/bin/env python3
"""
Display Module
Handles all output formatting and user interface
"""

from typing import Dict, Any
from datetime import datetime


class TradingDisplay:
    """Class for formatting and displaying trading analysis"""
    
    def __init__(self):
        self.colors = {
            "green": "\033[92m",
            "red": "\033[91m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "purple": "\033[95m",
            "cyan": "\033[96m",
            "white": "\033[97m",
            "bold": "\033[1m",
            "end": "\033[0m"
        }
        self.use_colors = True
    
    def colorize(self, text: str, color: str) -> str:
        """Add color to text if colors are enabled"""
        if not self.use_colors:
            return text
        return f"{self.colors.get(color, '')}{text}{self.colors['end']}"
    
    def print_header(self, title: str, width: int = 60):
        """Print a formatted header"""
        print("\n" + "=" * width)
        print(self.colorize(f"🚀 {title}", "bold"))
        print("=" * width)
    
    def print_section(self, title: str, emoji: str = "📊"):
        """Print a section header"""
        print(f"\n{emoji} {self.colorize(title, 'bold')}")
    
    def format_price(self, price: float, show_dollar: bool = True) -> str:
        """Format price with proper currency symbol"""
        symbol = "$" if show_dollar else ""
        return f"{symbol}{price:,.2f}"
    
    def format_percentage(self, percentage: float) -> str:
        """Format percentage with color coding"""
        color = "green" if percentage > 0 else "red" if percentage < 0 else "white"
        sign = "+" if percentage > 0 else ""
        return self.colorize(f"{sign}{percentage:.2f}%", color)
    
    def format_trend(self, trend: str) -> str:
        """Format trend with color coding"""
        trend_colors = {
            "BULLISH": "green",
            "BEARISH": "red",
            "NEUTRAL": "yellow",
            "UPTREND": "green",
            "DOWNTREND": "red",
            "SIDEWAYS": "yellow"
        }
        color = trend_colors.get(trend, "white")
        return self.colorize(trend, color)
    
    def format_action(self, action: str) -> str:
        """Format trading action with color coding"""
        action_colors = {
            "STRONG BUY": "green",
            "BUY": "green",
            "STRONG SELL": "red",
            "SELL": "red",
            "WAIT": "yellow",
            "HOLD": "blue"
        }
        color = action_colors.get(action, "white")
        return self.colorize(f"{action}", "bold") if color == "white" else self.colorize(f"{action}", color)
    
    def display_current_status(self, signal: Dict):
        """Display current market status"""
        self.print_section("CURRENT STATUS")
        print(f"Price: {self.format_price(signal['price'])}")
        print(f"Time: {signal['timestamp']}")
        
        # Add 24h change if available
        if 'market_data' in signal:
            market = signal['market_data']
            if 'change_24h' in market:
                print(f"24h Change: {self.format_percentage(market['change_24h'])}")
    
    def display_indicators(self, indicators: Dict):
        """Display technical indicators"""
        self.print_section("INDICATORS", "📈")
        
        # RSI
        rsi = indicators['RSI']
        rsi_status = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
        rsi_color = "green" if rsi < 30 else "red" if rsi > 70 else "yellow"
        print(f"RSI(14): {self.colorize(f'{rsi} ({rsi_status})', rsi_color)}")
        
        # MACD
        macd = indicators['MACD']
        macd_trend = self.format_trend(macd['trend'])
        print(f"MACD: {macd_trend} (M:{macd['macd']} S:{macd['signal']})")
        
        # Bollinger Bands
        bollinger = indicators['Bollinger']
        bb_position = self.format_trend(bollinger['position'])
        print(f"Bollinger: {bb_position}")
        print(f"  └─ Upper: {self.format_price(bollinger['upper'])} | "
              f"Middle: {self.format_price(bollinger['middle'])} | "
              f"Lower: {self.format_price(bollinger['lower'])}")
        
        # Elliott Wave
        elliott = indicators['Elliott']
        wave_info = elliott.get('wave', 'N/A')
        wave_desc = elliott.get('description', '')
        print(f"Elliott Wave: {self.colorize(wave_info, 'purple')}")
        if wave_desc:
            print(f"  └─ {wave_desc}")
        
        # Fibonacci
        fibonacci = indicators['Fibonacci']
        fib_action = fibonacci.get('action', 'WAIT')
        fib_trend = fibonacci.get('trend', 'N/A')
        print(f"Fibonacci: {self.format_action(fib_action)} ({self.format_trend(fib_trend)})")
        
        # Special Fibonacci highlights
        if fibonacci.get('golden_pocket'):
            print(f"  └─ {self.colorize('⭐ GOLDEN POCKET - High probability zone!', 'yellow')}")
        
        if fibonacci.get('closest_level'):
            closest = fibonacci['closest_level']
            print(f"  └─ Closest level: {closest['name']} at {self.format_price(closest['level'])}")
    
    def display_timeframes(self, timeframes: Dict):
        """Display multi-timeframe analysis"""
        if not timeframes:
            return
        
        self.print_section("MULTI-TIMEFRAME ANALYSIS", "⏰")
        
        for period, data in timeframes.items():
            trend = self.format_trend(data['trend'])
            strength = data.get('strength', 0)
            rsi = data.get('rsi', 'N/A')
            elliott = data.get('elliott_wave', 'N/A')
            
            print(f"{period}: {trend} ({strength}%) | RSI:{rsi} | {elliott}")
    
    def display_signal_scores(self, signal: Dict):
        """Display bullish/bearish scores"""
        self.print_section("SIGNAL SCORES", "🎯")
        
        bull_score = signal['bull_score']
        bear_score = signal['bear_score']
        confidence = signal.get('confidence', 0)
        
        bull_color = "green" if bull_score > bear_score else "white"
        bear_color = "red" if bear_score > bull_score else "white"
        
        print(f"Bullish Score: {self.colorize(str(bull_score), bull_color)}")
        print(f"Bearish Score: {self.colorize(str(bear_score), bear_color)}")
        print(f"Confidence: {self.colorize(f'{confidence}%', 'cyan')}")
    
    def display_recommendation(self, signal: Dict):
        """Display trading recommendation"""
        self.print_section("RECOMMENDATION", "💡")
        
        action = signal['action']
        print(f"Action: {self.format_action(action)}")
        
        # Display targets and stops for actionable signals
        if action != "WAIT" and 'targets' in signal:
            targets = signal['targets']
            
            if 'primary_target' in targets:
                print(f"Primary Target: {self.format_price(targets['primary_target'])}")
            
            if 'extended_target' in targets:
                print(f"Extended Target: {self.format_price(targets['extended_target'])}")
            
            if 'stop_loss' in targets:
                stop_color = "red"
                print(f"Stop Loss: {self.colorize(self.format_price(targets['stop_loss']), stop_color)}")
            
            if 'resistance_target' in targets:
                print(f"Resistance Target: {self.format_price(targets['resistance_target'])}")
            
            if 'support_target' in targets:
                print(f"Support Target: {self.format_price(targets['support_target'])}")
        
        # Position size
        if 'position_size' in signal:
            print(f"Position Size: {self.colorize(signal['position_size'], 'cyan')}")
        
        # Reason for the signal
        if 'reason' in signal:
            print(f"Reason: {signal['reason']}")
    
    def display_market_summary(self, market_data: Dict):
        """Display market summary information"""
        if not market_data:
            return
        
        self.print_section("MARKET SUMMARY", "📊")
        
        if 'current_price' in market_data:
            print(f"Price: {self.format_price(market_data['current_price'])}")
        
        if '24h_change' in market_data:
            print(f"24h Change: {self.format_percentage(market_data['24h_change'])}")
        
        if '24h_high' in market_data and '24h_low' in market_data:
            print(f"24h Range: {self.format_price(market_data['24h_low'])} - "
                  f"{self.format_price(market_data['24h_high'])}")
        
        if '24h_volume' in market_data:
            volume = market_data['24h_volume']
            print(f"24h Volume: {volume:,.0f} BNB")
    
    def display_full_analysis(self, signal: Dict):
        """Display complete trading analysis"""
        self.print_header("BNB ADVANCED TRADING ANALYSIS")
        
        # Current status
        self.display_current_status(signal)
        
        # Market summary if available
        if 'market_data' in signal:
            self.display_market_summary(signal['market_data'])
        
        # Technical indicators
        if 'indicators' in signal:
            self.display_indicators(signal['indicators'])
        
        # Multi-timeframe analysis
        if 'timeframes' in signal:
            self.display_timeframes(signal['timeframes'])
        
        # Signal scores
        self.display_signal_scores(signal)
        
        # Final recommendation
        self.display_recommendation(signal)
        
        print("\n" + "=" * 60)
    
    def display_menu(self) -> str:
        """Display menu options and get user choice"""
        print(f"\n{self.colorize('Options:', 'bold')}")
        print("1. Refresh analysis")
        print("2. Show detailed Fibonacci analysis")
        print("3. Show market summary")
        print("4. Toggle colors")
        print("5. Exit")
        
        return input(f"\n{self.colorize('Choice (1-5): ', 'cyan')}")
    
    def display_fibonacci_hint(self):
        """Display hint for Fibonacci analysis"""
        print(f"\n{self.colorize('💡 Tip:', 'yellow')} Use option 2 for detailed Fibonacci retracement levels")
    
    def toggle_colors(self):
        """Toggle color output on/off"""
        self.use_colors = not self.use_colors
        status = "enabled" if self.use_colors else "disabled"
        print(f"\nColors {status}")


# Example usage
if __name__ == "__main__":
    display = TradingDisplay()
    
    # Test display with sample data
    sample_signal = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "price": 856.75,
        "action": "BUY",
        "confidence": 75,
        "bull_score": 8,
        "bear_score": 4,
        "position_size": "25% with 2x leverage",
        "reason": "Oversold RSI | MACD bullish crossover",
        "indicators": {
            "RSI": 28.5,
            "MACD": {"macd": 2.3, "signal": 1.8, "trend": "BULLISH"},
            "Bollinger": {"upper": 870, "middle": 850, "lower": 830, "position": "OVERSOLD"},
            "Elliott": {"wave": "WAVE_2", "description": "Wave 2 pullback"},
            "Fibonacci": {"action": "BUY", "trend": "UPTREND", "closest_level": {"name": "61.8%", "level": 855}}
        },
        "targets": {
            "primary_target": 890,
            "stop_loss": 830
        }
    }
    
    display.display_full_analysis(sample_signal)
