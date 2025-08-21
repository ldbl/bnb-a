#!/usr/bin/env python3
"""
ASCII Charts Module
Simple ASCII chart generation for price visualization
"""

from typing import List, Dict
import math


class ASCIIChart:
    """Simple ASCII chart generator"""
    
    def __init__(self, width: int = 50, height: int = 10):
        self.width = width
        self.height = height
    
    def normalize_data(self, data: List[float]) -> List[int]:
        """Normalize data to chart height"""
        if not data:
            return []
        
        min_val = min(data)
        max_val = max(data)
        
        if max_val == min_val:
            return [self.height // 2] * len(data)
        
        normalized = []
        for val in data:
            # Scale to chart height
            scaled = ((val - min_val) / (max_val - min_val)) * (self.height - 1)
            normalized.append(int(scaled))
        
        return normalized
    
    def create_price_chart(self, prices: List[float], title: str = "Price Chart") -> str:
        """Create ASCII price chart"""
        if len(prices) < 2:
            return "❌ Not enough data for chart"
        
        # Take last N points that fit in width
        recent_prices = prices[-self.width:] if len(prices) > self.width else prices
        normalized = self.normalize_data(recent_prices)
        
        chart_lines = []
        
        # Add title
        chart_lines.append(f"📈 {title}")
        chart_lines.append("─" * (self.width + 10))
        
        # Build chart from top to bottom
        for row in range(self.height - 1, -1, -1):
            line = ""
            
            # Price scale on left
            if len(recent_prices) > 0:
                min_price = min(recent_prices)
                max_price = max(recent_prices)
                if max_price != min_price:
                    price_at_row = min_price + (max_price - min_price) * (row / (self.height - 1))
                    line += f"{price_at_row:6.0f}│"
                else:
                    line += f"{min_price:6.0f}│"
            else:
                line += "      │"
            
            # Chart data
            for i, height in enumerate(normalized):
                if height == row:
                    # Trend indicators
                    if i > 0:
                        prev_height = normalized[i-1]
                        if height > prev_height:
                            line += "↗"
                        elif height < prev_height:
                            line += "↘"
                        else:
                            line += "→"
                    else:
                        line += "●"
                elif height > row:
                    line += "│"
                else:
                    line += " "
            
            chart_lines.append(line)
        
        # Add bottom axis
        chart_lines.append("      └" + "─" * len(normalized))
        
        # Add current price info
        current = recent_prices[-1]
        previous = recent_prices[-2] if len(recent_prices) > 1 else current
        change = ((current - previous) / previous * 100) if previous != 0 else 0
        
        chart_lines.append(f"Current: ${current:.2f} ({change:+.2f}%)")
        
        return "\n".join(chart_lines)
    
    def create_rsi_chart(self, rsi_values: List[float]) -> str:
        """Create RSI momentum chart"""
        if not rsi_values:
            return "❌ No RSI data"
        
        chart_lines = []
        chart_lines.append("📊 RSI Momentum")
        chart_lines.append("─" * 30)
        
        current_rsi = rsi_values[-1]
        
        # RSI bar
        bar_length = 20
        rsi_pos = int((current_rsi / 100) * bar_length)
        
        bar = ""
        for i in range(bar_length):
            if i < 6:  # Oversold zone (0-30)
                bar += "🟢" if i < rsi_pos else "░"
            elif i > 14:  # Overbought zone (70-100)  
                bar += "🔴" if i < rsi_pos else "░"
            else:  # Neutral zone (30-70)
                bar += "🟡" if i < rsi_pos else "░"
        
        chart_lines.append(f"RSI: {bar} {current_rsi:.1f}")
        chart_lines.append("     🟢Oversold  🟡Neutral  🔴Overbought")
        
        return "\n".join(chart_lines)
    
    def create_volume_chart(self, volumes: List[float]) -> str:
        """Create volume bar chart"""
        if not volumes:
            return "❌ No volume data"
        
        chart_lines = []
        chart_lines.append("📈 Volume Trend")
        chart_lines.append("─" * 30)
        
        # Take last 10 periods
        recent_volumes = volumes[-10:] if len(volumes) > 10 else volumes
        max_vol = max(recent_volumes) if recent_volumes else 1
        
        for i, vol in enumerate(recent_volumes):
            bar_length = int((vol / max_vol) * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            chart_lines.append(f"Period {i+1:2d}: {bar}")
        
        return "\n".join(chart_lines)


def create_support_resistance_visual(current_price: float, support: List[float], 
                                   resistance: List[float]) -> str:
    """Create visual representation of support/resistance levels"""
    chart_lines = []
    chart_lines.append("🎯 Support/Resistance Levels")
    chart_lines.append("─" * 40)
    
    # Combine and sort all levels
    all_levels = []
    
    # Add resistance levels
    for level in resistance[:3]:
        if level > current_price:
            all_levels.append(("R", level))
    
    # Add current price
    all_levels.append(("CURRENT", current_price))
    
    # Add support levels  
    for level in support[:3]:
        if level < current_price:
            all_levels.append(("S", level))
    
    # Sort by price (highest first)
    all_levels.sort(key=lambda x: x[1], reverse=True)
    
    for level_type, price in all_levels:
        if level_type == "CURRENT":
            chart_lines.append(f"→ CURRENT: ${price:.2f} ← YOU ARE HERE")
        elif level_type == "R":
            distance = price - current_price
            chart_lines.append(f"  Resistance: ${price:.2f} (+${distance:.2f})")
        elif level_type == "S":
            distance = current_price - price
            chart_lines.append(f"  Support: ${price:.2f} (-${distance:.2f})")
    
    return "\n".join(chart_lines)


# Example usage
if __name__ == "__main__":
    # Test price chart
    test_prices = [840, 845, 842, 850, 848, 855, 851, 860, 857, 863]
    chart = ASCIIChart(width=30, height=8)
    
    print(chart.create_price_chart(test_prices, "BNB/USDT"))
    print("\n")
    
    # Test RSI chart
    print(chart.create_rsi_chart([65.5]))
    print("\n")
    
    # Test support/resistance
    print(create_support_resistance_visual(850, [840, 830, 820], [860, 870, 880]))
