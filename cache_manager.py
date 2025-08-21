#!/usr/bin/env python3
"""
Cache Manager Module
Simple caching system for API responses to improve performance
"""

import time
from typing import Dict, Any, Optional
import json


class SimpleCache:
    """Simple in-memory cache with TTL (Time To Live)"""
    
    def __init__(self):
        self.cache: Dict[str, Dict] = {}
        self.default_ttl = 60  # 1 minute default
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if time.time() > entry['expires']:
            del self.cache[key]
            return None
        
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL"""
        ttl = ttl or self.default_ttl
        expires = time.time() + ttl
        
        self.cache[key] = {
            'value': value,
            'expires': expires
        }
    
    def clear(self) -> None:
        """Clear all cached entries"""
        self.cache.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        now = time.time()
        expired = sum(1 for entry in self.cache.values() if now > entry['expires'])
        active = len(self.cache) - expired
        
        return {
            'total_entries': len(self.cache),
            'active_entries': active,
            'expired_entries': expired
        }


# Global cache instance
api_cache = SimpleCache()


def cache_key(symbol: str, interval: str, limit: int) -> str:
    """Generate cache key for API requests"""
    return f"{symbol}_{interval}_{limit}"


def get_cached_klines(symbol: str, interval: str, limit: int) -> Optional[list]:
    """Get cached klines data"""
    key = cache_key(symbol, interval, limit)
    return api_cache.get(key)


def cache_klines(symbol: str, interval: str, limit: int, data: list, ttl: int = 30) -> None:
    """Cache klines data"""
    key = cache_key(symbol, interval, limit)
    api_cache.set(key, data, ttl)


def get_cached_price(symbol: str) -> Optional[float]:
    """Get cached current price"""
    return api_cache.get(f"price_{symbol}")


def cache_price(symbol: str, price: float, ttl: int = 10) -> None:
    """Cache current price"""
    api_cache.set(f"price_{symbol}", price, ttl)


# Example usage
if __name__ == "__main__":
    cache = SimpleCache()
    
    # Test caching
    cache.set("test_key", {"data": "test_value"}, 5)
    print(f"Cached value: {cache.get('test_key')}")
    
    time.sleep(6)
    print(f"After expiry: {cache.get('test_key')}")  # Should be None
    
    print(f"Cache stats: {cache.get_stats()}")
