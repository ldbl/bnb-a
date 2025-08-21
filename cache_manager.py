#!/usr/bin/env python3
"""
Cache Manager Module
Simple caching system for API responses to improve performance
"""

import time
from typing import Dict, Any, Optional
import json
import threading
import atexit


class SimpleCache:
    """Thread-safe in-memory cache with TTL (Time To Live) and auto-cleanup"""
    
    def __init__(self, cleanup_interval: int = 300):  # 5 minutes
        self.cache: Dict[str, Dict] = {}
        self.default_ttl = 60  # 1 minute default
        self._lock = threading.RLock()  # Reentrant lock for nested operations
        self.cleanup_interval = cleanup_interval
        self._cleanup_timer = None
        self._start_cleanup_timer()
        
        # Register cleanup on exit
        atexit.register(self._stop_cleanup_timer)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired (thread-safe)"""
        with self._lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            if time.time() > entry['expires']:
                del self.cache[key]
                return None
            
            return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL (thread-safe)"""
        with self._lock:
            ttl = ttl or self.default_ttl
            expires = time.time() + ttl
            
            self.cache[key] = {
                'value': value,
                'expires': expires
            }
    
    def clear(self) -> None:
        """Clear all cached entries (thread-safe)"""
        with self._lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics (thread-safe)"""
        with self._lock:
            now = time.time()
            expired = sum(1 for entry in self.cache.values() if now > entry['expires'])
            active = len(self.cache) - expired
            
            return {
                'total_entries': len(self.cache),
                'active_entries': active,
                'expired_entries': expired
            }
    
    def _cleanup_expired(self) -> int:
        """Remove expired entries and return count removed"""
        with self._lock:
            now = time.time()
            expired_keys = [key for key, entry in self.cache.items() if now > entry['expires']]
            
            for key in expired_keys:
                del self.cache[key]
            
            return len(expired_keys)
    
    def _start_cleanup_timer(self) -> None:
        """Start periodic cleanup timer"""
        self._cleanup_timer = threading.Timer(self.cleanup_interval, self._periodic_cleanup)
        self._cleanup_timer.daemon = True  # Don't prevent program exit
        self._cleanup_timer.start()
    
    def _periodic_cleanup(self) -> None:
        """Periodic cleanup function"""
        try:
            removed_count = self._cleanup_expired()
            if removed_count > 0:
                # Could add logging here if needed
                pass
        except Exception:
            # Silently handle cleanup errors to avoid disrupting main application
            pass
        finally:
            # Schedule next cleanup
            if self._cleanup_timer is not None:
                self._start_cleanup_timer()
    
    def _stop_cleanup_timer(self) -> None:
        """Stop cleanup timer"""
        if self._cleanup_timer is not None:
            self._cleanup_timer.cancel()
            self._cleanup_timer = None
    
    def __del__(self):
        """Cleanup on destruction"""
        self._stop_cleanup_timer()


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
