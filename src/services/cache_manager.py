"""
Multi-level cache implementation for translation results.
"""

import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from uuid import uuid4

import redis.asyncio as redis

from src.config.config import config
from src.database.connection import get_db_session
from src.database.repositories import CacheRepository
from src.models.interfaces import CacheManager, CacheEntry
from src.utils.exceptions import CacheError
from src.utils.logging import TranslationLogger

logger = TranslationLogger(__name__, "cache-manager")


class MultiLevelCacheManager(CacheManager):
    """Multi-level cache implementation with L1 (memory), L2 (Redis), L3 (database)."""
    
    def __init__(self):
        # L1 Cache - In-memory
        self.l1_cache: Dict[str, Tuple[CacheEntry, float]] = {}  # key -> (entry, timestamp)
        self.l1_max_size = 1000
        self.l1_ttl_seconds = 300  # 5 minutes
        
        # L2 Cache - Redis
        self.redis_client = None
        self.l2_ttl_seconds = 3600  # 1 hour
        
        # L3 Cache - Database (persistent)
        self.l3_ttl_hours = config.performance.cache_ttl_hours
        
        # Cache statistics
        self.stats = {
            "l1_hits": 0,
            "l1_misses": 0,
            "l2_hits": 0,
            "l2_misses": 0,
            "l3_hits": 0,
            "l3_misses": 0,
            "total_requests": 0
        }
        
        self._cleanup_task = None
        self._running = False
        
        logger.info("Multi-level cache manager initialized")
    
    async def start(self):
        """Start cache manager and cleanup tasks."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Cache manager started")
    
    async def stop(self):
        """Stop cache manager and cleanup tasks."""
        if not self._running:
            return
        
        self._running = False
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Cache manager stopped")
    
    async def get(self, cache_key: str) -> Optional[CacheEntry]:
        """Get cached translation result from multi-level cache."""
        try:
            self.stats["total_requests"] += 1
            
            # L1 Cache check (in-memory)
            l1_entry = await self._get_from_l1(cache_key)
            if l1_entry:
                self.stats["l1_hits"] += 1
                logger.debug(f"L1 cache hit for key: {cache_key[:16]}...")
                return l1_entry
            
            self.stats["l1_misses"] += 1
            
            # L2 Cache check (Redis)
            l2_entry = await self._get_from_l2(cache_key)
            if l2_entry:
                self.stats["l2_hits"] += 1
                # Promote to L1
                await self._set_to_l1(cache_key, l2_entry)
                logger.debug(f"L2 cache hit for key: {cache_key[:16]}...")
                return l2_entry
            
            self.stats["l2_misses"] += 1
            
            # L3 Cache check (Database)
            l3_entry = await self._get_from_l3(cache_key)
            if l3_entry:
                self.stats["l3_hits"] += 1
                # Promote to L2 and L1
                await self._set_to_l2(cache_key, l3_entry)
                await self._set_to_l1(cache_key, l3_entry)
                logger.debug(f"L3 cache hit for key: {cache_key[:16]}...")
                return l3_entry
            
            self.stats["l3_misses"] += 1
            logger.debug(f"Cache miss for key: {cache_key[:16]}...")
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for key {cache_key[:16]}...: {str(e)}")
            return None
    
    async def set(self, cache_key: str, entry: CacheEntry, ttl: int = None) -> bool:
        """Store translation result in multi-level cache."""
        try:
            # Set in all cache levels
            success_l1 = await self._set_to_l1(cache_key, entry)
            success_l2 = await self._set_to_l2(cache_key, entry, ttl)
            success_l3 = await self._set_to_l3(cache_key, entry)
            
            logger.debug(
                f"Cache set for key {cache_key[:16]}...",
                metadata={
                    "l1_success": success_l1,
                    "l2_success": success_l2,
                    "l3_success": success_l3
                }
            )
            
            return success_l1 or success_l2 or success_l3
            
        except Exception as e:
            logger.error(f"Cache set error for key {cache_key[:16]}...: {str(e)}")
            return False
    
    async def invalidate(self, cache_key: str) -> bool:
        """Remove entry from all cache levels."""
        try:
            success_l1 = await self._invalidate_from_l1(cache_key)
            success_l2 = await self._invalidate_from_l2(cache_key)
            success_l3 = await self._invalidate_from_l3(cache_key)
            
            logger.debug(f"Cache invalidated for key: {cache_key[:16]}...")
            
            return success_l1 or success_l2 or success_l3
            
        except Exception as e:
            logger.error(f"Cache invalidation error for key {cache_key[:16]}...: {str(e)}")
            return False
    
    async def generate_cache_key(self, source_lang: str, target_lang: str, content: str, model_version: str) -> str:
        """Generate cache key for translation request."""
        # Create deterministic cache key
        key_data = f"{source_lang}:{target_lang}:{model_version}:{content.strip()}"
        cache_key = hashlib.sha256(key_data.encode('utf-8')).hexdigest()
        return f"translation:{cache_key}"
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_hits = self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["l3_hits"]
        total_misses = self.stats["l1_misses"] + self.stats["l2_misses"] + self.stats["l3_misses"]
        total_requests = self.stats["total_requests"]
        
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate_percentage": round(hit_rate, 2),
            "l1_cache": {
                "hits": self.stats["l1_hits"],
                "misses": self.stats["l1_misses"],
                "size": len(self.l1_cache),
                "max_size": self.l1_max_size,
                "hit_rate": round((self.stats["l1_hits"] / total_requests * 100) if total_requests > 0 else 0, 2)
            },
            "l2_cache": {
                "hits": self.stats["l2_hits"],
                "misses": self.stats["l2_misses"],
                "hit_rate": round((self.stats["l2_hits"] / total_requests * 100) if total_requests > 0 else 0, 2)
            },
            "l3_cache": {
                "hits": self.stats["l3_hits"],
                "misses": self.stats["l3_misses"],
                "hit_rate": round((self.stats["l3_hits"] / total_requests * 100) if total_requests > 0 else 0, 2)
            }
        }
    
    async def warm_cache(self, language_pairs: List[Tuple[str, str]], popular_phrases: List[str]) -> int:
        """Warm cache with popular language pairs and phrases."""
        warmed_count = 0
        
        try:
            for source_lang, target_lang in language_pairs:
                for phrase in popular_phrases:
                    # Check if already cached
                    cache_key = await self.generate_cache_key(source_lang, target_lang, phrase, "1.0.0")
                    existing = await self.get(cache_key)
                    
                    if not existing:
                        # Create placeholder cache entry for warming
                        # In real implementation, this would trigger actual translation
                        cache_entry = CacheEntry(
                            id=uuid4(),
                            content_hash=cache_key.split(":")[-1],
                            source_language=source_lang,
                            target_language=target_lang,
                            source_content=phrase,
                            translated_content=f"[Warmed] {phrase}",
                            model_version="1.0.0",
                            confidence_score=0.8,
                            created_at=datetime.utcnow(),
                            last_accessed=datetime.utcnow()
                        )
                        
                        success = await self.set(cache_key, cache_entry)
                        if success:
                            warmed_count += 1
            
            logger.info(f"Cache warming completed: {warmed_count} entries added")
            return warmed_count
            
        except Exception as e:
            logger.error(f"Cache warming failed: {str(e)}")
            return warmed_count
    
    async def _get_redis_client(self) -> redis.Redis:
        """Get Redis client connection."""
        if not self.redis_client:
            self.redis_client = redis.Redis(
                host=config.redis.host,
                port=config.redis.port,
                password=config.redis.password,
                db=config.redis.db + 1,  # Use different DB for cache
                decode_responses=True
            )
        return self.redis_client
    
    async def _get_from_l1(self, cache_key: str) -> Optional[CacheEntry]:
        """Get entry from L1 cache (in-memory)."""
        if cache_key in self.l1_cache:
            entry, timestamp = self.l1_cache[cache_key]
            
            # Check TTL
            if time.time() - timestamp < self.l1_ttl_seconds:
                # Update access time
                entry.last_accessed = datetime.utcnow()
                entry.access_count += 1
                return entry
            else:
                # Expired, remove from L1
                del self.l1_cache[cache_key]
        
        return None
    
    async def _set_to_l1(self, cache_key: str, entry: CacheEntry) -> bool:
        """Set entry in L1 cache (in-memory)."""
        try:
            # Check if L1 cache is full
            if len(self.l1_cache) >= self.l1_max_size:
                await self._evict_from_l1()
            
            self.l1_cache[cache_key] = (entry, time.time())
            return True
            
        except Exception as e:
            logger.error(f"L1 cache set error: {str(e)}")
            return False
    
    async def _invalidate_from_l1(self, cache_key: str) -> bool:
        """Remove entry from L1 cache."""
        if cache_key in self.l1_cache:
            del self.l1_cache[cache_key]
            return True
        return False
    
    async def _evict_from_l1(self):
        """Evict oldest entries from L1 cache."""
        if not self.l1_cache:
            return
        
        # Remove 10% of oldest entries
        entries_to_remove = max(1, len(self.l1_cache) // 10)
        
        # Sort by timestamp (oldest first)
        sorted_entries = sorted(self.l1_cache.items(), key=lambda x: x[1][1])
        
        for i in range(entries_to_remove):
            cache_key = sorted_entries[i][0]
            del self.l1_cache[cache_key]
    
    async def _get_from_l2(self, cache_key: str) -> Optional[CacheEntry]:
        """Get entry from L2 cache (Redis)."""
        try:
            redis_client = await self._get_redis_client()
            cached_data = await redis_client.get(cache_key)
            
            if cached_data:
                entry_dict = json.loads(cached_data)
                entry = self._dict_to_cache_entry(entry_dict)
                
                # Update access time
                entry.last_accessed = datetime.utcnow()
                entry.access_count += 1
                
                return entry
            
            return None
            
        except Exception as e:
            logger.error(f"L2 cache get error: {str(e)}")
            return None
    
    async def _set_to_l2(self, cache_key: str, entry: CacheEntry, ttl: int = None) -> bool:
        """Set entry in L2 cache (Redis)."""
        try:
            redis_client = await self._get_redis_client()
            
            entry_dict = self._cache_entry_to_dict(entry)
            cached_data = json.dumps(entry_dict, default=str)
            
            ttl_seconds = ttl or self.l2_ttl_seconds
            await redis_client.setex(cache_key, ttl_seconds, cached_data)
            
            return True
            
        except Exception as e:
            logger.error(f"L2 cache set error: {str(e)}")
            return False
    
    async def _invalidate_from_l2(self, cache_key: str) -> bool:
        """Remove entry from L2 cache."""
        try:
            redis_client = await self._get_redis_client()
            result = await redis_client.delete(cache_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"L2 cache invalidation error: {str(e)}")
            return False
    
    async def _get_from_l3(self, cache_key: str) -> Optional[CacheEntry]:
        """Get entry from L3 cache (Database)."""
        try:
            # Extract content hash from cache key
            content_hash = cache_key.split(":")[-1]
            
            async with get_db_session() as session:
                cache_repo = CacheRepository(session)
                
                # Find cache entry by content hash
                entries = await cache_repo.find_by({"content_hash": content_hash})
                
                if entries:
                    entry = entries[0]  # Get the first match
                    
                    # Check if entry is still valid
                    if entry.created_at and entry.created_at > datetime.utcnow() - timedelta(hours=self.l3_ttl_hours):
                        # Update access statistics
                        entry.last_accessed = datetime.utcnow()
                        entry.access_count += 1
                        await cache_repo.update(entry)
                        await session.commit()
                        
                        return entry
                    else:
                        # Entry expired, remove it
                        await cache_repo.delete(entry.id)
                        await session.commit()
            
            return None
            
        except Exception as e:
            logger.error(f"L3 cache get error: {str(e)}")
            return None
    
    async def _set_to_l3(self, cache_key: str, entry: CacheEntry) -> bool:
        """Set entry in L3 cache (Database)."""
        try:
            async with get_db_session() as session:
                cache_repo = CacheRepository(session)
                
                # Check if entry already exists
                existing_entries = await cache_repo.find_by({"content_hash": entry.content_hash})
                
                if existing_entries:
                    # Update existing entry
                    existing_entry = existing_entries[0]
                    existing_entry.last_accessed = datetime.utcnow()
                    existing_entry.access_count += 1
                    await cache_repo.update(existing_entry)
                else:
                    # Create new entry
                    entry.created_at = datetime.utcnow()
                    entry.last_accessed = datetime.utcnow()
                    await cache_repo.create(entry)
                
                await session.commit()
                return True
                
        except Exception as e:
            logger.error(f"L3 cache set error: {str(e)}")
            return False
    
    async def _invalidate_from_l3(self, cache_key: str) -> bool:
        """Remove entry from L3 cache."""
        try:
            content_hash = cache_key.split(":")[-1]
            
            async with get_db_session() as session:
                cache_repo = CacheRepository(session)
                
                entries = await cache_repo.find_by({"content_hash": content_hash})
                
                for entry in entries:
                    await cache_repo.delete(entry.id)
                
                await session.commit()
                return len(entries) > 0
                
        except Exception as e:
            logger.error(f"L3 cache invalidation error: {str(e)}")
            return False
    
    def _cache_entry_to_dict(self, entry: CacheEntry) -> Dict[str, Any]:
        """Convert CacheEntry to dictionary for serialization."""
        return {
            "id": str(entry.id),
            "content_hash": entry.content_hash,
            "source_language": entry.source_language,
            "target_language": entry.target_language,
            "source_content": entry.source_content,
            "translated_content": entry.translated_content,
            "model_version": entry.model_version,
            "confidence_score": entry.confidence_score,
            "access_count": entry.access_count,
            "last_accessed": entry.last_accessed.isoformat() if entry.last_accessed else None,
            "created_at": entry.created_at.isoformat() if entry.created_at else None
        }
    
    def _dict_to_cache_entry(self, entry_dict: Dict[str, Any]) -> CacheEntry:
        """Convert dictionary to CacheEntry."""
        return CacheEntry(
            id=entry_dict["id"],
            content_hash=entry_dict["content_hash"],
            source_language=entry_dict["source_language"],
            target_language=entry_dict["target_language"],
            source_content=entry_dict["source_content"],
            translated_content=entry_dict["translated_content"],
            model_version=entry_dict["model_version"],
            confidence_score=entry_dict["confidence_score"],
            access_count=entry_dict.get("access_count", 1),
            last_accessed=datetime.fromisoformat(entry_dict["last_accessed"]) if entry_dict.get("last_accessed") else None,
            created_at=datetime.fromisoformat(entry_dict["created_at"]) if entry_dict.get("created_at") else None
        )
    
    async def _cleanup_loop(self):
        """Periodic cleanup of expired cache entries."""
        while self._running:
            try:
                await self._cleanup_expired_entries()
                await asyncio.sleep(3600)  # Run every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {str(e)}")
                await asyncio.sleep(3600)
    
    async def _cleanup_expired_entries(self):
        """Clean up expired entries from all cache levels."""
        try:
            # L1 cleanup (in-memory)
            current_time = time.time()
            expired_keys = []
            
            for cache_key, (entry, timestamp) in self.l1_cache.items():
                if current_time - timestamp > self.l1_ttl_seconds:
                    expired_keys.append(cache_key)
            
            for key in expired_keys:
                del self.l1_cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired L1 cache entries")
            
            # L3 cleanup (database)
            async with get_db_session() as session:
                cache_repo = CacheRepository(session)
                cleaned_count = await cache_repo.cleanup_expired(self.l3_ttl_hours)
                await session.commit()
                
                if cleaned_count > 0:
                    logger.debug(f"Cleaned up {cleaned_count} expired L3 cache entries")
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {str(e)}")


def create_cache_manager() -> MultiLevelCacheManager:
    """Factory function to create cache manager."""
    return MultiLevelCacheManager()