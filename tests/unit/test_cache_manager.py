"""
Unit tests for cache manager.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from src.models.interfaces import CacheEntry
from src.services.cache_manager import MultiLevelCacheManager, create_cache_manager


class TestMultiLevelCacheManager:
    """Test multi-level cache manager functionality."""
    
    @pytest.fixture
    async def cache_manager(self):
        """Create cache manager instance."""
        manager = create_cache_manager()
        await manager.start()
        yield manager
        await manager.stop()
    
    @pytest.fixture
    def sample_cache_entry(self):
        """Create sample cache entry."""
        return CacheEntry(
            id=uuid4(),
            content_hash="test-hash-123",
            source_language="en",
            target_language="es",
            source_content="Hello world",
            translated_content="Hola mundo",
            model_version="1.0.0",
            confidence_score=0.95,
            access_count=1,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow()
        )
    
    @pytest.mark.asyncio
    async def test_cache_manager_initialization(self, cache_manager):
        """Test cache manager initialization."""
        assert cache_manager.l1_cache == {}
        assert cache_manager.l1_max_size > 0
        assert cache_manager.l1_ttl_seconds > 0
        assert cache_manager.l2_ttl_seconds > 0
        assert cache_manager.l3_ttl_hours > 0
        assert cache_manager._running is True
    
    @pytest.mark.asyncio
    async def test_generate_cache_key(self, cache_manager):
        """Test cache key generation."""
        key1 = await cache_manager.generate_cache_key("en", "es", "Hello world", "1.0.0")
        key2 = await cache_manager.generate_cache_key("en", "es", "Hello world", "1.0.0")
        key3 = await cache_manager.generate_cache_key("en", "fr", "Hello world", "1.0.0")
        
        # Same inputs should produce same key
        assert key1 == key2
        
        # Different inputs should produce different keys
        assert key1 != key3
        
        # Key should have expected format
        assert key1.startswith("translation:")
        assert len(key1) > 20  # Should be reasonably long hash
    
    @pytest.mark.asyncio
    async def test_l1_cache_operations(self, cache_manager, sample_cache_entry):
        """Test L1 cache operations."""
        cache_key = "test:l1:key"
        
        # Test set and get
        success = await cache_manager._set_to_l1(cache_key, sample_cache_entry)
        assert success is True
        
        retrieved = await cache_manager._get_from_l1(cache_key)
        assert retrieved is not None
        assert retrieved.content_hash == sample_cache_entry.content_hash
        assert retrieved.translated_content == sample_cache_entry.translated_content
        
        # Test invalidation
        success = await cache_manager._invalidate_from_l1(cache_key)
        assert success is True
        
        retrieved = await cache_manager._get_from_l1(cache_key)
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_l1_cache_ttl(self, cache_manager, sample_cache_entry):
        """Test L1 cache TTL expiration."""
        cache_key = "test:l1:ttl"
        
        # Set entry
        await cache_manager._set_to_l1(cache_key, sample_cache_entry)
        
        # Should be retrievable immediately
        retrieved = await cache_manager._get_from_l1(cache_key)
        assert retrieved is not None
        
        # Mock expired timestamp
        import time
        original_time = time.time()
        expired_time = original_time - cache_manager.l1_ttl_seconds - 1
        
        # Manually set expired timestamp
        cache_manager.l1_cache[cache_key] = (sample_cache_entry, expired_time)
        
        # Should not be retrievable after expiration
        retrieved = await cache_manager._get_from_l1(cache_key)
        assert retrieved is None
        
        # Entry should be removed from cache
        assert cache_key not in cache_manager.l1_cache
    
    @pytest.mark.asyncio
    async def test_l1_cache_eviction(self, cache_manager):
        """Test L1 cache eviction when full."""
        # Fill cache to max capacity
        for i in range(cache_manager.l1_max_size + 5):
            entry = CacheEntry(
                id=uuid4(),
                content_hash=f"hash-{i}",
                source_language="en",
                target_language="es",
                source_content=f"Content {i}",
                translated_content=f"Contenido {i}",
                model_version="1.0.0",
                confidence_score=0.9,
                access_count=1,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow()
            )
            
            await cache_manager._set_to_l1(f"key-{i}", entry)
        
        # Cache should not exceed max size
        assert len(cache_manager.l1_cache) <= cache_manager.l1_max_size
    
    @pytest.mark.asyncio
    async def test_multi_level_cache_get_miss(self, cache_manager):
        """Test cache miss across all levels."""
        with patch.object(cache_manager, '_get_from_l2', return_value=None), \
             patch.object(cache_manager, '_get_from_l3', return_value=None):
            
            result = await cache_manager.get("nonexistent:key")
            assert result is None
            
            # Check statistics
            stats = await cache_manager.get_cache_stats()
            assert stats["l1_misses"] >= 1
            assert stats["l2_misses"] >= 1
            assert stats["l3_misses"] >= 1
    
    @pytest.mark.asyncio
    async def test_multi_level_cache_l1_hit(self, cache_manager, sample_cache_entry):
        """Test L1 cache hit."""
        cache_key = "test:multi:l1"
        
        # Set in L1 cache
        await cache_manager._set_to_l1(cache_key, sample_cache_entry)
        
        # Should hit L1 and not check L2/L3
        with patch.object(cache_manager, '_get_from_l2') as mock_l2, \
             patch.object(cache_manager, '_get_from_l3') as mock_l3:
            
            result = await cache_manager.get(cache_key)
            
            assert result is not None
            assert result.content_hash == sample_cache_entry.content_hash
            
            # L2 and L3 should not be called
            mock_l2.assert_not_called()
            mock_l3.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_multi_level_cache_l2_hit(self, cache_manager, sample_cache_entry):
        """Test L2 cache hit with promotion to L1."""
        cache_key = "test:multi:l2"
        
        # Mock L2 hit
        with patch.object(cache_manager, '_get_from_l2', return_value=sample_cache_entry), \
             patch.object(cache_manager, '_get_from_l3') as mock_l3, \
             patch.object(cache_manager, '_set_to_l1') as mock_set_l1:
            
            result = await cache_manager.get(cache_key)
            
            assert result is not None
            assert result.content_hash == sample_cache_entry.content_hash
            
            # Should promote to L1
            mock_set_l1.assert_called_once_with(cache_key, sample_cache_entry)
            
            # L3 should not be called
            mock_l3.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_multi_level_cache_l3_hit(self, cache_manager, sample_cache_entry):
        """Test L3 cache hit with promotion to L1 and L2."""
        cache_key = "test:multi:l3"
        
        # Mock L3 hit
        with patch.object(cache_manager, '_get_from_l2', return_value=None), \
             patch.object(cache_manager, '_get_from_l3', return_value=sample_cache_entry), \
             patch.object(cache_manager, '_set_to_l1') as mock_set_l1, \
             patch.object(cache_manager, '_set_to_l2') as mock_set_l2:
            
            result = await cache_manager.get(cache_key)
            
            assert result is not None
            assert result.content_hash == sample_cache_entry.content_hash
            
            # Should promote to both L1 and L2
            mock_set_l1.assert_called_once_with(cache_key, sample_cache_entry)
            mock_set_l2.assert_called_once_with(cache_key, sample_cache_entry)
    
    @pytest.mark.asyncio
    async def test_cache_set_all_levels(self, cache_manager, sample_cache_entry):
        """Test setting cache entry in all levels."""
        cache_key = "test:set:all"
        
        with patch.object(cache_manager, '_set_to_l1', return_value=True) as mock_l1, \
             patch.object(cache_manager, '_set_to_l2', return_value=True) as mock_l2, \
             patch.object(cache_manager, '_set_to_l3', return_value=True) as mock_l3:
            
            result = await cache_manager.set(cache_key, sample_cache_entry)
            
            assert result is True
            
            # All levels should be called
            mock_l1.assert_called_once_with(cache_key, sample_cache_entry)
            mock_l2.assert_called_once_with(cache_key, sample_cache_entry, None)
            mock_l3.assert_called_once_with(cache_key, sample_cache_entry)
    
    @pytest.mark.asyncio
    async def test_cache_invalidate_all_levels(self, cache_manager):
        """Test invalidating cache entry from all levels."""
        cache_key = "test:invalidate:all"
        
        with patch.object(cache_manager, '_invalidate_from_l1', return_value=True) as mock_l1, \
             patch.object(cache_manager, '_invalidate_from_l2', return_value=True) as mock_l2, \
             patch.object(cache_manager, '_invalidate_from_l3', return_value=True) as mock_l3:
            
            result = await cache_manager.invalidate(cache_key)
            
            assert result is True
            
            # All levels should be called
            mock_l1.assert_called_once_with(cache_key)
            mock_l2.assert_called_once_with(cache_key)
            mock_l3.assert_called_once_with(cache_key)
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, cache_manager, sample_cache_entry):
        """Test cache statistics collection."""
        # Perform some cache operations
        cache_key = "test:stats"
        
        # Cache miss
        await cache_manager.get("nonexistent:key")
        
        # Cache set and hit
        await cache_manager._set_to_l1(cache_key, sample_cache_entry)
        await cache_manager.get(cache_key)
        
        stats = await cache_manager.get_cache_stats()
        
        assert "total_requests" in stats
        assert "total_hits" in stats
        assert "total_misses" in stats
        assert "hit_rate_percentage" in stats
        assert "l1_cache" in stats
        assert "l2_cache" in stats
        assert "l3_cache" in stats
        
        # Check L1 stats
        l1_stats = stats["l1_cache"]
        assert "hits" in l1_stats
        assert "misses" in l1_stats
        assert "size" in l1_stats
        assert "max_size" in l1_stats
        assert "hit_rate" in l1_stats
        
        assert stats["total_requests"] >= 2
        assert l1_stats["hits"] >= 1
    
    @pytest.mark.asyncio
    async def test_cache_warming(self, cache_manager):
        """Test cache warming functionality."""
        language_pairs = [("en", "es"), ("en", "fr")]
        popular_phrases = ["Hello", "Thank you", "Good morning"]
        
        with patch.object(cache_manager, 'get', return_value=None), \
             patch.object(cache_manager, 'set', return_value=True) as mock_set:
            
            warmed_count = await cache_manager.warm_cache(language_pairs, popular_phrases)
            
            expected_count = len(language_pairs) * len(popular_phrases)
            assert warmed_count == expected_count
            
            # Should have called set for each combination
            assert mock_set.call_count == expected_count
    
    @pytest.mark.asyncio
    async def test_cache_entry_serialization(self, cache_manager, sample_cache_entry):
        """Test cache entry serialization and deserialization."""
        # Test to dict
        entry_dict = cache_manager._cache_entry_to_dict(sample_cache_entry)
        
        assert isinstance(entry_dict, dict)
        assert entry_dict["content_hash"] == sample_cache_entry.content_hash
        assert entry_dict["source_language"] == sample_cache_entry.source_language
        assert entry_dict["target_language"] == sample_cache_entry.target_language
        assert entry_dict["translated_content"] == sample_cache_entry.translated_content
        
        # Test from dict
        reconstructed = cache_manager._dict_to_cache_entry(entry_dict)
        
        assert reconstructed.content_hash == sample_cache_entry.content_hash
        assert reconstructed.source_language == sample_cache_entry.source_language
        assert reconstructed.target_language == sample_cache_entry.target_language
        assert reconstructed.translated_content == sample_cache_entry.translated_content
    
    @pytest.mark.asyncio
    async def test_cache_error_handling(self, cache_manager):
        """Test cache error handling."""
        # Test with invalid cache key
        result = await cache_manager.get("")
        assert result is None
        
        # Test set with None entry should not crash
        result = await cache_manager.set("test:key", None)
        assert result is False or result is None
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(self, cache_manager):
        """Test concurrent cache operations."""
        entries = []
        for i in range(10):
            entry = CacheEntry(
                id=uuid4(),
                content_hash=f"concurrent-hash-{i}",
                source_language="en",
                target_language="es",
                source_content=f"Content {i}",
                translated_content=f"Contenido {i}",
                model_version="1.0.0",
                confidence_score=0.9,
                access_count=1,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow()
            )
            entries.append(entry)
        
        # Concurrent set operations
        set_tasks = [
            cache_manager._set_to_l1(f"concurrent:key:{i}", entry)
            for i, entry in enumerate(entries)
        ]
        set_results = await asyncio.gather(*set_tasks, return_exceptions=True)
        
        # Most should succeed
        successful_sets = [r for r in set_results if r is True]
        assert len(successful_sets) >= 8  # Allow some failures
        
        # Concurrent get operations
        get_tasks = [
            cache_manager._get_from_l1(f"concurrent:key:{i}")
            for i in range(10)
        ]
        get_results = await asyncio.gather(*get_tasks, return_exceptions=True)
        
        # Should retrieve most entries
        successful_gets = [r for r in get_results if r is not None and not isinstance(r, Exception)]
        assert len(successful_gets) >= 8