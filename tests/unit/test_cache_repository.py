"""
Unit tests for CacheRepository.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4

from src.database.models import TranslationCache
from tests.conftest import assert_cache_equals


class TestCacheRepository:
    """Test CacheRepository operations."""
    
    @pytest.mark.asyncio
    async def test_create_cache_entry(self, cache_repository, sample_cache_entry):
        """Test creating a cache entry."""
        created_entry = await cache_repository.create(sample_cache_entry)
        
        assert created_entry.id == sample_cache_entry.id
        assert created_entry.content_hash == sample_cache_entry.content_hash
        assert created_entry.source_language == sample_cache_entry.source_language
        assert created_entry.target_language == sample_cache_entry.target_language
        assert created_entry.created_at is not None
    
    @pytest.mark.asyncio
    async def test_get_cache_entry_by_id(self, cache_repository, sample_cache_entry):
        """Test getting a cache entry by ID."""
        # Create entry first
        await cache_repository.create(sample_cache_entry)
        
        # Retrieve entry
        retrieved_entry = await cache_repository.get_by_id(sample_cache_entry.id)
        
        assert retrieved_entry is not None
        assert_cache_equals(retrieved_entry, sample_cache_entry)
    
    @pytest.mark.asyncio
    async def test_get_by_hash(self, cache_repository, sample_cache_entry):
        """Test getting cache entry by content hash and languages."""
        # Create entry first
        await cache_repository.create(sample_cache_entry)
        
        # Retrieve by hash
        retrieved_entry = await cache_repository.get_by_hash(
            sample_cache_entry.content_hash,
            sample_cache_entry.source_language,
            sample_cache_entry.target_language,
            sample_cache_entry.model_version
        )
        
        assert retrieved_entry is not None
        assert_cache_equals(retrieved_entry, sample_cache_entry)
    
    @pytest.mark.asyncio
    async def test_get_by_hash_not_found(self, cache_repository):
        """Test getting non-existent cache entry by hash."""
        entry = await cache_repository.get_by_hash(
            "non-existent-hash",
            "en",
            "es",
            "v1.0.0"
        )
        assert entry is None
    
    @pytest.mark.asyncio
    async def test_get_by_hash_without_model_version(self, cache_repository):
        """Test getting cache entry without specifying model version."""
        # Create entries with different model versions
        entry1 = TranslationCache(
            id=uuid4(),
            content_hash="test-hash",
            source_language="en",
            target_language="es",
            source_content="Hello",
            translated_content="Hola",
            model_version="v1.0.0",
            confidence_score=Decimal('0.9')
        )
        
        entry2 = TranslationCache(
            id=uuid4(),
            content_hash="test-hash",
            source_language="en",
            target_language="es",
            source_content="Hello",
            translated_content="Hola",
            model_version="v2.0.0",
            confidence_score=Decimal('0.95')
        )
        
        await cache_repository.create(entry1)
        await cache_repository.create(entry2)
        
        # Get without model version - should return most recent
        retrieved_entry = await cache_repository.get_by_hash(
            "test-hash",
            "en",
            "es"
        )
        
        assert retrieved_entry is not None
        # Should return the more recent entry (entry2)
        assert retrieved_entry.model_version == "v2.0.0"
    
    @pytest.mark.asyncio
    async def test_update_access_stats(self, cache_repository, sample_cache_entry):
        """Test updating cache access statistics."""
        # Create entry
        await cache_repository.create(sample_cache_entry)
        original_access_count = sample_cache_entry.access_count
        
        # Update access stats
        success = await cache_repository.update_access_stats(sample_cache_entry.id)
        assert success is True
        
        # Verify update
        updated_entry = await cache_repository.get_by_id(sample_cache_entry.id)
        assert updated_entry.access_count == original_access_count + 1
        assert updated_entry.last_accessed > sample_cache_entry.last_accessed
    
    @pytest.mark.asyncio
    async def test_update_access_stats_not_found(self, cache_repository):
        """Test updating access stats for non-existent entry."""
        non_existent_id = uuid4()
        success = await cache_repository.update_access_stats(non_existent_id)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_cleanup_expired(self, cache_repository):
        """Test cleaning up expired cache entries."""
        # Create old entry with low access count
        old_time = datetime.utcnow() - timedelta(hours=25)
        old_entry = TranslationCache(
            id=uuid4(),
            content_hash="old-hash",
            source_language="en",
            target_language="es",
            source_content="Old content",
            translated_content="Contenido viejo",
            model_version="v1.0.0",
            confidence_score=Decimal('0.8'),
            access_count=1,
            last_accessed=old_time
        )
        
        # Create recent entry
        recent_entry = TranslationCache(
            id=uuid4(),
            content_hash="recent-hash",
            source_language="en",
            target_language="es",
            source_content="Recent content",
            translated_content="Contenido reciente",
            model_version="v1.0.0",
            confidence_score=Decimal('0.9'),
            access_count=1,
            last_accessed=datetime.utcnow()
        )
        
        # Create old entry with high access count (should not be deleted)
        popular_old_entry = TranslationCache(
            id=uuid4(),
            content_hash="popular-hash",
            source_language="en",
            target_language="es",
            source_content="Popular content",
            translated_content="Contenido popular",
            model_version="v1.0.0",
            confidence_score=Decimal('0.95'),
            access_count=10,
            last_accessed=old_time
        )
        
        await cache_repository.create(old_entry)
        await cache_repository.create(recent_entry)
        await cache_repository.create(popular_old_entry)
        
        # Cleanup expired entries (TTL 24 hours)
        deleted_count = await cache_repository.cleanup_expired(ttl_hours=24)
        assert deleted_count == 1  # Only the old entry with low access count
        
        # Verify correct entries were deleted/kept
        old_check = await cache_repository.get_by_id(old_entry.id)
        recent_check = await cache_repository.get_by_id(recent_entry.id)
        popular_check = await cache_repository.get_by_id(popular_old_entry.id)
        
        assert old_check is None  # Should be deleted
        assert recent_check is not None  # Should be kept
        assert popular_check is not None  # Should be kept (high access count)
    
    @pytest.mark.asyncio
    async def test_get_cache_statistics(self, cache_repository):
        """Test getting cache performance statistics."""
        # Create cache entries with different characteristics
        entries = []
        
        for i in range(5):
            entry = TranslationCache(
                id=uuid4(),
                content_hash=f"hash-{i}",
                source_language="en",
                target_language="es",
                source_content=f"Content {i}",
                translated_content=f"Contenido {i}",
                model_version="v1.0.0",
                confidence_score=Decimal(str(0.8 + i * 0.05)),
                access_count=i + 1
            )
            entries.append(entry)
            await cache_repository.create(entry)
        
        # Get statistics
        stats = await cache_repository.get_cache_statistics(days=7)
        
        assert stats["total_entries"] == 5
        assert stats["recent_entries"] == 5  # All created recently
        assert stats["total_accesses"] == 15  # 1+2+3+4+5
        assert stats["avg_accesses_per_entry"] == 3.0
        assert len(stats["popular_entries"]) <= 5
        assert len(stats["language_pairs"]) == 1
        assert stats["language_pairs"][0]["source_language"] == "en"
        assert stats["language_pairs"][0]["target_language"] == "es"
        assert stats["language_pairs"][0]["count"] == 5
    
    @pytest.mark.asyncio
    async def test_get_entries_by_language_pair(self, cache_repository):
        """Test getting cache entries by language pair."""
        # Create entries for different language pairs
        en_es_entry = TranslationCache(
            id=uuid4(),
            content_hash="hash-en-es",
            source_language="en",
            target_language="es",
            source_content="Hello",
            translated_content="Hola",
            model_version="v1.0.0"
        )
        
        en_fr_entry = TranslationCache(
            id=uuid4(),
            content_hash="hash-en-fr",
            source_language="en",
            target_language="fr",
            source_content="Hello",
            translated_content="Bonjour",
            model_version="v1.0.0"
        )
        
        await cache_repository.create(en_es_entry)
        await cache_repository.create(en_fr_entry)
        
        # Get entries for EN->ES
        en_es_entries = await cache_repository.get_entries_by_language_pair("en", "es")
        assert len(en_es_entries) == 1
        assert en_es_entries[0].id == en_es_entry.id
        
        # Get entries for EN->FR
        en_fr_entries = await cache_repository.get_entries_by_language_pair("en", "fr")
        assert len(en_fr_entries) == 1
        assert en_fr_entries[0].id == en_fr_entry.id
    
    @pytest.mark.asyncio
    async def test_get_entries_by_model_version(self, cache_repository):
        """Test getting cache entries by model version."""
        # Create entries with different model versions
        v1_entry = TranslationCache(
            id=uuid4(),
            content_hash="hash-v1",
            source_language="en",
            target_language="es",
            source_content="Hello",
            translated_content="Hola",
            model_version="v1.0.0"
        )
        
        v2_entry = TranslationCache(
            id=uuid4(),
            content_hash="hash-v2",
            source_language="en",
            target_language="es",
            source_content="Hello",
            translated_content="Hola",
            model_version="v2.0.0"
        )
        
        await cache_repository.create(v1_entry)
        await cache_repository.create(v2_entry)
        
        # Get entries for v1.0.0
        v1_entries = await cache_repository.get_entries_by_model_version("v1.0.0")
        assert len(v1_entries) == 1
        assert v1_entries[0].id == v1_entry.id
        
        # Get entries for v2.0.0
        v2_entries = await cache_repository.get_entries_by_model_version("v2.0.0")
        assert len(v2_entries) == 1
        assert v2_entries[0].id == v2_entry.id
    
    @pytest.mark.asyncio
    async def test_invalidate_by_model_version(self, cache_repository):
        """Test invalidating cache entries by model version."""
        # Create entries with different model versions
        v1_entry = TranslationCache(
            id=uuid4(),
            content_hash="hash-v1",
            source_language="en",
            target_language="es",
            source_content="Hello",
            translated_content="Hola",
            model_version="v1.0.0"
        )
        
        v2_entry = TranslationCache(
            id=uuid4(),
            content_hash="hash-v2",
            source_language="en",
            target_language="es",
            source_content="Hello",
            translated_content="Hola",
            model_version="v2.0.0"
        )
        
        await cache_repository.create(v1_entry)
        await cache_repository.create(v2_entry)
        
        # Invalidate v1.0.0 entries
        deleted_count = await cache_repository.invalidate_by_model_version("v1.0.0")
        assert deleted_count == 1
        
        # Verify v1 entry is deleted, v2 entry remains
        v1_check = await cache_repository.get_by_id(v1_entry.id)
        v2_check = await cache_repository.get_by_id(v2_entry.id)
        
        assert v1_check is None
        assert v2_check is not None
    
    @pytest.mark.asyncio
    async def test_get_low_confidence_entries(self, cache_repository):
        """Test getting cache entries with low confidence scores."""
        # Create entries with different confidence scores
        low_confidence_entry = TranslationCache(
            id=uuid4(),
            content_hash="hash-low",
            source_language="en",
            target_language="es",
            source_content="Ambiguous text",
            translated_content="Texto ambiguo",
            model_version="v1.0.0",
            confidence_score=Decimal('0.6')
        )
        
        high_confidence_entry = TranslationCache(
            id=uuid4(),
            content_hash="hash-high",
            source_language="en",
            target_language="es",
            source_content="Clear text",
            translated_content="Texto claro",
            model_version="v1.0.0",
            confidence_score=Decimal('0.95')
        )
        
        await cache_repository.create(low_confidence_entry)
        await cache_repository.create(high_confidence_entry)
        
        # Get low confidence entries (threshold 0.7)
        low_confidence_entries = await cache_repository.get_low_confidence_entries(0.7)
        assert len(low_confidence_entries) == 1
        assert low_confidence_entries[0].id == low_confidence_entry.id
    
    @pytest.mark.asyncio
    async def test_get_cache_size_stats(self, cache_repository):
        """Test getting cache size statistics."""
        # Create cache entries
        for i in range(3):
            entry = TranslationCache(
                id=uuid4(),
                content_hash=f"hash-{i}",
                source_language="en",
                target_language="es",
                source_content=f"Source content {i}" * 10,  # Make content longer
                translated_content=f"Contenido traducido {i}" * 10,
                model_version="v1.0.0"
            )
            await cache_repository.create(entry)
        
        # Get size statistics
        size_stats = await cache_repository.get_cache_size_stats()
        
        assert size_stats["total_entries"] == 3
        assert size_stats["total_content_size_bytes"] > 0
        assert size_stats["estimated_total_size_bytes"] > size_stats["total_content_size_bytes"]
        assert size_stats["estimated_total_size_mb"] > 0
        assert size_stats["avg_entry_size_bytes"] > 0
        assert size_stats["avg_entry_size_kb"] > 0
    
    @pytest.mark.asyncio
    async def test_cleanup_least_accessed(self, cache_repository):
        """Test cleaning up least accessed cache entries."""
        # Create entries with different access counts
        entries = []
        for i in range(5):
            entry = TranslationCache(
                id=uuid4(),
                content_hash=f"hash-{i}",
                source_language="en",
                target_language="es",
                source_content=f"Content {i}",
                translated_content=f"Contenido {i}",
                model_version="v1.0.0",
                access_count=i + 1  # Access counts: 1, 2, 3, 4, 5
            )
            entries.append(entry)
            await cache_repository.create(entry)
        
        # Keep only top 3 entries
        deleted_count = await cache_repository.cleanup_least_accessed(keep_count=3)
        assert deleted_count == 2  # Should delete 2 least accessed entries
        
        # Verify correct entries were kept/deleted
        remaining_entries = await cache_repository.get_all()
        assert len(remaining_entries) == 3
        
        # The remaining entries should have access counts 3, 4, 5
        access_counts = sorted([entry.access_count for entry in remaining_entries])
        assert access_counts == [3, 4, 5]
    
    @pytest.mark.asyncio
    async def test_warm_cache_for_language_pairs(self, cache_repository):
        """Test warming cache for specific language pairs."""
        # Create entries for different language pairs and model versions
        en_es_v1 = TranslationCache(
            id=uuid4(),
            content_hash="hash-en-es-v1",
            source_language="en",
            target_language="es",
            source_content="Hello",
            translated_content="Hola",
            model_version="v1.0.0",
            access_count=10
        )
        
        en_fr_v1 = TranslationCache(
            id=uuid4(),
            content_hash="hash-en-fr-v1",
            source_language="en",
            target_language="fr",
            source_content="Hello",
            translated_content="Bonjour",
            model_version="v1.0.0",
            access_count=5
        )
        
        en_es_v2 = TranslationCache(
            id=uuid4(),
            content_hash="hash-en-es-v2",
            source_language="en",
            target_language="es",
            source_content="Hello",
            translated_content="Hola",
            model_version="v2.0.0",
            access_count=3
        )
        
        await cache_repository.create(en_es_v1)
        await cache_repository.create(en_fr_v1)
        await cache_repository.create(en_es_v2)
        
        # Warm cache for specific language pairs and model version
        language_pairs = [("en", "es"), ("en", "fr")]
        warm_entries = await cache_repository.warm_cache_for_language_pairs(
            language_pairs, "v1.0.0"
        )
        
        assert len(warm_entries) == 2
        # Should be ordered by access count (descending)
        assert warm_entries[0].access_count >= warm_entries[1].access_count
        
        # Should only include v1.0.0 entries
        for entry in warm_entries:
            assert entry.model_version == "v1.0.0"