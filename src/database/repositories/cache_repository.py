"""
Repository for translation cache operations.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import TranslationCache
from src.database.repositories.base import BaseRepository, RepositoryMixin
from src.models.interfaces import CacheRepository as ICacheRepository
from src.utils.exceptions import DatabaseError
from src.utils.logging import TranslationLogger

logger = TranslationLogger(__name__, "cache-repository")


class CacheRepository(BaseRepository[TranslationCache], RepositoryMixin, ICacheRepository):
    """Repository for translation cache operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, TranslationCache)
    
    async def get_by_hash(self, content_hash: str, source_lang: str, 
                         target_lang: str, model_version: Optional[str] = None) -> Optional[TranslationCache]:
        """Get cache entry by content hash and languages."""
        try:
            filters = {
                'content_hash': content_hash,
                'source_language': source_lang,
                'target_language': target_lang
            }
            
            if model_version:
                filters['model_version'] = model_version
            
            # Get the most recent entry if model_version is not specified
            stmt = select(self.model_class)
            stmt = self._apply_filters(stmt, filters)
            
            if not model_version:
                stmt = stmt.order_by(desc(self.model_class.created_at))
            
            stmt = stmt.limit(1)
            result = await self.session.execute(stmt)
            return result.scalars().first()
            
        except Exception as e:
            logger.error(f"Error getting cache entry by hash: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get cache entry: {str(e)}")
    
    async def cleanup_expired(self, ttl_hours: int) -> int:
        """Remove expired cache entries."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=ttl_hours)
            
            # Delete entries that haven't been accessed recently and have low access count
            filters = {
                'last_accessed': {'lt': cutoff_time},
                'access_count': {'lt': 2}
            }
            
            deleted_count = await self.bulk_delete(filters)
            
            logger.info(f"Cleaned up {deleted_count} expired cache entries")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired cache entries: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to cleanup expired cache entries: {str(e)}")
    
    async def update_access_stats(self, cache_id: UUID) -> bool:
        """Update access count and last accessed time for a cache entry."""
        try:
            cache_entry = await self.get_by_id(cache_id)
            if not cache_entry:
                return False
            
            cache_entry.access_count += 1
            cache_entry.last_accessed = datetime.utcnow()
            
            await self.session.flush()
            return True
            
        except Exception as e:
            logger.error(f"Error updating cache access stats for {cache_id}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to update cache access stats: {str(e)}")
    
    async def get_cache_statistics(self, days: int = 7) -> Dict[str, any]:
        """Get cache performance statistics."""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            # Total cache entries
            total_entries_query = select(func.count()).select_from(self.model_class)
            total_result = await self.session.execute(total_entries_query)
            total_entries = total_result.scalar() or 0
            
            # Recent cache entries
            recent_entries_query = select(func.count()).where(
                self.model_class.created_at >= start_date
            )
            recent_result = await self.session.execute(recent_entries_query)
            recent_entries = recent_result.scalar() or 0
            
            # Total accesses
            total_accesses_query = select(func.sum(self.model_class.access_count))
            accesses_result = await self.session.execute(total_accesses_query)
            total_accesses = accesses_result.scalar() or 0
            
            # Average access count
            avg_access_query = select(func.avg(self.model_class.access_count))
            avg_result = await self.session.execute(avg_access_query)
            avg_accesses = float(avg_result.scalar() or 0)
            
            # Most accessed entries
            popular_entries_query = select(self.model_class).order_by(
                desc(self.model_class.access_count)
            ).limit(10)
            popular_result = await self.session.execute(popular_entries_query)
            popular_entries = list(popular_result.scalars().all())
            
            # Language pair distribution
            lang_pairs_query = select(
                self.model_class.source_language,
                self.model_class.target_language,
                func.count().label('count')
            ).group_by(
                self.model_class.source_language,
                self.model_class.target_language
            ).order_by(desc('count'))
            
            lang_pairs_result = await self.session.execute(lang_pairs_query)
            language_pairs = [
                {
                    'source_language': row.source_language,
                    'target_language': row.target_language,
                    'count': row.count
                }
                for row in lang_pairs_result
            ]
            
            # Model version distribution
            model_versions_query = select(
                self.model_class.model_version,
                func.count().label('count')
            ).group_by(self.model_class.model_version).order_by(desc('count'))
            
            model_versions_result = await self.session.execute(model_versions_query)
            model_versions = [
                {
                    'model_version': row.model_version,
                    'count': row.count
                }
                for row in model_versions_result
            ]
            
            return {
                'total_entries': total_entries,
                'recent_entries': recent_entries,
                'total_accesses': total_accesses,
                'avg_accesses_per_entry': avg_accesses,
                'popular_entries': [
                    {
                        'id': str(entry.id),
                        'source_language': entry.source_language,
                        'target_language': entry.target_language,
                        'access_count': entry.access_count,
                        'confidence_score': float(entry.confidence_score) if entry.confidence_score else None
                    }
                    for entry in popular_entries
                ],
                'language_pairs': language_pairs,
                'model_versions': model_versions
            }
            
        except Exception as e:
            logger.error(f"Error getting cache statistics: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get cache statistics: {str(e)}")
    
    async def get_entries_by_language_pair(self, source_lang: str, target_lang: str, 
                                         limit: int = 100) -> List[TranslationCache]:
        """Get cache entries for a specific language pair."""
        try:
            filters = {
                'source_language': source_lang,
                'target_language': target_lang
            }
            
            stmt = select(self.model_class)
            stmt = self._apply_filters(stmt, filters)
            stmt = stmt.order_by(desc(self.model_class.last_accessed)).limit(limit)
            
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
            
        except Exception as e:
            logger.error(f"Error getting cache entries by language pair: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get cache entries by language pair: {str(e)}")
    
    async def get_entries_by_model_version(self, model_version: str, 
                                         limit: int = 100) -> List[TranslationCache]:
        """Get cache entries for a specific model version."""
        try:
            filters = {'model_version': model_version}
            return await self.find_by(filters, limit=limit)
        except Exception as e:
            logger.error(f"Error getting cache entries by model version: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get cache entries by model version: {str(e)}")
    
    async def invalidate_by_model_version(self, model_version: str) -> int:
        """Invalidate all cache entries for a specific model version."""
        try:
            filters = {'model_version': model_version}
            deleted_count = await self.bulk_delete(filters)
            
            logger.info(f"Invalidated {deleted_count} cache entries for model version {model_version}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error invalidating cache by model version: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to invalidate cache by model version: {str(e)}")
    
    async def get_low_confidence_entries(self, confidence_threshold: float = 0.7, 
                                       limit: int = 100) -> List[TranslationCache]:
        """Get cache entries with low confidence scores."""
        try:
            filters = {
                'confidence_score': {'lt': confidence_threshold}
            }
            
            stmt = select(self.model_class)
            stmt = self._apply_filters(stmt, filters)
            stmt = stmt.order_by(self.model_class.confidence_score).limit(limit)
            
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
            
        except Exception as e:
            logger.error(f"Error getting low confidence cache entries: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get low confidence cache entries: {str(e)}")
    
    async def get_cache_size_stats(self) -> Dict[str, any]:
        """Get cache size and storage statistics."""
        try:
            # Estimate storage size based on content length
            size_query = select(
                func.sum(func.length(self.model_class.source_content) + 
                        func.length(self.model_class.translated_content)).label('total_content_size'),
                func.avg(func.length(self.model_class.source_content) + 
                        func.length(self.model_class.translated_content)).label('avg_entry_size'),
                func.count().label('total_entries')
            )
            
            result = await self.session.execute(size_query)
            row = result.first()
            
            total_content_size = row.total_content_size or 0
            avg_entry_size = float(row.avg_entry_size or 0)
            total_entries = row.total_entries or 0
            
            # Estimate total storage including metadata (multiply by 1.5 for overhead)
            estimated_total_size_bytes = int(total_content_size * 1.5)
            estimated_total_size_mb = estimated_total_size_bytes / (1024 * 1024)
            
            return {
                'total_entries': total_entries,
                'total_content_size_bytes': total_content_size,
                'estimated_total_size_bytes': estimated_total_size_bytes,
                'estimated_total_size_mb': round(estimated_total_size_mb, 2),
                'avg_entry_size_bytes': int(avg_entry_size),
                'avg_entry_size_kb': round(avg_entry_size / 1024, 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache size stats: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get cache size stats: {str(e)}")
    
    async def cleanup_least_accessed(self, keep_count: int = 10000) -> int:
        """Keep only the most accessed cache entries, remove the rest."""
        try:
            # Get total count
            total_count = await self.count()
            
            if total_count <= keep_count:
                return 0
            
            # Get the access_count threshold for the top entries
            threshold_query = select(self.model_class.access_count).order_by(
                desc(self.model_class.access_count)
            ).offset(keep_count - 1).limit(1)
            
            threshold_result = await self.session.execute(threshold_query)
            threshold_row = threshold_result.first()
            
            if not threshold_row:
                return 0
            
            threshold_access_count = threshold_row.access_count
            
            # Delete entries with access count below threshold
            filters = {
                'access_count': {'lt': threshold_access_count}
            }
            
            deleted_count = await self.bulk_delete(filters)
            
            logger.info(f"Cleaned up {deleted_count} least accessed cache entries, keeping top {keep_count}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up least accessed entries: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to cleanup least accessed entries: {str(e)}")
    
    async def warm_cache_for_language_pairs(self, language_pairs: List[tuple], 
                                          model_version: str) -> List[TranslationCache]:
        """Get existing cache entries for warming cache with specific language pairs."""
        try:
            # Build OR conditions for language pairs
            conditions = []
            for source_lang, target_lang in language_pairs:
                conditions.append(
                    and_(
                        self.model_class.source_language == source_lang,
                        self.model_class.target_language == target_lang,
                        self.model_class.model_version == model_version
                    )
                )
            
            if not conditions:
                return []
            
            from sqlalchemy import or_
            stmt = select(self.model_class).where(or_(*conditions)).order_by(
                desc(self.model_class.access_count)
            )
            
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
            
        except Exception as e:
            logger.error(f"Error warming cache for language pairs: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to warm cache for language pairs: {str(e)}")