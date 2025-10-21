"""
Base repository implementation with common CRUD operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import Select

from src.database.connection import Base
from src.utils.exceptions import DatabaseError
from src.utils.logging import TranslationLogger

logger = TranslationLogger(__name__, "repository")

# Type variable for model classes
ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType], ABC):
    """Abstract base repository with common CRUD operations."""
    
    def __init__(self, session: AsyncSession, model_class: Type[ModelType]):
        self.session = session
        self.model_class = model_class
    
    async def create(self, entity: ModelType) -> ModelType:
        """Create a new entity."""
        try:
            self.session.add(entity)
            await self.session.flush()
            await self.session.refresh(entity)
            return entity
        except Exception as e:
            logger.error(f"Error creating {self.model_class.__name__}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to create {self.model_class.__name__}: {str(e)}")
    
    async def get_by_id(self, entity_id: Union[UUID, str, int]) -> Optional[ModelType]:
        """Get entity by ID."""
        try:
            result = await self.session.get(self.model_class, entity_id)
            return result
        except Exception as e:
            logger.error(f"Error getting {self.model_class.__name__} by ID {entity_id}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get {self.model_class.__name__}: {str(e)}")
    
    async def update(self, entity: ModelType) -> bool:
        """Update existing entity."""
        try:
            await self.session.merge(entity)
            await self.session.flush()
            return True
        except Exception as e:
            logger.error(f"Error updating {self.model_class.__name__}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to update {self.model_class.__name__}: {str(e)}")
    
    async def delete(self, entity_id: Union[UUID, str, int]) -> bool:
        """Delete entity by ID."""
        try:
            stmt = delete(self.model_class).where(self.model_class.id == entity_id)
            result = await self.session.execute(stmt)
            return result.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting {self.model_class.__name__} {entity_id}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to delete {self.model_class.__name__}: {str(e)}")
    
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[ModelType]:
        """Get all entities with pagination."""
        try:
            stmt = select(self.model_class).limit(limit).offset(offset)
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error getting all {self.model_class.__name__}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get {self.model_class.__name__} list: {str(e)}")
    
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities with optional filters."""
        try:
            stmt = select(self.model_class)
            if filters:
                stmt = self._apply_filters(stmt, filters)
            
            # Use func.count() for counting
            from sqlalchemy import func
            count_stmt = select(func.count()).select_from(stmt.subquery())
            result = await self.session.execute(count_stmt)
            return result.scalar() or 0
        except Exception as e:
            logger.error(f"Error counting {self.model_class.__name__}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to count {self.model_class.__name__}: {str(e)}")
    
    async def find_by(self, filters: Dict[str, Any], limit: int = 100, offset: int = 0) -> List[ModelType]:
        """Find entities by filters."""
        try:
            stmt = select(self.model_class).limit(limit).offset(offset)
            stmt = self._apply_filters(stmt, filters)
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error finding {self.model_class.__name__} by filters: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to find {self.model_class.__name__}: {str(e)}")
    
    async def find_one_by(self, filters: Dict[str, Any]) -> Optional[ModelType]:
        """Find single entity by filters."""
        try:
            stmt = select(self.model_class)
            stmt = self._apply_filters(stmt, filters)
            result = await self.session.execute(stmt)
            return result.scalars().first()
        except Exception as e:
            logger.error(f"Error finding one {self.model_class.__name__} by filters: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to find {self.model_class.__name__}: {str(e)}")
    
    async def exists(self, filters: Dict[str, Any]) -> bool:
        """Check if entity exists with given filters."""
        try:
            stmt = select(self.model_class)
            stmt = self._apply_filters(stmt, filters)
            result = await self.session.execute(stmt)
            return result.scalars().first() is not None
        except Exception as e:
            logger.error(f"Error checking {self.model_class.__name__} existence: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to check {self.model_class.__name__} existence: {str(e)}")
    
    async def bulk_create(self, entities: List[ModelType]) -> List[ModelType]:
        """Create multiple entities in bulk."""
        try:
            self.session.add_all(entities)
            await self.session.flush()
            for entity in entities:
                await self.session.refresh(entity)
            return entities
        except Exception as e:
            logger.error(f"Error bulk creating {self.model_class.__name__}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to bulk create {self.model_class.__name__}: {str(e)}")
    
    async def bulk_update(self, filters: Dict[str, Any], values: Dict[str, Any]) -> int:
        """Update multiple entities in bulk."""
        try:
            stmt = update(self.model_class).values(**values)
            stmt = self._apply_filters_to_update(stmt, filters)
            result = await self.session.execute(stmt)
            return result.rowcount
        except Exception as e:
            logger.error(f"Error bulk updating {self.model_class.__name__}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to bulk update {self.model_class.__name__}: {str(e)}")
    
    async def bulk_delete(self, filters: Dict[str, Any]) -> int:
        """Delete multiple entities in bulk."""
        try:
            stmt = delete(self.model_class)
            stmt = self._apply_filters_to_delete(stmt, filters)
            result = await self.session.execute(stmt)
            return result.rowcount
        except Exception as e:
            logger.error(f"Error bulk deleting {self.model_class.__name__}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to bulk delete {self.model_class.__name__}: {str(e)}")
    
    def _apply_filters(self, stmt: Select, filters: Dict[str, Any]) -> Select:
        """Apply filters to a select statement."""
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                column = getattr(self.model_class, key)
                if isinstance(value, list):
                    stmt = stmt.where(column.in_(value))
                elif isinstance(value, dict):
                    # Handle range queries like {'gte': 10, 'lte': 20}
                    if 'gte' in value:
                        stmt = stmt.where(column >= value['gte'])
                    if 'lte' in value:
                        stmt = stmt.where(column <= value['lte'])
                    if 'gt' in value:
                        stmt = stmt.where(column > value['gt'])
                    if 'lt' in value:
                        stmt = stmt.where(column < value['lt'])
                    if 'ne' in value:
                        stmt = stmt.where(column != value['ne'])
                    if 'like' in value:
                        stmt = stmt.where(column.like(value['like']))
                    if 'ilike' in value:
                        stmt = stmt.where(column.ilike(value['ilike']))
                else:
                    stmt = stmt.where(column == value)
        return stmt
    
    def _apply_filters_to_update(self, stmt, filters: Dict[str, Any]):
        """Apply filters to an update statement."""
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                column = getattr(self.model_class, key)
                if isinstance(value, list):
                    stmt = stmt.where(column.in_(value))
                else:
                    stmt = stmt.where(column == value)
        return stmt
    
    def _apply_filters_to_delete(self, stmt, filters: Dict[str, Any]):
        """Apply filters to a delete statement."""
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                column = getattr(self.model_class, key)
                if isinstance(value, list):
                    stmt = stmt.where(column.in_(value))
                else:
                    stmt = stmt.where(column == value)
        return stmt
    
    async def get_with_relations(self, entity_id: Union[UUID, str, int], relations: List[str]) -> Optional[ModelType]:
        """Get entity with specified relations loaded."""
        try:
            stmt = select(self.model_class).where(self.model_class.id == entity_id)
            
            # Add selectinload for each relation
            for relation in relations:
                if hasattr(self.model_class, relation):
                    stmt = stmt.options(selectinload(getattr(self.model_class, relation)))
            
            result = await self.session.execute(stmt)
            return result.scalars().first()
        except Exception as e:
            logger.error(f"Error getting {self.model_class.__name__} with relations: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to get {self.model_class.__name__} with relations: {str(e)}")
    
    async def paginate(self, page: int = 1, per_page: int = 20, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Paginate results with metadata."""
        try:
            offset = (page - 1) * per_page
            
            # Get total count
            total = await self.count(filters)
            
            # Get items for current page
            items = await self.find_by(filters or {}, limit=per_page, offset=offset)
            
            # Calculate pagination metadata
            total_pages = (total + per_page - 1) // per_page
            has_prev = page > 1
            has_next = page < total_pages
            
            return {
                'items': items,
                'total': total,
                'page': page,
                'per_page': per_page,
                'total_pages': total_pages,
                'has_prev': has_prev,
                'has_next': has_next,
                'prev_page': page - 1 if has_prev else None,
                'next_page': page + 1 if has_next else None
            }
        except Exception as e:
            logger.error(f"Error paginating {self.model_class.__name__}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to paginate {self.model_class.__name__}: {str(e)}")


class RepositoryMixin:
    """Mixin class providing common repository functionality."""
    
    async def soft_delete(self, entity_id: Union[UUID, str, int]) -> bool:
        """Soft delete an entity (if it has deleted_at field)."""
        if not hasattr(self.model_class, 'deleted_at'):
            raise NotImplementedError("Model does not support soft delete")
        
        try:
            from datetime import datetime
            stmt = update(self.model_class).where(
                self.model_class.id == entity_id
            ).values(deleted_at=datetime.utcnow())
            
            result = await self.session.execute(stmt)
            return result.rowcount > 0
        except Exception as e:
            logger.error(f"Error soft deleting {self.model_class.__name__}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to soft delete {self.model_class.__name__}: {str(e)}")
    
    async def restore(self, entity_id: Union[UUID, str, int]) -> bool:
        """Restore a soft-deleted entity."""
        if not hasattr(self.model_class, 'deleted_at'):
            raise NotImplementedError("Model does not support soft delete")
        
        try:
            stmt = update(self.model_class).where(
                self.model_class.id == entity_id
            ).values(deleted_at=None)
            
            result = await self.session.execute(stmt)
            return result.rowcount > 0
        except Exception as e:
            logger.error(f"Error restoring {self.model_class.__name__}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to restore {self.model_class.__name__}: {str(e)}")
    
    async def get_active_only(self, filters: Optional[Dict[str, Any]] = None) -> List[ModelType]:
        """Get only non-deleted entities (if model supports soft delete)."""
        if hasattr(self.model_class, 'deleted_at'):
            filters = filters or {}
            filters['deleted_at'] = None
        
        return await self.find_by(filters or {})
    
    async def search(self, query: str, fields: List[str], limit: int = 100) -> List[ModelType]:
        """Search entities by text in specified fields."""
        try:
            stmt = select(self.model_class)
            
            # Build OR conditions for each field
            conditions = []
            for field in fields:
                if hasattr(self.model_class, field):
                    column = getattr(self.model_class, field)
                    conditions.append(column.ilike(f'%{query}%'))
            
            if conditions:
                from sqlalchemy import or_
                stmt = stmt.where(or_(*conditions))
            
            stmt = stmt.limit(limit)
            result = await self.session.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error searching {self.model_class.__name__}: {str(e)}", exc_info=True)
            raise DatabaseError(f"Failed to search {self.model_class.__name__}: {str(e)}")