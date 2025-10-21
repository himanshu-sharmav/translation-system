"""
Database connection management with connection pooling and transaction support.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from src.config.config import config
from src.utils.logging import TranslationLogger
from src.utils.exceptions import DatabaseError

logger = TranslationLogger(__name__, "database")


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self):
        self._engine = None
        self._session_factory = None
        self._pool = None
        
    async def initialize(self):
        """Initialize database connections and pools."""
        try:
            # Create SQLAlchemy async engine
            self._engine = create_async_engine(
                config.database.connection_string.replace("postgresql://", "postgresql+asyncpg://"),
                pool_size=config.database.pool_size,
                max_overflow=config.database.max_overflow,
                pool_pre_ping=True,
                echo=config.environment.value == "development"
            )
            
            # Create session factory
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create asyncpg connection pool for raw queries
            self._pool = await asyncpg.create_pool(
                config.database.connection_string,
                min_size=5,
                max_size=config.database.pool_size,
                command_timeout=60
            )
            
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {str(e)}", exc_info=True)
            raise DatabaseError(f"Database initialization failed: {str(e)}")
    
    async def close(self):
        """Close all database connections."""
        try:
            if self._pool:
                await self._pool.close()
                
            if self._engine:
                await self._engine.dispose()
                
            logger.info("Database connections closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {str(e)}", exc_info=True)
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session with automatic cleanup."""
        if not self._session_factory:
            raise DatabaseError("Database not initialized")
            
        async with self._session_factory() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {str(e)}", exc_info=True)
                raise DatabaseError(f"Database operation failed: {str(e)}")
            finally:
                await session.close()
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get a raw asyncpg connection for complex queries."""
        if not self._pool:
            raise DatabaseError("Database pool not initialized")
            
        async with self._pool.acquire() as connection:
            try:
                yield connection
            except Exception as e:
                logger.error(f"Database connection error: {str(e)}", exc_info=True)
                raise DatabaseError(f"Database operation failed: {str(e)}")
    
    async def execute_raw_query(self, query: str, *args) -> list:
        """Execute a raw SQL query and return results."""
        async with self.get_connection() as conn:
            try:
                result = await conn.fetch(query, *args)
                return [dict(row) for row in result]
            except Exception as e:
                logger.error(f"Raw query execution failed: {str(e)}", exc_info=True)
                raise DatabaseError(f"Query execution failed: {str(e)}")
    
    async def execute_raw_command(self, command: str, *args) -> str:
        """Execute a raw SQL command and return status."""
        async with self.get_connection() as conn:
            try:
                result = await conn.execute(command, *args)
                return result
            except Exception as e:
                logger.error(f"Raw command execution failed: {str(e)}", exc_info=True)
                raise DatabaseError(f"Command execution failed: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check database connectivity and health."""
        try:
            async with self.get_connection() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False
    
    @property
    def engine(self):
        """Get the SQLAlchemy engine."""
        return self._engine
    
    @property
    def session_factory(self):
        """Get the session factory."""
        return self._session_factory


# Global database manager instance
db_manager = DatabaseManager()


async def init_database():
    """Initialize the database manager."""
    await db_manager.initialize()


async def close_database():
    """Close the database manager."""
    await db_manager.close()


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Convenience function to get a database session."""
    async with db_manager.get_session() as session:
        yield session


@asynccontextmanager
async def get_db_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """Convenience function to get a database connection."""
    async with db_manager.get_connection() as connection:
        yield connection


class TransactionManager:
    """Manages database transactions with rollback support."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self._transaction = None
    
    async def __aenter__(self):
        self._transaction = await self.session.begin()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            await self._transaction.rollback()
            logger.warning(f"Transaction rolled back due to error: {exc_val}")
        else:
            await self._transaction.commit()
        
        self._transaction = None
    
    async def commit(self):
        """Manually commit the transaction."""
        if self._transaction:
            await self._transaction.commit()
    
    async def rollback(self):
        """Manually rollback the transaction."""
        if self._transaction:
            await self._transaction.rollback()


async def with_transaction(session: AsyncSession):
    """Create a transaction manager for the given session."""
    return TransactionManager(session)


# Database utilities
async def execute_migration(migration_sql: str) -> bool:
    """Execute a database migration script."""
    try:
        await db_manager.execute_raw_command(migration_sql)
        logger.info("Migration executed successfully")
        return True
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}", exc_info=True)
        return False


async def check_table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
    try:
        query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = $1
        )
        """
        result = await db_manager.execute_raw_query(query, table_name)
        return result[0]['exists'] if result else False
    except Exception as e:
        logger.error(f"Error checking table existence: {str(e)}")
        return False


async def get_table_row_count(table_name: str) -> int:
    """Get the number of rows in a table."""
    try:
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        result = await db_manager.execute_raw_query(query)
        return result[0]['count'] if result else 0
    except Exception as e:
        logger.error(f"Error getting row count: {str(e)}")
        return 0


async def cleanup_old_records(table_name: str, date_column: str, days_old: int) -> int:
    """Clean up old records from a table."""
    try:
        query = f"""
        DELETE FROM {table_name} 
        WHERE {date_column} < NOW() - INTERVAL '{days_old} days'
        """
        result = await db_manager.execute_raw_command(query)
        # Extract number from result string like "DELETE 5"
        deleted_count = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
        logger.info(f"Cleaned up {deleted_count} old records from {table_name}")
        return deleted_count
    except Exception as e:
        logger.error(f"Error cleaning up old records: {str(e)}")
        return 0