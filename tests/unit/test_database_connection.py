"""
Unit tests for database connection management.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.database.connection import DatabaseManager, TransactionManager
from src.utils.exceptions import DatabaseError


class TestDatabaseManager:
    """Test DatabaseManager functionality."""
    
    @pytest.mark.asyncio
    async def test_database_manager_initialization(self):
        """Test database manager initialization."""
        db_manager = DatabaseManager()
        
        assert db_manager._engine is None
        assert db_manager._session_factory is None
        assert db_manager._pool is None
    
    @pytest.mark.asyncio
    @patch('src.database.connection.create_async_engine')
    @patch('src.database.connection.async_sessionmaker')
    @patch('src.database.connection.asyncpg.create_pool')
    async def test_initialize_success(self, mock_create_pool, mock_sessionmaker, mock_create_engine):
        """Test successful database initialization."""
        # Setup mocks
        mock_engine = AsyncMock()
        mock_session_factory = MagicMock()
        mock_pool = AsyncMock()
        
        mock_create_engine.return_value = mock_engine
        mock_sessionmaker.return_value = mock_session_factory
        mock_create_pool.return_value = mock_pool
        
        # Initialize database manager
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        # Verify initialization
        assert db_manager._engine == mock_engine
        assert db_manager._session_factory == mock_session_factory
        assert db_manager._pool == mock_pool
        
        # Verify method calls
        mock_create_engine.assert_called_once()
        mock_sessionmaker.assert_called_once()
        mock_create_pool.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('src.database.connection.create_async_engine')
    async def test_initialize_failure(self, mock_create_engine):
        """Test database initialization failure."""
        # Setup mock to raise exception
        mock_create_engine.side_effect = Exception("Connection failed")
        
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Should raise DatabaseError
        with pytest.raises(DatabaseError) as exc_info:
            await db_manager.initialize()
        
        assert "Database initialization failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_close_database(self):
        """Test closing database connections."""
        db_manager = DatabaseManager()
        
        # Setup mocks
        mock_pool = AsyncMock()
        mock_engine = AsyncMock()
        
        db_manager._pool = mock_pool
        db_manager._engine = mock_engine
        
        # Close database
        await db_manager.close()
        
        # Verify close methods were called
        mock_pool.close.assert_called_once()
        mock_engine.dispose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_database_with_error(self):
        """Test closing database with error handling."""
        db_manager = DatabaseManager()
        
        # Setup mock to raise exception
        mock_pool = AsyncMock()
        mock_pool.close.side_effect = Exception("Close failed")
        
        db_manager._pool = mock_pool
        
        # Should not raise exception, just log error
        await db_manager.close()
        
        # Verify close was attempted
        mock_pool.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_session_not_initialized(self):
        """Test getting session when not initialized."""
        db_manager = DatabaseManager()
        
        with pytest.raises(DatabaseError) as exc_info:
            async with db_manager.get_session():
                pass
        
        assert "Database not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_connection_not_initialized(self):
        """Test getting connection when not initialized."""
        db_manager = DatabaseManager()
        
        with pytest.raises(DatabaseError) as exc_info:
            async with db_manager.get_connection():
                pass
        
        assert "Database pool not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_execute_raw_query(self):
        """Test executing raw SQL query."""
        db_manager = DatabaseManager()
        
        # Setup mock connection
        mock_connection = AsyncMock()
        mock_row = MagicMock()
        mock_row.__iter__ = MagicMock(return_value=iter([('col1', 'value1'), ('col2', 'value2')]))
        mock_connection.fetch.return_value = [mock_row]
        
        # Setup mock pool
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        
        db_manager._pool = mock_pool
        
        # Execute query
        result = await db_manager.execute_raw_query("SELECT * FROM test")
        
        # Verify result
        assert len(result) == 1
        mock_connection.fetch.assert_called_once_with("SELECT * FROM test")
    
    @pytest.mark.asyncio
    async def test_execute_raw_command(self):
        """Test executing raw SQL command."""
        db_manager = DatabaseManager()
        
        # Setup mock connection
        mock_connection = AsyncMock()
        mock_connection.execute.return_value = "INSERT 0 1"
        
        # Setup mock pool
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        
        db_manager._pool = mock_pool
        
        # Execute command
        result = await db_manager.execute_raw_command("INSERT INTO test VALUES (1)")
        
        # Verify result
        assert result == "INSERT 0 1"
        mock_connection.execute.assert_called_once_with("INSERT INTO test VALUES (1)")
    
    @pytest.mark.asyncio
    async def test_health_check_success(self):
        """Test successful database health check."""
        db_manager = DatabaseManager()
        
        # Setup mock connection
        mock_connection = AsyncMock()
        mock_connection.fetchval.return_value = 1
        
        # Setup mock pool
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        
        db_manager._pool = mock_pool
        
        # Perform health check
        is_healthy = await db_manager.health_check()
        
        # Verify result
        assert is_healthy is True
        mock_connection.fetchval.assert_called_once_with("SELECT 1")
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test failed database health check."""
        db_manager = DatabaseManager()
        
        # Setup mock connection to raise exception
        mock_connection = AsyncMock()
        mock_connection.fetchval.side_effect = Exception("Connection failed")
        
        # Setup mock pool
        mock_pool = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
        mock_pool.acquire.return_value.__aexit__.return_value = None
        
        db_manager._pool = mock_pool
        
        # Perform health check
        is_healthy = await db_manager.health_check()
        
        # Verify result
        assert is_healthy is False


class TestTransactionManager:
    """Test TransactionManager functionality."""
    
    @pytest.mark.asyncio
    async def test_transaction_manager_success(self):
        """Test successful transaction management."""
        # Setup mock session and transaction
        mock_transaction = AsyncMock()
        mock_session = AsyncMock()
        mock_session.begin.return_value = mock_transaction
        
        # Use transaction manager
        async with TransactionManager(mock_session) as tx_manager:
            # Simulate some work
            pass
        
        # Verify transaction was committed
        mock_session.begin.assert_called_once()
        mock_transaction.commit.assert_called_once()
        mock_transaction.rollback.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_transaction_manager_rollback_on_exception(self):
        """Test transaction rollback on exception."""
        # Setup mock session and transaction
        mock_transaction = AsyncMock()
        mock_session = AsyncMock()
        mock_session.begin.return_value = mock_transaction
        
        # Use transaction manager with exception
        with pytest.raises(ValueError):
            async with TransactionManager(mock_session) as tx_manager:
                raise ValueError("Test exception")
        
        # Verify transaction was rolled back
        mock_session.begin.assert_called_once()
        mock_transaction.rollback.assert_called_once()
        mock_transaction.commit.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_transaction_manager_manual_commit(self):
        """Test manual transaction commit."""
        # Setup mock session and transaction
        mock_transaction = AsyncMock()
        mock_session = AsyncMock()
        mock_session.begin.return_value = mock_transaction
        
        # Use transaction manager with manual commit
        async with TransactionManager(mock_session) as tx_manager:
            await tx_manager.commit()
        
        # Verify transaction was committed twice (manual + automatic)
        mock_session.begin.assert_called_once()
        assert mock_transaction.commit.call_count == 2
    
    @pytest.mark.asyncio
    async def test_transaction_manager_manual_rollback(self):
        """Test manual transaction rollback."""
        # Setup mock session and transaction
        mock_transaction = AsyncMock()
        mock_session = AsyncMock()
        mock_session.begin.return_value = mock_transaction
        
        # Use transaction manager with manual rollback
        async with TransactionManager(mock_session) as tx_manager:
            await tx_manager.rollback()
        
        # Verify transaction was rolled back
        mock_session.begin.assert_called_once()
        mock_transaction.rollback.assert_called_once()
        # Commit should still be called at the end, but transaction is already rolled back
        mock_transaction.commit.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions."""
    
    @pytest.mark.asyncio
    @patch('src.database.connection.db_manager')
    async def test_execute_migration_success(self, mock_db_manager):
        """Test successful migration execution."""
        from src.database.connection import execute_migration
        
        # Setup mock
        mock_db_manager.execute_raw_command.return_value = "CREATE TABLE"
        
        # Execute migration
        result = await execute_migration("CREATE TABLE test (id INT)")
        
        # Verify result
        assert result is True
        mock_db_manager.execute_raw_command.assert_called_once_with("CREATE TABLE test (id INT)")
    
    @pytest.mark.asyncio
    @patch('src.database.connection.db_manager')
    async def test_execute_migration_failure(self, mock_db_manager):
        """Test failed migration execution."""
        from src.database.connection import execute_migration
        
        # Setup mock to raise exception
        mock_db_manager.execute_raw_command.side_effect = Exception("Migration failed")
        
        # Execute migration
        result = await execute_migration("INVALID SQL")
        
        # Verify result
        assert result is False
    
    @pytest.mark.asyncio
    @patch('src.database.connection.db_manager')
    async def test_check_table_exists_true(self, mock_db_manager):
        """Test checking if table exists (returns True)."""
        from src.database.connection import check_table_exists
        
        # Setup mock
        mock_db_manager.execute_raw_query.return_value = [{'exists': True}]
        
        # Check table existence
        exists = await check_table_exists("test_table")
        
        # Verify result
        assert exists is True
    
    @pytest.mark.asyncio
    @patch('src.database.connection.db_manager')
    async def test_check_table_exists_false(self, mock_db_manager):
        """Test checking if table exists (returns False)."""
        from src.database.connection import check_table_exists
        
        # Setup mock
        mock_db_manager.execute_raw_query.return_value = [{'exists': False}]
        
        # Check table existence
        exists = await check_table_exists("nonexistent_table")
        
        # Verify result
        assert exists is False
    
    @pytest.mark.asyncio
    @patch('src.database.connection.db_manager')
    async def test_get_table_row_count(self, mock_db_manager):
        """Test getting table row count."""
        from src.database.connection import get_table_row_count
        
        # Setup mock
        mock_db_manager.execute_raw_query.return_value = [{'count': 42}]
        
        # Get row count
        count = await get_table_row_count("test_table")
        
        # Verify result
        assert count == 42
    
    @pytest.mark.asyncio
    @patch('src.database.connection.db_manager')
    async def test_cleanup_old_records(self, mock_db_manager):
        """Test cleaning up old records."""
        from src.database.connection import cleanup_old_records
        
        # Setup mock
        mock_db_manager.execute_raw_command.return_value = "DELETE 5"
        
        # Cleanup old records
        deleted_count = await cleanup_old_records("test_table", "created_at", 30)
        
        # Verify result
        assert deleted_count == 5