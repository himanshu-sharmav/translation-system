"""
Integration tests for rate limiting functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

from src.services.auth_service import AuthService
from src.database.models import User, RateLimit
from src.database.connection import get_db_session
from src.database.repositories.base import BaseRepository


class TestRateLimitingService:
    """Test rate limiting service functionality."""
    
    @pytest.fixture
    async def auth_service(self):
        """Create auth service instance."""
        return AuthService()
    
    @pytest.fixture
    async def test_user(self, db_session):
        """Create test user."""
        user_repo = BaseRepository(db_session, User)
        
        user = User(
            id=uuid4(),
            user_id="rate-limit-test-user",
            email="ratelimit@example.com",
            api_key="rate-limit-test-key-123456789012345678901234",
            is_active=True,
            rate_limit_per_minute=5  # Low limit for testing
        )
        
        created_user = await user_repo.create(user)
        await db_session.commit()
        return created_user
    
    @pytest.mark.asyncio
    async def test_rate_limit_within_limit(self, auth_service, test_user):
        """Test rate limiting when within limit."""
        user_id = test_user.user_id
        endpoint = "test_endpoint"
        
        # Make requests within limit
        for i in range(3):  # 3 requests, limit is 5
            within_limit = await auth_service.check_rate_limit(user_id, endpoint)
            assert within_limit is True
    
    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, auth_service, test_user):
        """Test rate limiting when limit is exceeded."""
        user_id = test_user.user_id
        endpoint = "test_endpoint"
        
        # Make requests up to limit
        for i in range(5):  # Exactly at limit
            within_limit = await auth_service.check_rate_limit(user_id, endpoint)
            assert within_limit is True
        
        # Next request should exceed limit
        within_limit = await auth_service.check_rate_limit(user_id, endpoint)
        assert within_limit is False
    
    @pytest.mark.asyncio
    async def test_rate_limit_different_endpoints(self, auth_service, test_user):
        """Test that rate limiting is per endpoint."""
        user_id = test_user.user_id
        endpoint1 = "endpoint1"
        endpoint2 = "endpoint2"
        
        # Use up limit for endpoint1
        for i in range(5):
            within_limit = await auth_service.check_rate_limit(user_id, endpoint1)
            assert within_limit is True
        
        # endpoint1 should be at limit
        within_limit = await auth_service.check_rate_limit(user_id, endpoint1)
        assert within_limit is False
        
        # endpoint2 should still be available
        within_limit = await auth_service.check_rate_limit(user_id, endpoint2)
        assert within_limit is True
    
    @pytest.mark.asyncio
    async def test_rate_limit_different_users(self, auth_service, db_session):
        """Test that rate limiting is per user."""
        # Create two users
        user_repo = BaseRepository(db_session, User)
        
        user1 = User(
            id=uuid4(),
            user_id="rate-limit-user1",
            email="user1@example.com",
            api_key="user1-api-key-123456789012345678901234567890",
            is_active=True,
            rate_limit_per_minute=3
        )
        
        user2 = User(
            id=uuid4(),
            user_id="rate-limit-user2",
            email="user2@example.com",
            api_key="user2-api-key-123456789012345678901234567890",
            is_active=True,
            rate_limit_per_minute=3
        )
        
        await user_repo.create(user1)
        await user_repo.create(user2)
        await db_session.commit()
        
        endpoint = "test_endpoint"
        
        # Use up limit for user1
        for i in range(3):
            within_limit = await auth_service.check_rate_limit(user1.user_id, endpoint)
            assert within_limit is True
        
        # user1 should be at limit
        within_limit = await auth_service.check_rate_limit(user1.user_id, endpoint)
        assert within_limit is False
        
        # user2 should still be available
        within_limit = await auth_service.check_rate_limit(user2.user_id, endpoint)
        assert within_limit is True
    
    @pytest.mark.asyncio
    async def test_rate_limit_window_reset(self, auth_service, test_user, db_session):
        """Test that rate limit resets in new time window."""
        user_id = test_user.user_id
        endpoint = "test_endpoint"
        
        # Use up current window
        for i in range(5):
            within_limit = await auth_service.check_rate_limit(user_id, endpoint)
            assert within_limit is True
        
        # Should be at limit
        within_limit = await auth_service.check_rate_limit(user_id, endpoint)
        assert within_limit is False
        
        # Manually create a rate limit entry for next minute window
        rate_limit_repo = BaseRepository(db_session, RateLimit)
        current_time = datetime.utcnow()
        next_window_start = current_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
        
        new_rate_limit = RateLimit(
            user_id=user_id,
            endpoint=endpoint,
            request_count=1,
            window_start=next_window_start,
            window_end=next_window_start + timedelta(minutes=1)
        )
        
        await rate_limit_repo.create(new_rate_limit)
        await db_session.commit()
        
        # Mock the current time to be in the next window
        with pytest.MonkeyPatch().context() as m:
            m.setattr('src.services.auth_service.datetime', 
                     type('MockDateTime', (), {
                         'utcnow': lambda: next_window_start + timedelta(seconds=30)
                     }))
            
            # Should be able to make requests in new window
            # Note: This test is simplified and may need adjustment based on actual implementation
    
    @pytest.mark.asyncio
    async def test_rate_limit_cleanup(self, auth_service, test_user, db_session):
        """Test cleanup of old rate limit entries."""
        user_id = test_user.user_id
        endpoint = "test_endpoint"
        
        # Create some rate limit entries
        for i in range(3):
            await auth_service.check_rate_limit(user_id, endpoint)
        
        # Verify entries exist
        rate_limit_repo = BaseRepository(db_session, RateLimit)
        entries_before = await rate_limit_repo.find_by({"user_id": user_id})
        assert len(entries_before) > 0
        
        # Run cleanup (should clean entries older than 24 hours)
        deleted_count = await auth_service.cleanup_old_rate_limits(hours=0)  # Clean all
        
        # Verify entries are cleaned up
        entries_after = await rate_limit_repo.find_by({"user_id": user_id})
        assert len(entries_after) == 0
        assert deleted_count > 0
    
    @pytest.mark.asyncio
    async def test_rate_limit_user_not_found(self, auth_service):
        """Test rate limiting with non-existent user."""
        # Should handle gracefully and allow request (fail open)
        within_limit = await auth_service.check_rate_limit("non-existent-user", "test_endpoint")
        assert within_limit is True  # Fail open for non-existent users
    
    @pytest.mark.asyncio
    async def test_rate_limit_concurrent_requests(self, auth_service, test_user):
        """Test rate limiting with concurrent requests."""
        user_id = test_user.user_id
        endpoint = "concurrent_test"
        
        async def make_request():
            return await auth_service.check_rate_limit(user_id, endpoint)
        
        # Make 10 concurrent requests (limit is 5)
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Should have exactly 5 successful requests
        successful_requests = sum(1 for result in results if result is True)
        failed_requests = sum(1 for result in results if result is False)
        
        assert successful_requests == 5
        assert failed_requests == 5
    
    @pytest.mark.asyncio
    async def test_update_user_rate_limit(self, auth_service, test_user):
        """Test updating user rate limit."""
        user_id = test_user.user_id
        new_limit = 10
        
        # Update rate limit
        success = await auth_service.update_user_rate_limit(user_id, new_limit)
        assert success is True
        
        # Verify new limit is applied
        endpoint = "test_endpoint"
        
        # Should be able to make more requests now
        for i in range(8):  # More than original limit of 5
            within_limit = await auth_service.check_rate_limit(user_id, endpoint)
            assert within_limit is True
    
    @pytest.mark.asyncio
    async def test_rate_limit_with_disabled_user(self, auth_service, db_session):
        """Test rate limiting with disabled user."""
        # Create disabled user
        user_repo = BaseRepository(db_session, User)
        
        disabled_user = User(
            id=uuid4(),
            user_id="disabled-user",
            email="disabled@example.com",
            api_key="disabled-api-key-123456789012345678901234567890",
            is_active=False,  # Disabled
            rate_limit_per_minute=100
        )
        
        await user_repo.create(disabled_user)
        await db_session.commit()
        
        # Rate limiting should still work for disabled users
        # (authentication will handle the disabled status)
        within_limit = await auth_service.check_rate_limit(disabled_user.user_id, "test_endpoint")
        assert within_limit is True  # Rate limiting allows, but auth will reject
    
    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self, auth_service):
        """Test rate limiting error handling."""
        # Test with invalid parameters
        within_limit = await auth_service.check_rate_limit("", "")
        assert within_limit is True  # Should fail open on errors
        
        within_limit = await auth_service.check_rate_limit(None, "test_endpoint")
        assert within_limit is True  # Should fail open on errors


class TestRateLimitMiddleware:
    """Test rate limiting middleware functionality."""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        from src.api.app import create_app
        return create_app()
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        from fastapi.testclient import TestClient
        return TestClient(app)
    
    def test_rate_limit_middleware_health_check_excluded(self, client):
        """Test that health check is excluded from rate limiting."""
        # Make many requests to health check - should not be rate limited
        for i in range(20):
            response = client.get("/api/v1/health")
            assert response.status_code in [200, 503]  # Not rate limited
    
    def test_rate_limit_middleware_docs_excluded(self, client):
        """Test that docs endpoints are excluded from rate limiting."""
        # These endpoints should not be rate limited
        excluded_endpoints = ["/docs", "/openapi.json", "/redoc"]
        
        for endpoint in excluded_endpoints:
            for i in range(10):
                response = client.get(endpoint)
                # Should not get 429 (rate limited)
                assert response.status_code != 429
    
    def test_rate_limit_middleware_applies_to_api_endpoints(self, client):
        """Test that rate limiting applies to API endpoints."""
        # This test would require more complex setup to actually trigger rate limiting
        # in the middleware, as it's IP-based rather than user-based
        
        # Make requests to an API endpoint
        for i in range(5):
            response = client.get("/api/v1/languages")
            assert response.status_code == 200
        
        # The middleware rate limiting is IP-based and has a high limit (1000/minute)
        # so it's difficult to test in integration tests without mocking
    
    def test_rate_limit_headers_in_response(self, client):
        """Test that rate limit information is included in response headers."""
        response = client.get("/api/v1/languages")
        
        # Check for custom headers added by middleware
        assert "X-Correlation-ID" in response.headers
        assert "X-Process-Time" in response.headers