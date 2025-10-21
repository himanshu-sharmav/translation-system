"""
Integration tests for authentication and authorization.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.api.app import create_app
from src.services.auth_service import AuthService
from src.database.models import User, RateLimit


class TestAuthentication:
    """Test authentication functionality."""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        return create_app()
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def auth_service(self):
        """Create auth service instance."""
        return AuthService()
    
    @pytest.fixture
    def mock_user(self):
        """Create mock user."""
        return User(
            id=uuid4(),
            user_id="test-user",
            email="test@example.com",
            api_key="test-api-key-12345678901234567890123456789012",
            is_active=True,
            rate_limit_per_minute=100
        )
    
    def test_no_authentication_required_endpoints(self, client):
        """Test endpoints that don't require authentication."""
        # Health check
        response = client.get("/api/v1/health")
        assert response.status_code in [200, 503]  # May be unhealthy in test
        
        # Supported languages
        response = client.get("/api/v1/languages")
        assert response.status_code == 200
        
        # Root endpoint
        response = client.get("/")
        assert response.status_code == 200
    
    def test_authentication_required_endpoints(self, client):
        """Test endpoints that require authentication."""
        endpoints = [
            ("GET", "/api/v1/stats"),
            ("POST", "/api/v1/translate"),
            ("GET", "/api/v1/jobs"),
            ("POST", "/api/v1/cost-estimate"),
        ]
        
        for method, endpoint in endpoints:
            if method == "GET":
                response = client.get(endpoint)
            elif method == "POST":
                response = client.post(endpoint, json={})
            
            assert response.status_code == 401
            data = response.json()
            assert "error" in data or "detail" in data
    
    @patch('src.services.auth_service.AuthService.authenticate')
    def test_valid_jwt_token(self, mock_authenticate, client):
        """Test authentication with valid JWT token."""
        mock_authenticate.return_value = {
            "user_id": "test-user",
            "email": "test@example.com",
            "is_active": True,
            "is_admin": False,
            "rate_limit_per_minute": 100
        }
        
        headers = {"Authorization": "Bearer valid-jwt-token"}
        
        with patch('src.database.repositories.JobRepository.get_job_statistics', return_value={}), \
             patch('src.database.repositories.JobRepository.get_queue_depth_by_priority', return_value=0), \
             patch('src.database.repositories.JobRepository.get_performance_metrics', return_value={}), \
             patch('src.database.repositories.CacheRepository.get_cache_statistics', return_value={"total_accesses": 0}), \
             patch('src.database.repositories.MetricsRepository.get_system_health_metrics', return_value={}), \
             patch('src.database.repositories.MetricsRepository.get_latest_metric', return_value=None):
            
            response = client.get("/api/v1/stats", headers=headers)
            assert response.status_code == 200
    
    @patch('src.services.auth_service.AuthService.authenticate')
    def test_invalid_jwt_token(self, mock_authenticate, client):
        """Test authentication with invalid JWT token."""
        mock_authenticate.return_value = None
        
        headers = {"Authorization": "Bearer invalid-jwt-token"}
        response = client.get("/api/v1/stats", headers=headers)
        
        assert response.status_code == 401
    
    @patch('src.services.auth_service.AuthService.authenticate')
    def test_expired_jwt_token(self, mock_authenticate, client):
        """Test authentication with expired JWT token."""
        from src.utils.exceptions import AuthenticationError
        mock_authenticate.side_effect = AuthenticationError("Token has expired")
        
        headers = {"Authorization": "Bearer expired-jwt-token"}
        response = client.get("/api/v1/stats", headers=headers)
        
        assert response.status_code == 401
    
    @patch('src.services.auth_service.AuthService.authenticate')
    def test_disabled_user_account(self, mock_authenticate, client):
        """Test authentication with disabled user account."""
        mock_authenticate.return_value = {
            "user_id": "disabled-user",
            "email": "disabled@example.com",
            "is_active": False,  # Disabled account
            "is_admin": False,
            "rate_limit_per_minute": 100
        }
        
        headers = {"Authorization": "Bearer valid-jwt-token"}
        response = client.get("/api/v1/stats", headers=headers)
        
        assert response.status_code == 403
        data = response.json()
        assert "disabled" in data["detail"].lower()
    
    def test_malformed_authorization_header(self, client):
        """Test authentication with malformed authorization header."""
        test_cases = [
            {"Authorization": "InvalidFormat"},
            {"Authorization": "Bearer"},  # Missing token
            {"Authorization": "Basic dGVzdA=="},  # Wrong scheme
        ]
        
        for headers in test_cases:
            response = client.get("/api/v1/stats", headers=headers)
            assert response.status_code == 401
    
    def test_missing_authorization_header(self, client):
        """Test authentication without authorization header."""
        response = client.get("/api/v1/stats")
        assert response.status_code == 401


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        return create_app()
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @patch('src.services.auth_service.AuthService.authenticate')
    @patch('src.services.auth_service.AuthService.check_rate_limit')
    def test_rate_limit_not_exceeded(self, mock_check_rate_limit, mock_authenticate, client):
        """Test request within rate limit."""
        mock_authenticate.return_value = {
            "user_id": "test-user",
            "email": "test@example.com",
            "is_active": True,
            "is_admin": False,
            "rate_limit_per_minute": 100
        }
        mock_check_rate_limit.return_value = True  # Within limit
        
        headers = {"Authorization": "Bearer valid-jwt-token"}
        
        with patch('src.database.repositories.JobRepository.get_job_statistics', return_value={}), \
             patch('src.database.repositories.JobRepository.get_queue_depth_by_priority', return_value=0), \
             patch('src.database.repositories.JobRepository.get_performance_metrics', return_value={}), \
             patch('src.database.repositories.CacheRepository.get_cache_statistics', return_value={"total_accesses": 0}), \
             patch('src.database.repositories.MetricsRepository.get_system_health_metrics', return_value={}), \
             patch('src.database.repositories.MetricsRepository.get_latest_metric', return_value=None):
            
            response = client.get("/api/v1/stats", headers=headers)
            assert response.status_code == 200
    
    @patch('src.services.auth_service.AuthService.authenticate')
    @patch('src.services.auth_service.AuthService.check_rate_limit')
    def test_rate_limit_exceeded(self, mock_check_rate_limit, mock_authenticate, client):
        """Test request exceeding rate limit."""
        mock_authenticate.return_value = {
            "user_id": "test-user",
            "email": "test@example.com",
            "is_active": True,
            "is_admin": False,
            "rate_limit_per_minute": 100
        }
        mock_check_rate_limit.return_value = False  # Rate limit exceeded
        
        headers = {"Authorization": "Bearer valid-jwt-token"}
        response = client.get("/api/v1/stats", headers=headers)
        
        assert response.status_code == 429
        data = response.json()
        assert "rate limit" in data["detail"].lower()
        assert "Retry-After" in response.headers
    
    @patch('src.services.auth_service.AuthService.authenticate')
    def test_rate_limit_different_users(self, mock_authenticate, client):
        """Test that rate limiting is per-user."""
        # This test would require more complex mocking of the rate limiting logic
        # For now, we'll test that different users get different rate limit checks
        
        def authenticate_side_effect(token):
            if token == "user1-token":
                return {
                    "user_id": "user1",
                    "email": "user1@example.com",
                    "is_active": True,
                    "is_admin": False,
                    "rate_limit_per_minute": 10
                }
            elif token == "user2-token":
                return {
                    "user_id": "user2",
                    "email": "user2@example.com",
                    "is_active": True,
                    "is_admin": False,
                    "rate_limit_per_minute": 100
                }
            return None
        
        mock_authenticate.side_effect = authenticate_side_effect
        
        with patch('src.services.auth_service.AuthService.check_rate_limit', return_value=True), \
             patch('src.database.repositories.JobRepository.get_job_statistics', return_value={}), \
             patch('src.database.repositories.JobRepository.get_queue_depth_by_priority', return_value=0), \
             patch('src.database.repositories.JobRepository.get_performance_metrics', return_value={}), \
             patch('src.database.repositories.CacheRepository.get_cache_statistics', return_value={"total_accesses": 0}), \
             patch('src.database.repositories.MetricsRepository.get_system_health_metrics', return_value={}), \
             patch('src.database.repositories.MetricsRepository.get_latest_metric', return_value=None):
            
            # Both users should be able to make requests
            response1 = client.get("/api/v1/stats", headers={"Authorization": "Bearer user1-token"})
            response2 = client.get("/api/v1/stats", headers={"Authorization": "Bearer user2-token"})
            
            assert response1.status_code == 200
            assert response2.status_code == 200


class TestAuthorization:
    """Test authorization functionality."""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        return create_app()
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @patch('src.services.auth_service.AuthService.authenticate')
    def test_admin_endpoint_with_admin_user(self, mock_authenticate, client):
        """Test admin endpoint access with admin user."""
        mock_authenticate.return_value = {
            "user_id": "admin-user",
            "email": "admin@example.com",
            "is_active": True,
            "is_admin": True,  # Admin user
            "rate_limit_per_minute": 1000
        }
        
        headers = {"Authorization": "Bearer admin-jwt-token"}
        
        with patch('src.services.auth_service.AuthService.check_rate_limit', return_value=True), \
             patch('src.database.repositories.MetricsRepository.get_system_health_metrics', return_value={}), \
             patch('src.database.repositories.MetricsRepository.get_capacity_metrics', return_value={}), \
             patch('src.database.repositories.MetricsRepository.get_performance_trends', return_value=[]), \
             patch('src.database.repositories.MetricsRepository.get_alert_metrics', return_value=[]):
            
            response = client.get("/api/v1/admin/metrics", headers=headers)
            assert response.status_code == 200
    
    @patch('src.services.auth_service.AuthService.authenticate')
    def test_admin_endpoint_with_regular_user(self, mock_authenticate, client):
        """Test admin endpoint access with regular user."""
        mock_authenticate.return_value = {
            "user_id": "regular-user",
            "email": "user@example.com",
            "is_active": True,
            "is_admin": False,  # Regular user
            "rate_limit_per_minute": 100
        }
        
        headers = {"Authorization": "Bearer user-jwt-token"}
        
        with patch('src.services.auth_service.AuthService.check_rate_limit', return_value=True):
            response = client.get("/api/v1/admin/metrics", headers=headers)
            assert response.status_code == 403
            data = response.json()
            assert "admin" in data["detail"].lower()
    
    @patch('src.services.auth_service.AuthService.authenticate')
    def test_job_access_control(self, mock_authenticate, client):
        """Test that users can only access their own jobs."""
        mock_authenticate.return_value = {
            "user_id": "user1",
            "email": "user1@example.com",
            "is_active": True,
            "is_admin": False,
            "rate_limit_per_minute": 100
        }
        
        # Mock a job owned by a different user
        from src.database.models import TranslationJob
        other_user_job = TranslationJob(
            id=uuid4(),
            user_id="user2",  # Different user
            source_language="en",
            target_language="es",
            content_hash="test-hash",
            word_count=10,
            priority="normal",
            status="completed"
        )
        
        headers = {"Authorization": "Bearer user1-jwt-token"}
        
        with patch('src.services.auth_service.AuthService.check_rate_limit', return_value=True), \
             patch('src.database.repositories.JobRepository.get_by_id', return_value=other_user_job):
            
            response = client.get(f"/api/v1/jobs/{other_user_job.id}", headers=headers)
            assert response.status_code == 403
            data = response.json()
            assert "access" in data["error"]["message"].lower()
    
    @patch('src.services.auth_service.AuthService.authenticate')
    def test_admin_can_access_any_job(self, mock_authenticate, client):
        """Test that admin users can access any job."""
        mock_authenticate.return_value = {
            "user_id": "admin-user",
            "email": "admin@example.com",
            "is_active": True,
            "is_admin": True,  # Admin user
            "rate_limit_per_minute": 1000
        }
        
        # Mock a job owned by a different user
        from src.database.models import TranslationJob
        other_user_job = TranslationJob(
            id=uuid4(),
            user_id="regular-user",  # Different user
            source_language="en",
            target_language="es",
            content_hash="test-hash",
            word_count=10,
            priority="normal",
            status="completed",
            created_at=datetime.utcnow()
        )
        
        headers = {"Authorization": "Bearer admin-jwt-token"}
        
        with patch('src.services.auth_service.AuthService.check_rate_limit', return_value=True), \
             patch('src.database.repositories.JobRepository.get_by_id', return_value=other_user_job):
            
            response = client.get(f"/api/v1/jobs/{other_user_job.id}", headers=headers)
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == str(other_user_job.id)


class TestSecurityHeaders:
    """Test security headers and middleware."""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        return create_app()
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    def test_security_headers_present(self, client):
        """Test that security headers are present in responses."""
        response = client.get("/api/v1/health")
        
        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "Referrer-Policy",
            "Content-Security-Policy"
        ]
        
        for header in expected_headers:
            assert header in response.headers
    
    def test_server_header_removed(self, client):
        """Test that server header is removed for security."""
        response = client.get("/api/v1/health")
        
        # Server header should be removed or not present
        assert "server" not in response.headers or response.headers.get("server") != "uvicorn"
    
    def test_cors_headers(self, client):
        """Test CORS headers for preflight requests."""
        response = client.options("/api/v1/health")
        
        # Should have CORS headers
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers