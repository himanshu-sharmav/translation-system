"""
Integration tests for API endpoints.
"""

import pytest
from datetime import datetime
from uuid import uuid4

from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from src.api.app import create_app
from src.database.models import User, TranslationJob, TranslationCache
from src.services.auth_service import AuthService


class TestTranslationAPI:
    """Test translation API endpoints."""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        return create_app()
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def auth_token(self):
        """Create test auth token."""
        auth_service = AuthService()
        # Mock token generation for testing
        return "test-jwt-token-12345"
    
    @pytest.fixture
    def auth_headers(self, auth_token):
        """Create auth headers."""
        return {"Authorization": f"Bearer {auth_token}"}
    
    @pytest.fixture
    def mock_user_data(self):
        """Mock user data."""
        return {
            "user_id": "test-user",
            "email": "test@example.com",
            "is_active": True,
            "is_admin": False,
            "rate_limit_per_minute": 100
        }
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        with patch('src.database.connection.db_manager.health_check', return_value=True):
            response = client.get("/api/v1/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] in ["healthy", "degraded"]
            assert "timestamp" in data
            assert "version" in data
            assert "database" in data
            assert "uptime_seconds" in data
    
    def test_health_check_unhealthy(self, client):
        """Test health check when services are unhealthy."""
        with patch('src.database.connection.db_manager.health_check', return_value=False):
            response = client.get("/api/v1/health")
            
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "degraded"
    
    def test_get_supported_languages(self, client):
        """Test get supported languages endpoint."""
        response = client.get("/api/v1/languages")
        
        assert response.status_code == 200
        data = response.json()
        assert "languages" in data
        assert "total_pairs" in data
        assert len(data["languages"]) > 0
        assert all("code" in lang and "name" in lang for lang in data["languages"])
    
    @patch('src.api.dependencies.get_current_user')
    def test_create_translation_job_success(self, mock_get_user, client, auth_headers, mock_user_data):
        """Test successful translation job creation."""
        mock_get_user.return_value = mock_user_data
        
        with patch('src.database.repositories.JobRepository.create') as mock_create, \
             patch('src.database.repositories.CacheRepository.get_by_hash', return_value=None), \
             patch('src.services.queue_service.QueueService.enqueue_job', return_value=True), \
             patch('src.database.repositories.JobRepository.get_queue_position', return_value=1):
            
            # Mock job creation
            job_id = uuid4()
            mock_job = TranslationJob(
                id=job_id,
                user_id=mock_user_data["user_id"],
                source_language="en",
                target_language="es",
                content_hash="test-hash",
                word_count=2,
                priority="normal",
                status="queued"
            )
            mock_create.return_value = mock_job
            
            request_data = {
                "source_language": "en",
                "target_language": "es",
                "content": "Hello world",
                "priority": "normal"
            }
            
            response = client.post("/api/v1/translate", json=request_data, headers=auth_headers)
            
            assert response.status_code == 201
            data = response.json()
            assert "job_id" in data
            assert data["status"] == "queued"
            assert data["queue_position"] == 1
            assert "estimated_completion" in data
    
    @patch('src.api.dependencies.get_current_user')
    def test_create_translation_job_cache_hit(self, mock_get_user, client, auth_headers, mock_user_data):
        """Test translation job creation with cache hit."""
        mock_get_user.return_value = mock_user_data
        
        # Mock cache hit
        cached_result = TranslationCache(
            id=uuid4(),
            content_hash="test-hash",
            source_language="en",
            target_language="es",
            source_content="Hello world",
            translated_content="Hola mundo",
            model_version="v1.0.0",
            confidence_score=0.95
        )
        
        with patch('src.database.repositories.CacheRepository.get_by_hash', return_value=cached_result), \
             patch('src.database.repositories.CacheRepository.update_access_stats', return_value=True):
            
            request_data = {
                "source_language": "en",
                "target_language": "es",
                "content": "Hello world",
                "priority": "normal"
            }
            
            response = client.post("/api/v1/translate", json=request_data, headers=auth_headers)
            
            assert response.status_code == 201
            data = response.json()
            assert data["status"] == "completed"
            assert "cache" in data["message"].lower()
    
    def test_create_translation_job_unauthorized(self, client):
        """Test translation job creation without authentication."""
        request_data = {
            "source_language": "en",
            "target_language": "es",
            "content": "Hello world"
        }
        
        response = client.post("/api/v1/translate", json=request_data)
        
        assert response.status_code == 401
    
    @patch('src.api.dependencies.get_current_user')
    def test_create_translation_job_invalid_language(self, mock_get_user, client, auth_headers, mock_user_data):
        """Test translation job creation with invalid language."""
        mock_get_user.return_value = mock_user_data
        
        request_data = {
            "source_language": "invalid",
            "target_language": "es",
            "content": "Hello world"
        }
        
        response = client.post("/api/v1/translate", json=request_data, headers=auth_headers)
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    @patch('src.api.dependencies.get_current_user')
    def test_create_translation_job_too_long(self, mock_get_user, client, auth_headers, mock_user_data):
        """Test translation job creation with content too long."""
        mock_get_user.return_value = mock_user_data
        
        # Create content with more than 15,000 words
        long_content = " ".join(["word"] * 15001)
        
        request_data = {
            "source_language": "en",
            "target_language": "es",
            "content": long_content
        }
        
        response = client.post("/api/v1/translate", json=request_data, headers=auth_headers)
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "15,000 words" in str(data)
    
    @patch('src.api.dependencies.get_current_user')
    def test_get_job_status_success(self, mock_get_user, client, auth_headers, mock_user_data):
        """Test successful job status retrieval."""
        mock_get_user.return_value = mock_user_data
        
        job_id = uuid4()
        mock_job = TranslationJob(
            id=job_id,
            user_id=mock_user_data["user_id"],
            source_language="en",
            target_language="es",
            content_hash="test-hash",
            word_count=2,
            priority="normal",
            status="processing",
            progress=50.0,
            created_at=datetime.utcnow()
        )
        
        with patch('src.database.repositories.JobRepository.get_by_id', return_value=mock_job):
            response = client.get(f"/api/v1/jobs/{job_id}", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == str(job_id)
            assert data["status"] == "processing"
            assert data["progress"] == 50.0
            assert data["word_count"] == 2
    
    @patch('src.api.dependencies.get_current_user')
    def test_get_job_status_not_found(self, mock_get_user, client, auth_headers, mock_user_data):
        """Test job status retrieval for non-existent job."""
        mock_get_user.return_value = mock_user_data
        
        job_id = uuid4()
        
        with patch('src.database.repositories.JobRepository.get_by_id', return_value=None):
            response = client.get(f"/api/v1/jobs/{job_id}", headers=auth_headers)
            
            assert response.status_code == 404
            data = response.json()
            assert "error" in data
            assert "not found" in data["error"]["message"].lower()
    
    @patch('src.api.dependencies.get_current_user')
    def test_get_job_status_access_denied(self, mock_get_user, client, auth_headers, mock_user_data):
        """Test job status retrieval for job owned by different user."""
        mock_get_user.return_value = mock_user_data
        
        job_id = uuid4()
        mock_job = TranslationJob(
            id=job_id,
            user_id="different-user",  # Different user
            source_language="en",
            target_language="es",
            content_hash="test-hash",
            word_count=2,
            priority="normal",
            status="processing"
        )
        
        with patch('src.database.repositories.JobRepository.get_by_id', return_value=mock_job):
            response = client.get(f"/api/v1/jobs/{job_id}", headers=auth_headers)
            
            assert response.status_code == 403
            data = response.json()
            assert "error" in data
            assert "access" in data["error"]["message"].lower()
    
    @patch('src.api.dependencies.get_current_user')
    def test_get_translation_result_success(self, mock_get_user, client, auth_headers, mock_user_data):
        """Test successful translation result retrieval."""
        mock_get_user.return_value = mock_user_data
        
        job_id = uuid4()
        mock_job = TranslationJob(
            id=job_id,
            user_id=mock_user_data["user_id"],
            source_language="en",
            target_language="es",
            content_hash="test-hash",
            word_count=2,
            priority="normal",
            status="completed",
            processing_time_ms=5000
        )
        
        mock_cache_result = TranslationCache(
            id=uuid4(),
            content_hash="test-hash",
            source_language="en",
            target_language="es",
            source_content="Hello world",
            translated_content="Hola mundo",
            model_version="v1.0.0",
            confidence_score=0.95
        )
        
        with patch('src.database.repositories.JobRepository.get_by_id', return_value=mock_job), \
             patch('src.database.repositories.CacheRepository.get_by_hash', return_value=mock_cache_result), \
             patch('src.database.repositories.CacheRepository.update_access_stats', return_value=True):
            
            response = client.get(f"/api/v1/jobs/{job_id}/result", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == str(job_id)
            assert data["translated_content"] == "Hola mundo"
            assert data["source_language"] == "en"
            assert data["target_language"] == "es"
            assert data["confidence_score"] == 0.95
            assert data["processing_time_ms"] == 5000
    
    @patch('src.api.dependencies.get_current_user')
    def test_get_translation_result_not_completed(self, mock_get_user, client, auth_headers, mock_user_data):
        """Test translation result retrieval for incomplete job."""
        mock_get_user.return_value = mock_user_data
        
        job_id = uuid4()
        mock_job = TranslationJob(
            id=job_id,
            user_id=mock_user_data["user_id"],
            source_language="en",
            target_language="es",
            content_hash="test-hash",
            word_count=2,
            priority="normal",
            status="processing"  # Not completed
        )
        
        with patch('src.database.repositories.JobRepository.get_by_id', return_value=mock_job):
            response = client.get(f"/api/v1/jobs/{job_id}/result", headers=auth_headers)
            
            assert response.status_code == 404
            data = response.json()
            assert "error" in data
            assert "not available" in data["error"]["message"].lower()
    
    @patch('src.api.dependencies.get_current_user')
    def test_list_user_jobs(self, mock_get_user, client, auth_headers, mock_user_data):
        """Test listing user jobs."""
        mock_get_user.return_value = mock_user_data
        
        # Mock paginated result
        mock_jobs = [
            TranslationJob(
                id=uuid4(),
                user_id=mock_user_data["user_id"],
                source_language="en",
                target_language="es",
                content_hash="test-hash-1",
                word_count=10,
                priority="normal",
                status="completed",
                created_at=datetime.utcnow()
            ),
            TranslationJob(
                id=uuid4(),
                user_id=mock_user_data["user_id"],
                source_language="en",
                target_language="fr",
                content_hash="test-hash-2",
                word_count=20,
                priority="high",
                status="processing",
                created_at=datetime.utcnow()
            )
        ]
        
        mock_paginated_result = {
            "items": mock_jobs,
            "total": 2,
            "page": 1,
            "per_page": 20,
            "total_pages": 1,
            "has_prev": False,
            "has_next": False
        }
        
        with patch('src.database.repositories.JobRepository.paginate', return_value=mock_paginated_result):
            response = client.get("/api/v1/jobs", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["jobs"]) == 2
            assert data["total"] == 2
            assert data["page"] == 1
            assert data["has_prev"] is False
            assert data["has_next"] is False
    
    @patch('src.api.dependencies.get_current_user')
    def test_list_user_jobs_with_filters(self, mock_get_user, client, auth_headers, mock_user_data):
        """Test listing user jobs with status filter."""
        mock_get_user.return_value = mock_user_data
        
        mock_paginated_result = {
            "items": [],
            "total": 0,
            "page": 1,
            "per_page": 20,
            "total_pages": 0,
            "has_prev": False,
            "has_next": False
        }
        
        with patch('src.database.repositories.JobRepository.paginate', return_value=mock_paginated_result):
            response = client.get("/api/v1/jobs?status_filter=completed&priority_filter=high", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 0
    
    @patch('src.api.dependencies.get_current_user')
    def test_cancel_translation_job(self, mock_get_user, client, auth_headers, mock_user_data):
        """Test cancelling a translation job."""
        mock_get_user.return_value = mock_user_data
        
        job_id = uuid4()
        mock_job = TranslationJob(
            id=job_id,
            user_id=mock_user_data["user_id"],
            source_language="en",
            target_language="es",
            content_hash="test-hash",
            word_count=2,
            priority="normal",
            status="queued"
        )
        
        with patch('src.database.repositories.JobRepository.get_by_id', return_value=mock_job), \
             patch('src.database.repositories.JobRepository.update_job_status', return_value=True), \
             patch('src.services.queue_service.QueueService.remove_job', return_value=True):
            
            response = client.delete(f"/api/v1/jobs/{job_id}", headers=auth_headers)
            
            assert response.status_code == 204
    
    @patch('src.api.dependencies.get_current_user')
    def test_cancel_completed_job(self, mock_get_user, client, auth_headers, mock_user_data):
        """Test cancelling a completed job (should fail)."""
        mock_get_user.return_value = mock_user_data
        
        job_id = uuid4()
        mock_job = TranslationJob(
            id=job_id,
            user_id=mock_user_data["user_id"],
            source_language="en",
            target_language="es",
            content_hash="test-hash",
            word_count=2,
            priority="normal",
            status="completed"  # Already completed
        )
        
        with patch('src.database.repositories.JobRepository.get_by_id', return_value=mock_job):
            response = client.delete(f"/api/v1/jobs/{job_id}", headers=auth_headers)
            
            assert response.status_code == 409
            data = response.json()
            assert "error" in data
            assert "cannot cancel" in data["error"]["message"].lower()
    
    @patch('src.api.dependencies.get_current_user')
    def test_batch_translation(self, mock_get_user, client, auth_headers, mock_user_data):
        """Test batch translation request."""
        mock_get_user.return_value = mock_user_data
        
        with patch('src.database.repositories.JobRepository.create') as mock_create, \
             patch('src.services.queue_service.QueueService.enqueue_job', return_value=True):
            
            # Mock job creation for each request
            def create_job_side_effect(job):
                return job
            mock_create.side_effect = create_job_side_effect
            
            request_data = {
                "requests": [
                    {
                        "source_language": "en",
                        "target_language": "es",
                        "content": "Hello world",
                        "priority": "normal"
                    },
                    {
                        "source_language": "en",
                        "target_language": "fr",
                        "content": "Good morning",
                        "priority": "high"
                    }
                ],
                "batch_priority": "normal"
            }
            
            response = client.post("/api/v1/translate/batch", json=request_data, headers=auth_headers)
            
            assert response.status_code == 201
            data = response.json()
            assert "batch_id" in data
            assert len(data["job_ids"]) == 2
            assert data["total_jobs"] == 2
            assert data["status"] == "accepted"


class TestSystemAPI:
    """Test system API endpoints."""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        return create_app()
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create auth headers."""
        return {"Authorization": "Bearer test-jwt-token"}
    
    @pytest.fixture
    def mock_user_data(self):
        """Mock user data."""
        return {
            "user_id": "test-user",
            "email": "test@example.com",
            "is_active": True,
            "is_admin": False,
            "rate_limit_per_minute": 100
        }
    
    @pytest.fixture
    def mock_admin_data(self):
        """Mock admin user data."""
        return {
            "user_id": "admin-user",
            "email": "admin@example.com",
            "is_active": True,
            "is_admin": True,
            "rate_limit_per_minute": 1000
        }
    
    @patch('src.api.dependencies.get_current_user')
    def test_get_system_stats(self, mock_get_user, client, auth_headers, mock_user_data):
        """Test system statistics endpoint."""
        mock_get_user.return_value = mock_user_data
        
        mock_job_stats = {
            "completed": 100,
            "failed": 5,
            "queued": 10,
            "processing": 2,
            "avg_processing_time_ms": 5000.0
        }
        
        mock_cache_stats = {
            "total_accesses": 150,
            "total_entries": 50
        }
        
        mock_performance_metrics = {
            "avg_words_per_minute": 1500.0
        }
        
        with patch('src.database.repositories.JobRepository.get_job_statistics', return_value=mock_job_stats), \
             patch('src.database.repositories.JobRepository.get_queue_depth_by_priority', return_value=5), \
             patch('src.database.repositories.JobRepository.get_performance_metrics', return_value=mock_performance_metrics), \
             patch('src.database.repositories.CacheRepository.get_cache_statistics', return_value=mock_cache_stats), \
             patch('src.database.repositories.MetricsRepository.get_system_health_metrics', return_value={}), \
             patch('src.database.repositories.MetricsRepository.get_latest_metric', return_value=None):
            
            response = client.get("/api/v1/stats", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert "total_jobs" in data
            assert "active_jobs" in data
            assert "completed_jobs" in data
            assert "failed_jobs" in data
            assert "queue_depth" in data
            assert "avg_processing_time_ms" in data
            assert "words_per_minute" in data
            assert "cache_hit_ratio" in data
            assert "active_instances" in data
    
    @patch('src.api.dependencies.get_current_user')
    def test_cost_estimate(self, mock_get_user, client, auth_headers, mock_user_data):
        """Test cost estimate endpoint."""
        mock_get_user.return_value = mock_user_data
        
        mock_estimate = {
            "daily_cost_usd": 10.54,
            "monthly_cost_usd": 316.20,
            "breakdown": {
                "compute_cost": 10.52,
                "storage_cost": 0.008,
                "network_cost": 0.009
            },
            "assumptions": {
                "cache_hit_ratio": 0.3,
                "words_per_minute": 1500
            }
        }
        
        with patch('src.services.cost_calculator.CostCalculator.calculate_cost_estimate', return_value=mock_estimate):
            request_data = {
                "words_per_day": 100000,
                "priority_distribution": {
                    "normal": 0.8,
                    "high": 0.15,
                    "critical": 0.05
                }
            }
            
            response = client.post("/api/v1/cost-estimate", json=request_data, headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["words_per_day"] == 100000
            assert data["estimated_daily_cost_usd"] == 10.54
            assert data["estimated_monthly_cost_usd"] == 316.20
            assert "breakdown" in data
            assert "assumptions" in data
    
    def test_cost_estimate_invalid_distribution(self, client, auth_headers):
        """Test cost estimate with invalid priority distribution."""
        with patch('src.api.dependencies.get_current_user'):
            request_data = {
                "words_per_day": 100000,
                "priority_distribution": {
                    "normal": 0.5,
                    "high": 0.3,
                    "critical": 0.3  # Sum > 1.0
                }
            }
            
            response = client.post("/api/v1/cost-estimate", json=request_data, headers=auth_headers)
            
            assert response.status_code == 422  # Validation error
    
    @patch('src.api.dependencies.get_admin_user')
    def test_admin_metrics(self, mock_get_admin, client, auth_headers, mock_admin_data):
        """Test admin metrics endpoint."""
        mock_get_admin.return_value = mock_admin_data
        
        mock_health_metrics = {
            "system_metrics": {},
            "instance_metrics": {}
        }
        
        mock_capacity_metrics = {
            "queue_depth": {"avg": 10},
            "gpu_utilization": {"avg": 75}
        }
        
        with patch('src.database.repositories.MetricsRepository.get_system_health_metrics', return_value=mock_health_metrics), \
             patch('src.database.repositories.MetricsRepository.get_capacity_metrics', return_value=mock_capacity_metrics), \
             patch('src.database.repositories.MetricsRepository.get_performance_trends', return_value=[]), \
             patch('src.database.repositories.MetricsRepository.get_alert_metrics', return_value=[]):
            
            response = client.get("/api/v1/admin/metrics", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert "health_metrics" in data
            assert "capacity_metrics" in data
            assert "performance_trends" in data
            assert "active_alerts" in data
            assert "summary" in data
    
    @patch('src.api.dependencies.get_current_user')
    def test_admin_metrics_non_admin(self, mock_get_user, client, auth_headers, mock_user_data):
        """Test admin metrics endpoint with non-admin user."""
        mock_get_user.return_value = mock_user_data  # Non-admin user
        
        response = client.get("/api/v1/admin/metrics", headers=auth_headers)
        
        assert response.status_code == 403
        data = response.json()
        assert "error" in data
        assert "admin" in data["error"]["message"].lower()
    
    @patch('src.api.dependencies.get_admin_user')
    def test_admin_cleanup(self, mock_get_admin, client, auth_headers, mock_admin_data):
        """Test admin cleanup endpoint."""
        mock_get_admin.return_value = mock_admin_data
        
        with patch('src.database.repositories.JobRepository.cleanup_old_jobs', return_value=10), \
             patch('src.database.repositories.CacheRepository.cleanup_expired', return_value=5), \
             patch('src.database.repositories.MetricsRepository.cleanup_old_metrics', return_value=100):
            
            response = client.post("/api/v1/admin/cleanup?days=30", headers=auth_headers)
            
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert "results" in data
            assert data["results"]["deleted_jobs"] == 10
            assert data["results"]["deleted_cache_entries"] == 5
            assert data["results"]["deleted_metrics"] == 100
    
    def test_admin_cleanup_invalid_days(self, client, auth_headers):
        """Test admin cleanup with invalid days parameter."""
        with patch('src.api.dependencies.get_admin_user'):
            response = client.post("/api/v1/admin/cleanup?days=500", headers=auth_headers)  # > 365
            
            assert response.status_code == 400
            data = response.json()
            assert "error" in data


class TestErrorHandling:
    """Test API error handling."""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        return create_app()
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    def test_404_endpoint(self, client):
        """Test 404 for non-existent endpoint."""
        response = client.get("/api/v1/nonexistent")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test 405 for wrong HTTP method."""
        response = client.put("/api/v1/health")
        
        assert response.status_code == 405
    
    def test_validation_error(self, client):
        """Test validation error handling."""
        with patch('src.api.dependencies.get_current_user'):
            # Missing required fields
            response = client.post("/api/v1/translate", json={})
            
            assert response.status_code == 422
            data = response.json()
            assert "error" in data
            assert data["error"]["code"] == "VALIDATION_ERROR"
    
    def test_correlation_id_in_response(self, client):
        """Test that correlation ID is included in responses."""
        response = client.get("/api/v1/health")
        
        assert "X-Correlation-ID" in response.headers
        assert len(response.headers["X-Correlation-ID"]) > 0
    
    def test_process_time_in_response(self, client):
        """Test that process time is included in responses."""
        response = client.get("/api/v1/health")
        
        assert "X-Process-Time" in response.headers
        assert float(response.headers["X-Process-Time"]) >= 0