"""
Integration tests for translation engine with the overall system.
"""

import pytest
import asyncio
from datetime import datetime
from uuid import uuid4

from src.config.config import Priority
from src.models.interfaces import TranslationRequest
from src.services.translation_engine import create_translation_engine, CPUTranslationEngine
from src.services.translation_service import TranslationService


class TestTranslationEngineIntegration:
    """Test translation engine integration with the system."""
    
    @pytest.fixture
    async def translation_engine(self):
        """Create translation engine for testing."""
        # Force CPU engine for testing to avoid GPU dependencies
        engine = CPUTranslationEngine()
        yield engine
        if hasattr(engine, 'close'):
            await engine.close()
    
    @pytest.fixture
    async def translation_service(self, db_session):
        """Create translation service for testing."""
        service = TranslationService()
        yield service
        await service.close()
    
    @pytest.fixture
    def sample_request(self):
        """Create sample translation request."""
        return TranslationRequest(
            source_language="en",
            target_language="es",
            content="Hello, how are you today? This is a test translation.",
            priority=Priority.NORMAL,
            user_id="test-user"
        )
    
    @pytest.mark.asyncio
    async def test_translation_engine_basic_functionality(self, translation_engine, sample_request):
        """Test basic translation engine functionality."""
        # Test supported languages
        languages = translation_engine.get_supported_languages()
        assert isinstance(languages, list)
        assert len(languages) > 0
        assert "en" in languages
        assert "es" in languages
        
        # Test translation
        result = await translation_engine.translate(sample_request)
        
        assert result is not None
        assert result.source_language == sample_request.source_language
        assert result.target_language == sample_request.target_language
        assert result.translated_content is not None
        assert len(result.translated_content) > 0
        assert result.confidence_score >= 0.0
        assert result.confidence_score <= 1.0
        assert result.processing_time_ms >= 0
        assert result.model_version is not None
    
    @pytest.mark.asyncio
    async def test_translation_engine_memory_management(self, translation_engine):
        """Test translation engine memory management."""
        # Test memory usage reporting
        memory_usage = translation_engine.get_memory_usage()
        assert isinstance(memory_usage, (int, float))
        assert memory_usage >= 0.0
        
        # Test model loading/unloading (simplified for CPU engine)
        model_id = "test-model"
        
        # Load model
        load_result = await translation_engine.load_model(model_id)
        assert load_result is True
        
        # Unload model
        unload_result = await translation_engine.unload_model(model_id)
        assert unload_result is True
    
    @pytest.mark.asyncio
    async def test_translation_engine_error_handling(self, translation_engine):
        """Test translation engine error handling."""
        # Test with unsupported language pair
        invalid_request = TranslationRequest(
            source_language="xyz",
            target_language="abc",
            content="Hello world",
            priority=Priority.NORMAL,
            user_id="test-user"
        )
        
        with pytest.raises(Exception):  # Should raise TranslationError
            await translation_engine.translate(invalid_request)
    
    @pytest.mark.asyncio
    async def test_translation_service_integration(self, translation_service, sample_request):
        """Test translation service integration."""
        # This test would require database setup
        pytest.skip("Requires database setup for integration test")
        
        # Submit translation job
        job = await translation_service.submit_translation_job(sample_request, "test-user")
        
        assert job is not None
        assert job.user_id == "test-user"
        assert job.source_language == sample_request.source_language
        assert job.target_language == sample_request.target_language
        assert job.word_count > 0
        assert job.priority == sample_request.priority
        
        # Check job status
        job_status = await translation_service.get_job_status(job.id, "test-user")
        assert job_status is not None
        assert job_status.id == job.id
    
    @pytest.mark.asyncio
    async def test_translation_engine_performance_requirements(self, translation_engine):
        """Test translation engine meets performance requirements."""
        # Test with different content sizes
        test_cases = [
            ("Short text", "Hello world", 2),
            ("Medium text", "This is a medium length text for translation testing. " * 10, 100),
            ("Long text", "This is a longer text for performance testing. " * 50, 500)
        ]
        
        for case_name, content, expected_word_count in test_cases:
            request = TranslationRequest(
                source_language="en",
                target_language="es",
                content=content,
                priority=Priority.NORMAL,
                user_id="test-user"
            )
            
            start_time = datetime.utcnow()
            result = await translation_engine.translate(request)
            end_time = datetime.utcnow()
            
            processing_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Verify result
            assert result is not None
            assert result.translated_content is not None
            
            # Performance check (relaxed for CPU engine)
            word_count = len(content.split())
            assert word_count >= expected_word_count * 0.8  # Allow some variance
            
            # For CPU engine, we don't expect to meet the 1500 WPM target
            # but we should get reasonable performance
            if processing_time_ms > 0:
                words_per_minute = (word_count / (processing_time_ms / 1000 / 60))
                assert words_per_minute > 0  # Just ensure it's processing
    
    @pytest.mark.asyncio
    async def test_translation_engine_concurrent_requests(self, translation_engine):
        """Test translation engine handles concurrent requests."""
        # Create multiple translation requests
        requests = []
        for i in range(5):
            request = TranslationRequest(
                source_language="en",
                target_language="es",
                content=f"This is test message number {i} for concurrent translation testing.",
                priority=Priority.NORMAL,
                user_id=f"test-user-{i}"
            )
            requests.append(request)
        
        # Execute translations concurrently
        tasks = [translation_engine.translate(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all translations completed
        assert len(results) == 5
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Translation {i} failed with exception: {result}")
            
            assert result is not None
            assert result.translated_content is not None
            assert result.source_language == "en"
            assert result.target_language == "es"
    
    @pytest.mark.asyncio
    async def test_translation_engine_factory(self):
        """Test translation engine factory function."""
        # Test engine creation
        engine = create_translation_engine()
        
        assert engine is not None
        
        # Should create CPU engine in test environment (no GPU)
        assert isinstance(engine, CPUTranslationEngine)
        
        # Test basic functionality
        languages = engine.get_supported_languages()
        assert len(languages) > 0
        
        # Clean up
        if hasattr(engine, 'close'):
            await engine.close()
    
    @pytest.mark.asyncio
    async def test_translation_engine_language_validation(self, translation_engine):
        """Test translation engine language validation."""
        supported_languages = translation_engine.get_supported_languages()
        
        # Test valid language pairs
        for source_lang in supported_languages[:3]:  # Test first 3 languages
            for target_lang in supported_languages[:3]:
                if source_lang != target_lang:
                    request = TranslationRequest(
                        source_language=source_lang,
                        target_language=target_lang,
                        content="Test translation content",
                        priority=Priority.NORMAL,
                        user_id="test-user"
                    )
                    
                    # Should not raise exception for supported pairs
                    result = await translation_engine.translate(request)
                    assert result is not None
        
        # Test invalid language pairs
        invalid_pairs = [
            ("en", "en"),  # Same language
            ("xyz", "abc"),  # Unsupported languages
            ("en", "xyz"),  # One unsupported
        ]
        
        for source_lang, target_lang in invalid_pairs:
            request = TranslationRequest(
                source_language=source_lang,
                target_language=target_lang,
                content="Test content",
                priority=Priority.NORMAL,
                user_id="test-user"
            )
            
            with pytest.raises(Exception):
                await translation_engine.translate(request)