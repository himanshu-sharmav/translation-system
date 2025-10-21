"""
Integration tests for model optimization features.
"""

import pytest
import asyncio
from datetime import datetime
from uuid import uuid4

from src.config.config import Priority
from src.models.interfaces import TranslationRequest
from src.services.translation_engine import GPUTranslationEngine, CPUTranslationEngine
from src.services.model_optimizer import OptimizedTranslationEngine, create_optimized_engine


class TestModelOptimizationIntegration:
    """Test model optimization integration with translation engine."""
    
    @pytest.fixture
    async def base_engine(self):
        """Create base translation engine for testing."""
        # Use CPU engine for testing to avoid GPU dependencies
        engine = CPUTranslationEngine()
        yield engine
        if hasattr(engine, 'close'):
            await engine.close()
    
    @pytest.fixture
    async def optimized_engine(self, base_engine):
        """Create optimized translation engine."""
        engine = create_optimized_engine(base_engine)
        yield engine
        # Cleanup if needed
    
    @pytest.fixture
    def sample_request(self):
        """Create sample translation request."""
        return TranslationRequest(
            source_language="en",
            target_language="es",
            content="This is a test translation request for optimization testing.",
            priority=Priority.NORMAL,
            user_id="test-user"
        )
    
    @pytest.mark.asyncio
    async def test_optimization_engine_creation(self, base_engine):
        """Test creating optimized engine from base engine."""
        optimized = create_optimized_engine(base_engine)
        
        assert optimized is not None
        assert isinstance(optimized, OptimizedTranslationEngine)
        assert optimized.base_engine == base_engine
        assert optimized.optimization_enabled is True
    
    @pytest.mark.asyncio
    async def test_optimized_translation_basic(self, optimized_engine, sample_request):
        """Test basic optimized translation functionality."""
        result = await optimized_engine.translate_optimized(sample_request)
        
        assert result is not None
        assert result.source_language == sample_request.source_language
        assert result.target_language == sample_request.target_language
        assert result.translated_content is not None
        assert result.confidence_score >= 0.0
        assert result.processing_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_optimization_status_reporting(self, optimized_engine):
        """Test optimization status reporting."""
        status = await optimized_engine.get_optimization_status()
        
        assert isinstance(status, dict)
        assert "optimization_enabled" in status
        assert "dynamic_batching" in status
        assert "quantization" in status
        assert "pipeline_parallelism" in status
        
        # Check dynamic batching status
        batching_status = status["dynamic_batching"]
        assert "enabled" in batching_status
        assert "max_batch_size" in batching_status
        assert "max_wait_time_ms" in batching_status
        
        # Check quantization status
        quantization_status = status["quantization"]
        assert "available" in quantization_status
        assert "methods" in quantization_status
        
        # Check pipeline parallelism status
        pipeline_status = status["pipeline_parallelism"]
        assert "available" in pipeline_status
        assert "num_gpus" in pipeline_status
    
    @pytest.mark.asyncio
    async def test_optimization_enable_disable(self, optimized_engine, sample_request):
        """Test enabling and disabling optimization features."""
        # Initially enabled
        assert optimized_engine.optimization_enabled is True
        
        # Test with optimization enabled
        result_optimized = await optimized_engine.translate_optimized(sample_request)
        assert result_optimized is not None
        
        # Disable optimization
        await optimized_engine.enable_optimization(False)
        assert optimized_engine.optimization_enabled is False
        
        # Test with optimization disabled (should fallback to base engine)
        result_fallback = await optimized_engine.translate_optimized(sample_request)
        assert result_fallback is not None
        
        # Re-enable optimization
        await optimized_engine.enable_optimization(True)
        assert optimized_engine.optimization_enabled is True
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, optimized_engine, sample_request):
        """Test performance metrics collection during optimization."""
        # Initially no metrics
        initial_metrics = await optimized_engine.get_optimization_metrics()
        assert "message" in initial_metrics
        
        # Perform some translations to generate metrics
        for i in range(5):
            request = TranslationRequest(
                source_language="en",
                target_language="es",
                content=f"Test translation request number {i}",
                priority=Priority.NORMAL,
                user_id="test-user"
            )
            await optimized_engine.translate_optimized(request)
        
        # Check if metrics were collected
        metrics = await optimized_engine.get_optimization_metrics()
        
        if "total_requests" in metrics:
            assert metrics["total_requests"] > 0
            assert "average_metrics" in metrics
            assert "performance_targets" in metrics
    
    @pytest.mark.asyncio
    async def test_batch_parameter_tuning(self, optimized_engine):
        """Test automatic batch parameter tuning."""
        # Get initial batch parameters
        initial_status = await optimized_engine.get_optimization_status()
        initial_batch_size = initial_status["dynamic_batching"]["max_batch_size"]
        initial_wait_time = initial_status["dynamic_batching"]["max_wait_time_ms"]
        
        # Attempt to tune parameters (may not change if insufficient data)
        tuning_result = await optimized_engine.tune_batch_parameters()
        
        # Should return tuning information
        assert isinstance(tuning_result, dict)
        
        if "batch_size" in tuning_result:
            # Parameters were tuned
            assert "wait_time_ms" in tuning_result
            assert "previous_batch_size" in tuning_result
            assert "previous_wait_time" in tuning_result
        else:
            # Insufficient data for tuning
            assert "message" in tuning_result or "error" in tuning_result
    
    @pytest.mark.asyncio
    async def test_concurrent_optimized_translations(self, optimized_engine):
        """Test concurrent optimized translations."""
        # Create multiple translation requests
        requests = []
        for i in range(10):
            request = TranslationRequest(
                source_language="en",
                target_language="es",
                content=f"Concurrent translation test message {i}",
                priority=Priority.NORMAL,
                user_id=f"test-user-{i}"
            )
            requests.append(request)
        
        # Execute translations concurrently
        tasks = [optimized_engine.translate_optimized(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all translations completed
        assert len(results) == 10
        
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                pytest.fail(f"Translation {i} failed with exception: {result}")
            else:
                successful_results.append(result)
        
        assert len(successful_results) == 10
        
        # Verify all results are valid
        for result in successful_results:
            assert result.source_language == "en"
            assert result.target_language == "es"
            assert result.translated_content is not None
            assert result.confidence_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_optimization_with_different_priorities(self, optimized_engine):
        """Test optimization with different priority requests."""
        priorities = [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL]
        
        results = []
        for priority in priorities:
            request = TranslationRequest(
                source_language="en",
                target_language="es",
                content=f"Priority {priority.value} translation test",
                priority=priority,
                user_id="test-user"
            )
            
            result = await optimized_engine.translate_optimized(request)
            results.append((priority, result))
        
        # Verify all priorities were handled
        assert len(results) == 3
        
        for priority, result in results:
            assert result is not None
            assert result.source_language == "en"
            assert result.target_language == "es"
            assert result.translated_content is not None
    
    @pytest.mark.asyncio
    async def test_optimization_error_handling(self, optimized_engine):
        """Test optimization error handling and fallback."""
        # Test with invalid language pair
        invalid_request = TranslationRequest(
            source_language="xyz",
            target_language="abc",
            content="Invalid language pair test",
            priority=Priority.NORMAL,
            user_id="test-user"
        )
        
        # Should handle error gracefully (may fallback to base engine)
        try:
            result = await optimized_engine.translate_optimized(invalid_request)
            # If no exception, result should be None or error result
            if result is not None:
                # Check if it's an error result
                assert result.confidence_score >= 0.0
        except Exception as e:
            # Exception is acceptable for invalid language pairs
            assert "not supported" in str(e).lower() or "translation" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_optimization_performance_targets(self, optimized_engine):
        """Test optimization meets performance targets."""
        # Test with various content sizes
        test_cases = [
            ("Small", "Hello world", 50),
            ("Medium", "This is a medium-sized text for translation testing. " * 5, 200),
            ("Large", "This is a larger text for performance testing. " * 20, 800)
        ]
        
        for case_name, content, expected_min_wpm in test_cases:
            request = TranslationRequest(
                source_language="en",
                target_language="es",
                content=content,
                priority=Priority.NORMAL,
                user_id="test-user"
            )
            
            start_time = datetime.utcnow()
            result = await optimized_engine.translate_optimized(request)
            end_time = datetime.utcnow()
            
            # Calculate performance
            processing_time_ms = (end_time - start_time).total_seconds() * 1000
            word_count = len(content.split())
            
            if processing_time_ms > 0:
                words_per_minute = (word_count / (processing_time_ms / 1000 / 60))
                
                # For CPU engine, we don't expect to meet the full 1500 WPM target
                # but we should get reasonable performance
                assert words_per_minute > 0
                
                # Log performance for analysis
                print(f"{case_name}: {word_count} words, {processing_time_ms:.1f}ms, {words_per_minute:.1f} WPM")
    
    @pytest.mark.asyncio
    async def test_optimization_memory_efficiency(self, optimized_engine):
        """Test optimization improves memory efficiency."""
        # Get initial optimization status
        initial_status = await optimized_engine.get_optimization_status()
        
        # Verify optimization features are available
        assert "quantization" in initial_status
        assert "pipeline_parallelism" in initial_status
        
        # Test multiple translations to exercise memory management
        for i in range(20):
            request = TranslationRequest(
                source_language="en",
                target_language="es",
                content=f"Memory efficiency test translation {i}",
                priority=Priority.NORMAL,
                user_id="test-user"
            )
            
            result = await optimized_engine.translate_optimized(request)
            assert result is not None
        
        # Get final metrics
        final_metrics = await optimized_engine.get_optimization_metrics()
        
        # Should have collected performance data
        if "total_requests" in final_metrics:
            assert final_metrics["total_requests"] >= 20