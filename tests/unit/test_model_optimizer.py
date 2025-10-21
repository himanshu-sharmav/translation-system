"""
Unit tests for ModelOptimizer and optimization features.
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.config.config import Priority
from src.models.interfaces import TranslationRequest, TranslationResult
from src.services.model_optimizer import (
    DynamicBatcher, 
    ModelQuantizer, 
    PipelineParallelizer,
    OptimizedTranslationEngine,
    BatchRequest,
    OptimizationMetrics
)


class TestDynamicBatcher:
    """Test dynamic batching functionality."""
    
    @pytest.fixture
    def batcher(self):
        """Create dynamic batcher instance."""
        return DynamicBatcher(max_batch_size=4, max_wait_time_ms=50)
    
    @pytest.fixture
    def sample_request(self):
        """Create sample translation request."""
        return TranslationRequest(
            source_language="en",
            target_language="es",
            content="Hello world",
            priority=Priority.NORMAL,
            user_id="test-user"
        )
    
    @pytest.mark.asyncio
    async def test_batcher_initialization(self, batcher):
        """Test batcher initialization."""
        assert batcher.max_batch_size == 4
        assert batcher.max_wait_time_ms == 50
        assert len(batcher.pending_requests) == 0
        assert batcher.processing is False
    
    @pytest.mark.asyncio
    async def test_collect_single_request(self, batcher, sample_request):
        """Test collecting a single request into batch."""
        # Add request to pending
        batch_request = BatchRequest(
            request=sample_request,
            request_id="test-1",
            timestamp=datetime.utcnow(),
            future=asyncio.Future()
        )
        batcher.pending_requests.append(batch_request)
        
        # Collect batch
        batch = await batcher._collect_batch()
        
        assert len(batch) == 1
        assert batch[0].request_id == "test-1"
        assert len(batcher.pending_requests) == 0
    
    @pytest.mark.asyncio
    async def test_collect_multiple_requests(self, batcher, sample_request):
        """Test collecting multiple requests into batch."""
        # Add multiple requests
        for i in range(3):
            batch_request = BatchRequest(
                request=sample_request,
                request_id=f"test-{i}",
                timestamp=datetime.utcnow(),
                future=asyncio.Future()
            )
            batcher.pending_requests.append(batch_request)
        
        # Collect batch
        batch = await batcher._collect_batch()
        
        assert len(batch) == 3
        assert len(batcher.pending_requests) == 0
    
    @pytest.mark.asyncio
    async def test_collect_batch_size_limit(self, batcher, sample_request):
        """Test batch collection respects size limit."""
        # Add more requests than max batch size
        for i in range(6):
            batch_request = BatchRequest(
                request=sample_request,
                request_id=f"test-{i}",
                timestamp=datetime.utcnow(),
                future=asyncio.Future()
            )
            batcher.pending_requests.append(batch_request)
        
        # Collect batch
        batch = await batcher._collect_batch()
        
        # Should only collect up to max batch size
        assert len(batch) == 4
        assert len(batcher.pending_requests) == 2  # Remaining requests
    
    @pytest.mark.asyncio
    async def test_collect_batch_timeout(self, batcher, sample_request):
        """Test batch collection respects timeout."""
        # Add one request
        batch_request = BatchRequest(
            request=sample_request,
            request_id="test-1",
            timestamp=datetime.utcnow(),
            future=asyncio.Future()
        )
        batcher.pending_requests.append(batch_request)
        
        # Mock time to simulate timeout
        with patch('time.time', side_effect=[0, 0.1]):  # 100ms elapsed
            batch = await batcher._collect_batch()
        
        # Should collect the single request due to timeout
        assert len(batch) == 1


class TestModelQuantizer:
    """Test model quantization functionality."""
    
    @pytest.fixture
    def quantizer(self):
        """Create model quantizer instance."""
        return ModelQuantizer()
    
    @pytest.fixture
    def mock_model(self):
        """Create mock PyTorch model."""
        model = MagicMock()
        model.half.return_value = model
        model.eval.return_value = model
        return model
    
    def test_quantizer_initialization(self, quantizer):
        """Test quantizer initialization."""
        assert isinstance(quantizer.quantization_methods, list)
        assert "int8" in quantizer.quantization_methods
        assert "fp16" in quantizer.quantization_methods
        assert "dynamic" in quantizer.quantization_methods
    
    def test_quantize_model_fp16(self, quantizer, mock_model):
        """Test FP16 quantization."""
        with patch('torch.cuda.is_available', return_value=True):
            result = quantizer.quantize_model(mock_model, "fp16")
            
            mock_model.half.assert_called_once()
            assert result == mock_model
    
    def test_quantize_model_int8(self, quantizer, mock_model):
        """Test INT8 quantization."""
        with patch('torch.quantization.quantize_dynamic') as mock_quantize:
            mock_quantize.return_value = mock_model
            
            result = quantizer.quantize_model(mock_model, "int8")
            
            mock_model.eval.assert_called_once()
            mock_quantize.assert_called_once()
            assert result == mock_model
    
    def test_quantize_model_dynamic(self, quantizer, mock_model):
        """Test dynamic quantization."""
        with patch('torch.quantization.quantize_dynamic') as mock_quantize:
            mock_quantize.return_value = mock_model
            
            result = quantizer.quantize_model(mock_model, "dynamic")
            
            mock_model.eval.assert_called_once()
            mock_quantize.assert_called_once()
            assert result == mock_model
    
    def test_quantize_model_unknown_method(self, quantizer, mock_model):
        """Test quantization with unknown method."""
        result = quantizer.quantize_model(mock_model, "unknown")
        
        # Should return original model
        assert result == mock_model
    
    def test_estimate_memory_savings(self, quantizer):
        """Test memory savings estimation."""
        original_size = 1000  # MB
        
        # Test different quantization methods
        fp16_savings = quantizer.estimate_memory_savings(original_size, "fp16")
        assert fp16_savings == 500  # 50% reduction
        
        int8_savings = quantizer.estimate_memory_savings(original_size, "int8")
        assert int8_savings == 750  # 75% reduction
        
        dynamic_savings = quantizer.estimate_memory_savings(original_size, "dynamic")
        assert dynamic_savings == 600  # 60% reduction
        
        unknown_savings = quantizer.estimate_memory_savings(original_size, "unknown")
        assert unknown_savings == 0  # No savings for unknown method


class TestPipelineParallelizer:
    """Test pipeline parallelism functionality."""
    
    @pytest.fixture
    def parallelizer(self):
        """Create pipeline parallelizer instance."""
        return PipelineParallelizer(num_stages=2)
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model with encoder/decoder."""
        model = MagicMock()
        model.encoder = MagicMock()
        model.decoder = MagicMock()
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        return MagicMock()
    
    def test_parallelizer_initialization(self, parallelizer):
        """Test parallelizer initialization."""
        assert parallelizer.num_stages == 2
        assert parallelizer.stages == []
        assert parallelizer.stage_devices == []
    
    @pytest.mark.asyncio
    async def test_setup_pipeline_no_gpu(self, parallelizer, mock_model, mock_tokenizer):
        """Test pipeline setup without multiple GPUs."""
        with patch('torch.cuda.is_available', return_value=False):
            result = await parallelizer.setup_pipeline(mock_model, mock_tokenizer)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_setup_pipeline_single_gpu(self, parallelizer, mock_model, mock_tokenizer):
        """Test pipeline setup with single GPU."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1):
            
            result = await parallelizer.setup_pipeline(mock_model, mock_tokenizer)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_setup_pipeline_multiple_gpus(self, parallelizer, mock_model, mock_tokenizer):
        """Test pipeline setup with multiple GPUs."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=2):
            
            # Mock the to() method for stages
            mock_model.encoder.to = MagicMock()
            mock_model.decoder.to = MagicMock()
            
            result = await parallelizer.setup_pipeline(mock_model, mock_tokenizer)
            
            assert result is True
            assert len(parallelizer.stages) == 2
            assert len(parallelizer.stage_devices) == 2
    
    def test_split_model_encoder_decoder(self, parallelizer):
        """Test model splitting for encoder-decoder architecture."""
        mock_model = MagicMock()
        mock_model.encoder = MagicMock()
        mock_model.decoder = MagicMock()
        
        stages = parallelizer._split_model_into_stages(mock_model)
        
        assert len(stages) == 2
        assert stages[0] == mock_model.encoder
        assert stages[1] == mock_model.decoder
    
    def test_assign_stages_to_devices(self, parallelizer):
        """Test stage assignment to GPU devices."""
        with patch('torch.cuda.device_count', return_value=4):
            devices = parallelizer._assign_stages_to_devices()
            
            assert len(devices) == 2  # num_stages
            assert all(device.type == "cuda" for device in devices)


class TestOptimizedTranslationEngine:
    """Test optimized translation engine."""
    
    @pytest.fixture
    def mock_base_engine(self):
        """Create mock base translation engine."""
        engine = AsyncMock()
        engine.translate.return_value = TranslationResult(
            job_id="test-job",
            translated_content="Hola mundo",
            source_language="en",
            target_language="es",
            confidence_score=0.9,
            model_version="1.0.0",
            processing_time_ms=100
        )
        engine.get_model_info.return_value = {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "memory_mb": 1000
        }
        return engine
    
    @pytest.fixture
    def optimized_engine(self, mock_base_engine):
        """Create optimized translation engine."""
        return OptimizedTranslationEngine(mock_base_engine)
    
    @pytest.fixture
    def sample_request(self):
        """Create sample translation request."""
        return TranslationRequest(
            source_language="en",
            target_language="es",
            content="Hello world",
            priority=Priority.NORMAL,
            user_id="test-user"
        )
    
    def test_optimized_engine_initialization(self, optimized_engine, mock_base_engine):
        """Test optimized engine initialization."""
        assert optimized_engine.base_engine == mock_base_engine
        assert optimized_engine.batcher is not None
        assert optimized_engine.quantizer is not None
        assert optimized_engine.parallelizer is not None
        assert optimized_engine.optimization_enabled is True
        assert optimized_engine.performance_metrics == []
    
    @pytest.mark.asyncio
    async def test_translate_optimized_disabled(self, optimized_engine, sample_request, mock_base_engine):
        """Test optimized translation when optimization is disabled."""
        optimized_engine.optimization_enabled = False
        
        result = await optimized_engine.translate_optimized(sample_request)
        
        mock_base_engine.translate.assert_called_once_with(sample_request)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_optimize_model_quantization(self, optimized_engine, mock_base_engine):
        """Test model optimization with quantization."""
        model_id = "test-model"
        optimization_config = {
            "quantization": {
                "enabled": True,
                "method": "fp16"
            }
        }
        
        mock_model = MagicMock()
        mock_model.half.return_value = mock_model
        
        mock_base_engine.get_model_info.return_value = {
            "model": mock_model,
            "tokenizer": MagicMock(),
            "memory_mb": 1000
        }
        
        with patch('torch.cuda.is_available', return_value=True):
            result = await optimized_engine.optimize_model(model_id, optimization_config)
        
        assert result is True
        mock_model.half.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_optimize_model_pipeline(self, optimized_engine, mock_base_engine):
        """Test model optimization with pipeline parallelism."""
        model_id = "test-model"
        optimization_config = {
            "pipeline": {
                "enabled": True,
                "stages": 3
            }
        }
        
        mock_model = MagicMock()
        mock_model.encoder = MagicMock()
        mock_model.decoder = MagicMock()
        
        mock_base_engine.get_model_info.return_value = {
            "model": mock_model,
            "tokenizer": MagicMock(),
            "memory_mb": 1000
        }
        
        with patch.object(optimized_engine.parallelizer, 'setup_pipeline', return_value=True):
            result = await optimized_engine.optimize_model(model_id, optimization_config)
        
        assert result is True
        assert optimized_engine.parallelizer.num_stages == 3
    
    @pytest.mark.asyncio
    async def test_get_optimization_metrics_no_data(self, optimized_engine):
        """Test getting optimization metrics with no data."""
        metrics = await optimized_engine.get_optimization_metrics()
        
        assert "message" in metrics
        assert metrics["message"] == "No metrics available"
    
    @pytest.mark.asyncio
    async def test_get_optimization_metrics_with_data(self, optimized_engine):
        """Test getting optimization metrics with performance data."""
        # Add some mock metrics
        for i in range(10):
            metrics = OptimizationMetrics(
                batch_size=4,
                processing_time_ms=100 + i * 10,
                throughput_wpm=1500 + i * 50,
                gpu_utilization=70.0 + i,
                memory_usage_mb=2000 + i * 100,
                quantization_enabled=i % 2 == 0,
                pipeline_stages=2 if i % 3 == 0 else 1
            )
            optimized_engine.performance_metrics.append(metrics)
        
        result = await optimized_engine.get_optimization_metrics()
        
        assert "total_requests" in result
        assert "recent_requests" in result
        assert "average_metrics" in result
        assert "optimization_usage" in result
        assert "performance_targets" in result
        
        assert result["total_requests"] == 10
        assert result["recent_requests"] == 10
        assert "batch_size" in result["average_metrics"]
        assert "throughput_wpm" in result["average_metrics"]
    
    @pytest.mark.asyncio
    async def test_tune_batch_parameters_insufficient_data(self, optimized_engine):
        """Test batch parameter tuning with insufficient data."""
        result = await optimized_engine.tune_batch_parameters()
        
        assert "message" in result
        assert result["message"] == "Insufficient data for tuning"
    
    @pytest.mark.asyncio
    async def test_tune_batch_parameters_high_latency(self, optimized_engine):
        """Test batch parameter tuning for high latency scenario."""
        # Add metrics with high latency
        for i in range(50):
            metrics = OptimizationMetrics(
                batch_size=8,
                processing_time_ms=300,  # High latency
                throughput_wpm=1200,     # Low throughput
                gpu_utilization=60.0,
                memory_usage_mb=2000,
                quantization_enabled=True,
                pipeline_stages=1
            )
            optimized_engine.performance_metrics.append(metrics)
        
        original_batch_size = optimized_engine.batcher.max_batch_size
        original_wait_time = optimized_engine.batcher.max_wait_time_ms
        
        result = await optimized_engine.tune_batch_parameters(target_latency_ms=200)
        
        # Should reduce batch size and wait time
        assert result["batch_size"] < original_batch_size
        assert result["wait_time_ms"] < original_wait_time
    
    @pytest.mark.asyncio
    async def test_tune_batch_parameters_low_throughput(self, optimized_engine):
        """Test batch parameter tuning for low throughput scenario."""
        # Add metrics with low throughput
        for i in range(50):
            metrics = OptimizationMetrics(
                batch_size=4,
                processing_time_ms=100,  # Good latency
                throughput_wpm=1200,     # Low throughput
                gpu_utilization=60.0,
                memory_usage_mb=2000,
                quantization_enabled=True,
                pipeline_stages=1
            )
            optimized_engine.performance_metrics.append(metrics)
        
        original_batch_size = optimized_engine.batcher.max_batch_size
        original_wait_time = optimized_engine.batcher.max_wait_time_ms
        
        result = await optimized_engine.tune_batch_parameters()
        
        # Should increase batch size and wait time
        assert result["batch_size"] > original_batch_size
        assert result["wait_time_ms"] > original_wait_time
    
    @pytest.mark.asyncio
    async def test_enable_optimization(self, optimized_engine):
        """Test enabling/disabling optimization."""
        # Initially enabled
        assert optimized_engine.optimization_enabled is True
        
        # Disable optimization
        await optimized_engine.enable_optimization(False)
        assert optimized_engine.optimization_enabled is False
        
        # Re-enable optimization
        await optimized_engine.enable_optimization(True)
        assert optimized_engine.optimization_enabled is True
    
    @pytest.mark.asyncio
    async def test_get_optimization_status(self, optimized_engine):
        """Test getting optimization status."""
        status = await optimized_engine.get_optimization_status()
        
        assert "optimization_enabled" in status
        assert "dynamic_batching" in status
        assert "quantization" in status
        assert "pipeline_parallelism" in status
        
        assert status["optimization_enabled"] is True
        assert "max_batch_size" in status["dynamic_batching"]
        assert "available" in status["quantization"]
        assert "available" in status["pipeline_parallelism"]