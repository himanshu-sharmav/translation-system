"""
Unit tests for TranslationEngine.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.config.config import Priority
from src.models.interfaces import TranslationRequest, TranslationResult
from src.services.translation_engine import GPUTranslationEngine, CPUTranslationEngine, create_translation_engine
from src.utils.exceptions import TranslationError, ModelLoadError


class TestGPUTranslationEngine:
    """Test GPU translation engine functionality."""
    
    @pytest.fixture
    def translation_engine(self):
        """Create GPU translation engine instance."""
        with patch('torch.cuda.is_available', return_value=True):
            return GPUTranslationEngine()
    
    @pytest.fixture
    def sample_request(self):
        """Create sample translation request."""
        return TranslationRequest(
            source_language="en",
            target_language="es",
            content="Hello, how are you today?",
            priority=Priority.NORMAL,
            user_id="test-user"
        )
    
    @pytest.mark.asyncio
    async def test_translation_engine_initialization(self, translation_engine):
        """Test translation engine initialization."""
        assert translation_engine.device is not None
        assert translation_engine.loaded_models == {}
        assert translation_engine.model_cache_size > 0
        assert translation_engine.max_memory_usage > 0
        assert len(translation_engine.supported_languages) > 0
    
    @pytest.mark.asyncio
    async def test_get_supported_languages(self, translation_engine):
        """Test getting supported languages."""
        languages = translation_engine.get_supported_languages()
        
        assert isinstance(languages, list)
        assert len(languages) > 0
        assert "en" in languages
        assert "es" in languages
    
    @pytest.mark.asyncio
    async def test_is_language_pair_supported(self, translation_engine):
        """Test language pair support validation."""
        # Valid language pairs
        assert translation_engine._is_language_pair_supported("en", "es") is True
        assert translation_engine._is_language_pair_supported("es", "en") is True
        
        # Invalid language pairs
        assert translation_engine._is_language_pair_supported("en", "en") is False
        assert translation_engine._is_language_pair_supported("xyz", "abc") is False
        assert translation_engine._is_language_pair_supported("en", "xyz") is False
    
    @pytest.mark.asyncio
    async def test_get_model_id(self, translation_engine):
        """Test model ID retrieval for language pairs."""
        # Should return a model ID for supported language pairs
        model_id = translation_engine._get_model_id("en", "es")
        assert model_id is not None
        assert isinstance(model_id, str)
        
        # Should raise error for unsupported pairs
        with pytest.raises(TranslationError):
            translation_engine._get_model_id("xyz", "abc")
    
    @pytest.mark.asyncio
    async def test_generate_content_hash(self, translation_engine):
        """Test content hash generation."""
        hash1 = translation_engine._generate_content_hash("Hello world", "en", "es")
        hash2 = translation_engine._generate_content_hash("Hello world", "en", "es")
        hash3 = translation_engine._generate_content_hash("Hello world", "es", "en")
        
        # Same content should produce same hash
        assert hash1 == hash2
        
        # Different language pair should produce different hash
        assert hash1 != hash3
        
        # Hash should be consistent format
        assert len(hash1) == 64  # SHA256 hex length
    
    @pytest.mark.asyncio
    async def test_calculate_confidence_score(self, translation_engine):
        """Test confidence score calculation."""
        # Normal translation
        score1 = translation_engine._calculate_confidence_score("Hello world", "Hola mundo")
        assert 0.0 <= score1 <= 1.0
        
        # Empty translation
        score2 = translation_engine._calculate_confidence_score("Hello world", "")
        assert score2 == 0.0
        
        # Identical text (likely untranslated)
        score3 = translation_engine._calculate_confidence_score("Hello world", "Hello world")
        assert score3 < 0.5
        
        # Very repetitive translation
        score4 = translation_engine._calculate_confidence_score("Hello world", "hello hello hello")
        assert score4 < 0.5
    
    @pytest.mark.asyncio
    async def test_prepare_input_text(self, translation_engine):
        """Test input text preparation for different model types."""
        text = "Hello world"
        
        # Test with different model configurations
        model_configs = {
            "prefix_model": {"requires_language_prefix": True},
            "target_model": {"requires_target_prefix": True},
            "direct_model": {}
        }
        
        translation_engine.model_configs.update(model_configs)
        
        # Language prefix model
        result1 = translation_engine._prepare_input_text(text, "en", "es", "prefix_model")
        assert "translate en to es:" in result1
        
        # Target prefix model
        result2 = translation_engine._prepare_input_text(text, "en", "es", "target_model")
        assert ">>es<<" in result2
        
        # Direct model
        result3 = translation_engine._prepare_input_text(text, "en", "es", "direct_model")
        assert result3 == text
    
    @pytest.mark.asyncio
    async def test_estimate_model_memory(self, translation_engine):
        """Test model memory estimation."""
        # Create a mock model
        mock_model = MagicMock()
        mock_param = MagicMock()
        mock_param.nelement.return_value = 1000
        mock_param.element_size.return_value = 4  # 4 bytes per element
        
        mock_buffer = MagicMock()
        mock_buffer.nelement.return_value = 100
        mock_buffer.element_size.return_value = 4
        
        mock_model.parameters.return_value = [mock_param]
        mock_model.buffers.return_value = [mock_buffer]
        
        memory_mb = translation_engine._estimate_model_memory(mock_model)
        
        assert memory_mb > 0
        assert isinstance(memory_mb, int)
        
        # Should account for parameters + buffers + overhead
        expected_bytes = (1000 + 100) * 4 * 1.5  # 1.5x overhead
        expected_mb = expected_bytes / (1024 * 1024)
        assert memory_mb == int(expected_mb)
    
    @pytest.mark.asyncio
    async def test_load_model_mock(self, translation_engine):
        """Test model loading with mocked dependencies."""
        model_id = "test-model"
        
        # Mock the model loading components
        with patch('src.services.translation_engine.AutoTokenizer') as mock_tokenizer, \
             patch('src.services.translation_engine.AutoModelForSeq2SeqLM') as mock_model, \
             patch('src.services.translation_engine.pipeline') as mock_pipeline:
            
            mock_tokenizer.from_pretrained.return_value = MagicMock()
            mock_model_instance = MagicMock()
            mock_model.from_pretrained.return_value = mock_model_instance
            mock_pipeline.return_value = MagicMock()
            
            # Add test model config
            translation_engine.model_configs[model_id] = {
                "model_name": "test/model",
                "version": "1.0.0"
            }
            
            # Mock memory estimation
            translation_engine._estimate_model_memory = MagicMock(return_value=1000)
            
            result = await translation_engine.load_model(model_id)
            
            assert result is True
            assert model_id in translation_engine.loaded_models
            
            model_info = translation_engine.loaded_models[model_id]
            assert model_info["version"] == "1.0.0"
            assert model_info["usage_count"] == 0
            assert model_info["memory_mb"] == 1000
    
    @pytest.mark.asyncio
    async def test_load_model_already_loaded(self, translation_engine):
        """Test loading a model that's already loaded."""
        model_id = "test-model"
        
        # Pre-load model
        translation_engine.loaded_models[model_id] = {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "translator": MagicMock(),
            "version": "1.0.0",
            "loaded_at": datetime.utcnow(),
            "last_used": datetime.utcnow(),
            "usage_count": 0,
            "memory_mb": 1000
        }
        
        result = await translation_engine.load_model(model_id)
        
        assert result is True
        # Should not create duplicate entry
        assert len([k for k in translation_engine.loaded_models.keys() if k == model_id]) == 1
    
    @pytest.mark.asyncio
    async def test_unload_model(self, translation_engine):
        """Test model unloading."""
        model_id = "test-model"
        
        # Pre-load model
        translation_engine.loaded_models[model_id] = {
            "model": MagicMock(),
            "tokenizer": MagicMock(),
            "translator": MagicMock(),
            "version": "1.0.0",
            "loaded_at": datetime.utcnow(),
            "last_used": datetime.utcnow(),
            "usage_count": 5,
            "memory_mb": 1000
        }
        
        with patch('gc.collect'), patch('torch.cuda.empty_cache'):
            result = await translation_engine.unload_model(model_id)
        
        assert result is True
        assert model_id not in translation_engine.loaded_models
    
    @pytest.mark.asyncio
    async def test_unload_model_not_loaded(self, translation_engine):
        """Test unloading a model that's not loaded."""
        model_id = "nonexistent-model"
        
        result = await translation_engine.unload_model(model_id)
        
        assert result is True  # Should succeed even if not loaded
    
    @pytest.mark.asyncio
    async def test_get_memory_usage_gpu(self, translation_engine):
        """Test GPU memory usage calculation."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.memory_allocated', return_value=1000), \
             patch('torch.cuda.memory_reserved', return_value=2000), \
             patch('torch.cuda.get_device_properties') as mock_props:
            
            mock_device = MagicMock()
            mock_device.total_memory = 10000
            mock_props.return_value = mock_device
            
            usage = translation_engine.get_memory_usage()
            
            assert usage == 20.0  # (2000 / 10000) * 100
    
    @pytest.mark.asyncio
    async def test_get_memory_usage_cpu_fallback(self, translation_engine):
        """Test CPU memory usage fallback."""
        with patch('torch.cuda.is_available', return_value=False), \
             patch('psutil.virtual_memory') as mock_memory:
            
            mock_mem = MagicMock()
            mock_mem.percent = 75.5
            mock_memory.return_value = mock_mem
            
            usage = translation_engine.get_memory_usage()
            
            assert usage == 75.5
    
    @pytest.mark.asyncio
    async def test_health_check(self, translation_engine):
        """Test translation engine health check."""
        with patch.object(translation_engine, 'get_memory_usage', return_value=50.0):
            health = await translation_engine.health_check()
            
            assert health["status"] == "healthy"
            assert health["memory_usage_percent"] == 50.0
            assert health["loaded_models"] == 0
            assert "timestamp" in health
    
    @pytest.mark.asyncio
    async def test_health_check_high_memory(self, translation_engine):
        """Test health check with high memory usage."""
        with patch.object(translation_engine, 'get_memory_usage', return_value=96.0):
            health = await translation_engine.health_check()
            
            assert health["status"] == "warning"
            assert "warning" in health
    
    @pytest.mark.asyncio
    async def test_optimize_memory(self, translation_engine):
        """Test memory optimization."""
        # Pre-load some models with different usage times
        old_time = datetime.utcnow()
        recent_time = datetime.utcnow()
        
        translation_engine.loaded_models = {
            "old_model": {
                "last_used": old_time,
                "memory_mb": 500,
                "model": MagicMock(),
                "tokenizer": MagicMock(),
                "translator": MagicMock()
            },
            "recent_model": {
                "last_used": recent_time,
                "memory_mb": 300,
                "model": MagicMock(),
                "tokenizer": MagicMock(),
                "translator": MagicMock()
            }
        }
        
        with patch.object(translation_engine, 'get_memory_usage', side_effect=[90.0, 70.0]), \
             patch('gc.collect'), \
             patch('torch.cuda.empty_cache'):
            
            result = await translation_engine.optimize_memory()
            
            assert result["unloaded_models"] >= 0
            assert result["memory_freed_mb"] >= 0


class TestCPUTranslationEngine:
    """Test CPU translation engine functionality."""
    
    @pytest.fixture
    def cpu_engine(self):
        """Create CPU translation engine instance."""
        return CPUTranslationEngine()
    
    @pytest.fixture
    def sample_request(self):
        """Create sample translation request."""
        return TranslationRequest(
            source_language="en",
            target_language="es",
            content="Hello, how are you today?",
            priority=Priority.NORMAL,
            user_id="test-user"
        )
    
    @pytest.mark.asyncio
    async def test_cpu_engine_initialization(self, cpu_engine):
        """Test CPU engine initialization."""
        assert str(cpu_engine.device) == "cpu"
        assert cpu_engine.loaded_models == {}
        assert len(cpu_engine.supported_languages) > 0
    
    @pytest.mark.asyncio
    async def test_cpu_translate(self, cpu_engine, sample_request):
        """Test CPU translation (simplified implementation)."""
        result = await cpu_engine.translate(sample_request)
        
        assert isinstance(result, TranslationResult)
        assert result.source_language == sample_request.source_language
        assert result.target_language == sample_request.target_language
        assert result.translated_content is not None
        assert result.confidence_score > 0
        assert result.processing_time_ms >= 0
    
    @pytest.mark.asyncio
    async def test_cpu_unsupported_language(self, cpu_engine):
        """Test CPU engine with unsupported language."""
        request = TranslationRequest(
            source_language="xyz",
            target_language="abc",
            content="Hello world",
            priority=Priority.NORMAL,
            user_id="test-user"
        )
        
        with pytest.raises(TranslationError):
            await cpu_engine.translate(request)
    
    @pytest.mark.asyncio
    async def test_cpu_get_memory_usage(self, cpu_engine):
        """Test CPU memory usage."""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_mem = MagicMock()
            mock_mem.percent = 60.0
            mock_memory.return_value = mock_mem
            
            usage = cpu_engine.get_memory_usage()
            assert usage == 60.0
    
    @pytest.mark.asyncio
    async def test_cpu_get_memory_usage_no_psutil(self, cpu_engine):
        """Test CPU memory usage without psutil."""
        with patch('psutil.virtual_memory', side_effect=ImportError):
            usage = cpu_engine.get_memory_usage()
            assert usage == 0.0


class TestTranslationEngineFactory:
    """Test translation engine factory function."""
    
    @pytest.mark.asyncio
    async def test_create_gpu_engine(self):
        """Test creating GPU engine when available."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('src.config.config.config.translation.use_gpu', True):
            
            engine = create_translation_engine()
            assert isinstance(engine, GPUTranslationEngine)
    
    @pytest.mark.asyncio
    async def test_create_cpu_engine_no_gpu(self):
        """Test creating CPU engine when GPU not available."""
        with patch('torch.cuda.is_available', return_value=False):
            engine = create_translation_engine()
            assert isinstance(engine, CPUTranslationEngine)
    
    @pytest.mark.asyncio
    async def test_create_cpu_engine_gpu_disabled(self):
        """Test creating CPU engine when GPU disabled in config."""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('src.config.config.config.translation.use_gpu', False):
            
            engine = create_translation_engine()
            assert isinstance(engine, CPUTranslationEngine)