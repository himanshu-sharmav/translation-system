"""
GPU-based translation engine implementation for the machine translation system.
"""

import asyncio
import gc
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from accelerate import Accelerator

from src.config.config import config
from src.models.interfaces import TranslationEngine, TranslationRequest, TranslationResult
from src.utils.exceptions import TranslationError, ModelLoadError, ResourceError
from src.utils.logging import TranslationLogger

logger = TranslationLogger(__name__, "translation-engine")


class GPUTranslationEngine(TranslationEngine):
    """GPU-accelerated translation engine with model management."""
    
    def __init__(self):
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.loaded_models: Dict[str, Dict] = {}
        self.model_cache_size = config.translation.max_models_in_memory
        self.max_memory_usage = config.translation.max_memory_usage_percent
        self.supported_languages = config.translation.supported_languages
        self.model_configs = config.translation.model_configs
        self.batch_size = config.translation.batch_size
        self.max_length = config.translation.max_sequence_length
        self._model_lock = asyncio.Lock()
        self._translation_lock = asyncio.Lock()
        
        # Optimization features
        self.optimization_enabled = True
        self.dynamic_batching_enabled = True
        self.quantization_enabled = True
        self.pipeline_parallelism_enabled = torch.cuda.device_count() > 1
        
        logger.info(
            f"Translation engine initialized",
            metadata={
                "device": str(self.device),
                "max_models": self.model_cache_size,
                "max_memory_percent": self.max_memory_usage,
                "supported_languages": len(self.supported_languages)
            }
        )
    
    async def translate(self, request: TranslationRequest) -> TranslationResult:
        """Translate text from source to target language."""
        start_time = time.time()
        
        try:
            # Validate language pair
            if not self._is_language_pair_supported(request.source_language, request.target_language):
                raise TranslationError(
                    f"Language pair {request.source_language}->{request.target_language} not supported"
                )
            
            # Get model for language pair
            model_id = self._get_model_id(request.source_language, request.target_language)
            
            # Load model if not already loaded
            await self._ensure_model_loaded(model_id)
            
            # Perform translation
            async with self._translation_lock:
                translated_text, confidence_score = await self._perform_translation(
                    request.content,
                    request.source_language,
                    request.target_language,
                    model_id
                )
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Calculate words per minute
            word_count = len(request.content.split())
            words_per_minute = (word_count / (processing_time_ms / 1000 / 60)) if processing_time_ms > 0 else 0
            
            logger.info(
                f"Translation completed",
                metadata={
                    "source_language": request.source_language,
                    "target_language": request.target_language,
                    "word_count": word_count,
                    "processing_time_ms": processing_time_ms,
                    "words_per_minute": words_per_minute,
                    "confidence_score": confidence_score,
                    "model_id": model_id
                }
            )
            
            return TranslationResult(
                job_id=uuid4(),  # This will be set by the calling service
                translated_content=translated_text,
                source_language=request.source_language,
                target_language=request.target_language,
                confidence_score=confidence_score,
                model_version=self.loaded_models[model_id]["version"],
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            logger.error(
                f"Translation failed",
                metadata={
                    "source_language": request.source_language,
                    "target_language": request.target_language,
                    "error": str(e)
                },
                exc_info=True
            )
            raise TranslationError(f"Translation failed: {str(e)}")
    
    async def load_model(self, model_id: str) -> bool:
        """Load a specific translation model."""
        async with self._model_lock:
            try:
                if model_id in self.loaded_models:
                    logger.debug(f"Model {model_id} already loaded")
                    return True
                
                # Check memory usage before loading
                current_memory = self.get_memory_usage()
                if current_memory > self.max_memory_usage:
                    logger.warning(
                        f"Memory usage too high ({current_memory:.1f}%), unloading models"
                    )
                    await self._unload_least_used_model()
                
                # Check if we need to unload models to make space
                if len(self.loaded_models) >= self.model_cache_size:
                    await self._unload_least_used_model()
                
                logger.info(f"Loading model {model_id}")
                start_time = time.time()
                
                # Get model configuration
                model_config = self.model_configs.get(model_id)
                if not model_config:
                    raise ModelLoadError(f"Model configuration not found for {model_id}")
                
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(
                    model_config["model_name"],
                    cache_dir=config.translation.model_cache_dir
                )
                
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_config["model_name"],
                    cache_dir=config.translation.model_cache_dir,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    device_map="auto" if self.device.type == "cuda" else None,
                    low_cpu_mem_usage=True
                )
                
                # Move to device if not using device_map
                if self.device.type != "cuda":
                    model = model.to(self.device)
                
                # Create translation pipeline
                translator = pipeline(
                    "translation",
                    model=model,
                    tokenizer=tokenizer,
                    device=self.device if self.device.type != "cuda" else None,
                    batch_size=self.batch_size,
                    max_length=self.max_length
                )
                
                load_time = time.time() - start_time
                
                # Store model information
                self.loaded_models[model_id] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "translator": translator,
                    "version": model_config.get("version", "1.0.0"),
                    "loaded_at": datetime.utcnow(),
                    "last_used": datetime.utcnow(),
                    "usage_count": 0,
                    "load_time_seconds": load_time,
                    "memory_mb": self._estimate_model_memory(model)
                }
                
                logger.info(
                    f"Model {model_id} loaded successfully",
                    metadata={
                        "load_time_seconds": load_time,
                        "estimated_memory_mb": self.loaded_models[model_id]["memory_mb"],
                        "total_loaded_models": len(self.loaded_models)
                    }
                )
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {str(e)}", exc_info=True)
                raise ModelLoadError(f"Failed to load model {model_id}: {str(e)}")
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload a translation model to free memory."""
        async with self._model_lock:
            try:
                if model_id not in self.loaded_models:
                    logger.debug(f"Model {model_id} not loaded")
                    return True
                
                logger.info(f"Unloading model {model_id}")
                
                # Get model info before deletion
                model_info = self.loaded_models[model_id]
                
                # Delete model components
                del model_info["model"]
                del model_info["tokenizer"]
                del model_info["translator"]
                del self.loaded_models[model_id]
                
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(
                    f"Model {model_id} unloaded successfully",
                    metadata={
                        "usage_count": model_info["usage_count"],
                        "estimated_memory_freed_mb": model_info["memory_mb"],
                        "remaining_loaded_models": len(self.loaded_models)
                    }
                )
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to unload model {model_id}: {str(e)}", exc_info=True)
                return False
    
    def get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        try:
            if torch.cuda.is_available():
                # GPU memory usage
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                total = torch.cuda.get_device_properties(0).total_memory
                return (reserved / total) * 100
            else:
                # CPU memory usage (simplified)
                import psutil
                return psutil.virtual_memory().percent
        except Exception as e:
            logger.error(f"Failed to get memory usage: {str(e)}")
            return 0.0
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return list(self.supported_languages)
    
    async def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get information about a loaded model."""
        return self.loaded_models.get(model_id)
    
    async def get_all_loaded_models(self) -> Dict[str, Dict]:
        """Get information about all loaded models."""
        return {
            model_id: {
                "version": info["version"],
                "loaded_at": info["loaded_at"].isoformat(),
                "last_used": info["last_used"].isoformat(),
                "usage_count": info["usage_count"],
                "memory_mb": info["memory_mb"]
            }
            for model_id, info in self.loaded_models.items()
        }
    
    async def optimize_memory(self) -> Dict[str, int]:
        """Optimize memory usage by unloading unused models."""
        async with self._model_lock:
            unloaded_count = 0
            memory_freed = 0
            
            # Sort models by last used time (oldest first)
            sorted_models = sorted(
                self.loaded_models.items(),
                key=lambda x: x[1]["last_used"]
            )
            
            current_memory = self.get_memory_usage()
            target_memory = self.max_memory_usage * 0.8  # Target 80% of max
            
            for model_id, model_info in sorted_models:
                if current_memory <= target_memory:
                    break
                
                memory_freed += model_info["memory_mb"]
                await self.unload_model(model_id)
                unloaded_count += 1
                current_memory = self.get_memory_usage()
            
            logger.info(
                f"Memory optimization completed",
                metadata={
                    "unloaded_models": unloaded_count,
                    "memory_freed_mb": memory_freed,
                    "current_memory_percent": current_memory
                }
            )
            
            return {
                "unloaded_models": unloaded_count,
                "memory_freed_mb": memory_freed
            }
    
    def _is_language_pair_supported(self, source_lang: str, target_lang: str) -> bool:
        """Check if language pair is supported."""
        return (source_lang in self.supported_languages and 
                target_lang in self.supported_languages and
                source_lang != target_lang)
    
    def _get_model_id(self, source_lang: str, target_lang: str) -> str:
        """Get model ID for language pair."""
        # Create a consistent model ID for language pair
        lang_pair = f"{source_lang}-{target_lang}"
        
        # Check if we have a specific model for this pair
        for model_id, config in self.model_configs.items():
            if lang_pair in config.get("language_pairs", []):
                return model_id
        
        # Fall back to multilingual model
        for model_id, config in self.model_configs.items():
            if config.get("multilingual", False):
                return model_id
        
        raise TranslationError(f"No model available for language pair {lang_pair}")
    
    async def _ensure_model_loaded(self, model_id: str) -> None:
        """Ensure model is loaded, loading if necessary."""
        if model_id not in self.loaded_models:
            await self.load_model(model_id)
        else:
            # Update last used time
            self.loaded_models[model_id]["last_used"] = datetime.utcnow()
    
    async def _perform_translation(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str, 
        model_id: str
    ) -> Tuple[str, float]:
        """Perform the actual translation."""
        try:
            model_info = self.loaded_models[model_id]
            translator = model_info["translator"]
            
            # Update usage statistics
            model_info["usage_count"] += 1
            model_info["last_used"] = datetime.utcnow()
            
            # Prepare input text with language codes if needed
            input_text = self._prepare_input_text(text, source_lang, target_lang, model_id)
            
            # Perform translation
            result = translator(
                input_text,
                max_length=self.max_length,
                num_return_sequences=1,
                do_sample=False,  # Use greedy decoding for consistency
                early_stopping=True
            )
            
            if not result or len(result) == 0:
                raise TranslationError("Translation model returned empty result")
            
            translated_text = result[0]["translation_text"]
            
            # Calculate confidence score (simplified)
            confidence_score = self._calculate_confidence_score(text, translated_text)
            
            return translated_text, confidence_score
            
        except Exception as e:
            logger.error(f"Translation execution failed: {str(e)}", exc_info=True)
            raise TranslationError(f"Translation execution failed: {str(e)}")
    
    def _prepare_input_text(self, text: str, source_lang: str, target_lang: str, model_id: str) -> str:
        """Prepare input text with language codes if needed."""
        model_config = self.model_configs.get(model_id, {})
        
        if model_config.get("requires_language_prefix", False):
            # Some models require language prefixes
            return f"translate {source_lang} to {target_lang}: {text}"
        elif model_config.get("requires_target_prefix", False):
            # Some models require target language prefix
            return f">>{target_lang}<< {text}"
        else:
            # Direct text input
            return text
    
    def _calculate_confidence_score(self, source_text: str, translated_text: str) -> float:
        """Calculate confidence score for translation (simplified implementation)."""
        try:
            # Simple heuristic based on length ratio and content
            source_len = len(source_text.split())
            translated_len = len(translated_text.split())
            
            if source_len == 0 or translated_len == 0:
                return 0.5
            
            # Length ratio score (closer to 1.0 is better for most language pairs)
            length_ratio = min(source_len, translated_len) / max(source_len, translated_len)
            
            # Basic content score (check for obvious issues)
            content_score = 1.0
            if translated_text.strip() == "":
                content_score = 0.0
            elif translated_text == source_text:
                content_score = 0.3  # Likely untranslated
            elif len(set(translated_text.split())) < 2:
                content_score = 0.4  # Too repetitive
            
            # Combine scores
            confidence = (length_ratio * 0.3 + content_score * 0.7)
            
            # Clamp to reasonable range
            return max(0.1, min(0.99, confidence))
            
        except Exception:
            return 0.5  # Default confidence if calculation fails
    
    async def _unload_least_used_model(self) -> None:
        """Unload the least recently used model."""
        if not self.loaded_models:
            return
        
        # Find least recently used model
        least_used_model = min(
            self.loaded_models.items(),
            key=lambda x: x[1]["last_used"]
        )
        
        model_id = least_used_model[0]
        await self.unload_model(model_id)
    
    def _estimate_model_memory(self, model: nn.Module) -> int:
        """Estimate model memory usage in MB."""
        try:
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            total_bytes = param_size + buffer_size
            total_mb = total_bytes / (1024 * 1024)
            
            # Add overhead estimate (activations, gradients, etc.)
            total_mb *= 1.5
            
            return int(total_mb)
            
        except Exception:
            return 1000  # Default estimate if calculation fails
    
    async def health_check(self) -> Dict[str, any]:
        """Perform health check on the translation engine."""
        try:
            health_status = {
                "status": "healthy",
                "device": str(self.device),
                "memory_usage_percent": self.get_memory_usage(),
                "loaded_models": len(self.loaded_models),
                "max_models": self.model_cache_size,
                "supported_languages": len(self.supported_languages),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check if memory usage is too high
            if health_status["memory_usage_percent"] > 95:
                health_status["status"] = "warning"
                health_status["warning"] = "High memory usage"
            
            # Check if GPU is available when expected
            if config.translation.require_gpu and not torch.cuda.is_available():
                health_status["status"] = "error"
                health_status["error"] = "GPU not available"
            
            return health_status
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def apply_model_optimizations(self, model_id: str, optimizations: Dict[str, Any]) -> bool:
        """Apply optimizations to a loaded model."""
        try:
            if model_id not in self.loaded_models:
                logger.error(f"Model {model_id} not loaded")
                return False
            
            model_info = self.loaded_models[model_id]
            model = model_info["model"]
            
            # Apply quantization if requested
            if optimizations.get("quantization", {}).get("enabled", False):
                method = optimizations["quantization"].get("method", "fp16")
                optimized_model = await self._apply_quantization(model, method)
                if optimized_model is not model:
                    model_info["model"] = optimized_model
                    model_info["quantization_method"] = method
                    logger.info(f"Applied {method} quantization to model {model_id}")
            
            # Apply memory optimization
            if optimizations.get("memory_optimization", False):
                await self._optimize_model_memory(model_id)
            
            # Enable gradient checkpointing for large models
            if optimizations.get("gradient_checkpointing", False):
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                    logger.info(f"Enabled gradient checkpointing for model {model_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply optimizations to model {model_id}: {str(e)}")
            return False
    
    async def _apply_quantization(self, model: nn.Module, method: str) -> nn.Module:
        """Apply quantization to model."""
        try:
            if method == "fp16" and torch.cuda.is_available():
                return model.half()
            elif method == "int8":
                model.eval()
                return torch.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
            elif method == "dynamic":
                model.eval()
                return torch.quantization.quantize_dynamic(
                    model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=torch.qint8
                )
            else:
                logger.warning(f"Unknown quantization method: {method}")
                return model
        except Exception as e:
            logger.error(f"Quantization failed: {str(e)}")
            return model
    
    async def _optimize_model_memory(self, model_id: str):
        """Optimize model memory usage."""
        try:
            model_info = self.loaded_models[model_id]
            model = model_info["model"]
            
            # Enable memory efficient attention if available
            if hasattr(model.config, "use_memory_efficient_attention"):
                model.config.use_memory_efficient_attention = True
            
            # Optimize attention computation
            if hasattr(model, "enable_memory_efficient_attention"):
                model.enable_memory_efficient_attention()
            
            # Set model to eval mode for inference optimization
            model.eval()
            
            logger.info(f"Applied memory optimizations to model {model_id}")
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {str(e)}")
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        return {
            "optimization_enabled": self.optimization_enabled,
            "dynamic_batching_enabled": self.dynamic_batching_enabled,
            "quantization_enabled": self.quantization_enabled,
            "pipeline_parallelism_enabled": self.pipeline_parallelism_enabled,
            "device_info": {
                "device": str(self.device),
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "memory_allocated_mb": torch.cuda.memory_allocated() // (1024 * 1024) if torch.cuda.is_available() else 0,
                "memory_reserved_mb": torch.cuda.memory_reserved() // (1024 * 1024) if torch.cuda.is_available() else 0
            },
            "loaded_models": {
                model_id: {
                    "quantization_method": info.get("quantization_method"),
                    "memory_mb": info.get("memory_mb", 0),
                    "usage_count": info.get("usage_count", 0)
                }
                for model_id, info in self.loaded_models.items()
            }
        }
    
    async def enable_dynamic_batching(self, enabled: bool = True, batch_size: int = None):
        """Enable or disable dynamic batching."""
        self.dynamic_batching_enabled = enabled
        if batch_size:
            self.batch_size = batch_size
        
        logger.info(f"Dynamic batching {'enabled' if enabled else 'disabled'}, batch size: {self.batch_size}")
    
    async def process_batch(self, requests: List[TranslationRequest]) -> List[TranslationResult]:
        """Process multiple translation requests as a batch."""
        if not requests:
            return []
        
        try:
            # Group requests by language pair and model
            batches_by_model = {}
            for request in requests:
                model_id = self._get_model_id(request.source_language, request.target_language)
                if model_id not in batches_by_model:
                    batches_by_model[model_id] = []
                batches_by_model[model_id].append(request)
            
            # Process each model's batch
            all_results = []
            for model_id, model_requests in batches_by_model.items():
                batch_results = await self._process_model_batch(model_id, model_requests)
                all_results.extend(batch_results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            # Fallback to individual processing
            results = []
            for request in requests:
                try:
                    result = await self.translate(request)
                    results.append(result)
                except Exception as req_e:
                    logger.error(f"Individual request failed: {str(req_e)}")
                    # Create error result
                    error_result = TranslationResult(
                        job_id=uuid4(),
                        translated_content="",
                        source_language=request.source_language,
                        target_language=request.target_language,
                        confidence_score=0.0,
                        model_version="error",
                        processing_time_ms=0
                    )
                    results.append(error_result)
            return results
    
    async def _process_model_batch(self, model_id: str, requests: List[TranslationRequest]) -> List[TranslationResult]:
        """Process a batch of requests for a specific model."""
        try:
            await self._ensure_model_loaded(model_id)
            model_info = self.loaded_models[model_id]
            translator = model_info["translator"]
            
            # Prepare batch inputs
            input_texts = []
            for request in requests:
                input_text = self._prepare_input_text(
                    request.content, 
                    request.source_language, 
                    request.target_language, 
                    model_id
                )
                input_texts.append(input_text)
            
            # Process batch
            start_time = time.time()
            batch_results = translator(
                input_texts,
                max_length=self.max_length,
                num_return_sequences=1,
                do_sample=False,
                early_stopping=True,
                batch_size=len(input_texts)
            )
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Create results
            results = []
            for i, (request, translation_result) in enumerate(zip(requests, batch_results)):
                translated_text = translation_result["translation_text"]
                confidence_score = self._calculate_confidence_score(request.content, translated_text)
                
                result = TranslationResult(
                    job_id=uuid4(),
                    translated_content=translated_text,
                    source_language=request.source_language,
                    target_language=request.target_language,
                    confidence_score=confidence_score,
                    model_version=model_info["version"],
                    processing_time_ms=processing_time_ms // len(requests)  # Distribute time across batch
                )
                results.append(result)
            
            # Update model usage statistics
            model_info["usage_count"] += len(requests)
            model_info["last_used"] = datetime.utcnow()
            
            logger.info(
                f"Batch processing completed",
                metadata={
                    "model_id": model_id,
                    "batch_size": len(requests),
                    "total_processing_time_ms": processing_time_ms,
                    "avg_time_per_request_ms": processing_time_ms // len(requests)
                }
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Model batch processing failed for {model_id}: {str(e)}")
            raise
    
    async def close(self) -> None:
        """Clean up resources and unload all models."""
        logger.info("Shutting down translation engine")
        
        # Unload all models
        model_ids = list(self.loaded_models.keys())
        for model_id in model_ids:
            await self.unload_model(model_id)
        
        # Clear any remaining references
        self.loaded_models.clear()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Translation engine shutdown complete")


class CPUTranslationEngine(TranslationEngine):
    """CPU-based translation engine for fallback scenarios."""
    
    def __init__(self):
        self.device = torch.device("cpu")
        self.loaded_models: Dict[str, Dict] = {}
        self.supported_languages = config.translation.supported_languages
        self.model_configs = config.translation.model_configs
        self.max_length = config.translation.max_sequence_length
        self._model_lock = asyncio.Lock()
        
        logger.info(
            f"CPU Translation engine initialized",
            metadata={
                "device": str(self.device),
                "supported_languages": len(self.supported_languages)
            }
        )
    
    async def translate(self, request: TranslationRequest) -> TranslationResult:
        """Translate text using CPU-based models."""
        # Similar implementation to GPU engine but optimized for CPU
        # This is a simplified version for demonstration
        start_time = time.time()
        
        try:
            if not self._is_language_pair_supported(request.source_language, request.target_language):
                raise TranslationError(
                    f"Language pair {request.source_language}->{request.target_language} not supported"
                )
            
            # For CPU, we'll use a simpler approach
            # In a real implementation, this would use CPU-optimized models
            translated_text = f"[CPU Translation] {request.content}"
            confidence_score = 0.8
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return TranslationResult(
                job_id=uuid4(),
                translated_content=translated_text,
                source_language=request.source_language,
                target_language=request.target_language,
                confidence_score=confidence_score,
                model_version="cpu-1.0.0",
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            logger.error(f"CPU translation failed: {str(e)}", exc_info=True)
            raise TranslationError(f"CPU translation failed: {str(e)}")
    
    async def load_model(self, model_id: str) -> bool:
        """Load CPU-optimized model."""
        # Simplified implementation
        return True
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload CPU model."""
        # Simplified implementation
        return True
    
    def get_memory_usage(self) -> float:
        """Get CPU memory usage."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0
    
    def get_supported_languages(self) -> List[str]:
        """Get supported languages."""
        return list(self.supported_languages)
    
    def _is_language_pair_supported(self, source_lang: str, target_lang: str) -> bool:
        """Check if language pair is supported."""
        return (source_lang in self.supported_languages and 
                target_lang in self.supported_languages and
                source_lang != target_lang)


def create_translation_engine() -> TranslationEngine:
    """Factory function to create appropriate translation engine."""
    if torch.cuda.is_available() and config.translation.use_gpu:
        logger.info("Creating GPU translation engine")
        return GPUTranslationEngine()
    else:
        logger.info("Creating CPU translation engine (GPU not available or disabled)")
        return CPUTranslationEngine()