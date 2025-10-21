"""
Model optimization and performance features for the translation engine.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from uuid import uuid4

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from accelerate import Accelerator

from src.config.config import config
from src.models.interfaces import TranslationRequest, TranslationResult
from src.utils.exceptions import TranslationError, ModelLoadError
from src.utils.logging import TranslationLogger

logger = TranslationLogger(__name__, "model-optimizer")


@dataclass
class BatchRequest:
    """Request in a batch with metadata."""
    request: TranslationRequest
    request_id: str
    timestamp: datetime
    future: asyncio.Future


@dataclass
class OptimizationMetrics:
    """Metrics for model optimization."""
    batch_size: int
    processing_time_ms: int
    throughput_wpm: float
    gpu_utilization: float
    memory_usage_mb: int
    quantization_enabled: bool
    pipeline_stages: int


class DynamicBatcher:
    """Dynamic batching system for improved GPU utilization."""
    
    def __init__(self, max_batch_size: int = 16, max_wait_time_ms: int = 100):
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.pending_requests: deque = deque()
        self.batch_lock = asyncio.Lock()
        self.processing = False
        
    async def add_request(self, request: TranslationRequest) -> TranslationResult:
        """Add request to batch and wait for result."""
        request_id = str(uuid4())
        future = asyncio.Future()
        
        batch_request = BatchRequest(
            request=request,
            request_id=request_id,
            timestamp=datetime.utcnow(),
            future=future
        )
        
        async with self.batch_lock:
            self.pending_requests.append(batch_request)
            
            # Start batch processing if not already running
            if not self.processing:
                asyncio.create_task(self._process_batches())
        
        # Wait for result
        return await future
    
    async def _process_batches(self):
        """Process batches of requests."""
        self.processing = True
        
        try:
            while True:
                batch = await self._collect_batch()
                if not batch:
                    break
                
                # Process batch
                await self._process_batch(batch)
                
        finally:
            self.processing = False
    
    async def _collect_batch(self) -> List[BatchRequest]:
        """Collect requests into a batch."""
        batch = []
        start_time = time.time()
        
        while len(batch) < self.max_batch_size:
            async with self.batch_lock:
                if self.pending_requests:
                    batch.append(self.pending_requests.popleft())
                elif batch:
                    # Have some requests, check if we should wait for more
                    elapsed_ms = (time.time() - start_time) * 1000
                    if elapsed_ms >= self.max_wait_time_ms:
                        break
                else:
                    # No requests at all
                    break
            
            # Small delay to allow more requests to accumulate
            if len(batch) < self.max_batch_size:
                await asyncio.sleep(0.001)  # 1ms
        
        return batch
    
    async def _process_batch(self, batch: List[BatchRequest]):
        """Process a batch of requests."""
        # This will be implemented by the translation engine
        # For now, just set empty results
        for batch_request in batch:
            if not batch_request.future.done():
                batch_request.future.set_result(None)


class ModelQuantizer:
    """Model quantization for memory optimization."""
    
    def __init__(self):
        self.quantization_enabled = torch.cuda.is_available()
        self.quantization_methods = ["int8", "fp16", "dynamic"]
        
    def quantize_model(self, model: nn.Module, method: str = "fp16") -> nn.Module:
        """Quantize model for memory optimization."""
        try:
            if not self.quantization_enabled:
                logger.warning("Quantization not available, returning original model")
                return model
            
            logger.info(f"Quantizing model using {method}")
            
            if method == "fp16":
                return self._quantize_fp16(model)
            elif method == "int8":
                return self._quantize_int8(model)
            elif method == "dynamic":
                return self._quantize_dynamic(model)
            else:
                logger.warning(f"Unknown quantization method: {method}")
                return model
                
        except Exception as e:
            logger.error(f"Model quantization failed: {str(e)}")
            return model
    
    def _quantize_fp16(self, model: nn.Module) -> nn.Module:
        """Apply FP16 quantization."""
        if torch.cuda.is_available():
            model = model.half()
            logger.info("Applied FP16 quantization")
        return model
    
    def _quantize_int8(self, model: nn.Module) -> nn.Module:
        """Apply INT8 quantization."""
        try:
            # Use torch.quantization for INT8
            model.eval()
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            logger.info("Applied INT8 quantization")
            return quantized_model
        except Exception as e:
            logger.warning(f"INT8 quantization failed: {str(e)}")
            return model
    
    def _quantize_dynamic(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization."""
        try:
            model.eval()
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=torch.qint8
            )
            logger.info("Applied dynamic quantization")
            return quantized_model
        except Exception as e:
            logger.warning(f"Dynamic quantization failed: {str(e)}")
            return model
    
    def estimate_memory_savings(self, original_size_mb: int, method: str) -> int:
        """Estimate memory savings from quantization."""
        savings_ratios = {
            "fp16": 0.5,    # 50% reduction
            "int8": 0.75,   # 75% reduction
            "dynamic": 0.6  # 60% reduction
        }
        
        ratio = savings_ratios.get(method, 0.0)
        return int(original_size_mb * ratio)


class PipelineParallelizer:
    """Pipeline parallelism for large model handling."""
    
    def __init__(self, num_stages: int = 2):
        self.num_stages = num_stages
        self.stages = []
        self.stage_devices = []
        
    def setup_pipeline(self, model: nn.Module, tokenizer: AutoTokenizer) -> bool:
        """Setup pipeline parallelism for model."""
        try:
            if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                logger.warning("Pipeline parallelism requires multiple GPUs")
                return False
            
            logger.info(f"Setting up pipeline with {self.num_stages} stages")
            
            # Split model into stages
            self.stages = self._split_model_into_stages(model)
            
            # Assign stages to different devices
            self.stage_devices = self._assign_stages_to_devices()
            
            # Move stages to respective devices
            for i, (stage, device) in enumerate(zip(self.stages, self.stage_devices)):
                stage.to(device)
                logger.info(f"Stage {i} assigned to device {device}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline setup failed: {str(e)}")
            return False
    
    def _split_model_into_stages(self, model: nn.Module) -> List[nn.Module]:
        """Split model into pipeline stages."""
        # This is a simplified implementation
        # In practice, this would analyze the model architecture
        # and split it at appropriate boundaries
        
        stages = []
        
        if hasattr(model, 'encoder') and hasattr(model, 'decoder'):
            # Encoder-decoder model
            stages.append(model.encoder)
            stages.append(model.decoder)
        else:
            # Split model layers roughly in half
            layers = list(model.children())
            mid_point = len(layers) // 2
            
            stage1 = nn.Sequential(*layers[:mid_point])
            stage2 = nn.Sequential(*layers[mid_point:])
            
            stages.extend([stage1, stage2])
        
        return stages
    
    def _assign_stages_to_devices(self) -> List[torch.device]:
        """Assign pipeline stages to GPU devices."""
        devices = []
        num_gpus = torch.cuda.device_count()
        
        for i in range(self.num_stages):
            device_id = i % num_gpus
            devices.append(torch.device(f"cuda:{device_id}"))
        
        return devices
    
    async def process_pipeline(self, inputs: torch.Tensor) -> torch.Tensor:
        """Process inputs through pipeline stages."""
        try:
            current_input = inputs
            
            for i, (stage, device) in enumerate(zip(self.stages, self.stage_devices)):
                # Move input to stage device
                current_input = current_input.to(device)
                
                # Process through stage
                with torch.no_grad():
                    current_input = stage(current_input)
                
                logger.debug(f"Completed pipeline stage {i}")
            
            return current_input
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {str(e)}")
            raise TranslationError(f"Pipeline processing failed: {str(e)}")


class OptimizedTranslationEngine:
    """Enhanced translation engine with optimization features."""
    
    def __init__(self, base_engine):
        self.base_engine = base_engine
        self.batcher = DynamicBatcher(
            max_batch_size=config.translation.batch_size * 2,  # Larger batches for optimization
            max_wait_time_ms=50  # Shorter wait time for responsiveness
        )
        self.quantizer = ModelQuantizer()
        self.parallelizer = PipelineParallelizer()
        self.optimization_enabled = True
        self.performance_metrics = []
        
        logger.info("Optimized translation engine initialized")
    
    async def translate_optimized(self, request: TranslationRequest) -> TranslationResult:
        """Translate with optimization features."""
        if not self.optimization_enabled:
            return await self.base_engine.translate(request)
        
        start_time = time.time()
        
        try:
            # Use dynamic batching for better GPU utilization
            result = await self.batcher.add_request(request)
            
            if result is None:
                # Fallback to base engine if batching fails
                result = await self.base_engine.translate(request)
            
            # Record performance metrics
            processing_time = (time.time() - start_time) * 1000
            await self._record_performance_metrics(request, result, processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Optimized translation failed: {str(e)}")
            # Fallback to base engine
            return await self.base_engine.translate(request)
    
    async def optimize_model(self, model_id: str, optimization_config: Dict[str, Any]) -> bool:
        """Apply optimizations to a loaded model."""
        try:
            model_info = await self.base_engine.get_model_info(model_id)
            if not model_info:
                logger.error(f"Model {model_id} not found for optimization")
                return False
            
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            
            # Apply quantization if requested
            if optimization_config.get("quantization", {}).get("enabled", False):
                method = optimization_config["quantization"].get("method", "fp16")
                original_size = model_info.get("memory_mb", 0)
                
                optimized_model = self.quantizer.quantize_model(model, method)
                model_info["model"] = optimized_model
                model_info["quantization_method"] = method
                
                # Estimate memory savings
                savings = self.quantizer.estimate_memory_savings(original_size, method)
                logger.info(f"Model quantization completed, estimated savings: {savings}MB")
            
            # Setup pipeline parallelism if requested
            if optimization_config.get("pipeline", {}).get("enabled", False):
                num_stages = optimization_config["pipeline"].get("stages", 2)
                self.parallelizer.num_stages = num_stages
                
                success = self.parallelizer.setup_pipeline(model, tokenizer)
                if success:
                    model_info["pipeline_enabled"] = True
                    model_info["pipeline_stages"] = num_stages
                    logger.info(f"Pipeline parallelism enabled with {num_stages} stages")
            
            return True
            
        except Exception as e:
            logger.error(f"Model optimization failed: {str(e)}")
            return False
    
    async def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization performance metrics."""
        if not self.performance_metrics:
            return {"message": "No metrics available"}
        
        # Calculate aggregate metrics
        recent_metrics = self.performance_metrics[-100:]  # Last 100 requests
        
        avg_batch_size = sum(m.batch_size for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.throughput_wpm for m in recent_metrics) / len(recent_metrics)
        avg_gpu_util = sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        
        quantized_requests = sum(1 for m in recent_metrics if m.quantization_enabled)
        pipeline_requests = sum(1 for m in recent_metrics if m.pipeline_stages > 1)
        
        return {
            "total_requests": len(self.performance_metrics),
            "recent_requests": len(recent_metrics),
            "average_metrics": {
                "batch_size": round(avg_batch_size, 2),
                "throughput_wpm": round(avg_throughput, 2),
                "gpu_utilization": round(avg_gpu_util, 2),
                "memory_usage_mb": round(avg_memory, 2)
            },
            "optimization_usage": {
                "quantization_percentage": round((quantized_requests / len(recent_metrics)) * 100, 2),
                "pipeline_percentage": round((pipeline_requests / len(recent_metrics)) * 100, 2)
            },
            "performance_targets": {
                "target_wpm": 1500,
                "current_wpm": round(avg_throughput, 2),
                "target_met": avg_throughput >= 1500
            }
        }
    
    async def tune_batch_parameters(self, target_latency_ms: int = 200) -> Dict[str, int]:
        """Auto-tune batching parameters for optimal performance."""
        try:
            # Analyze recent performance
            if len(self.performance_metrics) < 10:
                return {"message": "Insufficient data for tuning"}
            
            recent_metrics = self.performance_metrics[-50:]
            avg_latency = sum(m.processing_time_ms for m in recent_metrics) / len(recent_metrics)
            avg_throughput = sum(m.throughput_wpm for m in recent_metrics) / len(recent_metrics)
            
            # Adjust batch size based on performance
            current_batch_size = self.batcher.max_batch_size
            current_wait_time = self.batcher.max_wait_time_ms
            
            if avg_latency > target_latency_ms:
                # Reduce batch size or wait time to improve latency
                new_batch_size = max(1, current_batch_size - 2)
                new_wait_time = max(10, current_wait_time - 10)
            elif avg_throughput < 1500:
                # Increase batch size to improve throughput
                new_batch_size = min(32, current_batch_size + 2)
                new_wait_time = min(100, current_wait_time + 10)
            else:
                # Performance is good, no changes needed
                new_batch_size = current_batch_size
                new_wait_time = current_wait_time
            
            # Apply new parameters
            self.batcher.max_batch_size = new_batch_size
            self.batcher.max_wait_time_ms = new_wait_time
            
            logger.info(
                f"Batch parameters tuned",
                metadata={
                    "old_batch_size": current_batch_size,
                    "new_batch_size": new_batch_size,
                    "old_wait_time": current_wait_time,
                    "new_wait_time": new_wait_time,
                    "avg_latency_ms": avg_latency,
                    "avg_throughput_wpm": avg_throughput
                }
            )
            
            return {
                "batch_size": new_batch_size,
                "wait_time_ms": new_wait_time,
                "previous_batch_size": current_batch_size,
                "previous_wait_time": current_wait_time
            }
            
        except Exception as e:
            logger.error(f"Batch parameter tuning failed: {str(e)}")
            return {"error": str(e)}
    
    async def _record_performance_metrics(
        self, 
        request: TranslationRequest, 
        result: TranslationResult, 
        processing_time_ms: float
    ):
        """Record performance metrics for optimization analysis."""
        try:
            word_count = len(request.content.split())
            throughput_wpm = (word_count / (processing_time_ms / 1000 / 60)) if processing_time_ms > 0 else 0
            
            # Get current GPU utilization (simplified)
            gpu_util = 0.0
            memory_usage = 0
            if torch.cuda.is_available():
                gpu_util = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 50.0
                memory_usage = torch.cuda.memory_allocated() // (1024 * 1024)  # MB
            
            metrics = OptimizationMetrics(
                batch_size=self.batcher.max_batch_size,
                processing_time_ms=int(processing_time_ms),
                throughput_wpm=throughput_wpm,
                gpu_utilization=gpu_util,
                memory_usage_mb=memory_usage,
                quantization_enabled=hasattr(result, 'quantization_method'),
                pipeline_stages=getattr(result, 'pipeline_stages', 1)
            )
            
            self.performance_metrics.append(metrics)
            
            # Keep only recent metrics to prevent memory growth
            if len(self.performance_metrics) > 1000:
                self.performance_metrics = self.performance_metrics[-500:]
            
        except Exception as e:
            logger.debug(f"Failed to record performance metrics: {str(e)}")
    
    async def enable_optimization(self, enabled: bool = True):
        """Enable or disable optimization features."""
        self.optimization_enabled = enabled
        logger.info(f"Optimization features {'enabled' if enabled else 'disabled'}")
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        return {
            "optimization_enabled": self.optimization_enabled,
            "dynamic_batching": {
                "enabled": True,
                "max_batch_size": self.batcher.max_batch_size,
                "max_wait_time_ms": self.batcher.max_wait_time_ms,
                "pending_requests": len(self.batcher.pending_requests)
            },
            "quantization": {
                "available": self.quantizer.quantization_enabled,
                "methods": self.quantizer.quantization_methods
            },
            "pipeline_parallelism": {
                "available": torch.cuda.is_available() and torch.cuda.device_count() > 1,
                "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "stages": self.parallelizer.num_stages
            }
        }


def create_optimized_engine(base_engine) -> OptimizedTranslationEngine:
    """Factory function to create optimized translation engine."""
    return OptimizedTranslationEngine(base_engine)