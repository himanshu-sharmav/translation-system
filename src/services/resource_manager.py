"""
Resource manager for auto-scaling compute instances based on demand.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4

from src.config.config import config
from src.database.connection import get_db_session
from src.database.models import ComputeInstance, QueueMetric
from src.database.repositories.base import BaseRepository
from src.database.repositories import MetricsRepository
from src.models.interfaces import ResourceManager
from src.services.queue_service import QueueService
from src.utils.exceptions import ResourceScalingError, InsufficientResourcesError
from src.utils.logging import TranslationLogger

logger = TranslationLogger(__name__, "resource-manager")


class AutoScalingResourceManager(ResourceManager):
    """Auto-scaling resource manager for compute instances."""
    
    def __init__(self):
        self.queue_service = QueueService()
        self.active_instances: Dict[str, ComputeInstance] = {}
        self.scaling_in_progress: Set[str] = set()
        self.last_scale_action = datetime.utcnow()
        self.min_scale_interval_seconds = 300  # 5 minutes between scaling actions
        self.idle_timeout_minutes = config.compute.idle_timeout_minutes
        self.scale_up_threshold = config.compute.scale_up_threshold
        self.scale_down_threshold = config.compute.scale_down_threshold
        self.min_instances = config.compute.min_instances
        self.max_instances = config.compute.max_instances
        self._monitoring_task = None
        self._running = False
        
        logger.info(
            f"Resource manager initialized",
            metadata={
                "min_instances": self.min_instances,
                "max_instances": self.max_instances,
                "scale_up_threshold": self.scale_up_threshold,
                "scale_down_threshold": self.scale_down_threshold,
                "idle_timeout_minutes": self.idle_timeout_minutes
            }
        )
    
    async def start_monitoring(self):
        """Start resource monitoring and auto-scaling."""
        if self._running:
            logger.warning("Resource manager monitoring already running")
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Load existing instances from database
        await self._load_existing_instances()
        
        # Ensure minimum instances are running
        await self._ensure_minimum_instances()
        
        logger.info("Resource manager monitoring started")
    
    async def stop_monitoring(self):
        """Stop resource monitoring."""
        if not self._running:
            return
        
        self._running = False
        
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Resource manager monitoring stopped")
    
    async def scale_up(self, instance_count: int = 1) -> List[str]:
        """Add new compute instances."""
        try:
            if not self._can_scale():
                logger.warning("Scaling action blocked due to recent scaling activity")
                return []
            
            current_count = len(self.active_instances)
            if current_count + instance_count > self.max_instances:
                available_slots = self.max_instances - current_count
                if available_slots <= 0:
                    logger.warning(f"Cannot scale up: already at maximum instances ({self.max_instances})")
                    return []
                instance_count = available_slots
                logger.info(f"Scaling up limited to {instance_count} instances due to max limit")
            
            logger.info(f"Scaling up {instance_count} instances")
            
            new_instance_ids = []
            for i in range(instance_count):
                instance_id = await self._create_compute_instance()
                if instance_id:
                    new_instance_ids.append(instance_id)
                    self.scaling_in_progress.add(instance_id)
            
            self.last_scale_action = datetime.utcnow()
            
            # Wait for instances to become ready
            await self._wait_for_instances_ready(new_instance_ids)
            
            # Remove from scaling in progress
            for instance_id in new_instance_ids:
                self.scaling_in_progress.discard(instance_id)
            
            logger.info(
                f"Scale up completed",
                metadata={
                    "new_instances": len(new_instance_ids),
                    "total_instances": len(self.active_instances),
                    "instance_ids": new_instance_ids
                }
            )
            
            return new_instance_ids
            
        except Exception as e:
            logger.error(f"Scale up failed: {str(e)}", exc_info=True)
            raise ResourceScalingError("scale_up", f"Failed to scale up: {str(e)}")
    
    async def scale_down(self, instance_count: int = 1) -> List[str]:
        """Remove compute instances."""
        try:
            if not self._can_scale():
                logger.warning("Scaling action blocked due to recent scaling activity")
                return []
            
            current_count = len(self.active_instances)
            if current_count - instance_count < self.min_instances:
                available_for_removal = current_count - self.min_instances
                if available_for_removal <= 0:
                    logger.warning(f"Cannot scale down: already at minimum instances ({self.min_instances})")
                    return []
                instance_count = available_for_removal
                logger.info(f"Scaling down limited to {instance_count} instances due to min limit")
            
            logger.info(f"Scaling down {instance_count} instances")
            
            # Select instances to remove (prefer idle instances)
            instances_to_remove = await self._select_instances_for_removal(instance_count)
            
            if not instances_to_remove:
                logger.warning("No suitable instances found for removal")
                return []
            
            removed_instance_ids = []
            for instance in instances_to_remove:
                success = await self._terminate_compute_instance(instance.id)
                if success:
                    removed_instance_ids.append(instance.id)
                    self.active_instances.pop(instance.id, None)
            
            self.last_scale_action = datetime.utcnow()
            
            logger.info(
                f"Scale down completed",
                metadata={
                    "removed_instances": len(removed_instance_ids),
                    "total_instances": len(self.active_instances),
                    "instance_ids": removed_instance_ids
                }
            )
            
            return removed_instance_ids
            
        except Exception as e:
            logger.error(f"Scale down failed: {str(e)}", exc_info=True)
            raise ResourceScalingError("scale_down", f"Failed to scale down: {str(e)}")
    
    async def get_instance_metrics(self, instance_id: str) -> Dict[str, float]:
        """Get metrics for a specific instance."""
        try:
            instance = self.active_instances.get(instance_id)
            if not instance:
                return {}
            
            # Get latest metrics from database
            async with get_db_session() as session:
                metrics_repo = MetricsRepository(session)
                
                # Get recent metrics for this instance
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(minutes=5)
                
                cpu_metrics = await metrics_repo.get_metrics("cpu_utilization", start_time, end_time)
                gpu_metrics = await metrics_repo.get_metrics("gpu_utilization", start_time, end_time)
                memory_metrics = await metrics_repo.get_metrics("memory_usage", start_time, end_time)
                
                # Filter metrics for this instance
                instance_cpu = [m for m in cpu_metrics if m.instance_id == instance_id]
                instance_gpu = [m for m in gpu_metrics if m.instance_id == instance_id]
                instance_memory = [m for m in memory_metrics if m.instance_id == instance_id]
                
                # Calculate averages
                avg_cpu = sum(m.metric_value for m in instance_cpu) / len(instance_cpu) if instance_cpu else 0.0
                avg_gpu = sum(m.metric_value for m in instance_gpu) / len(instance_gpu) if instance_gpu else 0.0
                avg_memory = sum(m.metric_value for m in instance_memory) / len(instance_memory) if instance_memory else 0.0
                
                return {
                    "cpu_utilization": avg_cpu,
                    "gpu_utilization": avg_gpu,
                    "memory_usage": avg_memory,
                    "active_jobs": instance.active_jobs,
                    "max_jobs": instance.max_concurrent_jobs,
                    "utilization_ratio": instance.active_jobs / instance.max_concurrent_jobs if instance.max_concurrent_jobs > 0 else 0.0
                }
                
        except Exception as e:
            logger.error(f"Failed to get metrics for instance {instance_id}: {str(e)}")
            return {}
    
    async def get_all_instances(self) -> List[ComputeInstance]:
        """Get all active compute instances."""
        return list(self.active_instances.values())
    
    async def health_check(self, instance_id: str) -> bool:
        """Check if instance is healthy."""
        try:
            instance = self.active_instances.get(instance_id)
            if not instance:
                return False
            
            # Check if instance has recent heartbeat
            if instance.last_heartbeat:
                time_since_heartbeat = datetime.utcnow() - instance.last_heartbeat
                if time_since_heartbeat > timedelta(minutes=5):
                    logger.warning(f"Instance {instance_id} has stale heartbeat")
                    return False
            
            # Check instance status
            if instance.status != "running":
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for instance {instance_id}: {str(e)}")
            return False
    
    async def get_scaling_metrics(self) -> Dict[str, any]:
        """Get current scaling metrics and status."""
        try:
            # Get queue metrics
            queue_stats = await self.queue_service.get_queue_stats()
            total_queued = sum(queue_stats.get(priority, 0) for priority in ["critical", "high", "normal"])
            
            # Get instance utilization
            total_capacity = 0
            total_active_jobs = 0
            healthy_instances = 0
            
            for instance in self.active_instances.values():
                if await self.health_check(instance.id):
                    healthy_instances += 1
                    total_capacity += instance.max_concurrent_jobs
                    total_active_jobs += instance.active_jobs
            
            utilization_ratio = total_active_jobs / total_capacity if total_capacity > 0 else 0.0
            
            # Calculate scaling recommendation
            scaling_recommendation = await self._calculate_scaling_recommendation(
                total_queued, utilization_ratio, healthy_instances
            )
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "instances": {
                    "total": len(self.active_instances),
                    "healthy": healthy_instances,
                    "scaling_in_progress": len(self.scaling_in_progress),
                    "min_instances": self.min_instances,
                    "max_instances": self.max_instances
                },
                "queue": {
                    "total_queued": total_queued,
                    "critical": queue_stats.get("critical", 0),
                    "high": queue_stats.get("high", 0),
                    "normal": queue_stats.get("normal", 0),
                    "processing": queue_stats.get("processing", 0)
                },
                "utilization": {
                    "total_capacity": total_capacity,
                    "active_jobs": total_active_jobs,
                    "utilization_ratio": utilization_ratio,
                    "scale_up_threshold": self.scale_up_threshold,
                    "scale_down_threshold": self.scale_down_threshold
                },
                "scaling": {
                    "recommendation": scaling_recommendation,
                    "can_scale": self._can_scale(),
                    "last_scale_action": self.last_scale_action.isoformat(),
                    "min_scale_interval_seconds": self.min_scale_interval_seconds
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get scaling metrics: {str(e)}")
            return {"error": str(e)}
    
    async def _monitoring_loop(self):
        """Main monitoring loop for auto-scaling."""
        while self._running:
            try:
                await self._perform_scaling_check()
                await self._cleanup_unhealthy_instances()
                await self._update_instance_heartbeats()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _perform_scaling_check(self):
        """Check if scaling is needed and perform scaling actions."""
        try:
            if not self._can_scale():
                return
            
            metrics = await self.get_scaling_metrics()
            recommendation = metrics.get("scaling", {}).get("recommendation")
            
            if recommendation == "scale_up":
                # Determine how many instances to add
                queue_depth = metrics["queue"]["total_queued"]
                current_instances = metrics["instances"]["healthy"]
                
                # Scale up based on queue depth
                if queue_depth > 20:
                    instances_needed = min(3, self.max_instances - current_instances)
                elif queue_depth > 10:
                    instances_needed = min(2, self.max_instances - current_instances)
                else:
                    instances_needed = min(1, self.max_instances - current_instances)
                
                if instances_needed > 0:
                    logger.info(f"Auto-scaling up {instances_needed} instances due to high demand")
                    await self.scale_up(instances_needed)
            
            elif recommendation == "scale_down":
                # Determine how many instances to remove
                current_instances = metrics["instances"]["healthy"]
                utilization = metrics["utilization"]["utilization_ratio"]
                
                # Scale down conservatively
                if utilization < 0.1 and current_instances > self.min_instances:
                    instances_to_remove = min(1, current_instances - self.min_instances)
                    logger.info(f"Auto-scaling down {instances_to_remove} instances due to low utilization")
                    await self.scale_down(instances_to_remove)
            
        except Exception as e:
            logger.error(f"Error in scaling check: {str(e)}", exc_info=True)
    
    async def _calculate_scaling_recommendation(
        self, 
        queue_depth: int, 
        utilization_ratio: float, 
        healthy_instances: int
    ) -> str:
        """Calculate scaling recommendation based on metrics."""
        
        # Scale up conditions
        if queue_depth > 5 and utilization_ratio > self.scale_up_threshold:
            return "scale_up"
        elif queue_depth > 15:  # High queue depth regardless of utilization
            return "scale_up"
        elif healthy_instances < self.min_instances:
            return "scale_up"
        
        # Scale down conditions
        elif (queue_depth == 0 and 
              utilization_ratio < self.scale_down_threshold and 
              healthy_instances > self.min_instances):
            return "scale_down"
        
        return "maintain"
    
    async def _load_existing_instances(self):
        """Load existing instances from database."""
        try:
            async with get_db_session() as session:
                instance_repo = BaseRepository(session, ComputeInstance)
                instances = await instance_repo.find_by({"status": "running"})
                
                for instance in instances:
                    self.active_instances[instance.id] = instance
                
                logger.info(f"Loaded {len(instances)} existing instances from database")
                
        except Exception as e:
            logger.error(f"Failed to load existing instances: {str(e)}")
    
    async def _ensure_minimum_instances(self):
        """Ensure minimum number of instances are running."""
        try:
            current_count = len(self.active_instances)
            if current_count < self.min_instances:
                needed = self.min_instances - current_count
                logger.info(f"Starting {needed} instances to meet minimum requirement")
                await self.scale_up(needed)
        except Exception as e:
            logger.error(f"Failed to ensure minimum instances: {str(e)}")
    
    async def _create_compute_instance(self) -> Optional[str]:
        """Create a new compute instance."""
        try:
            instance_id = f"instance-{uuid4().hex[:8]}"
            
            # Determine instance type based on availability and configuration
            instance_type = await self._select_instance_type()
            
            # Create instance record in database
            instance = ComputeInstance(
                id=instance_id,
                instance_type=instance_type,
                status="starting",
                gpu_utilization=0.0,
                memory_usage=0.0,
                active_jobs=0,
                max_concurrent_jobs=self._get_max_jobs_for_instance_type(instance_type),
                created_at=datetime.utcnow(),
                last_heartbeat=datetime.utcnow()
            )
            
            async with get_db_session() as session:
                instance_repo = BaseRepository(session, ComputeInstance)
                created_instance = await instance_repo.create(instance)
                await session.commit()
            
            # Simulate instance startup (in real implementation, this would call cloud provider API)
            await self._start_instance(instance_id, instance_type)
            
            # Update instance status to running
            created_instance.status = "running"
            async with get_db_session() as session:
                instance_repo = BaseRepository(session, ComputeInstance)
                await instance_repo.update(created_instance)
                await session.commit()
            
            self.active_instances[instance_id] = created_instance
            
            logger.info(
                f"Created compute instance",
                instance_id=instance_id,
                metadata={
                    "instance_type": instance_type,
                    "max_concurrent_jobs": created_instance.max_concurrent_jobs
                }
            )
            
            return instance_id
            
        except Exception as e:
            logger.error(f"Failed to create compute instance: {str(e)}", exc_info=True)
            return None
    
    async def _terminate_compute_instance(self, instance_id: str) -> bool:
        """Terminate a compute instance."""
        try:
            instance = self.active_instances.get(instance_id)
            if not instance:
                logger.warning(f"Instance {instance_id} not found for termination")
                return False
            
            # Check if instance has active jobs
            if instance.active_jobs > 0:
                logger.warning(f"Cannot terminate instance {instance_id} with {instance.active_jobs} active jobs")
                return False
            
            # Simulate instance termination (in real implementation, this would call cloud provider API)
            await self._stop_instance(instance_id)
            
            # Update instance status in database
            instance.status = "terminated"
            async with get_db_session() as session:
                instance_repo = BaseRepository(session, ComputeInstance)
                await instance_repo.update(instance)
                await session.commit()
            
            logger.info(f"Terminated compute instance", instance_id=instance_id)
            return True
            
        except Exception as e:
            logger.error(f"Failed to terminate instance {instance_id}: {str(e)}", exc_info=True)
            return False
    
    async def _select_instances_for_removal(self, count: int) -> List[ComputeInstance]:
        """Select instances for removal based on utilization and idle time."""
        candidates = []
        
        for instance in self.active_instances.values():
            # Skip instances with active jobs
            if instance.active_jobs > 0:
                continue
            
            # Skip instances that are scaling in progress
            if instance.id in self.scaling_in_progress:
                continue
            
            # Calculate idle time
            idle_time = datetime.utcnow() - (instance.last_heartbeat or instance.created_at)
            
            candidates.append({
                "instance": instance,
                "idle_minutes": idle_time.total_seconds() / 60,
                "utilization": instance.gpu_utilization + instance.memory_usage
            })
        
        # Sort by idle time (longest idle first) and utilization (lowest first)
        candidates.sort(key=lambda x: (-x["idle_minutes"], x["utilization"]))
        
        return [c["instance"] for c in candidates[:count]]
    
    async def _select_instance_type(self) -> str:
        """Select appropriate instance type based on current needs."""
        # Simple logic: prefer GPU instances if available, fall back to CPU
        # In real implementation, this would check cloud provider availability
        
        gpu_instances = [i for i in self.active_instances.values() if "gpu" in i.instance_type.lower()]
        cpu_instances = [i for i in self.active_instances.values() if "cpu" in i.instance_type.lower()]
        
        # Balance between GPU and CPU instances
        if len(gpu_instances) < len(cpu_instances) * 2:
            return config.compute.gpu_instance_type
        else:
            return config.compute.cpu_instance_type
    
    def _get_max_jobs_for_instance_type(self, instance_type: str) -> int:
        """Get maximum concurrent jobs for instance type."""
        # GPU instances can handle more concurrent jobs
        if "gpu" in instance_type.lower() or "g4" in instance_type or "p3" in instance_type:
            return 4
        else:
            return 2
    
    async def _start_instance(self, instance_id: str, instance_type: str):
        """Start compute instance (simulated)."""
        # In real implementation, this would call cloud provider API
        # For now, simulate startup time
        startup_time = 30 if "gpu" in instance_type.lower() else 15  # GPU instances take longer
        await asyncio.sleep(startup_time)
        
        logger.debug(f"Instance {instance_id} started (simulated)")
    
    async def _stop_instance(self, instance_id: str):
        """Stop compute instance (simulated)."""
        # In real implementation, this would call cloud provider API
        await asyncio.sleep(5)  # Simulate shutdown time
        
        logger.debug(f"Instance {instance_id} stopped (simulated)")
    
    async def _wait_for_instances_ready(self, instance_ids: List[str], timeout_minutes: int = 10):
        """Wait for instances to become ready."""
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while time.time() - start_time < timeout_seconds:
            all_ready = True
            
            for instance_id in instance_ids:
                if not await self.health_check(instance_id):
                    all_ready = False
                    break
            
            if all_ready:
                logger.info(f"All {len(instance_ids)} instances are ready")
                return
            
            await asyncio.sleep(10)  # Check every 10 seconds
        
        logger.warning(f"Timeout waiting for instances to become ready: {instance_ids}")
    
    async def _cleanup_unhealthy_instances(self):
        """Remove unhealthy instances from tracking."""
        unhealthy_instances = []
        
        for instance_id, instance in self.active_instances.items():
            if not await self.health_check(instance_id):
                unhealthy_instances.append(instance_id)
        
        for instance_id in unhealthy_instances:
            logger.warning(f"Removing unhealthy instance from tracking: {instance_id}")
            self.active_instances.pop(instance_id, None)
            
            # Update instance status in database
            try:
                async with get_db_session() as session:
                    instance_repo = BaseRepository(session, ComputeInstance)
                    instance = await instance_repo.get_by_id(instance_id)
                    if instance:
                        instance.status = "unhealthy"
                        await instance_repo.update(instance)
                        await session.commit()
            except Exception as e:
                logger.error(f"Failed to update unhealthy instance status: {str(e)}")
    
    async def _update_instance_heartbeats(self):
        """Update instance heartbeats (simulated)."""
        # In real implementation, instances would send heartbeats
        # For simulation, we'll update heartbeats for healthy instances
        
        for instance in self.active_instances.values():
            if instance.status == "running":
                instance.last_heartbeat = datetime.utcnow()
                
                # Simulate some resource utilization changes
                import random
                instance.gpu_utilization = max(0, min(100, instance.gpu_utilization + random.uniform(-5, 5)))
                instance.memory_usage = max(0, min(100, instance.memory_usage + random.uniform(-3, 3)))
                
                try:
                    async with get_db_session() as session:
                        instance_repo = BaseRepository(session, ComputeInstance)
                        await instance_repo.update(instance)
                        await session.commit()
                except Exception as e:
                    logger.debug(f"Failed to update instance heartbeat: {str(e)}")
    
    def _can_scale(self) -> bool:
        """Check if scaling action is allowed."""
        time_since_last_scale = datetime.utcnow() - self.last_scale_action
        return time_since_last_scale.total_seconds() >= self.min_scale_interval_seconds
    
    async def close(self):
        """Clean up resources."""
        await self.stop_monitoring()
        await self.queue_service.close()
        
        logger.info("Resource manager closed")