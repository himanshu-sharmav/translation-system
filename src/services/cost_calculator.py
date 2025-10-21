"""
Cost calculation service for estimating translation costs.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from src.config.config import config
from src.database.connection import get_db_session
from src.database.repositories import JobRepository, MetricsRepository
from src.utils.logging import TranslationLogger

logger = TranslationLogger(__name__, "cost-calculator")


class CostCalculator:
    """Service for calculating translation costs and estimates."""
    
    def __init__(self):
        # Cost constants (in USD)
        self.gpu_instance_cost_per_hour = 0.526  # g4dn.xlarge
        self.cpu_instance_cost_per_hour = 0.192  # r5.8xlarge
        self.storage_cost_per_gb_month = 0.023   # EBS gp3
        self.network_cost_per_gb = 0.09          # Data transfer
        
        # Performance assumptions
        self.words_per_minute_gpu = 1500
        self.words_per_minute_cpu = 800
        self.avg_model_size_gb = 75
        self.cache_hit_ratio = 0.3  # 30% cache hit rate
        
        # Priority multipliers for cost calculation
        self.priority_multipliers = {
            "normal": 1.0,
            "high": 1.2,
            "critical": 1.5
        }
    
    async def calculate_cost_estimate(self, words_per_day: int, 
                                    priority_distribution: Dict[str, float] = None) -> Dict[str, Any]:
        """Calculate cost estimate for given daily word volume."""
        try:
            if priority_distribution is None:
                priority_distribution = {"normal": 0.8, "high": 0.15, "critical": 0.05}
            
            # Calculate effective words after cache hits
            effective_words = words_per_day * (1 - self.cache_hit_ratio)
            
            # Calculate weighted processing time based on priority distribution
            weighted_multiplier = sum(
                priority_distribution.get(priority, 0) * multiplier
                for priority, multiplier in self.priority_multipliers.items()
            )
            
            # Calculate processing time in minutes
            processing_time_minutes = (effective_words / self.words_per_minute_gpu) * weighted_multiplier
            
            # Calculate compute costs
            gpu_hours = processing_time_minutes / 60
            compute_cost = gpu_hours * self.gpu_instance_cost_per_hour
            
            # Add overhead for auto-scaling and idle time (20%)
            compute_cost *= 1.2
            
            # Calculate storage costs (model storage + cache + logs)
            storage_gb_per_day = (
                self.avg_model_size_gb +  # Model storage
                (words_per_day * 0.001) +  # Cache storage (rough estimate)
                (words_per_day * 0.0001)   # Logs and metadata
            )
            storage_cost = (storage_gb_per_day * self.storage_cost_per_gb_month) / 30
            
            # Calculate network costs (API requests + results)
            network_gb = words_per_day * 0.00001  # Rough estimate: 10KB per 1000 words
            network_cost = network_gb * self.network_cost_per_gb
            
            # Total daily cost
            daily_cost = compute_cost + storage_cost + network_cost
            monthly_cost = daily_cost * 30
            
            # Create breakdown
            breakdown = {
                "compute_cost": round(compute_cost, 4),
                "storage_cost": round(storage_cost, 4),
                "network_cost": round(network_cost, 4)
            }
            
            # Create assumptions
            assumptions = {
                "cache_hit_ratio": self.cache_hit_ratio,
                "words_per_minute": self.words_per_minute_gpu,
                "gpu_instance_cost_per_hour": self.gpu_instance_cost_per_hour,
                "avg_processing_time_ms": (processing_time_minutes * 60 * 1000) / (effective_words or 1),
                "priority_distribution": priority_distribution,
                "overhead_factor": 1.2,
                "model_size_gb": self.avg_model_size_gb
            }
            
            logger.info(
                f"Cost estimate calculated",
                metadata={
                    "words_per_day": words_per_day,
                    "daily_cost_usd": daily_cost,
                    "monthly_cost_usd": monthly_cost,
                    "effective_words": effective_words,
                    "processing_time_minutes": processing_time_minutes
                }
            )
            
            return {
                "daily_cost_usd": round(daily_cost, 2),
                "monthly_cost_usd": round(monthly_cost, 2),
                "breakdown": breakdown,
                "assumptions": assumptions
            }
            
        except Exception as e:
            logger.error(f"Error calculating cost estimate: {str(e)}", exc_info=True)
            raise
    
    async def calculate_actual_daily_cost(self, date: datetime = None) -> Dict[str, Any]:
        """Calculate actual cost for a specific date based on usage."""
        try:
            if date is None:
                date = datetime.utcnow().date()
            
            start_time = datetime.combine(date, datetime.min.time())
            end_time = start_time + timedelta(days=1)
            
            async with get_db_session() as session:
                job_repo = JobRepository(session)
                metrics_repo = MetricsRepository(session)
                
                # Get completed jobs for the day
                completed_jobs = await job_repo.find_by({
                    "status": "completed",
                    "completed_at": {"gte": start_time, "lt": end_time}
                })
                
                # Calculate total processing time and words
                total_processing_time_ms = 0
                total_words = 0
                
                for job in completed_jobs:
                    if job.processing_time_ms:
                        total_processing_time_ms += job.processing_time_ms
                    total_words += job.word_count
                
                # Get instance usage metrics
                instance_metrics = await metrics_repo.get_metrics(
                    "active_instances_count", start_time, end_time
                )
                
                # Calculate average instance hours
                avg_instances = 1  # Default
                if instance_metrics:
                    avg_instances = sum(m.metric_value for m in instance_metrics) / len(instance_metrics)
                
                instance_hours = avg_instances * 24  # 24 hours in a day
                
                # Calculate costs
                compute_cost = instance_hours * self.gpu_instance_cost_per_hour
                
                # Storage cost (rough estimate based on usage)
                storage_cost = (self.avg_model_size_gb * self.storage_cost_per_gb_month) / 30
                
                # Network cost (based on processed words)
                network_gb = total_words * 0.00001
                network_cost = network_gb * self.network_cost_per_gb
                
                total_cost = compute_cost + storage_cost + network_cost
                
                # Calculate efficiency metrics
                cost_per_word = total_cost / total_words if total_words > 0 else 0
                words_per_dollar = total_words / total_cost if total_cost > 0 else 0
                
                logger.info(
                    f"Actual daily cost calculated",
                    metadata={
                        "date": date.isoformat(),
                        "total_cost_usd": total_cost,
                        "total_words": total_words,
                        "completed_jobs": len(completed_jobs),
                        "instance_hours": instance_hours
                    }
                )
                
                return {
                    "date": date.isoformat(),
                    "total_cost_usd": round(total_cost, 2),
                    "breakdown": {
                        "compute_cost": round(compute_cost, 2),
                        "storage_cost": round(storage_cost, 4),
                        "network_cost": round(network_cost, 4)
                    },
                    "usage_stats": {
                        "total_words": total_words,
                        "completed_jobs": len(completed_jobs),
                        "total_processing_time_ms": total_processing_time_ms,
                        "instance_hours": round(instance_hours, 2),
                        "avg_instances": round(avg_instances, 2)
                    },
                    "efficiency_metrics": {
                        "cost_per_word": round(cost_per_word, 6),
                        "words_per_dollar": round(words_per_dollar, 2),
                        "utilization_rate": min(1.0, total_processing_time_ms / (instance_hours * 3600 * 1000)) if instance_hours > 0 else 0
                    }
                }
                
        except Exception as e:
            logger.error(f"Error calculating actual daily cost: {str(e)}", exc_info=True)
            raise
    
    async def get_cost_optimization_recommendations(self, days: int = 7) -> Dict[str, Any]:
        """Get cost optimization recommendations based on usage patterns."""
        try:
            recommendations = []
            potential_savings = 0.0
            
            # Analyze recent usage
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=days)
            
            daily_costs = []
            for i in range(days):
                date = start_date + timedelta(days=i)
                daily_cost = await self.calculate_actual_daily_cost(date)
                daily_costs.append(daily_cost)
            
            # Calculate averages
            avg_daily_cost = sum(cost["total_cost_usd"] for cost in daily_costs) / len(daily_costs)
            avg_utilization = sum(cost["efficiency_metrics"]["utilization_rate"] for cost in daily_costs) / len(daily_costs)
            avg_words_per_day = sum(cost["usage_stats"]["total_words"] for cost in daily_costs) / len(daily_costs)
            
            # Recommendation 1: Instance right-sizing
            if avg_utilization < 0.3:
                potential_savings += avg_daily_cost * 0.2  # 20% savings
                recommendations.append({
                    "type": "instance_rightsizing",
                    "title": "Consider using smaller instances",
                    "description": f"Current utilization is {avg_utilization:.1%}. Using smaller instances could save ~20%.",
                    "potential_daily_savings": round(avg_daily_cost * 0.2, 2),
                    "priority": "high"
                })
            
            # Recommendation 2: Spot instances
            if avg_daily_cost > 10:  # Only recommend for higher usage
                spot_savings = avg_daily_cost * 0.6  # 60% savings with spot instances
                potential_savings += spot_savings
                recommendations.append({
                    "type": "spot_instances",
                    "title": "Use spot instances for non-critical workloads",
                    "description": "Spot instances can provide up to 60% cost savings for fault-tolerant workloads.",
                    "potential_daily_savings": round(spot_savings, 2),
                    "priority": "medium"
                })
            
            # Recommendation 3: Cache optimization
            current_cache_hit_ratio = 0.3  # This would come from actual metrics
            if current_cache_hit_ratio < 0.5:
                cache_savings = avg_daily_cost * 0.15  # 15% savings with better caching
                potential_savings += cache_savings
                recommendations.append({
                    "type": "cache_optimization",
                    "title": "Improve cache hit ratio",
                    "description": f"Current cache hit ratio is {current_cache_hit_ratio:.1%}. Optimizing cache could save ~15%.",
                    "potential_daily_savings": round(cache_savings, 2),
                    "priority": "medium"
                })
            
            # Recommendation 4: Batch processing
            if avg_words_per_day > 50000:
                batch_savings = avg_daily_cost * 0.1  # 10% savings with batching
                potential_savings += batch_savings
                recommendations.append({
                    "type": "batch_processing",
                    "title": "Implement batch processing",
                    "description": "Batching similar requests can improve efficiency and reduce costs by ~10%.",
                    "potential_daily_savings": round(batch_savings, 2),
                    "priority": "low"
                })
            
            # Recommendation 5: Reserved instances
            if avg_daily_cost > 20:  # Only for high usage
                reserved_savings = avg_daily_cost * 0.3  # 30% savings with reserved instances
                potential_savings += reserved_savings
                recommendations.append({
                    "type": "reserved_instances",
                    "title": "Consider reserved instances",
                    "description": "Reserved instances can provide up to 30% savings for predictable workloads.",
                    "potential_daily_savings": round(reserved_savings, 2),
                    "priority": "high"
                })
            
            logger.info(
                f"Cost optimization recommendations generated",
                metadata={
                    "days_analyzed": days,
                    "avg_daily_cost": avg_daily_cost,
                    "avg_utilization": avg_utilization,
                    "total_recommendations": len(recommendations),
                    "potential_daily_savings": potential_savings
                }
            )
            
            return {
                "analysis_period_days": days,
                "current_metrics": {
                    "avg_daily_cost_usd": round(avg_daily_cost, 2),
                    "avg_utilization_rate": round(avg_utilization, 3),
                    "avg_words_per_day": round(avg_words_per_day, 0)
                },
                "recommendations": recommendations,
                "potential_savings": {
                    "daily_usd": round(potential_savings, 2),
                    "monthly_usd": round(potential_savings * 30, 2),
                    "annual_usd": round(potential_savings * 365, 2)
                },
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating cost optimization recommendations: {str(e)}", exc_info=True)
            raise
    
    async def track_cost_trends(self, days: int = 30) -> Dict[str, Any]:
        """Track cost trends over time."""
        try:
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=days)
            
            daily_costs = []
            total_cost = 0
            total_words = 0
            
            for i in range(days):
                date = start_date + timedelta(days=i)
                daily_cost = await self.calculate_actual_daily_cost(date)
                daily_costs.append({
                    "date": date.isoformat(),
                    "cost": daily_cost["total_cost_usd"],
                    "words": daily_cost["usage_stats"]["total_words"]
                })
                total_cost += daily_cost["total_cost_usd"]
                total_words += daily_cost["usage_stats"]["total_words"]
            
            # Calculate trends
            avg_daily_cost = total_cost / days
            avg_daily_words = total_words / days
            avg_cost_per_word = total_cost / total_words if total_words > 0 else 0
            
            # Calculate week-over-week change
            if days >= 14:
                recent_week_cost = sum(day["cost"] for day in daily_costs[-7:])
                previous_week_cost = sum(day["cost"] for day in daily_costs[-14:-7])
                week_over_week_change = ((recent_week_cost - previous_week_cost) / previous_week_cost * 100) if previous_week_cost > 0 else 0
            else:
                week_over_week_change = 0
            
            return {
                "period_days": days,
                "daily_costs": daily_costs,
                "summary": {
                    "total_cost_usd": round(total_cost, 2),
                    "avg_daily_cost_usd": round(avg_daily_cost, 2),
                    "total_words": total_words,
                    "avg_daily_words": round(avg_daily_words, 0),
                    "avg_cost_per_word": round(avg_cost_per_word, 6),
                    "week_over_week_change_percent": round(week_over_week_change, 1)
                },
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error tracking cost trends: {str(e)}", exc_info=True)
            raise