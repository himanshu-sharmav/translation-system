# Terraform Outputs

# Network Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

# Load Balancer Outputs
output "load_balancer_dns_name" {
  description = "DNS name of the load balancer"
  value       = aws_lb.main.dns_name
}

output "load_balancer_zone_id" {
  description = "Zone ID of the load balancer"
  value       = aws_lb.main.zone_id
}

output "api_endpoint" {
  description = "API endpoint URL"
  value       = "http://${aws_lb.main.dns_name}/api/v1"
}

# Database Outputs
output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.main.endpoint
  sensitive   = true
}

output "database_port" {
  description = "RDS instance port"
  value       = aws_db_instance.main.port
}

output "database_name" {
  description = "Database name"
  value       = aws_db_instance.main.db_name
}

output "database_username" {
  description = "Database username"
  value       = aws_db_instance.main.username
  sensitive   = true
}

# Redis Outputs
output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_replication_group.main.primary_endpoint_address
  sensitive   = true
}

output "redis_port" {
  description = "ElastiCache Redis port"
  value       = aws_elasticache_replication_group.main.port
}

# ECS Outputs
output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "Name of the ECS service"
  value       = aws_ecs_service.api.name
}

# ECR Outputs
output "ecr_api_repository_url" {
  description = "URL of the API ECR repository"
  value       = aws_ecr_repository.api.repository_url
}

output "ecr_gpu_engine_repository_url" {
  description = "URL of the GPU engine ECR repository"
  value       = aws_ecr_repository.gpu_engine.repository_url
}

output "ecr_cpu_engine_repository_url" {
  description = "URL of the CPU engine ECR repository"
  value       = aws_ecr_repository.cpu_engine.repository_url
}

# S3 Outputs
output "models_bucket_name" {
  description = "Name of the models S3 bucket"
  value       = aws_s3_bucket.models.bucket
}

output "translations_bucket_name" {
  description = "Name of the translations S3 bucket"
  value       = aws_s3_bucket.translations.bucket
}

output "backups_bucket_name" {
  description = "Name of the backups S3 bucket"
  value       = aws_s3_bucket.backups.bucket
}

# Security Outputs
output "jwt_secret_arn" {
  description = "ARN of the JWT secret in Secrets Manager"
  value       = aws_secretsmanager_secret.jwt_secret.arn
  sensitive   = true
}

output "encryption_key_arn" {
  description = "ARN of the encryption key in Secrets Manager"
  value       = aws_secretsmanager_secret.encryption_key.arn
  sensitive   = true
}

output "kms_key_id" {
  description = "ID of the KMS key"
  value       = aws_kms_key.main.key_id
}

output "kms_key_arn" {
  description = "ARN of the KMS key"
  value       = aws_kms_key.main.arn
}

# Auto Scaling Outputs
output "gpu_autoscaling_group_name" {
  description = "Name of the GPU auto scaling group"
  value       = aws_autoscaling_group.gpu_instances.name
}

output "gpu_launch_template_id" {
  description = "ID of the GPU launch template"
  value       = aws_launch_template.gpu_instances.id
}

# Monitoring Outputs
output "cloudwatch_dashboard_url" {
  description = "URL of the CloudWatch dashboard"
  value       = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${aws_cloudwatch_dashboard.main.dashboard_name}"
}

output "sns_alerts_topic_arn" {
  description = "ARN of the SNS alerts topic"
  value       = aws_sns_topic.alerts.arn
}

# Cost Monitoring
output "budget_name" {
  description = "Name of the cost budget"
  value       = aws_budgets_budget.monthly_cost.name
}

# Connection Information
output "connection_info" {
  description = "Connection information for the deployed infrastructure"
  value = {
    api_endpoint    = "http://${aws_lb.main.dns_name}/api/v1"
    health_check    = "http://${aws_lb.main.dns_name}/api/v1/system/health"
    dashboard_url   = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${aws_cloudwatch_dashboard.main.dashboard_name}"
    environment     = var.environment
    region          = var.aws_region
  }
}

# Deployment Commands
output "deployment_commands" {
  description = "Commands to deploy the application"
  value = {
    docker_login = "aws ecr get-login-password --region ${var.aws_region} | docker login --username AWS --password-stdin ${aws_ecr_repository.api.repository_url}"
    build_api    = "docker build -f Dockerfile.api -t ${aws_ecr_repository.api.repository_url}:latest ."
    push_api     = "docker push ${aws_ecr_repository.api.repository_url}:latest"
    update_service = "aws ecs update-service --cluster ${aws_ecs_cluster.main.name} --service ${aws_ecs_service.api.name} --force-new-deployment"
  }
}

# Environment Variables for Application
output "environment_variables" {
  description = "Environment variables for the application"
  value = {
    DATABASE_URL = "postgresql+asyncpg://${aws_db_instance.main.username}:${random_password.db_password.result}@${aws_db_instance.main.endpoint}/${aws_db_instance.main.db_name}"
    REDIS_HOST   = aws_elasticache_replication_group.main.primary_endpoint_address
    REDIS_PORT   = tostring(aws_elasticache_replication_group.main.port)
    ENVIRONMENT  = var.environment
    AWS_REGION   = var.aws_region
    MODELS_BUCKET = aws_s3_bucket.models.bucket
    TRANSLATIONS_BUCKET = aws_s3_bucket.translations.bucket
  }
  sensitive = true
}