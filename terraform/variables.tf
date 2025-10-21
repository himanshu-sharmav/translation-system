# Variables for Machine Translation Backend Infrastructure

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "translation-backend"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.20.0/24"]
}

# Database Configuration
variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.large"
}

variable "db_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 100
}

variable "db_max_allocated_storage" {
  description = "RDS maximum allocated storage in GB"
  type        = number
  default     = 1000
}

variable "db_backup_retention_period" {
  description = "RDS backup retention period in days"
  type        = number
  default     = 7
}

variable "db_multi_az" {
  description = "Enable RDS Multi-AZ deployment"
  type        = bool
  default     = true
}

# Redis Configuration
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.r6g.large"
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes"
  type        = number
  default     = 2
}

variable "redis_parameter_group_name" {
  description = "Redis parameter group name"
  type        = string
  default     = "default.redis7"
}

# ECS Configuration
variable "ecs_cpu" {
  description = "CPU units for ECS tasks"
  type        = number
  default     = 2048
}

variable "ecs_memory" {
  description = "Memory for ECS tasks in MB"
  type        = number
  default     = 4096
}

variable "ecs_desired_count" {
  description = "Desired number of ECS tasks"
  type        = number
  default     = 2
}

variable "ecs_min_capacity" {
  description = "Minimum number of ECS tasks"
  type        = number
  default     = 1
}

variable "ecs_max_capacity" {
  description = "Maximum number of ECS tasks"
  type        = number
  default     = 10
}

# GPU Instance Configuration
variable "gpu_instance_type" {
  description = "EC2 instance type for GPU workloads"
  type        = string
  default     = "g4dn.xlarge"
}

variable "gpu_min_size" {
  description = "Minimum number of GPU instances"
  type        = number
  default     = 0
}

variable "gpu_max_size" {
  description = "Maximum number of GPU instances"
  type        = number
  default     = 5
}

variable "gpu_desired_capacity" {
  description = "Desired number of GPU instances"
  type        = number
  default     = 1
}

# Monitoring Configuration
variable "enable_monitoring" {
  description = "Enable CloudWatch monitoring"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

# Cost Optimization
variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = true
}

variable "spot_price" {
  description = "Maximum spot price"
  type        = string
  default     = "0.50"
}

# SSL Certificate
variable "domain_name" {
  description = "Domain name for SSL certificate"
  type        = string
  default     = ""
}

variable "create_ssl_cert" {
  description = "Create SSL certificate"
  type        = bool
  default     = false
}

# Backup Configuration
variable "backup_schedule" {
  description = "Backup schedule expression"
  type        = string
  default     = "cron(0 2 * * ? *)"  # Daily at 2 AM
}

variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 30
}