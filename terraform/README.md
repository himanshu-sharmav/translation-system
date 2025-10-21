# Terraform Infrastructure for Machine Translation Backend

This directory contains Terraform configurations to deploy the complete infrastructure for the Machine Translation Backend system on AWS.

## Architecture Overview

The infrastructure includes:

- **VPC with public/private subnets** across multiple AZs
- **Application Load Balancer** for traffic distribution
- **ECS Fargate cluster** for API services
- **Auto Scaling Group** with GPU instances for translation processing
- **RDS PostgreSQL** with read replicas for data persistence
- **ElastiCache Redis** for caching and queue management
- **S3 buckets** for model storage, translations, and backups
- **ECR repositories** for container images
- **CloudWatch** monitoring and alerting
- **IAM roles and policies** with least privilege access
- **KMS encryption** for data at rest
- **Secrets Manager** for sensitive configuration

## Prerequisites

1. **AWS CLI** configured with appropriate credentials
2. **Terraform** >= 1.0 installed
3. **SSH key pair** for EC2 access (place public key at `~/.ssh/id_rsa.pub`)
4. **Docker** for building and pushing container images

## Quick Start

### 1. Initialize Terraform

```bash
cd terraform/
terraform init
```

### 2. Configure Variables

Copy the example variables file and customize:

```bash
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` with your specific configuration:

```hcl
# Basic Configuration
aws_region   = "us-east-1"
environment  = "dev"
project_name = "my-translation-backend"

# Network Configuration
vpc_cidr = "10.0.0.0/16"

# Database Configuration
db_instance_class = "db.r6g.large"
db_multi_az      = true

# GPU Configuration
gpu_instance_type = "g4dn.xlarge"
enable_spot_instances = true

# Domain (optional)
domain_name = "api.example.com"
create_ssl_cert = true
```

### 3. Plan and Apply

```bash
# Review the planned changes
terraform plan

# Apply the infrastructure
terraform apply
```

### 4. Build and Deploy Application

After infrastructure is created, build and deploy the application:

```bash
# Get ECR login command from Terraform output
terraform output deployment_commands

# Build and push API image
docker build -f Dockerfile.api -t $(terraform output -raw ecr_api_repository_url):latest .
docker push $(terraform output -raw ecr_api_repository_url):latest

# Update ECS service
aws ecs update-service \
  --cluster $(terraform output -raw ecs_cluster_name) \
  --service $(terraform output -raw ecs_service_name) \
  --force-new-deployment
```

## Configuration Files

### Core Infrastructure

- `main.tf` - VPC, networking, and security groups
- `ecs.tf` - ECS cluster, services, and load balancer
- `rds.tf` - PostgreSQL database with read replicas
- `redis.tf` - ElastiCache Redis cluster
- `gpu-instances.tf` - Auto Scaling Group for GPU instances
- `storage.tf` - S3 buckets and EFS (optional)
- `iam.tf` - IAM roles, policies, and Secrets Manager
- `ecr.tf` - Container registries
- `monitoring.tf` - CloudWatch dashboards, alarms, and logging

### Configuration

- `variables.tf` - Input variables with descriptions
- `outputs.tf` - Output values for integration
- `terraform.tfvars.example` - Example configuration

### Scripts

- `scripts/gpu-userdata.sh` - GPU instance initialization script

## Environment-Specific Deployments

### Development Environment

```bash
# Use smaller instances for cost optimization
terraform apply -var="environment=dev" \
  -var="db_instance_class=db.t3.micro" \
  -var="redis_node_type=cache.t3.micro" \
  -var="ecs_desired_count=1"
```

### Production Environment

```bash
# Use production-grade configuration
terraform apply -var="environment=prod" \
  -var="db_multi_az=true" \
  -var="redis_num_cache_nodes=3" \
  -var="ecs_desired_count=3" \
  -var="enable_monitoring=true"
```

## Monitoring and Observability

### CloudWatch Dashboard

Access the monitoring dashboard:

```bash
# Get dashboard URL
terraform output cloudwatch_dashboard_url
```

### Key Metrics Monitored

- **Translation Performance**: Words per minute, processing time
- **System Health**: CPU, memory, GPU utilization
- **Application Metrics**: API latency, error rates, cache hit rates
- **Cost Tracking**: Resource usage and spending

### Alerts Configuration

Alerts are configured for:

- High error rates (>5%)
- Low translation speed (<1000 WPM)
- High resource utilization (>80%)
- Cost threshold exceeded
- Database connection issues

## Security Features

### Network Security

- Private subnets for application and database tiers
- Security groups with minimal required access
- NAT gateways for outbound internet access
- VPC flow logs for network monitoring

### Data Protection

- Encryption at rest using KMS
- Encryption in transit with TLS
- Secrets stored in AWS Secrets Manager
- S3 bucket policies preventing public access

### Access Control

- IAM roles with least privilege principles
- Service-specific roles for ECS tasks and EC2 instances
- Cross-service access through IAM policies
- MFA requirements for sensitive operations

## Cost Optimization

### Spot Instances

Enable spot instances for GPU workloads:

```hcl
enable_spot_instances = true
spot_price           = "0.50"
```

### Auto Scaling

Automatic scaling based on:

- Queue depth
- CPU/GPU utilization
- Custom application metrics
- Time-based patterns

### Storage Lifecycle

- S3 lifecycle policies for cost optimization
- Automated transition to cheaper storage classes
- Intelligent tiering for frequently accessed data

## Backup and Disaster Recovery

### Automated Backups

- RDS automated backups with point-in-time recovery
- S3 cross-region replication for critical data
- EBS snapshots for EC2 instances
- Database export to S3 for long-term retention

### Recovery Procedures

```bash
# Restore RDS from backup
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier restored-db \
  --db-snapshot-identifier snapshot-id

# Restore from S3 backup
aws s3 sync s3://backup-bucket/database/ ./restore/
```

## Troubleshooting

### Common Issues

1. **ECS Service Won't Start**
   ```bash
   # Check service events
   aws ecs describe-services --cluster cluster-name --services service-name
   
   # Check task logs
   aws logs get-log-events --log-group-name /ecs/translation-backend/api
   ```

2. **GPU Instances Not Scaling**
   ```bash
   # Check Auto Scaling Group
   aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names gpu-asg
   
   # Check CloudWatch alarms
   aws cloudwatch describe-alarms --alarm-names gpu-high-cpu
   ```

3. **Database Connection Issues**
   ```bash
   # Check security groups
   aws ec2 describe-security-groups --group-ids sg-xxxxx
   
   # Test connectivity
   telnet database-endpoint 5432
   ```

### Debugging Commands

```bash
# View Terraform state
terraform show

# Check resource status
terraform refresh

# Validate configuration
terraform validate

# Format configuration files
terraform fmt -recursive
```

## Maintenance

### Regular Tasks

1. **Update AMIs**: Regularly update GPU instance AMIs
2. **Security Patches**: Apply OS and application updates
3. **Certificate Renewal**: Renew SSL certificates before expiration
4. **Backup Verification**: Test backup restoration procedures
5. **Cost Review**: Monitor and optimize resource usage

### Scaling Operations

```bash
# Scale ECS service
aws ecs update-service --cluster cluster-name --service service-name --desired-count 5

# Scale GPU Auto Scaling Group
aws autoscaling update-auto-scaling-group --auto-scaling-group-name gpu-asg --desired-capacity 3

# Scale RDS (requires downtime)
aws rds modify-db-instance --db-instance-identifier db-name --db-instance-class db.r6g.xlarge
```

## Cleanup

To destroy all resources:

```bash
# Destroy infrastructure (be careful!)
terraform destroy

# Confirm destruction
terraform show  # Should show no resources
```

## Support

For issues and questions:

1. Check CloudWatch logs and metrics
2. Review Terraform state and outputs
3. Consult AWS documentation
4. Contact the development team

## Cost Estimation

### Monthly Cost Estimates

| Component | Development | Production |
|-----------|-------------|------------|
| ECS Fargate | $50 | $200 |
| RDS PostgreSQL | $100 | $400 |
| ElastiCache Redis | $50 | $150 |
| GPU Instances (g4dn.xlarge) | $200 | $800 |
| Load Balancer | $25 | $25 |
| S3 Storage | $20 | $100 |
| Data Transfer | $10 | $50 |
| CloudWatch | $10 | $30 |
| **Total** | **$465** | **$1,755** |

*Estimates based on moderate usage patterns and may vary significantly based on actual workload.*