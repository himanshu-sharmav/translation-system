#!/bin/bash

# GPU Instance User Data Script for Translation Backend
# This script sets up a GPU instance for translation processing

set -e

# Variables passed from Terraform
REGION="${region}"
CLUSTER_NAME="${cluster_name}"
PROJECT_NAME="${project_name}"
ECR_REPOSITORY="${ecr_repository}"

# Update system
yum update -y

# Install Docker
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install NVIDIA Docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | \
  tee /etc/yum.repos.d/nvidia-docker.repo

yum clean expire-cache
yum install -y nvidia-docker2
systemctl restart docker

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# Install CloudWatch agent
yum install -y amazon-cloudwatch-agent

# Configure CloudWatch agent
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << EOF
{
  "agent": {
    "metrics_collection_interval": 60,
    "run_as_user": "cwagent"
  },
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/translation-engine.log",
            "log_group_name": "/ec2/${PROJECT_NAME}/gpu-instances",
            "log_stream_name": "{instance_id}/translation-engine",
            "timezone": "UTC"
          },
          {
            "file_path": "/var/log/docker",
            "log_group_name": "/ec2/${PROJECT_NAME}/gpu-instances",
            "log_stream_name": "{instance_id}/docker",
            "timezone": "UTC"
          }
        ]
      }
    }
  },
  "metrics": {
    "namespace": "${PROJECT_NAME}/GPU",
    "metrics_collected": {
      "cpu": {
        "measurement": [
          "cpu_usage_idle",
          "cpu_usage_iowait",
          "cpu_usage_user",
          "cpu_usage_system"
        ],
        "metrics_collection_interval": 60
      },
      "disk": {
        "measurement": [
          "used_percent"
        ],
        "metrics_collection_interval": 60,
        "resources": [
          "*"
        ]
      },
      "diskio": {
        "measurement": [
          "io_time"
        ],
        "metrics_collection_interval": 60,
        "resources": [
          "*"
        ]
      },
      "mem": {
        "measurement": [
          "mem_used_percent"
        ],
        "metrics_collection_interval": 60
      },
      "netstat": {
        "measurement": [
          "tcp_established",
          "tcp_time_wait"
        ],
        "metrics_collection_interval": 60
      },
      "swap": {
        "measurement": [
          "swap_used_percent"
        ],
        "metrics_collection_interval": 60
      }
    }
  }
}
EOF

# Start CloudWatch agent
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config -m ec2 -s \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json

# Install nvidia-ml-py for GPU monitoring
pip3 install nvidia-ml-py3

# Create GPU monitoring script
cat > /usr/local/bin/gpu-monitor.py << 'EOF'
#!/usr/bin/env python3

import time
import boto3
import pynvml
from datetime import datetime

def get_gpu_metrics():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    
    metrics = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        
        # Get GPU utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        # Get memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        # Get temperature
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        
        # Get power usage
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
        
        metrics.append({
            'gpu_id': i,
            'gpu_utilization': util.gpu,
            'memory_utilization': (mem_info.used / mem_info.total) * 100,
            'memory_used_mb': mem_info.used / 1024 / 1024,
            'memory_total_mb': mem_info.total / 1024 / 1024,
            'temperature': temp,
            'power_watts': power
        })
    
    return metrics

def send_metrics_to_cloudwatch(metrics):
    cloudwatch = boto3.client('cloudwatch', region_name='${REGION}')
    
    for gpu_metric in metrics:
        cloudwatch.put_metric_data(
            Namespace='${PROJECT_NAME}/GPU',
            MetricData=[
                {
                    'MetricName': 'GPUUtilization',
                    'Dimensions': [
                        {
                            'Name': 'InstanceId',
                            'Value': boto3.Session().region_name
                        },
                        {
                            'Name': 'GPUId',
                            'Value': str(gpu_metric['gpu_id'])
                        }
                    ],
                    'Value': gpu_metric['gpu_utilization'],
                    'Unit': 'Percent',
                    'Timestamp': datetime.utcnow()
                },
                {
                    'MetricName': 'GPUMemoryUtilization',
                    'Dimensions': [
                        {
                            'Name': 'InstanceId',
                            'Value': boto3.Session().region_name
                        },
                        {
                            'Name': 'GPUId',
                            'Value': str(gpu_metric['gpu_id'])
                        }
                    ],
                    'Value': gpu_metric['memory_utilization'],
                    'Unit': 'Percent',
                    'Timestamp': datetime.utcnow()
                },
                {
                    'MetricName': 'GPUTemperature',
                    'Dimensions': [
                        {
                            'Name': 'InstanceId',
                            'Value': boto3.Session().region_name
                        },
                        {
                            'Name': 'GPUId',
                            'Value': str(gpu_metric['gpu_id'])
                        }
                    ],
                    'Value': gpu_metric['temperature'],
                    'Unit': 'None',
                    'Timestamp': datetime.utcnow()
                },
                {
                    'MetricName': 'GPUPowerUsage',
                    'Dimensions': [
                        {
                            'Name': 'InstanceId',
                            'Value': boto3.Session().region_name
                        },
                        {
                            'Name': 'GPUId',
                            'Value': str(gpu_metric['gpu_id'])
                        }
                    ],
                    'Value': gpu_metric['power_watts'],
                    'Unit': 'None',
                    'Timestamp': datetime.utcnow()
                }
            ]
        )

if __name__ == "__main__":
    while True:
        try:
            metrics = get_gpu_metrics()
            send_metrics_to_cloudwatch(metrics)
            time.sleep(60)  # Send metrics every minute
        except Exception as e:
            print(f"Error collecting GPU metrics: {e}")
            time.sleep(60)
EOF

chmod +x /usr/local/bin/gpu-monitor.py

# Create systemd service for GPU monitoring
cat > /etc/systemd/system/gpu-monitor.service << EOF
[Unit]
Description=GPU Monitoring Service
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/local/bin/gpu-monitor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl enable gpu-monitor
systemctl start gpu-monitor

# Login to ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_REPOSITORY}

# Pull and run the translation engine container
docker pull ${ECR_REPOSITORY}:latest

# Create translation engine service
cat > /etc/systemd/system/translation-engine.service << EOF
[Unit]
Description=Translation Engine Service
After=docker.service
Requires=docker.service

[Service]
Type=simple
User=root
ExecStartPre=-/usr/bin/docker stop translation-engine
ExecStartPre=-/usr/bin/docker rm translation-engine
ExecStart=/usr/bin/docker run --name translation-engine \
  --runtime=nvidia \
  --gpus all \
  -p 8080:8080 \
  -e AWS_DEFAULT_REGION=${REGION} \
  -e PROJECT_NAME=${PROJECT_NAME} \
  --log-driver=awslogs \
  --log-opt awslogs-group=/ec2/${PROJECT_NAME}/gpu-instances \
  --log-opt awslogs-stream=translation-engine \
  --log-opt awslogs-region=${REGION} \
  ${ECR_REPOSITORY}:latest
ExecStop=/usr/bin/docker stop translation-engine
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl enable translation-engine
systemctl start translation-engine

# Create health check script
cat > /usr/local/bin/health-check.sh << 'EOF'
#!/bin/bash

# Health check for translation engine
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health)

if [ "$response" = "200" ]; then
    echo "Translation engine is healthy"
    exit 0
else
    echo "Translation engine is unhealthy (HTTP $response)"
    exit 1
fi
EOF

chmod +x /usr/local/bin/health-check.sh

# Set up cron job for health checks
echo "*/1 * * * * /usr/local/bin/health-check.sh >> /var/log/health-check.log 2>&1" | crontab -

# Signal that the instance is ready
/opt/aws/bin/cfn-signal -e $? --stack ${PROJECT_NAME}-stack --resource AutoScalingGroup --region ${REGION}

echo "GPU instance setup completed successfully"