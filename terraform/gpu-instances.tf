# GPU Instances for Translation Engine Processing

# Launch Template for GPU Instances
resource "aws_launch_template" "gpu_instances" {
  name_prefix   = "${var.project_name}-gpu-"
  image_id      = data.aws_ami.gpu_optimized.id
  instance_type = var.gpu_instance_type
  key_name      = aws_key_pair.main.key_name

  vpc_security_group_ids = [aws_security_group.gpu_instances.id]

  # IAM instance profile
  iam_instance_profile {
    name = aws_iam_instance_profile.gpu_instance_profile.name
  }

  # User data script for GPU instance setup
  user_data = base64encode(templatefile("${path.module}/scripts/gpu-userdata.sh", {
    region           = var.aws_region
    cluster_name     = aws_ecs_cluster.main.name
    project_name     = var.project_name
    ecr_repository   = aws_ecr_repository.gpu_engine.repository_url
  }))

  # Block device mappings
  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = 100
      volume_type           = "gp3"
      encrypted             = true
      delete_on_termination = true
    }
  }

  # Instance metadata options
  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  # Monitoring
  monitoring {
    enabled = true
  }

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "${var.project_name}-gpu-instance"
      Type = "gpu-translation-engine"
    }
  }

  tags = {
    Name = "${var.project_name}-gpu-launch-template"
  }
}

# Auto Scaling Group for GPU Instances
resource "aws_autoscaling_group" "gpu_instances" {
  name                = "${var.project_name}-gpu-asg"
  vpc_zone_identifier = aws_subnet.private[*].id
  target_group_arns   = [aws_lb_target_group.gpu_engine.arn]
  health_check_type   = "ELB"
  health_check_grace_period = 300

  min_size         = var.gpu_min_size
  max_size         = var.gpu_max_size
  desired_capacity = var.gpu_desired_capacity

  # Mixed instances policy for cost optimization
  mixed_instances_policy {
    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.gpu_instances.id
        version            = "$Latest"
      }

      # Override for spot instances
      dynamic "override" {
        for_each = var.enable_spot_instances ? [1] : []
        content {
          instance_type     = var.gpu_instance_type
          spot_max_price    = var.spot_price
        }
      }
    }

    instances_distribution {
      on_demand_base_capacity                  = var.enable_spot_instances ? 1 : 100
      on_demand_percentage_above_base_capacity = var.enable_spot_instances ? 0 : 100
      spot_allocation_strategy                 = "diversified"
    }
  }

  # Instance refresh
  instance_refresh {
    strategy = "Rolling"
    preferences {
      min_healthy_percentage = 50
    }
  }

  tag {
    key                 = "Name"
    value               = "${var.project_name}-gpu-instance"
    propagate_at_launch = true
  }

  tag {
    key                 = "Type"
    value               = "gpu-translation-engine"
    propagate_at_launch = true
  }
}

# Auto Scaling Policies
resource "aws_autoscaling_policy" "gpu_scale_up" {
  name                   = "${var.project_name}-gpu-scale-up"
  scaling_adjustment     = 1
  adjustment_type        = "ChangeInCapacity"
  cooldown              = 300
  autoscaling_group_name = aws_autoscaling_group.gpu_instances.name
}

resource "aws_autoscaling_policy" "gpu_scale_down" {
  name                   = "${var.project_name}-gpu-scale-down"
  scaling_adjustment     = -1
  adjustment_type        = "ChangeInCapacity"
  cooldown              = 300
  autoscaling_group_name = aws_autoscaling_group.gpu_instances.name
}

# CloudWatch Alarms for Auto Scaling
resource "aws_cloudwatch_metric_alarm" "gpu_high_cpu" {
  alarm_name          = "${var.project_name}-gpu-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors GPU instance CPU utilization"
  alarm_actions       = [aws_autoscaling_policy.gpu_scale_up.arn]

  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.gpu_instances.name
  }

  tags = {
    Name = "${var.project_name}-gpu-high-cpu-alarm"
  }
}

resource "aws_cloudwatch_metric_alarm" "gpu_low_cpu" {
  alarm_name          = "${var.project_name}-gpu-low-cpu"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"
  statistic           = "Average"
  threshold           = "20"
  alarm_description   = "This metric monitors GPU instance CPU utilization"
  alarm_actions       = [aws_autoscaling_policy.gpu_scale_down.arn]

  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.gpu_instances.name
  }

  tags = {
    Name = "${var.project_name}-gpu-low-cpu-alarm"
  }
}

# Security Group for GPU Instances
resource "aws_security_group" "gpu_instances" {
  name_prefix = "${var.project_name}-gpu-"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "HTTP from ALB"
    from_port       = 8080
    to_port         = 8080
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-gpu-instances-sg"
  }
}

# Target Group for GPU Engine
resource "aws_lb_target_group" "gpu_engine" {
  name     = "${var.project_name}-gpu-tg"
  port     = 8080
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = {
    Name = "${var.project_name}-gpu-target-group"
  }
}

# Load Balancer Listener Rule for GPU Engine
resource "aws_lb_listener_rule" "gpu_engine" {
  listener_arn = aws_lb_listener.api.arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.gpu_engine.arn
  }

  condition {
    path_pattern {
      values = ["/api/v1/translate/gpu/*"]
    }
  }
}

# AMI for GPU-optimized instances
data "aws_ami" "gpu_optimized" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning AMI GPU PyTorch*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Key Pair for SSH access
resource "aws_key_pair" "main" {
  key_name   = "${var.project_name}-keypair"
  public_key = file("~/.ssh/id_rsa.pub")  # Assumes you have an SSH key

  tags = {
    Name = "${var.project_name}-keypair"
  }
}