# CloudWatch Monitoring and Alerting

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${var.project_name}/api"
  retention_in_days = var.log_retention_days

  tags = {
    Name = "${var.project_name}-api-logs"
  }
}

resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${var.project_name}/cluster"
  retention_in_days = var.log_retention_days

  tags = {
    Name = "${var.project_name}-ecs-logs"
  }
}

resource "aws_cloudwatch_log_group" "gpu_instances" {
  name              = "/ec2/${var.project_name}/gpu-instances"
  retention_in_days = var.log_retention_days

  tags = {
    Name = "${var.project_name}-gpu-instances-logs"
  }
}

# SNS Topic for Alerts
resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-alerts"

  tags = {
    Name = "${var.project_name}-alerts-topic"
  }
}

# CloudWatch Dashboard
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.project_name}-dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/ECS", "CPUUtilization", "ServiceName", aws_ecs_service.api.name, "ClusterName", aws_ecs_cluster.main.name],
            [".", "MemoryUtilization", ".", ".", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "ECS Service Metrics"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/RDS", "CPUUtilization", "DBInstanceIdentifier", aws_db_instance.main.id],
            [".", "DatabaseConnections", ".", "."],
            [".", "ReadLatency", ".", "."],
            [".", "WriteLatency", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "RDS Metrics"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 12
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/ElastiCache", "CPUUtilization", "CacheClusterId", "${aws_elasticache_replication_group.main.replication_group_id}-001"],
            [".", "DatabaseMemoryUsagePercentage", ".", "."],
            [".", "CurrConnections", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "ElastiCache Metrics"
          period  = 300
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 18
        width  = 12
        height = 6

        properties = {
          metrics = [
            ["AWS/ApplicationELB", "RequestCount", "LoadBalancer", aws_lb.main.arn_suffix],
            [".", "TargetResponseTime", ".", "."],
            [".", "HTTPCode_Target_2XX_Count", ".", "."],
            [".", "HTTPCode_Target_4XX_Count", ".", "."],
            [".", "HTTPCode_Target_5XX_Count", ".", "."]
          ]
          view    = "timeSeries"
          stacked = false
          region  = var.aws_region
          title   = "Load Balancer Metrics"
          period  = 300
        }
      }
    ]
  })
}

# Custom CloudWatch Metrics for Translation Performance
resource "aws_cloudwatch_log_metric_filter" "translation_speed" {
  name           = "${var.project_name}-translation-speed"
  log_group_name = aws_cloudwatch_log_group.api.name
  pattern        = "[timestamp, level=\"INFO\", service, event=\"translation_completed\", correlation_id, job_id, user_id, source_language, target_language, word_count, processing_time_ms, words_per_minute, ...]"

  metric_transformation {
    name      = "TranslationSpeed"
    namespace = "${var.project_name}/Translation"
    value     = "$words_per_minute"
  }
}

resource "aws_cloudwatch_log_metric_filter" "translation_errors" {
  name           = "${var.project_name}-translation-errors"
  log_group_name = aws_cloudwatch_log_group.api.name
  pattern        = "[timestamp, level=\"ERROR\", service, event=\"translation_failed\", ...]"

  metric_transformation {
    name      = "TranslationErrors"
    namespace = "${var.project_name}/Translation"
    value     = "1"
  }
}

resource "aws_cloudwatch_log_metric_filter" "cache_hits" {
  name           = "${var.project_name}-cache-hits"
  log_group_name = aws_cloudwatch_log_group.api.name
  pattern        = "[timestamp, level=\"INFO\", service, event=\"cache_hit\", ...]"

  metric_transformation {
    name      = "CacheHits"
    namespace = "${var.project_name}/Cache"
    value     = "1"
  }
}

resource "aws_cloudwatch_log_metric_filter" "cache_misses" {
  name           = "${var.project_name}-cache-misses"
  log_group_name = aws_cloudwatch_log_group.api.name
  pattern        = "[timestamp, level=\"INFO\", service, event=\"cache_miss\", ...]"

  metric_transformation {
    name      = "CacheMisses"
    namespace = "${var.project_name}/Cache"
    value     = "1"
  }
}

# CloudWatch Alarms for Application Metrics
resource "aws_cloudwatch_metric_alarm" "high_error_rate" {
  alarm_name          = "${var.project_name}-high-error-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "TranslationErrors"
  namespace           = "${var.project_name}/Translation"
  period              = "300"
  statistic           = "Sum"
  threshold           = "10"
  alarm_description   = "This metric monitors translation error rate"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  tags = {
    Name = "${var.project_name}-high-error-rate-alarm"
  }
}

resource "aws_cloudwatch_metric_alarm" "low_translation_speed" {
  alarm_name          = "${var.project_name}-low-translation-speed"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "3"
  metric_name         = "TranslationSpeed"
  namespace           = "${var.project_name}/Translation"
  period              = "300"
  statistic           = "Average"
  threshold           = "1000"
  alarm_description   = "This metric monitors translation speed performance"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  tags = {
    Name = "${var.project_name}-low-translation-speed-alarm"
  }
}

resource "aws_cloudwatch_metric_alarm" "low_cache_hit_rate" {
  alarm_name          = "${var.project_name}-low-cache-hit-rate"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CacheHitRate"
  namespace           = "${var.project_name}/Cache"
  period              = "300"
  statistic           = "Average"
  threshold           = "0.8"  # 80% hit rate
  alarm_description   = "This metric monitors cache hit rate"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  # Calculate cache hit rate using metric math
  metric_query {
    id = "m1"
    metric {
      metric_name = "CacheHits"
      namespace   = "${var.project_name}/Cache"
      period      = 300
      stat        = "Sum"
    }
  }

  metric_query {
    id = "m2"
    metric {
      metric_name = "CacheMisses"
      namespace   = "${var.project_name}/Cache"
      period      = 300
      stat        = "Sum"
    }
  }

  metric_query {
    id          = "e1"
    expression  = "m1/(m1+m2)"
    label       = "Cache Hit Rate"
    return_data = true
  }

  tags = {
    Name = "${var.project_name}-low-cache-hit-rate-alarm"
  }
}

# Cost Monitoring
resource "aws_budgets_budget" "monthly_cost" {
  name         = "${var.project_name}-monthly-budget"
  budget_type  = "COST"
  limit_amount = "1000"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filters {
    tag {
      key = "Project"
      values = [var.project_name]
    }
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 80
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_email_addresses = ["admin@example.com"]  # Replace with actual email
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 100
    threshold_type            = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = ["admin@example.com"]  # Replace with actual email
  }
}

# X-Ray Tracing (optional)
resource "aws_xray_sampling_rule" "main" {
  count = var.enable_monitoring ? 1 : 0

  rule_name      = "${var.project_name}-sampling-rule"
  priority       = 9000
  version        = 1
  reservoir_size = 1
  fixed_rate     = 0.1
  url_path       = "*"
  host           = "*"
  http_method    = "*"
  service_type   = "*"
  service_name   = "*"
  resource_arn   = "*"

  tags = {
    Name = "${var.project_name}-xray-sampling-rule"
  }
}