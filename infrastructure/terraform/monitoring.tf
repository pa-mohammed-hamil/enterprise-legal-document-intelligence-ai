# ==============================================================================
# Enterprise Legal Document Intelligence AI
# Monitoring / Observability Infrastructure
#
# File:
#   infrastructure/terraform/monitoring.tf
#
# Covers:
#   - CloudWatch log groups
#   - Application / worker / AI logs
#   - RAG and model metrics
#   - Security metrics
#   - CloudWatch alarms
#   - SNS alerting
#   - CloudWatch dashboard
#   - RDS / Redis monitoring
#   - ALB monitoring
#   - ECS monitoring
#   - Log metric filters
#   - AI/RAG security event detection
# ==============================================================================


# ------------------------------------------------------------------------------
# SNS topic for infrastructure / application alerts
# ------------------------------------------------------------------------------

resource "aws_sns_topic" "alerts" {
  name = "${local.name_prefix}-alerts"

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-alerts"
      Component = "Monitoring"
    }
  )
}


# ------------------------------------------------------------------------------
# Optional email alert subscription
# ------------------------------------------------------------------------------

resource "aws_sns_topic_subscription" "email_alerts" {
  count = var.alert_email != null && var.alert_email != "" ? 1 : 0

  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}


# ==============================================================================
# CloudWatch Log Groups
# ==============================================================================


# ------------------------------------------------------------------------------
# Backend API logs
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "backend" {
  name = "${local.resource_names.logs}/backend"

  retention_in_days = var.log_retention_days

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-backend-logs"
      Component = "Backend"
    }
  )
}


# ------------------------------------------------------------------------------
# Frontend logs
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "frontend" {
  name = "${local.resource_names.logs}/frontend"

  retention_in_days = var.log_retention_days

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-frontend-logs"
      Component = "Frontend"
    }
  )
}


# ------------------------------------------------------------------------------
# Worker logs
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "worker" {
  name = "${local.resource_names.logs}/worker"

  retention_in_days = var.log_retention_days

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-worker-logs"
      Component = "Worker"
    }
  )
}


# ------------------------------------------------------------------------------
# RAG logs
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "rag" {
  name = "${local.resource_names.logs}/rag"

  retention_in_days = var.log_retention_days

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-rag-logs"
      Component = "RAG"
    }
  )
}


# ------------------------------------------------------------------------------
# AI model logs
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "models" {
  name = "${local.resource_names.logs}/models"

  retention_in_days = var.log_retention_days

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-model-logs"
      Component = "Models"
    }
  )
}


# ------------------------------------------------------------------------------
# Agent logs
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "agents" {
  name = "${local.resource_names.logs}/agents"

  retention_in_days = var.log_retention_days

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-agent-logs"
      Component = "Agents"
    }
  )
}


# ------------------------------------------------------------------------------
# Security logs
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "security" {
  name = "${local.resource_names.logs}/security"

  retention_in_days = var.security_log_retention_days

  tags = merge(
    local.common_tags,
    {
      Name               = "${local.name_prefix}-security-logs"
      Component          = "Security"
      DataClassification = "SecuritySensitive"
    }
  )
}


# ------------------------------------------------------------------------------
# Audit logs
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "audit" {
  name = "${local.resource_names.logs}/audit"

  retention_in_days = var.audit_log_retention_days

  tags = merge(
    local.common_tags,
    {
      Name               = "${local.name_prefix}-audit-logs"
      Component          = "Audit"
      DataClassification = "Confidential"
    }
  )
}


# ------------------------------------------------------------------------------
# Document ingestion logs
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "ingestion" {
  name = "${local.resource_names.logs}/ingestion"

  retention_in_days = var.log_retention_days

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-ingestion-logs"
      Component = "DocumentIngestion"
    }
  )
}


# ------------------------------------------------------------------------------
# Evaluation logs
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "evaluation" {
  name = "${local.resource_names.logs}/evaluation"

  retention_in_days = var.log_retention_days

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-evaluation-logs"
      Component = "Evaluation"
    }
  )
}


# ==============================================================================
# Log Metric Filters
# ==============================================================================


# ------------------------------------------------------------------------------
# Backend application errors
#
# Expected structured log:
#
# {
#   "level": "ERROR"
# }
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_log_metric_filter" "backend_errors" {
  name = "${local.name_prefix}-backend-errors"

  log_group_name = aws_cloudwatch_log_group.backend.name

  pattern = "{ $.level = \"ERROR\" }"

  metric_transformation {
    name      = "BackendErrors"
    namespace = "${var.project_name}/${var.environment}/Application"
    value     = "1"
  }
}


# ------------------------------------------------------------------------------
# RAG retrieval failures
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_log_metric_filter" "rag_retrieval_failure" {
  name = "${local.name_prefix}-rag-retrieval-failures"

  log_group_name = aws_cloudwatch_log_group.rag.name

  pattern = "{ $.event = \"retrieval_failure\" }"

  metric_transformation {
    name      = "RAGRetrievalFailures"
    namespace = "${var.project_name}/${var.environment}/RAG"
    value     = "1"
  }
}


# ------------------------------------------------------------------------------
# Hallucination / answer validation failures
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_log_metric_filter" "hallucination_detected" {
  name = "${local.name_prefix}-hallucination-detected"

  log_group_name = aws_cloudwatch_log_group.rag.name

  pattern = "{ $.event = \"hallucination_detected\" }"

  metric_transformation {
    name      = "HallucinationDetected"
    namespace = "${var.project_name}/${var.environment}/RAG"
    value     = "1"
  }
}


# ------------------------------------------------------------------------------
# Citation validation failures
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_log_metric_filter" "citation_failure" {
  name = "${local.name_prefix}-citation-validation-failure"

  log_group_name = aws_cloudwatch_log_group.rag.name

  pattern = "{ $.event = \"citation_validation_failure\" }"

  metric_transformation {
    name      = "CitationValidationFailures"
    namespace = "${var.project_name}/${var.environment}/RAG"
    value     = "1"
  }
}


# ------------------------------------------------------------------------------
# LLM / model inference failures
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_log_metric_filter" "model_inference_failure" {
  name = "${local.name_prefix}-model-inference-failure"

  log_group_name = aws_cloudwatch_log_group.models.name

  pattern = "{ $.event = \"model_inference_failure\" }"

  metric_transformation {
    name      = "ModelInferenceFailures"
    namespace = "${var.project_name}/${var.environment}/Models"
    value     = "1"
  }
}


# ------------------------------------------------------------------------------
# Agent execution failures
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_log_metric_filter" "agent_failure" {
  name = "${local.name_prefix}-agent-failure"

  log_group_name = aws_cloudwatch_log_group.agents.name

  pattern = "{ $.event = \"agent_failure\" }"

  metric_transformation {
    name      = "AgentFailures"
    namespace = "${var.project_name}/${var.environment}/Agents"
    value     = "1"
  }
}


# ==============================================================================
# AI Security Metric Filters
# ==============================================================================


# ------------------------------------------------------------------------------
# Prompt injection detection
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_log_metric_filter" "prompt_injection" {
  name = "${local.name_prefix}-prompt-injection"

  log_group_name = aws_cloudwatch_log_group.security.name

  pattern = "{ $.event = \"prompt_injection_detected\" }"

  metric_transformation {
    name      = "PromptInjectionDetected"
    namespace = "${var.project_name}/${var.environment}/Security"
    value     = "1"
  }
}


# ------------------------------------------------------------------------------
# Unauthorized document access
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_log_metric_filter" "unauthorized_document_access" {
  name = "${local.name_prefix}-unauthorized-document-access"

  log_group_name = aws_cloudwatch_log_group.security.name

  pattern = "{ $.event = \"unauthorized_document_access\" }"

  metric_transformation {
    name      = "UnauthorizedDocumentAccess"
    namespace = "${var.project_name}/${var.environment}/Security"
    value     = "1"
  }
}


# ------------------------------------------------------------------------------
# Tenant isolation violation
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_log_metric_filter" "tenant_isolation_violation" {
  name = "${local.name_prefix}-tenant-isolation-violation"

  log_group_name = aws_cloudwatch_log_group.security.name

  pattern = "{ $.event = \"tenant_isolation_violation\" }"

  metric_transformation {
    name      = "TenantIsolationViolation"
    namespace = "${var.project_name}/${var.environment}/Security"
    value     = "1"
  }
}


# ------------------------------------------------------------------------------
# PII exposure detection
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_log_metric_filter" "pii_exposure" {
  name = "${local.name_prefix}-pii-exposure"

  log_group_name = aws_cloudwatch_log_group.security.name

  pattern = "{ $.event = \"pii_exposure_detected\" }"

  metric_transformation {
    name      = "PIIExposureDetected"
    namespace = "${var.project_name}/${var.environment}/Security"
    value     = "1"
  }
}


# ==============================================================================
# Application Alarms
# ==============================================================================


# ------------------------------------------------------------------------------
# Backend application errors
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "backend_errors" {
  alarm_name = "${local.name_prefix}-backend-errors"

  alarm_description = "Backend application error rate exceeded threshold."

  namespace   = "${var.project_name}/${var.environment}/Application"
  metric_name = "BackendErrors"

  statistic = "Sum"

  period = 300

  evaluation_periods = 1

  threshold = var.backend_error_threshold

  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "notBreaching"

  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]

  ok_actions = [
    aws_sns_topic.alerts.arn
  ]

  tags = local.common_tags
}


# ------------------------------------------------------------------------------
# RAG retrieval failures
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "rag_retrieval_failures" {
  alarm_name = "${local.name_prefix}-rag-retrieval-failures"

  alarm_description = "RAG retrieval failures exceeded the configured threshold."

  namespace   = "${var.project_name}/${var.environment}/RAG"
  metric_name = "RAGRetrievalFailures"

  statistic = "Sum"

  period = 300

  evaluation_periods = 1

  threshold = var.rag_failure_threshold

  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "notBreaching"

  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]

  tags = local.common_tags
}


# ------------------------------------------------------------------------------
# Hallucination alarm
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "hallucination" {
  alarm_name = "${local.name_prefix}-hallucination-detection"

  alarm_description = "Legal AI answer validation detected hallucinated output."

  namespace   = "${var.project_name}/${var.environment}/RAG"
  metric_name = "HallucinationDetected"

  statistic = "Sum"

  period = 300

  evaluation_periods = 1

  threshold = var.hallucination_threshold

  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "notBreaching"

  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]

  tags = local.common_tags
}


# ------------------------------------------------------------------------------
# Citation validation alarm
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "citation_validation" {
  alarm_name = "${local.name_prefix}-citation-validation-failures"

  alarm_description = "Legal RAG citation validation failures exceeded threshold."

  namespace   = "${var.project_name}/${var.environment}/RAG"
  metric_name = "CitationValidationFailures"

  statistic = "Sum"

  period = 300

  evaluation_periods = 1

  threshold = var.citation_failure_threshold

  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "notBreaching"

  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]

  tags = local.common_tags
}


# ------------------------------------------------------------------------------
# Model inference failures
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "model_inference_failure" {
  alarm_name = "${local.name_prefix}-model-inference-failure"

  alarm_description = "Model inference failures exceeded threshold."

  namespace   = "${var.project_name}/${var.environment}/Models"
  metric_name = "ModelInferenceFailures"

  statistic = "Sum"

  period = 300

  evaluation_periods = 1

  threshold = var.model_failure_threshold

  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "notBreaching"

  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]

  tags = local.common_tags
}


# ==============================================================================
# Security Alarms
# ==============================================================================


# ------------------------------------------------------------------------------
# Prompt injection
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "prompt_injection" {
  alarm_name = "${local.name_prefix}-prompt-injection"

  alarm_description = "Prompt injection attempts detected by AI security controls."

  namespace   = "${var.project_name}/${var.environment}/Security"
  metric_name = "PromptInjectionDetected"

  statistic = "Sum"

  period = 300

  evaluation_periods = 1

  threshold = var.prompt_injection_threshold

  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "notBreaching"

  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]

  tags = local.common_tags
}


# ------------------------------------------------------------------------------
# Unauthorized document access
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "unauthorized_document_access" {
  alarm_name = "${local.name_prefix}-unauthorized-document-access"

  alarm_description = "Unauthorized legal document access was detected."

  namespace   = "${var.project_name}/${var.environment}/Security"
  metric_name = "UnauthorizedDocumentAccess"

  statistic = "Sum"

  period = 300

  evaluation_periods = 1

  threshold = 1

  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "notBreaching"

  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]

  tags = local.common_tags
}


# ------------------------------------------------------------------------------
# Tenant isolation violation
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "tenant_isolation_violation" {
  alarm_name = "${local.name_prefix}-tenant-isolation-violation"

  alarm_description = "Possible cross-tenant data access was detected."

  namespace   = "${var.project_name}/${var.environment}/Security"
  metric_name = "TenantIsolationViolation"

  statistic = "Sum"

  period = 60

  evaluation_periods = 1

  threshold = 1

  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "notBreaching"

  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]

  tags = local.common_tags
}


# ------------------------------------------------------------------------------
# PII exposure
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "pii_exposure" {
  alarm_name = "${local.name_prefix}-pii-exposure"

  alarm_description = "Potential PII exposure was detected in AI output."

  namespace   = "${var.project_name}/${var.environment}/Security"
  metric_name = "PIIExposureDetected"

  statistic = "Sum"

  period = 60

  evaluation_periods = 1

  threshold = 1

  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "notBreaching"

  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]

  tags = local.common_tags
}


# ==============================================================================
# ALB Monitoring
#
# These resources expect compute.tf to expose:
#
# aws_lb.main
# aws_lb_target_group.backend
# ==============================================================================


resource "aws_cloudwatch_metric_alarm" "alb_5xx" {
  alarm_name = "${local.name_prefix}-alb-5xx"

  alarm_description = "Application Load Balancer is returning elevated 5xx errors."

  namespace   = "AWS/ApplicationELB"
  metric_name = "HTTPCode_ELB_5XX_Count"

  statistic = "Sum"

  period = 300

  evaluation_periods = 2

  threshold = var.alb_5xx_threshold

  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "notBreaching"

  dimensions = {
    LoadBalancer = aws_lb.main.arn_suffix
  }

  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]

  tags = local.common_tags
}


# ------------------------------------------------------------------------------
# Backend target response time
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "backend_latency" {
  alarm_name = "${local.name_prefix}-backend-latency"

  alarm_description = "Backend target response time is elevated."

  namespace   = "AWS/ApplicationELB"
  metric_name = "TargetResponseTime"

  extended_statistic = "p95"

  period = 300

  evaluation_periods = 2

  threshold = var.backend_latency_threshold

  comparison_operator = "GreaterThanThreshold"

  treat_missing_data = "notBreaching"

  dimensions = {
    LoadBalancer = aws_lb.main.arn_suffix

    TargetGroup = aws_lb_target_group.backend.arn_suffix
  }

  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]

  tags = local.common_tags
}


# ------------------------------------------------------------------------------
# Unhealthy backend hosts
# ------------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "backend_unhealthy_hosts" {
  alarm_name = "${local.name_prefix}-backend-unhealthy-hosts"

  alarm_description = "One or more backend targets are unhealthy."

  namespace   = "AWS/ApplicationELB"
  metric_name = "UnHealthyHostCount"

  statistic = "Average"

  period = 60

  evaluation_periods = 2

  threshold = 1

  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "breaching"

  dimensions = {
    LoadBalancer = aws_lb.main.arn_suffix

    TargetGroup = aws_lb_target_group.backend.arn_suffix
  }

  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]

  tags = local.common_tags
}


# ==============================================================================
# ECS Monitoring
#
# Expects compute.tf:
#
# aws_ecs_cluster.main
# aws_ecs_service.backend
# aws_ecs_service.worker
# ==============================================================================


resource "aws_cloudwatch_metric_alarm" "backend_cpu" {
  alarm_name = "${local.name_prefix}-backend-high-cpu"

  alarm_description = "Backend ECS CPU utilization is high."

  namespace   = "AWS/ECS"
  metric_name = "CPUUtilization"

  statistic = "Average"

  period = 300

  evaluation_periods = 2

  threshold = var.ecs_cpu_alarm_threshold

  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "notBreaching"

  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
    ServiceName = aws_ecs_service.backend.name
  }

  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]

  tags = local.common_tags
}


resource "aws_cloudwatch_metric_alarm" "backend_memory" {
  alarm_name = "${local.name_prefix}-backend-high-memory"

  alarm_description = "Backend ECS memory utilization is high."

  namespace   = "AWS/ECS"
  metric_name = "MemoryUtilization"

  statistic = "Average"

  period = 300

  evaluation_periods = 2

  threshold = var.ecs_memory_alarm_threshold

  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "notBreaching"

  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
    ServiceName = aws_ecs_service.backend.name
  }

  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]

  tags = local.common_tags
}


resource "aws_cloudwatch_metric_alarm" "worker_cpu" {
  alarm_name = "${local.name_prefix}-worker-high-cpu"

  alarm_description = "Document/AI worker CPU utilization is high."

  namespace   = "AWS/ECS"
  metric_name = "CPUUtilization"

  statistic = "Average"

  period = 300

  evaluation_periods = 2

  threshold = var.ecs_cpu_alarm_threshold

  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "notBreaching"

  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
    ServiceName = aws_ecs_service.worker.name
  }

  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]

  tags = local.common_tags
}


resource "aws_cloudwatch_metric_alarm" "worker_memory" {
  alarm_name = "${local.name_prefix}-worker-high-memory"

  alarm_description = "Document/AI worker memory utilization is high."

  namespace   = "AWS/ECS"
  metric_name = "MemoryUtilization"

  statistic = "Average"

  period = 300

  evaluation_periods = 2

  threshold = var.ecs_memory_alarm_threshold

  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "notBreaching"

  dimensions = {
    ClusterName = aws_ecs_cluster.main.name
    ServiceName = aws_ecs_service.worker.name
  }

  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]

  tags = local.common_tags
}


# ==============================================================================
# PostgreSQL / RDS Monitoring
#
# Expects database.tf:
#
# aws_db_instance.postgres
# ==============================================================================


resource "aws_cloudwatch_metric_alarm" "database_cpu" {
  alarm_name = "${local.name_prefix}-database-high-cpu"

  alarm_description = "PostgreSQL CPU utilization is high."

  namespace   = "AWS/RDS"
  metric_name = "CPUUtilization"

  statistic = "Average"

  period = 300

  evaluation_periods = 2

  threshold = var.database_cpu_alarm_threshold

  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "notBreaching"

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.postgres.identifier
  }

  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]

  tags = local.common_tags
}


resource "aws_cloudwatch_metric_alarm" "database_connections" {
  alarm_name = "${local.name_prefix}-database-connections"

  alarm_description = "PostgreSQL connection count is high."

  namespace   = "AWS/RDS"
  metric_name = "DatabaseConnections"

  statistic = "Average"

  period = 300

  evaluation_periods = 2

  threshold = var.database_connection_alarm_threshold

  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "notBreaching"

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.postgres.identifier
  }

  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]

  tags = local.common_tags
}


resource "aws_cloudwatch_metric_alarm" "database_storage" {
  alarm_name = "${local.name_prefix}-database-low-storage"

  alarm_description = "PostgreSQL free storage space is below threshold."

  namespace   = "AWS/RDS"
  metric_name = "FreeStorageSpace"

  statistic = "Average"

  period = 300

  evaluation_periods = 2

  threshold = var.database_free_storage_threshold_bytes

  comparison_operator = "LessThanThreshold"

  treat_missing_data = "notBreaching"

  dimensions = {
    DBInstanceIdentifier = aws_db_instance.postgres.identifier
  }

  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]

  tags = local.common_tags
}


# ==============================================================================
# Redis Monitoring
#
# Expects database.tf:
#
# aws_elasticache_replication_group.redis
# ==============================================================================


resource "aws_cloudwatch_metric_alarm" "redis_cpu" {
  alarm_name = "${local.name_prefix}-redis-high-cpu"

  alarm_description = "Redis CPU utilization is high."

  namespace   = "AWS/ElastiCache"
  metric_name = "EngineCPUUtilization"

  statistic = "Average"

  period = 300

  evaluation_periods = 2

  threshold = var.redis_cpu_alarm_threshold

  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "notBreaching"

  dimensions = {
    ReplicationGroupId = aws_elasticache_replication_group.redis.replication_group_id
  }

  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]

  tags = local.common_tags
}


resource "aws_cloudwatch_metric_alarm" "redis_memory" {
  alarm_name = "${local.name_prefix}-redis-memory"

  alarm_description = "Redis database memory usage is high."

  namespace   = "AWS/ElastiCache"
  metric_name = "DatabaseMemoryUsagePercentage"

  statistic = "Average"

  period = 300

  evaluation_periods = 2

  threshold = var.redis_memory_alarm_threshold

  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "notBreaching"

  dimensions = {
    ReplicationGroupId = aws_elasticache_replication_group.redis.replication_group_id
  }

  alarm_actions = [
    aws_sns_topic.alerts.arn
  ]

  tags = local.common_tags
}


# ==============================================================================
# CloudWatch Dashboard
# ==============================================================================


resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${local.name_prefix}-observability"

  dashboard_body = jsonencode({
    widgets = [

      # ------------------------------------------------------------------------
      # Backend ECS CPU
      # ------------------------------------------------------------------------

      {
        type = "metric"

        x      = 0
        y      = 0
        width  = 12
        height = 6

        properties = {
          title  = "Backend CPU"
          region = local.aws_region

          metrics = [
            [
              "AWS/ECS",
              "CPUUtilization",
              "ClusterName",
              aws_ecs_cluster.main.name,
              "ServiceName",
              aws_ecs_service.backend.name
            ]
          ]

          period = 300
          stat   = "Average"
        }
      },

      # ------------------------------------------------------------------------
      # Backend memory
      # ------------------------------------------------------------------------

      {
        type = "metric"

        x      = 12
        y      = 0
        width  = 12
        height = 6

        properties = {
          title  = "Backend Memory"
          region = local.aws_region

          metrics = [
            [
              "AWS/ECS",
              "MemoryUtilization",
              "ClusterName",
              aws_ecs_cluster.main.name,
              "ServiceName",
              aws_ecs_service.backend.name
            ]
          ]

          period = 300
          stat   = "Average"
        }
      },

      # ------------------------------------------------------------------------
      # Backend response time
      # ------------------------------------------------------------------------

      {
        type = "metric"

        x      = 0
        y      = 6
        width  = 12
        height = 6

        properties = {
          title  = "Backend Response Time"
          region = local.aws_region

          metrics = [
            [
              "AWS/ApplicationELB",
              "TargetResponseTime",
              "LoadBalancer",
              aws_lb.main.arn_suffix,
              "TargetGroup",
              aws_lb_target_group.backend.arn_suffix
            ]
          ]

          stat   = "p95"
          period = 300
        }
      },

      # ------------------------------------------------------------------------
      # ALB errors
      # ------------------------------------------------------------------------

      {
        type = "metric"

        x      = 12
        y      = 6
        width  = 12
        height = 6

        properties = {
          title  = "ALB 5xx Errors"
          region = local.aws_region

          metrics = [
            [
              "AWS/ApplicationELB",
              "HTTPCode_ELB_5XX_Count",
              "LoadBalancer",
              aws_lb.main.arn_suffix
            ]
          ]

          stat   = "Sum"
          period = 300
        }
      },

      # ------------------------------------------------------------------------
      # RAG failures
      # ------------------------------------------------------------------------

      {
        type = "metric"

        x      = 0
        y      = 12
        width  = 8
        height = 6

        properties = {
          title  = "RAG Retrieval Failures"
          region = local.aws_region

          metrics = [
            [
              "${var.project_name}/${var.environment}/RAG",
              "RAGRetrievalFailures"
            ]
          ]

          stat   = "Sum"
          period = 300
        }
      },

      # ------------------------------------------------------------------------
      # Hallucinations
      # ------------------------------------------------------------------------

      {
        type = "metric"

        x      = 8
        y      = 12
        width  = 8
        height = 6

        properties = {
          title  = "Hallucination Detection"
          region = local.aws_region

          metrics = [
            [
              "${var.project_name}/${var.environment}/RAG",
              "HallucinationDetected"
            ]
          ]

          stat   = "Sum"
          period = 300
        }
      },

      # ------------------------------------------------------------------------
      # Citation failures
      # ------------------------------------------------------------------------

      {
        type = "metric"

        x      = 16
        y      = 12
        width  = 8
        height = 6

        properties = {
          title  = "Citation Validation Failures"
          region = local.aws_region

          metrics = [
            [
              "${var.project_name}/${var.environment}/RAG",
              "CitationValidationFailures"
            ]
          ]

          stat   = "Sum"
          period = 300
        }
      },

      # ------------------------------------------------------------------------
      # Security events
      # ------------------------------------------------------------------------

      {
        type = "metric"

        x      = 0
        y      = 18
        width  = 12
        height = 6

        properties = {
          title  = "AI Security Events"
          region = local.aws_region

          metrics = [
            [
              "${var.project_name}/${var.environment}/Security",
              "PromptInjectionDetected"
            ],

            [
              ".",
              "UnauthorizedDocumentAccess"
            ],

            [
              ".",
              "TenantIsolationViolation"
            ],

            [
              ".",
              "PIIExposureDetected"
            ]
          ]

          stat   = "Sum"
          period = 300
        }
      },

      # ------------------------------------------------------------------------
      # Model inference failures
      # ------------------------------------------------------------------------

      {
        type = "metric"

        x      = 12
        y      = 18
        width  = 12
        height = 6

        properties = {
          title  = "Model Inference Failures"
          region = local.aws_region

          metrics = [
            [
              "${var.project_name}/${var.environment}/Models",
              "ModelInferenceFailures"
            ]
          ]

          stat   = "Sum"
          period = 300
        }
      },

      # ------------------------------------------------------------------------
      # PostgreSQL CPU
      # ------------------------------------------------------------------------

      {
        type = "metric"

        x      = 0
        y      = 24
        width  = 8
        height = 6

        properties = {
          title  = "PostgreSQL CPU"
          region = local.aws_region

          metrics = [
            [
              "AWS/RDS",
              "CPUUtilization",
              "DBInstanceIdentifier",
              aws_db_instance.postgres.identifier
            ]
          ]

          stat   = "Average"
          period = 300
        }
      },

      # ------------------------------------------------------------------------
      # PostgreSQL connections
      # ------------------------------------------------------------------------

      {
        type = "metric"

        x      = 8
        y      = 24
        width  = 8
        height = 6

        properties = {
          title  = "PostgreSQL Connections"
          region = local.aws_region

          metrics = [
            [
              "AWS/RDS",
              "DatabaseConnections",
              "DBInstanceIdentifier",
              aws_db_instance.postgres.identifier
            ]
          ]

          stat   = "Average"
          period = 300
        }
      },

      # ------------------------------------------------------------------------
      # Redis memory
      # ------------------------------------------------------------------------

      {
        type = "metric"

        x      = 16
        y      = 24
        width  = 8
        height = 6

        properties = {
          title  = "Redis Memory"
          region = local.aws_region

          metrics = [
            [
              "AWS/ElastiCache",
              "DatabaseMemoryUsagePercentage",
              "ReplicationGroupId",
              aws_elasticache_replication_group.redis.replication_group_id
            ]
          ]

          stat   = "Average"
          period = 300
        }
      },

      # ------------------------------------------------------------------------
      # Backend error logs
      # ------------------------------------------------------------------------

      {
        type = "log"

        x      = 0
        y      = 30
        width  = 12
        height = 6

        properties = {
          title  = "Recent Backend Errors"
          region = local.aws_region

          query = join(
            "\n",
            [
              "SOURCE '${aws_cloudwatch_log_group.backend.name}'",
              "| fields @timestamp, level, message, request_id",
              "| filter level = \"ERROR\"",
              "| sort @timestamp desc",
              "| limit 50"
            ]
          )

          view = "table"
        }
      },

      # ------------------------------------------------------------------------
      # Security logs
      # ------------------------------------------------------------------------

      {
        type = "log"

        x      = 12
        y      = 30
        width  = 12
        height = 6

        properties = {
          title  = "Recent Security Events"
          region = local.aws_region

          query = join(
            "\n",
            [
              "SOURCE '${aws_cloudwatch_log_group.security.name}'",
              "| fields @timestamp, event, severity, user_id, organization_id, request_id",
              "| sort @timestamp desc",
              "| limit 50"
            ]
          )

          view = "table"
        }
      }
    ]
  })
}


# ==============================================================================
# Optional AWS Cost Anomaly Monitoring
# ==============================================================================


resource "aws_ce_anomaly_monitor" "service" {
  count = var.enable_cost_anomaly_detection ? 1 : 0

  name = "${local.name_prefix}-service-cost-monitor"

  monitor_type      = "DIMENSIONAL"
  monitor_dimension = "SERVICE"
}


resource "aws_ce_anomaly_subscription" "alerts" {
  count = (
    var.enable_cost_anomaly_detection &&
    var.alert_email != null &&
    var.alert_email != ""
  ) ? 1 : 0

  name = "${local.name_prefix}-cost-anomaly-alerts"

  frequency = "DAILY"

  monitor_arn_list = [
    aws_ce_anomaly_monitor.service[0].arn
  ]

  subscriber {
    type    = "EMAIL"
    address = var.alert_email
  }

  threshold_expression {
    dimension {
      key           = "ANOMALY_TOTAL_IMPACT_ABSOLUTE"
      values        = [tostring(var.cost_anomaly_threshold)]
      match_options = ["GREATER_THAN_OR_EQUAL"]
    }
  }

  tags = local.common_tags
}


# ==============================================================================
# Monitoring Locals
# ==============================================================================


locals {
  monitoring_log_groups = {
    backend = aws_cloudwatch_log_group.backend.name

    frontend = aws_cloudwatch_log_group.frontend.name

    worker = aws_cloudwatch_log_group.worker.name

    rag = aws_cloudwatch_log_group.rag.name

    models = aws_cloudwatch_log_group.models.name

    agents = aws_cloudwatch_log_group.agents.name

    security = aws_cloudwatch_log_group.security.name

    audit = aws_cloudwatch_log_group.audit.name

    ingestion = aws_cloudwatch_log_group.ingestion.name

    evaluation = aws_cloudwatch_log_group.evaluation.name
  }


  monitoring_namespaces = {
    application = "${var.project_name}/${var.environment}/Application"

    rag = "${var.project_name}/${var.environment}/RAG"

    models = "${var.project_name}/${var.environment}/Models"

    agents = "${var.project_name}/${var.environment}/Agents"

    security = "${var.project_name}/${var.environment}/Security"

    ingestion = "${var.project_name}/${var.environment}/Ingestion"

    evaluation = "${var.project_name}/${var.environment}/Evaluation"
  }
}


# ==============================================================================
# Monitoring Outputs
# ==============================================================================


output "monitoring_sns_topic_arn" {
  description = "SNS topic ARN used for infrastructure and AI alerts."

  value = aws_sns_topic.alerts.arn
}


output "cloudwatch_dashboard_name" {
  description = "Main CloudWatch observability dashboard."

  value = aws_cloudwatch_dashboard.main.dashboard_name
}


output "backend_log_group_name" {
  description = "Backend CloudWatch log group."

  value = aws_cloudwatch_log_group.backend.name
}


output "rag_log_group_name" {
  description = "RAG CloudWatch log group."

  value = aws_cloudwatch_log_group.rag.name
}


output "model_log_group_name" {
  description = "Model-serving CloudWatch log group."

  value = aws_cloudwatch_log_group.models.name
}


output "security_log_group_name" {
  description = "AI security CloudWatch log group."

  value = aws_cloudwatch_log_group.security.name
}


output "audit_log_group_name" {
  description = "Audit CloudWatch log group."

  value = aws_cloudwatch_log_group.audit.name
}