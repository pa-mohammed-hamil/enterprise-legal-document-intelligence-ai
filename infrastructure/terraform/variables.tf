# ==============================================================================
# Enterprise Legal Document Intelligence AI
# Terraform Variables
#
# File:
#   infrastructure/terraform/variables.tf
#
# Used by:
#   main.tf
#   providers.tf
#   networking.tf
#   database.tf
#   storage.tf
#   compute.tf
#   monitoring.tf
#   outputs.tf
# ==============================================================================


# ==============================================================================
# Project
# ==============================================================================

variable "project_name" {
  description = "Short project identifier used in AWS resource names."
  type        = string
  default     = "legal-intelligence"

  validation {
    condition = (
      length(var.project_name) >= 3 &&
      length(var.project_name) <= 32 &&
      can(regex("^[a-z0-9-]+$", var.project_name))
    )

    error_message = "project_name must contain 3-32 lowercase letters, numbers, or hyphens."
  }
}


variable "environment" {
  description = "Deployment environment."
  type        = string
  default     = "development"

  validation {
    condition = contains(
      [
        "development",
        "staging",
        "production"
      ],
      var.environment
    )

    error_message = "environment must be development, staging, or production."
  }
}


variable "additional_tags" {
  description = "Additional tags applied to infrastructure resources."
  type        = map(string)
  default     = {}
}


# ==============================================================================
# AWS
# ==============================================================================

variable "aws_region" {
  description = "Primary AWS deployment region."
  type        = string
  default     = "ap-south-1"

  validation {
    condition     = length(trimspace(var.aws_region)) > 0
    error_message = "aws_region cannot be empty."
  }
}


variable "secondary_aws_region" {
  description = "Secondary AWS region for disaster recovery and replication."
  type        = string
  default     = "ap-southeast-1"

  validation {
    condition     = length(trimspace(var.secondary_aws_region)) > 0
    error_message = "secondary_aws_region cannot be empty."
  }
}


variable "enable_cross_region_replication" {
  description = "Enable cross-region storage replication."
  type        = bool
  default     = false
}


variable "allowed_aws_account_ids" {
  description = "Optional AWS account allow-list. Empty allows the authenticated account."
  type        = list(string)
  default     = []

  validation {
    condition = alltrue([
      for account_id in var.allowed_aws_account_ids :
      can(regex("^[0-9]{12}$", account_id))
    ])

    error_message = "Each AWS account ID must contain exactly 12 digits."
  }
}


variable "production_allowed_regions" {
  description = "AWS regions allowed for production deployment. Empty disables this restriction."
  type        = list(string)

  default = [
    "ap-south-1"
  ]
}


# ==============================================================================
# Availability
# ==============================================================================

variable "availability_zone_count" {
  description = "Number of availability zones used by the deployment."
  type        = number
  default     = 2

  validation {
    condition = (
      var.availability_zone_count >= 2 &&
      var.availability_zone_count <= 3
    )

    error_message = "availability_zone_count must be between 2 and 3."
  }
}


# ==============================================================================
# Networking
# ==============================================================================

variable "vpc_cidr" {
  description = "CIDR range allocated to the application VPC."
  type        = string
  default     = "10.20.0.0/16"

  validation {
    condition = can(
      cidrhost(var.vpc_cidr, 0)
    )

    error_message = "vpc_cidr must be a valid IPv4 CIDR block."
  }
}


variable "single_nat_gateway" {
  description = "Use one NAT gateway instead of one per AZ. Recommended only for non-production."
  type        = bool
  default     = true
}


variable "allowed_http_cidr_blocks" {
  description = "CIDR ranges allowed to access the HTTP ALB listener."
  type        = list(string)

  default = [
    "0.0.0.0/0"
  ]

  validation {
    condition = alltrue([
      for cidr in var.allowed_http_cidr_blocks :
      can(cidrhost(cidr, 0))
    ])

    error_message = "Every allowed HTTP entry must be a valid CIDR."
  }
}


variable "allowed_https_cidr_blocks" {
  description = "CIDR ranges allowed to access the HTTPS ALB listener."
  type        = list(string)

  default = [
    "0.0.0.0/0"
  ]

  validation {
    condition = alltrue([
      for cidr in var.allowed_https_cidr_blocks :
      can(cidrhost(cidr, 0))
    ])

    error_message = "Every allowed HTTPS entry must be a valid CIDR."
  }
}


variable "enable_vpc_endpoints" {
  description = "Create private VPC endpoints for supported AWS services."
  type        = bool
  default     = true
}


variable "enable_vpc_flow_logs" {
  description = "Enable VPC Flow Logs."
  type        = bool
  default     = true
}


variable "vpc_flow_log_retention_days" {
  description = "CloudWatch retention period for VPC Flow Logs."
  type        = number
  default     = 30

  validation {
    condition = contains(
      [
        1, 3, 5, 7, 14, 30, 60, 90, 120,
        150, 180, 365, 400, 545, 731,
        1096, 1827, 2192, 2557, 2922,
        3288, 3653
      ],
      var.vpc_flow_log_retention_days
    )

    error_message = "vpc_flow_log_retention_days must be a supported CloudWatch retention value."
  }
}


# ==============================================================================
# Application Ports
# ==============================================================================

variable "backend_port" {
  description = "FastAPI backend container port."
  type        = number
  default     = 8000

  validation {
    condition = (
      var.backend_port >= 1 &&
      var.backend_port <= 65535
    )

    error_message = "backend_port must be between 1 and 65535."
  }
}


variable "frontend_port" {
  description = "Frontend container port."
  type        = number
  default     = 80

  validation {
    condition = (
      var.frontend_port >= 1 &&
      var.frontend_port <= 65535
    )

    error_message = "frontend_port must be between 1 and 65535."
  }
}


# ==============================================================================
# ECS Backend
# ==============================================================================

variable "backend_cpu" {
  description = "CPU units allocated to the backend ECS task."
  type        = number
  default     = 1024
}


variable "backend_memory" {
  description = "Memory in MiB allocated to the backend ECS task."
  type        = number
  default     = 2048
}


variable "backend_desired_count" {
  description = "Desired number of backend ECS tasks."
  type        = number
  default     = 2

  validation {
    condition     = var.backend_desired_count >= 1
    error_message = "backend_desired_count must be at least 1."
  }
}


variable "backend_min_capacity" {
  description = "Minimum backend ECS autoscaling capacity."
  type        = number
  default     = 2
}


variable "backend_max_capacity" {
  description = "Maximum backend ECS autoscaling capacity."
  type        = number
  default     = 10
}


# ==============================================================================
# ECS Frontend
# ==============================================================================

variable "frontend_cpu" {
  description = "CPU units allocated to the frontend ECS task."
  type        = number
  default     = 256
}


variable "frontend_memory" {
  description = "Memory in MiB allocated to the frontend ECS task."
  type        = number
  default     = 512
}


variable "frontend_desired_count" {
  description = "Desired number of frontend ECS tasks."
  type        = number
  default     = 2

  validation {
    condition     = var.frontend_desired_count >= 1
    error_message = "frontend_desired_count must be at least 1."
  }
}


variable "frontend_min_capacity" {
  description = "Minimum frontend ECS autoscaling capacity."
  type        = number
  default     = 2
}


variable "frontend_max_capacity" {
  description = "Maximum frontend ECS autoscaling capacity."
  type        = number
  default     = 6
}


# ==============================================================================
# ECS Worker
# ==============================================================================

variable "worker_cpu" {
  description = "CPU units allocated to document/AI worker tasks."
  type        = number
  default     = 2048
}


variable "worker_memory" {
  description = "Memory in MiB allocated to document/AI worker tasks."
  type        = number
  default     = 4096
}


variable "worker_desired_count" {
  description = "Desired number of worker ECS tasks."
  type        = number
  default     = 2

  validation {
    condition     = var.worker_desired_count >= 1
    error_message = "worker_desired_count must be at least 1."
  }
}


variable "worker_min_capacity" {
  description = "Minimum worker autoscaling capacity."
  type        = number
  default     = 1
}


variable "worker_max_capacity" {
  description = "Maximum worker autoscaling capacity."
  type        = number
  default     = 10
}


# ==============================================================================
# Container Images
# ==============================================================================

variable "backend_image_tag" {
  description = "Backend container image tag."
  type        = string
  default     = "latest"
}


variable "frontend_image_tag" {
  description = "Frontend container image tag."
  type        = string
  default     = "latest"
}


variable "worker_image_tag" {
  description = "Worker container image tag."
  type        = string
  default     = "latest"
}


# ==============================================================================
# Application Load Balancer
# ==============================================================================

variable "enable_https" {
  description = "Enable the HTTPS ALB listener."
  type        = bool
  default     = false
}


variable "certificate_arn" {
  description = "ACM certificate ARN used by the HTTPS listener."
  type        = string
  default     = null
  nullable    = true
}


variable "alb_idle_timeout" {
  description = "ALB idle timeout in seconds."
  type        = number
  default     = 60
}


variable "alb_deletion_protection" {
  description = "Enable ALB deletion protection."
  type        = bool
  default     = false
}


# ==============================================================================
# ECS Autoscaling
# ==============================================================================

variable "ecs_target_cpu_utilization" {
  description = "Target ECS CPU utilization percentage."
  type        = number
  default     = 65

  validation {
    condition = (
      var.ecs_target_cpu_utilization > 0 &&
      var.ecs_target_cpu_utilization <= 100
    )

    error_message = "ecs_target_cpu_utilization must be between 1 and 100."
  }
}


variable "ecs_target_memory_utilization" {
  description = "Target ECS memory utilization percentage."
  type        = number
  default     = 70

  validation {
    condition = (
      var.ecs_target_memory_utilization > 0 &&
      var.ecs_target_memory_utilization <= 100
    )

    error_message = "ecs_target_memory_utilization must be between 1 and 100."
  }
}


# ==============================================================================
# PostgreSQL
# ==============================================================================

variable "database_name" {
  description = "PostgreSQL application database name."
  type        = string
  default     = "legal_intelligence"

  validation {
    condition = can(
      regex("^[A-Za-z][A-Za-z0-9_]*$", var.database_name)
    )

    error_message = "database_name must begin with a letter and contain only letters, numbers, and underscores."
  }
}


variable "database_username" {
  description = "PostgreSQL master/application username."
  type        = string
  default     = "legal_admin"

  sensitive = true

  validation {
    condition = (
      length(var.database_username) >= 1 &&
      length(var.database_username) <= 63 &&
      can(regex("^[A-Za-z][A-Za-z0-9_]*$", var.database_username))
    )

    error_message = "database_username must be a valid PostgreSQL-style identifier."
  }
}


variable "database_engine_version" {
  description = "PostgreSQL major/minor engine version."
  type        = string
  default     = "16"
}


variable "database_instance_class" {
  description = "RDS PostgreSQL instance class."
  type        = string
  default     = "db.t4g.medium"
}


variable "database_allocated_storage" {
  description = "Initial RDS storage allocation in GiB."
  type        = number
  default     = 50

  validation {
    condition     = var.database_allocated_storage >= 20
    error_message = "database_allocated_storage must be at least 20 GiB."
  }
}


variable "database_max_allocated_storage" {
  description = "Maximum RDS autoscaled storage in GiB."
  type        = number
  default     = 200
}


variable "database_storage_type" {
  description = "RDS storage type."
  type        = string
  default     = "gp3"
}


variable "database_multi_az" {
  description = "Enable RDS Multi-AZ."
  type        = bool
  default     = false
}


variable "database_backup_retention_days" {
  description = "RDS automated backup retention."
  type        = number
  default     = 7

  validation {
    condition = (
      var.database_backup_retention_days >= 0 &&
      var.database_backup_retention_days <= 35
    )

    error_message = "database_backup_retention_days must be between 0 and 35."
  }
}


variable "database_deletion_protection" {
  description = "Enable RDS deletion protection."
  type        = bool
  default     = false
}


variable "database_skip_final_snapshot" {
  description = "Skip final snapshot when destroying RDS."
  type        = bool
  default     = true
}


variable "database_performance_insights" {
  description = "Enable RDS Performance Insights."
  type        = bool
  default     = true
}


# ==============================================================================
# Redis
# ==============================================================================

variable "redis_node_type" {
  description = "ElastiCache Redis node type."
  type        = string
  default     = "cache.t4g.small"
}


variable "redis_engine_version" {
  description = "Redis/Valkey-compatible engine version expected by database.tf."
  type        = string
  default     = "7.1"
}


variable "redis_num_cache_clusters" {
  description = "Number of cache nodes in the Redis replication group."
  type        = number
  default     = 2

  validation {
    condition     = var.redis_num_cache_clusters >= 1
    error_message = "redis_num_cache_clusters must be at least 1."
  }
}


variable "redis_automatic_failover" {
  description = "Enable automatic failover for Redis."
  type        = bool
  default     = true
}


variable "redis_multi_az" {
  description = "Enable Multi-AZ Redis deployment."
  type        = bool
  default     = true
}


variable "redis_snapshot_retention_limit" {
  description = "Number of days Redis snapshots are retained."
  type        = number
  default     = 7
}


# ==============================================================================
# Qdrant
# ==============================================================================

variable "qdrant_image" {
  description = "Qdrant container image."
  type        = string
  default     = "qdrant/qdrant:v1.12.1"
}


variable "qdrant_cpu" {
  description = "CPU units allocated to Qdrant."
  type        = number
  default     = 1024
}


variable "qdrant_memory" {
  description = "Memory in MiB allocated to Qdrant."
  type        = number
  default     = 2048
}


variable "qdrant_desired_count" {
  description = "Desired number of Qdrant tasks."
  type        = number
  default     = 1
}


variable "qdrant_storage_size" {
  description = "Persistent Qdrant storage size in GiB."
  type        = number
  default     = 50
}


# ==============================================================================
# Storage / KMS
# ==============================================================================

variable "kms_deletion_window_days" {
  description = "KMS key deletion waiting period."
  type        = number
  default     = 30

  validation {
    condition = (
      var.kms_deletion_window_days >= 7 &&
      var.kms_deletion_window_days <= 30
    )

    error_message = "kms_deletion_window_days must be between 7 and 30."
  }
}


variable "secret_recovery_window_days" {
  description = "Secrets Manager recovery window."
  type        = number
  default     = 30

  validation {
    condition = (
      var.secret_recovery_window_days >= 7 &&
      var.secret_recovery_window_days <= 30
    )

    error_message = "secret_recovery_window_days must be between 7 and 30."
  }
}


variable "enable_s3_access_logging" {
  description = "Enable S3 server access logging."
  type        = bool
  default     = true
}


variable "s3_access_log_retention_days" {
  description = "Retention period for S3 server access logs."
  type        = number
  default     = 90
}


# ==============================================================================
# Document Storage Lifecycle
# ==============================================================================

variable "document_standard_ia_after_days" {
  description = "Days before legal documents move to STANDARD_IA."
  type        = number
  default     = 30
}


variable "document_glacier_after_days" {
  description = "Days before legal documents move to GLACIER_IR."
  type        = number
  default     = 90
}


variable "document_noncurrent_ia_after_days" {
  description = "Days before previous document versions move to STANDARD_IA."
  type        = number
  default     = 30
}


variable "document_noncurrent_glacier_after_days" {
  description = "Days before previous document versions move to GLACIER_IR."
  type        = number
  default     = 90
}


variable "document_noncurrent_expiration_days" {
  description = "Days before old legal-document versions expire."
  type        = number
  default     = 365
}


# ==============================================================================
# Model Storage Lifecycle
# ==============================================================================

variable "model_noncurrent_ia_after_days" {
  description = "Days before previous model versions move to STANDARD_IA."
  type        = number
  default     = 30
}


variable "model_noncurrent_glacier_after_days" {
  description = "Days before previous model versions move to GLACIER_IR."
  type        = number
  default     = 90
}


variable "model_noncurrent_expiration_days" {
  description = "Days before previous model versions expire."
  type        = number
  default     = 365
}


# ==============================================================================
# Backup Storage
# ==============================================================================

variable "enable_backups" {
  description = "Enable application backup functionality."
  type        = bool
  default     = true
}


variable "backup_retention_days" {
  description = "Backup retention period."
  type        = number
  default     = 365

  validation {
    condition     = var.backup_retention_days > 0
    error_message = "backup_retention_days must be greater than zero."
  }
}


variable "backup_standard_ia_after_days" {
  description = "Days before backup objects transition to STANDARD_IA."
  type        = number
  default     = 30
}


variable "backup_glacier_after_days" {
  description = "Days before backup objects transition to Glacier."
  type        = number
  default     = 90
}


variable "backup_noncurrent_expiration_days" {
  description = "Days before old backup object versions expire."
  type        = number
  default     = 365
}


# ==============================================================================
# AI / ML Models
# ==============================================================================

variable "embedding_model" {
  description = "Default Hugging Face embedding model."
  type        = string
  default     = "BAAI/bge-base-en-v1.5"
}


variable "reranker_model" {
  description = "Default Hugging Face reranker model."
  type        = string
  default     = "BAAI/bge-reranker-base"
}


variable "document_classifier_model" {
  description = "Document classification model identifier."
  type        = string
  default     = "legal-document-classifier"
}


variable "legal_ner_model" {
  description = "Legal named-entity recognition model identifier."
  type        = string
  default     = "legal-ner"
}


# ==============================================================================
# RAG
# ==============================================================================

variable "rag_top_k" {
  description = "Initial number of RAG retrieval candidates."
  type        = number
  default     = 20
}


variable "rag_rerank_top_k" {
  description = "Number of RAG chunks retained after reranking."
  type        = number
  default     = 8
}


variable "rag_similarity_threshold" {
  description = "Minimum vector similarity threshold."
  type        = number
  default     = 0.60

  validation {
    condition = (
      var.rag_similarity_threshold >= 0 &&
      var.rag_similarity_threshold <= 1
    )

    error_message = "rag_similarity_threshold must be between 0 and 1."
  }
}


# ==============================================================================
# Application
# ==============================================================================

variable "log_level" {
  description = "Application logging level."
  type        = string
  default     = "INFO"

  validation {
    condition = contains(
      [
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL"
      ],
      upper(var.log_level)
    )

    error_message = "log_level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL."
  }
}


# ==============================================================================
# Monitoring
# ==============================================================================

variable "enable_prometheus" {
  description = "Enable Prometheus-related monitoring integration."
  type        = bool
  default     = true
}


variable "enable_grafana" {
  description = "Enable Grafana-related monitoring integration."
  type        = bool
  default     = true
}


variable "alert_email" {
  description = "Email address receiving infrastructure/security alerts."
  type        = string
  default     = null
  nullable    = true

  validation {
    condition = (
      var.alert_email == null ||
      can(
        regex(
          "^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$",
          var.alert_email
        )
      )
    )

    error_message = "alert_email must be null or a valid email address."
  }
}


variable "log_retention_days" {
  description = "CloudWatch application log retention."
  type        = number
  default     = 30
}


variable "security_log_retention_days" {
  description = "CloudWatch security log retention."
  type        = number
  default     = 90
}


variable "audit_log_retention_days" {
  description = "CloudWatch audit log retention."
  type        = number
  default     = 365
}


# ==============================================================================
# Application Monitoring Thresholds
# ==============================================================================

variable "backend_error_threshold" {
  description = "Backend errors per alarm evaluation period."
  type        = number
  default     = 10
}


variable "rag_failure_threshold" {
  description = "RAG retrieval failures required to trigger an alarm."
  type        = number
  default     = 10
}


variable "hallucination_threshold" {
  description = "Detected hallucination events required to trigger an alarm."
  type        = number
  default     = 5
}


variable "citation_failure_threshold" {
  description = "Citation validation failures required to trigger an alarm."
  type        = number
  default     = 10
}


variable "model_failure_threshold" {
  description = "Model inference failures required to trigger an alarm."
  type        = number
  default     = 10
}


variable "prompt_injection_threshold" {
  description = "Prompt injection detections required to trigger an alarm."
  type        = number
  default     = 5
}


# ==============================================================================
# ALB Monitoring
# ==============================================================================

variable "alb_5xx_threshold" {
  description = "ALB 5xx count required to trigger an alarm."
  type        = number
  default     = 10
}


variable "backend_latency_threshold" {
  description = "Maximum acceptable backend p95 response time in seconds."
  type        = number
  default     = 2
}


# ==============================================================================
# ECS Monitoring
# ==============================================================================

variable "ecs_cpu_alarm_threshold" {
  description = "ECS CPU utilization alarm percentage."
  type        = number
  default     = 80
}


variable "ecs_memory_alarm_threshold" {
  description = "ECS memory utilization alarm percentage."
  type        = number
  default     = 80
}


# ==============================================================================
# Database Monitoring
# ==============================================================================

variable "database_cpu_alarm_threshold" {
  description = "RDS CPU utilization alarm threshold."
  type        = number
  default     = 80
}


variable "database_connection_alarm_threshold" {
  description = "RDS connection count alarm threshold."
  type        = number
  default     = 100
}


variable "database_free_storage_threshold_bytes" {
  description = "RDS low free-storage alarm threshold in bytes."
  type        = number

  # 5 GiB
  default = 5368709120
}


# ==============================================================================
# Redis Monitoring
# ==============================================================================

variable "redis_cpu_alarm_threshold" {
  description = "Redis engine CPU alarm threshold."
  type        = number
  default     = 80
}


variable "redis_memory_alarm_threshold" {
  description = "Redis memory utilization alarm threshold."
  type        = number
  default     = 80
}


# ==============================================================================
# Cost Monitoring
# ==============================================================================

variable "enable_cost_anomaly_detection" {
  description = "Enable AWS Cost Anomaly Detection."
  type        = bool
  default     = false
}


variable "cost_anomaly_threshold" {
  description = "Absolute USD cost anomaly threshold."
  type        = number
  default     = 20

  validation {
    condition     = var.cost_anomaly_threshold > 0
    error_message = "cost_anomaly_threshold must be greater than zero."
  }
}


# ==============================================================================
# ECR
# ==============================================================================

variable "ecr_image_retention_count" {
  description = "Number of tagged container images retained in ECR."
  type        = number
  default     = 20
}


variable "ecr_untagged_expiration_days" {
  description = "Days before untagged ECR images are deleted."
  type        = number
  default     = 7
}


# ==============================================================================
# Health Checks
# ==============================================================================

variable "backend_health_check_path" {
  description = "Backend ALB health-check path."
  type        = string
  default     = "/health"
}


variable "frontend_health_check_path" {
  description = "Frontend ALB health-check path."
  type        = string
  default     = "/"
}


variable "health_check_interval" {
  description = "ALB health-check interval in seconds."
  type        = number
  default     = 30
}


variable "health_check_timeout" {
  description = "ALB health-check timeout in seconds."
  type        = number
  default     = 5
}


variable "healthy_threshold" {
  description = "Successful health checks required before a target is healthy."
  type        = number
  default     = 2
}


variable "unhealthy_threshold" {
  description = "Failed health checks required before a target is unhealthy."
  type        = number
  default     = 3
}


# ==============================================================================
# ECS Deployment
# ==============================================================================

variable "deployment_minimum_healthy_percent" {
  description = "Minimum healthy ECS task percentage during deployments."
  type        = number
  default     = 100
}


variable "deployment_maximum_percent" {
  description = "Maximum ECS task percentage during deployments."
  type        = number
  default     = 200
}


variable "enable_execute_command" {
  description = "Enable ECS Exec for controlled operational troubleshooting."
  type        = bool
  default     = false
}


# ==============================================================================
# Security
# ==============================================================================

variable "enable_container_insights" {
  description = "Enable ECS Container Insights."
  type        = bool
  default     = true
}


variable "enable_ecr_image_scanning" {
  description = "Enable ECR image scanning on push."
  type        = bool
  default     = true
}


# ==============================================================================
# Validation Guards
# ==============================================================================

check "backend_capacity_range" {
  assert {
    condition = (
      var.backend_min_capacity >= 1 &&
      var.backend_max_capacity >= var.backend_min_capacity &&
      var.backend_desired_count >= var.backend_min_capacity &&
      var.backend_desired_count <= var.backend_max_capacity
    )

    error_message = "Backend desired/min/max capacity values are inconsistent."
  }
}


check "frontend_capacity_range" {
  assert {
    condition = (
      var.frontend_min_capacity >= 1 &&
      var.frontend_max_capacity >= var.frontend_min_capacity &&
      var.frontend_desired_count >= var.frontend_min_capacity &&
      var.frontend_desired_count <= var.frontend_max_capacity
    )

    error_message = "Frontend desired/min/max capacity values are inconsistent."
  }
}


check "worker_capacity_range" {
  assert {
    condition = (
      var.worker_min_capacity >= 1 &&
      var.worker_max_capacity >= var.worker_min_capacity &&
      var.worker_desired_count >= var.worker_min_capacity &&
      var.worker_desired_count <= var.worker_max_capacity
    )

    error_message = "Worker desired/min/max capacity values are inconsistent."
  }
}


check "database_storage_range" {
  assert {
    condition = (
      var.database_max_allocated_storage >=
      var.database_allocated_storage
    )

    error_message = "database_max_allocated_storage must be greater than or equal to database_allocated_storage."
  }
}


check "https_certificate" {
  assert {
    condition = (
      var.enable_https == false ||
      (
        var.certificate_arn != null &&
        length(trimspace(var.certificate_arn)) > 0
      )
    )

    error_message = "certificate_arn is required when enable_https=true."
  }
}


check "rag_result_limits" {
  assert {
    condition = (
      var.rag_top_k >= 1 &&
      var.rag_rerank_top_k >= 1 &&
      var.rag_rerank_top_k <= var.rag_top_k
    )

    error_message = "rag_rerank_top_k must be between 1 and rag_top_k."
  }
}


check "production_database_protection" {
  assert {
    condition = (
      var.environment != "production" ||
      (
        var.database_multi_az == true &&
        var.database_deletion_protection == true &&
        var.database_skip_final_snapshot == false
      )
    )

    error_message = "Production requires Multi-AZ RDS, deletion protection, and a final snapshot."
  }
}


check "production_load_balancer_protection" {
  assert {
    condition = (
      var.environment != "production" ||
      var.alb_deletion_protection == true
    )

    error_message = "Production requires ALB deletion protection."
  }
}