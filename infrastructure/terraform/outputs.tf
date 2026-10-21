# ==============================================================================
# Enterprise Legal Document Intelligence AI
# Terraform Outputs
#
# File:
#   infrastructure/terraform/outputs.tf
#
# NOTE:
# Keep this file as the canonical output location.
# If outputs with the same names currently exist in networking.tf or
# monitoring.tf, remove those duplicates before terraform validate.
# ==============================================================================


# ==============================================================================
# Deployment
# ==============================================================================

output "project_name" {
  description = "Project name."
  value       = var.project_name
}

output "environment" {
  description = "Deployment environment."
  value       = var.environment
}

output "aws_region" {
  description = "AWS region used for deployment."
  value       = local.aws_region
}

output "aws_account_id" {
  description = "AWS account ID."
  value       = local.aws_account_id
}

output "deployment_id" {
  description = "Unique deployment suffix."
  value       = local.deployment_suffix
}

output "terraform_workspace" {
  description = "Active Terraform workspace."
  value       = terraform.workspace
}


# ==============================================================================
# Networking
# ==============================================================================

output "vpc_id" {
  description = "Application VPC ID."
  value       = aws_vpc.main.id
}

output "vpc_arn" {
  description = "Application VPC ARN."
  value       = aws_vpc.main.arn
}

output "vpc_cidr" {
  description = "Application VPC CIDR block."
  value       = aws_vpc.main.cidr_block
}

output "availability_zones" {
  description = "Availability zones used by the platform."
  value       = local.availability_zones
}

output "public_subnet_ids" {
  description = "Public subnet IDs."
  value       = aws_subnet.public[*].id
}

output "private_app_subnet_ids" {
  description = "Private application subnet IDs."
  value       = aws_subnet.private_app[*].id
}

output "private_data_subnet_ids" {
  description = "Private data subnet IDs."
  value       = aws_subnet.private_data[*].id
}

output "internet_gateway_id" {
  description = "Internet gateway ID."
  value       = aws_internet_gateway.main.id
}

output "nat_gateway_ids" {
  description = "NAT gateway IDs."
  value       = aws_nat_gateway.main[*].id
}

output "database_subnet_group_name" {
  description = "RDS database subnet group."
  value       = aws_db_subnet_group.database.name
}

output "redis_subnet_group_name" {
  description = "ElastiCache subnet group."
  value       = aws_elasticache_subnet_group.redis.name
}


# ==============================================================================
# Security Groups
# ==============================================================================

output "alb_security_group_id" {
  description = "Application Load Balancer security group."
  value       = aws_security_group.alb.id
}

output "backend_security_group_id" {
  description = "Backend ECS security group."
  value       = aws_security_group.backend.id
}

output "frontend_security_group_id" {
  description = "Frontend ECS security group."
  value       = aws_security_group.frontend.id
}

output "worker_security_group_id" {
  description = "Worker ECS security group."
  value       = aws_security_group.worker.id
}

output "postgres_security_group_id" {
  description = "PostgreSQL security group."
  value       = aws_security_group.postgres.id
}

output "redis_security_group_id" {
  description = "Redis security group."
  value       = aws_security_group.redis.id
}

output "qdrant_security_group_id" {
  description = "Qdrant security group."
  value       = aws_security_group.qdrant.id
}

output "vpc_endpoint_security_group_id" {
  description = "VPC interface endpoint security group."
  value       = aws_security_group.vpc_endpoints.id
}


# ==============================================================================
# Load Balancer
#
# Expected resources from compute.tf:
#
#   aws_lb.main
#   aws_lb_target_group.backend
#   aws_lb_target_group.frontend
# ==============================================================================

output "load_balancer_arn" {
  description = "Application Load Balancer ARN."
  value       = aws_lb.main.arn
}

output "load_balancer_dns_name" {
  description = "Application Load Balancer DNS name."
  value       = aws_lb.main.dns_name
}

output "load_balancer_zone_id" {
  description = "Application Load Balancer hosted zone ID."
  value       = aws_lb.main.zone_id
}

output "load_balancer_url" {
  description = "Base URL for the Application Load Balancer."
  value       = "http://${aws_lb.main.dns_name}"
}

output "backend_target_group_arn" {
  description = "Backend ALB target group ARN."
  value       = aws_lb_target_group.backend.arn
}

output "frontend_target_group_arn" {
  description = "Frontend ALB target group ARN."
  value       = aws_lb_target_group.frontend.arn
}


# ==============================================================================
# ECS
#
# Expected resources from compute.tf:
#
#   aws_ecs_cluster.main
#   aws_ecs_service.backend
#   aws_ecs_service.frontend
#   aws_ecs_service.worker
# ==============================================================================

output "ecs_cluster_id" {
  description = "ECS cluster ID."
  value       = aws_ecs_cluster.main.id
}

output "ecs_cluster_arn" {
  description = "ECS cluster ARN."
  value       = aws_ecs_cluster.main.arn
}

output "ecs_cluster_name" {
  description = "ECS cluster name."
  value       = aws_ecs_cluster.main.name
}

output "backend_service_name" {
  description = "Backend ECS service name."
  value       = aws_ecs_service.backend.name
}

output "frontend_service_name" {
  description = "Frontend ECS service name."
  value       = aws_ecs_service.frontend.name
}

output "worker_service_name" {
  description = "Worker ECS service name."
  value       = aws_ecs_service.worker.name
}


# ==============================================================================
# Container Repositories
#
# Expected resources from compute.tf:
#
#   aws_ecr_repository.backend
#   aws_ecr_repository.frontend
#   aws_ecr_repository.worker
# ==============================================================================

output "backend_ecr_repository_url" {
  description = "Backend ECR repository URL."
  value       = aws_ecr_repository.backend.repository_url
}

output "frontend_ecr_repository_url" {
  description = "Frontend ECR repository URL."
  value       = aws_ecr_repository.frontend.repository_url
}

output "worker_ecr_repository_url" {
  description = "Worker ECR repository URL."
  value       = aws_ecr_repository.worker.repository_url
}


# ==============================================================================
# PostgreSQL
#
# Expected resource from database.tf:
#
#   aws_db_instance.postgres
# ==============================================================================

output "database_identifier" {
  description = "PostgreSQL RDS instance identifier."
  value       = aws_db_instance.postgres.identifier
}

output "database_address" {
  description = "PostgreSQL hostname."
  value       = aws_db_instance.postgres.address
}

output "database_port" {
  description = "PostgreSQL port."
  value       = aws_db_instance.postgres.port
}

output "database_name" {
  description = "Application database name."
  value       = var.database_name
}

output "database_username" {
  description = "PostgreSQL application username."
  value       = var.database_username
  sensitive   = true
}

output "database_endpoint" {
  description = "PostgreSQL endpoint."
  value       = aws_db_instance.postgres.endpoint
  sensitive   = true
}

output "database_connection_summary" {
  description = "Non-password PostgreSQL connection information."

  value = {
    host     = aws_db_instance.postgres.address
    port     = aws_db_instance.postgres.port
    database = var.database_name
  }
}


# ==============================================================================
# Redis
#
# Expected resource:
#
#   aws_elasticache_replication_group.redis
# ==============================================================================

output "redis_replication_group_id" {
  description = "Redis replication group ID."
  value       = aws_elasticache_replication_group.redis.replication_group_id
}

output "redis_primary_endpoint" {
  description = "Redis primary endpoint."
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
}

output "redis_reader_endpoint" {
  description = "Redis reader endpoint."
  value       = aws_elasticache_replication_group.redis.reader_endpoint_address
}

output "redis_port" {
  description = "Redis service port."
  value       = local.ports.redis
}


# ==============================================================================
# Object Storage
#
# Expected resources from storage.tf:
#
#   aws_s3_bucket.documents
#   aws_s3_bucket.models
#   aws_s3_bucket.backups
# ==============================================================================

output "documents_bucket_name" {
  description = "S3 bucket containing legal documents."
  value       = aws_s3_bucket.documents.bucket
}

output "documents_bucket_arn" {
  description = "Legal document bucket ARN."
  value       = aws_s3_bucket.documents.arn
}

output "models_bucket_name" {
  description = "S3 bucket containing AI/ML model artifacts."
  value       = aws_s3_bucket.models.bucket
}

output "models_bucket_arn" {
  description = "Model artifact bucket ARN."
  value       = aws_s3_bucket.models.arn
}

output "backups_bucket_name" {
  description = "S3 backup bucket."
  value       = aws_s3_bucket.backups.bucket
}

output "backups_bucket_arn" {
  description = "Backup bucket ARN."
  value       = aws_s3_bucket.backups.arn
}


# ==============================================================================
# Secrets
#
# Expected resources from storage/database configuration:
#
#   aws_secretsmanager_secret.application
#   aws_secretsmanager_secret.database
#   aws_secretsmanager_secret.redis
#
# Only ARNs are exposed here. Secret values should never be ordinary outputs.
# ==============================================================================

output "application_secret_arn" {
  description = "Secrets Manager ARN containing application secrets."
  value       = aws_secretsmanager_secret.application.arn
}

output "database_secret_arn" {
  description = "Secrets Manager ARN containing database credentials."
  value       = aws_secretsmanager_secret.database.arn
}

output "redis_secret_arn" {
  description = "Secrets Manager ARN containing Redis credentials."
  value       = aws_secretsmanager_secret.redis.arn
}


# ==============================================================================
# Monitoring
# ==============================================================================

output "monitoring_sns_topic_arn" {
  description = "SNS topic used for operational/security alerts."
  value       = aws_sns_topic.alerts.arn
}

output "cloudwatch_dashboard_name" {
  description = "Main CloudWatch observability dashboard."
  value       = aws_cloudwatch_dashboard.main.dashboard_name
}

output "backend_log_group_name" {
  description = "Backend CloudWatch log group."
  value       = aws_cloudwatch_log_group.backend.name
}

output "frontend_log_group_name" {
  description = "Frontend CloudWatch log group."
  value       = aws_cloudwatch_log_group.frontend.name
}

output "worker_log_group_name" {
  description = "Worker CloudWatch log group."
  value       = aws_cloudwatch_log_group.worker.name
}

output "rag_log_group_name" {
  description = "RAG CloudWatch log group."
  value       = aws_cloudwatch_log_group.rag.name
}

output "model_log_group_name" {
  description = "AI model CloudWatch log group."
  value       = aws_cloudwatch_log_group.models.name
}

output "agent_log_group_name" {
  description = "AI agent CloudWatch log group."
  value       = aws_cloudwatch_log_group.agents.name
}

output "security_log_group_name" {
  description = "Security CloudWatch log group."
  value       = aws_cloudwatch_log_group.security.name
}

output "audit_log_group_name" {
  description = "Audit CloudWatch log group."
  value       = aws_cloudwatch_log_group.audit.name
}

output "ingestion_log_group_name" {
  description = "Document ingestion CloudWatch log group."
  value       = aws_cloudwatch_log_group.ingestion.name
}

output "evaluation_log_group_name" {
  description = "AI evaluation CloudWatch log group."
  value       = aws_cloudwatch_log_group.evaluation.name
}


# ==============================================================================
# VPC Endpoints
# ==============================================================================

output "s3_vpc_endpoint_id" {
  description = "S3 gateway endpoint ID."

  value = try(
    aws_vpc_endpoint.s3[0].id,
    null
  )
}

output "ecr_api_vpc_endpoint_id" {
  description = "ECR API endpoint ID."

  value = try(
    aws_vpc_endpoint.ecr_api[0].id,
    null
  )
}

output "ecr_dkr_vpc_endpoint_id" {
  description = "ECR Docker endpoint ID."

  value = try(
    aws_vpc_endpoint.ecr_dkr[0].id,
    null
  )
}

output "cloudwatch_logs_vpc_endpoint_id" {
  description = "CloudWatch Logs endpoint ID."

  value = try(
    aws_vpc_endpoint.logs[0].id,
    null
  )
}

output "secretsmanager_vpc_endpoint_id" {
  description = "Secrets Manager endpoint ID."

  value = try(
    aws_vpc_endpoint.secretsmanager[0].id,
    null
  )
}

output "kms_vpc_endpoint_id" {
  description = "KMS endpoint ID."

  value = try(
    aws_vpc_endpoint.kms[0].id,
    null
  )
}


# ==============================================================================
# Application Endpoints
# ==============================================================================

output "application_endpoints" {
  description = "Primary platform endpoints."

  value = {
    load_balancer = "http://${aws_lb.main.dns_name}"

    frontend = "http://${aws_lb.main.dns_name}/"

    backend = "http://${aws_lb.main.dns_name}/api"

    health = "http://${aws_lb.main.dns_name}/health"

    docs = "http://${aws_lb.main.dns_name}/docs"
  }
}


# ==============================================================================
# AI / RAG Configuration
# ==============================================================================

output "ai_configuration" {
  description = "Non-secret AI/RAG deployment configuration."

  value = {
    embedding_model = var.embedding_model

    reranker_model = var.reranker_model

    document_classifier = var.document_classifier_model

    legal_ner_model = var.legal_ner_model

    vector_database = "qdrant"

    hybrid_search = true

    reranking = true

    permission_aware_search = true

    citation_validation = true

    uncertainty_detection = true
  }
}


# ==============================================================================
# Infrastructure Summary
# ==============================================================================

output "infrastructure_summary" {
  description = "High-level infrastructure deployment summary."

  value = {
    project = var.project_name

    environment = var.environment

    region = local.aws_region

    vpc_id = aws_vpc.main.id

    availability_zones = local.availability_zones

    public_subnets = length(
      aws_subnet.public
    )

    application_subnets = length(
      aws_subnet.private_app
    )

    data_subnets = length(
      aws_subnet.private_data
    )

    ecs_cluster = aws_ecs_cluster.main.name

    load_balancer = aws_lb.main.dns_name

    database = aws_db_instance.postgres.identifier

    redis = aws_elasticache_replication_group.redis.replication_group_id

    documents_bucket = aws_s3_bucket.documents.bucket

    models_bucket = aws_s3_bucket.models.bucket

    monitoring_dashboard = aws_cloudwatch_dashboard.main.dashboard_name
  }
}