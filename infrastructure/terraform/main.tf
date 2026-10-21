

# ==============================================================================
# Enterprise Legal Document Intelligence AI
# Terraform Root Configuration
#
# File:
#   infrastructure/terraform/main.tf
#
# Purpose:
#   Root Terraform configuration for provisioning the cloud infrastructure used
#   by the Enterprise Legal Document Intelligence AI platform.
#
# Resource implementations are separated into:
#   providers.tf
#   networking.tf
#   storage.tf
#   database.tf
#   compute.tf
#   monitoring.tf
#   variables.tf
#   outputs.tf
# ==============================================================================


# ------------------------------------------------------------------------------
# Terraform configuration
# ------------------------------------------------------------------------------

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }

    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }

    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }

    null = {
      source  = "hashicorp/null"
      version = "~> 3.2"
    }
  }
}


# ------------------------------------------------------------------------------
# Local project configuration
# ------------------------------------------------------------------------------

locals {
  project_name = var.project_name
  environment  = var.environment

  name_prefix = "${var.project_name}-${var.environment}"

  application_name = "enterprise-legal-document-intelligence-ai"

  common_tags = merge(
    {
      Project     = var.project_name
      Application = local.application_name
      Environment = var.environment

      ManagedBy = "Terraform"

      Repository = "enterprise-legal-document-intelligence-ai"

      Service = "LegalDocumentIntelligence"

      DataClassification = "Confidential"

      AIPlatform = "true"
    },
    var.additional_tags
  )
}


# ------------------------------------------------------------------------------
# Random identifiers
#
# Useful for globally unique resources such as:
#
#   S3 buckets
#   log buckets
#   backup buckets
# ------------------------------------------------------------------------------

resource "random_id" "deployment" {
  byte_length = 4

  keepers = {
    project     = var.project_name
    environment = var.environment
  }
}


# ------------------------------------------------------------------------------
# Secure application secret
#
# Used when an external secret is not explicitly supplied.
#
# The secret should normally be stored in AWS Secrets Manager by the
# appropriate infrastructure file.
# ------------------------------------------------------------------------------

resource "random_password" "application_secret" {
  length  = 64
  special = true

  min_lower   = 8
  min_upper   = 8
  min_numeric = 8
  min_special = 4

  override_special = "!#$%&*+-=?@^_"
}


# ------------------------------------------------------------------------------
# Database password
#
# Generated automatically when database infrastructure requires a password.
#
# Do NOT expose this value through non-sensitive Terraform outputs.
# ------------------------------------------------------------------------------

resource "random_password" "database_password" {
  length  = 32
  special = true

  min_lower   = 4
  min_upper   = 4
  min_numeric = 4
  min_special = 2

  override_special = "!#$%&*+-=?@^_"
}


# ------------------------------------------------------------------------------
# Redis authentication token
# ------------------------------------------------------------------------------

resource "random_password" "redis_auth_token" {
  length  = 32
  special = false

  min_lower   = 8
  min_upper   = 8
  min_numeric = 8
}


# ------------------------------------------------------------------------------
# Resource naming
# ------------------------------------------------------------------------------

locals {
  deployment_suffix = random_id.deployment.hex

  resource_names = {
    vpc = "${local.name_prefix}-vpc"

    ecs_cluster = "${local.name_prefix}-cluster"

    backend = "${local.name_prefix}-backend"

    frontend = "${local.name_prefix}-frontend"

    worker = "${local.name_prefix}-worker"

    database = "${local.name_prefix}-postgres"

    redis = "${local.name_prefix}-redis"

    qdrant = "${local.name_prefix}-qdrant"

    documents_bucket = "${local.name_prefix}-documents-${local.deployment_suffix}"

    models_bucket = "${local.name_prefix}-models-${local.deployment_suffix}"

    backups_bucket = "${local.name_prefix}-backups-${local.deployment_suffix}"

    alb = "${local.name_prefix}-alb"

    logs = "/${var.project_name}/${var.environment}"
  }
}


# ------------------------------------------------------------------------------
# Availability zones
# ------------------------------------------------------------------------------

data "aws_availability_zones" "available" {
  state = "available"
}


locals {
  availability_zones = slice(
    data.aws_availability_zones.available.names,
    0,
    min(
      var.availability_zone_count,
      length(data.aws_availability_zones.available.names)
    )
  )
}


# ------------------------------------------------------------------------------
# AWS account information
# ------------------------------------------------------------------------------

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

data "aws_partition" "current" {}


# ------------------------------------------------------------------------------
# Current AWS metadata
# ------------------------------------------------------------------------------

locals {
  aws_account_id = data.aws_caller_identity.current.account_id

  aws_region = data.aws_region.current.name

  aws_partition = data.aws_partition.current.partition
}


# ------------------------------------------------------------------------------
# Standard application ports
# ------------------------------------------------------------------------------

locals {
  ports = {
    http     = 80
    https    = 443

    backend  = var.backend_port
    frontend = var.frontend_port

    postgres = 5432
    redis    = 6379

    qdrant_http = 6333
    qdrant_grpc = 6334

    prometheus = 9090
    grafana    = 3000
  }
}


# ------------------------------------------------------------------------------
# Application service definitions
#
# compute.tf can iterate over these definitions to create:
#
#   ECS services
#   task definitions
#   log groups
#   autoscaling policies
# ------------------------------------------------------------------------------

locals {
  application_services = {
    backend = {
      name = local.resource_names.backend

      container_name = "backend"

      container_port = var.backend_port

      cpu = var.backend_cpu

      memory = var.backend_memory

      desired_count = var.backend_desired_count

      health_check_path = "/health"
    }

    frontend = {
      name = local.resource_names.frontend

      container_name = "frontend"

      container_port = var.frontend_port

      cpu = var.frontend_cpu

      memory = var.frontend_memory

      desired_count = var.frontend_desired_count

      health_check_path = "/"
    }

    worker = {
      name = local.resource_names.worker

      container_name = "worker"

      container_port = null

      cpu = var.worker_cpu

      memory = var.worker_memory

      desired_count = var.worker_desired_count

      health_check_path = null
    }
  }
}


# ------------------------------------------------------------------------------
# Legal AI workload configuration
# ------------------------------------------------------------------------------

locals {
  legal_ai_services = {
    rag = {
      enabled = true
    }

    embeddings = {
      enabled = true
    }

    reranker = {
      enabled = true
    }

    legal_llm = {
      enabled = true
    }

    document_classification = {
      enabled = true
    }

    clause_classification = {
      enabled = true
    }

    legal_ner = {
      enabled = true
    }

    risk_classification = {
      enabled = true
    }

    agents = {
      enabled = true
    }
  }
}


# ------------------------------------------------------------------------------
# Data storage paths
# ------------------------------------------------------------------------------

locals {
  storage_prefixes = {
    uploaded_documents = "documents/uploads"

    processed_documents = "documents/processed"

    document_versions = "documents/versions"

    extracted_text = "documents/extracted-text"

    ocr_results = "documents/ocr"

    tables = "documents/tables"

    embeddings = "rag/embeddings"

    indexes = "rag/indexes"

    models = "models"

    model_artifacts = "models/artifacts"

    fine_tuned_models = "models/fine-tuned"

    evaluation = "evaluation"

    audit_exports = "audit"

    backups = "backups"
  }
}


# ------------------------------------------------------------------------------
# Security configuration
# ------------------------------------------------------------------------------

locals {
  security_configuration = {
    encryption_at_rest = true

    encryption_in_transit = true

    private_database = true

    private_redis = true

    tenant_isolation = true

    audit_logging = true

    pii_protection = true

    prompt_injection_protection = true
  }
}


# ------------------------------------------------------------------------------
# Observability configuration
# ------------------------------------------------------------------------------

locals {
  observability = {
    cloudwatch_enabled = true

    prometheus_enabled = var.enable_prometheus

    grafana_enabled = var.enable_grafana

    application_metrics = true

    rag_metrics = true

    model_metrics = true

    security_metrics = true

    infrastructure_metrics = true
  }
}


# ------------------------------------------------------------------------------
# RAG observability labels
# ------------------------------------------------------------------------------

locals {
  rag_metric_dimensions = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "RAG"
  }

  model_metric_dimensions = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "Models"
  }

  security_metric_dimensions = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "Security"
  }
}


# ------------------------------------------------------------------------------
# Backup configuration
# ------------------------------------------------------------------------------

locals {
  backup_configuration = {
    enabled = var.enable_backups

    retention_days = var.backup_retention_days

    database_backup_enabled = true

    document_backup_enabled = true

    model_backup_enabled = true
  }
}


# ------------------------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------------------------

locals {
  application_environment = {
    APP_NAME = local.application_name

    APP_ENV = var.environment

    AWS_REGION = local.aws_region

    LOG_LEVEL = var.log_level

    BACKEND_PORT = tostring(var.backend_port)

    FRONTEND_PORT = tostring(var.frontend_port)

    ENABLE_RAG = "true"

    ENABLE_AGENTS = "true"

    ENABLE_MODEL_EVALUATION = "true"

    ENABLE_AUDIT_LOGGING = "true"

    ENABLE_TENANT_ISOLATION = "true"

    ENABLE_PROMPT_INJECTION_PROTECTION = "true"

    ENABLE_PII_PROTECTION = "true"
  }
}


# ------------------------------------------------------------------------------
# AI / RAG environment configuration
# ------------------------------------------------------------------------------

locals {
  ai_environment = {
    EMBEDDING_MODEL = var.embedding_model

    RERANKER_MODEL = var.reranker_model

    DOCUMENT_CLASSIFIER_MODEL = var.document_classifier_model

    LEGAL_NER_MODEL = var.legal_ner_model

    VECTOR_DATABASE = "qdrant"

    HYBRID_SEARCH_ENABLED = "true"

    BM25_ENABLED = "true"

    VECTOR_SEARCH_ENABLED = "true"

    RERANKING_ENABLED = "true"

    PERMISSION_AWARE_SEARCH = "true"

    CITATION_GENERATION_ENABLED = "true"

    ANSWER_VALIDATION_ENABLED = "true"

    UNCERTAINTY_DETECTION_ENABLED = "true"
  }
}


# ------------------------------------------------------------------------------
# Database configuration
# ------------------------------------------------------------------------------

locals {
  database_configuration = {
    engine = "postgres"

    port = local.ports.postgres

    database_name = var.database_name

    username = var.database_username

    ssl_enabled = true
  }
}


# ------------------------------------------------------------------------------
# Vector database configuration
# ------------------------------------------------------------------------------

locals {
  vector_database_configuration = {
    engine = "qdrant"

    http_port = local.ports.qdrant_http

    grpc_port = local.ports.qdrant_grpc

    collection_prefix = "${var.project_name}-${var.environment}"

    tenant_isolation = true
  }
}


# ------------------------------------------------------------------------------
# Deployment metadata
# ------------------------------------------------------------------------------

locals {
  deployment_metadata = {
    project = var.project_name

    environment = var.environment

    region = local.aws_region

    account_id = local.aws_account_id

    deployment_id = local.deployment_suffix

    terraform_workspace = terraform.workspace
  }
}


# ------------------------------------------------------------------------------
# Validation checks
# ------------------------------------------------------------------------------

check "minimum_availability_zones" {
  assert {
    condition = length(
      data.aws_availability_zones.available.names
    ) >= 2

    error_message = "At least two AWS availability zones are required."
  }
}


check "production_multi_az" {
  assert {
    condition = (
      var.environment != "production" ||
      var.availability_zone_count >= 2
    )

    error_message = "Production deployments must use at least two availability zones."
  }
}


check "production_database_backup" {
  assert {
    condition = (
      var.environment != "production" ||
      var.enable_backups
    )

    error_message = "Backups must be enabled for production deployments."
  }
}


check "valid_backend_port" {
  assert {
    condition = (
      var.backend_port > 0 &&
      var.backend_port <= 65535
    )

    error_message = "backend_port must be between 1 and 65535."
  }
}


check "valid_frontend_port" {
  assert {
    condition = (
      var.frontend_port > 0 &&
      var.frontend_port <= 65535
    )

    error_message = "frontend_port must be between 1 and 65535."
  }
}


# ------------------------------------------------------------------------------
# Infrastructure relationship
#
# Terraform automatically builds the dependency graph from resource references.
#
# Expected infrastructure flow:
#
#        AWS
#         │
#         ▼
#        VPC
#         │
#     ┌───┴───────────────┐
#     │                   │
#     ▼                   ▼
# Public Subnets      Private Subnets
#     │                   │
#     ▼                   ├───────────────┐
#    ALB                  ECS             │
#     │                   │               │
#     ▼                   │               ▼
# Frontend ─────────► Backend            RDS
#                         │               │
#                         ├──────────────►Redis
#                         │
#                         ├──────────────►Qdrant
#                         │
#                         ├──────────────►S3
#                         │
#                         └──────────────►AI Models
#
#
# Legal AI processing:
#
# Document
#    │
#    ▼
# S3
#    │
#    ▼
# Worker
#    │
#    ├── OCR
#    ├── Parser
#    ├── Legal NER
#    ├── Clause Classifier
#    ├── Risk Classifier
#    └── Chunking
#          │
#          ▼
#      Embeddings
#          │
#          ▼
#       Qdrant
#          │
#          ▼
# Hybrid Retrieval
#          │
#          ▼
#      Reranker
#          │
#          ▼
#      Legal LLM
#          │
#          ▼
# Citations + Validation
#
# ------------------------------------------------------------------------------