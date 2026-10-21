# ==============================================================================
# Enterprise Legal Document Intelligence AI
# Terraform Provider Configuration
#
# File:
#   infrastructure/terraform/providers.tf
#
# Providers:
#   - AWS
#   - Random
#   - TLS
#   - Null
#
# IMPORTANT:
# Provider version constraints are defined in main.tf under:
# terraform.required_providers
# ==============================================================================


# ==============================================================================
# AWS Provider
# ==============================================================================

provider "aws" {
  region = var.aws_region

  # ---------------------------------------------------------------------------
  # Default tags
  #
  # Automatically applied to supported AWS resources.
  # ---------------------------------------------------------------------------

  default_tags {
    tags = {
      Project     = var.project_name
      Application = "enterprise-legal-document-intelligence-ai"
      Environment = var.environment
      ManagedBy   = "Terraform"

      Service = "LegalDocumentIntelligence"

      DataClassification = "Confidential"

      AIPlatform = "true"
    }
  }
}


# ==============================================================================
# AWS Provider - Secondary Region
#
# Optional provider alias for:
#   - disaster recovery
#   - cross-region backups
#   - replicated model artifacts
#   - legal-document backup replication
#   - multi-region deployments
#
# Resources can use:
#
# provider = aws.secondary
# ==============================================================================

provider "aws" {
  alias = "secondary"

  region = var.secondary_aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Application = "enterprise-legal-document-intelligence-ai"
      Environment = var.environment
      ManagedBy   = "Terraform"

      Service = "LegalDocumentIntelligence"

      RegionRole = "Secondary"

      DataClassification = "Confidential"

      AIPlatform = "true"
    }
  }
}


# ==============================================================================
# Random Provider
#
# Used by:
#   random_id.deployment
#   random_password.application_secret
#   random_password.database_password
#   random_password.redis_auth_token
# ==============================================================================

provider "random" {}


# ==============================================================================
# TLS Provider
#
# Available for:
#   - generated certificates
#   - private keys
#   - internal service certificates
#   - development PKI
#
# Production public certificates should normally use AWS ACM.
# ==============================================================================

provider "tls" {}


# ==============================================================================
# Null Provider
#
# Reserved for exceptional orchestration/integration tasks.
#
# Prefer native Terraform resources instead of local-exec/remote-exec wherever
# possible.
# ==============================================================================

provider "null" {}


# ==============================================================================
# AWS Region Validation
# ==============================================================================

data "aws_regions" "available" {
  all_regions = false
}


check "primary_region_available" {
  assert {
    condition = contains(
      data.aws_regions.available.names,
      var.aws_region
    )

    error_message = "The configured aws_region is not available for this AWS account."
  }
}


# ==============================================================================
# Account Validation
#
# Optionally restrict deployment to explicitly approved AWS accounts.
# ==============================================================================

check "approved_aws_account" {
  assert {
    condition = (
      length(var.allowed_aws_account_ids) == 0 ||
      contains(
        var.allowed_aws_account_ids,
        data.aws_caller_identity.current.account_id
      )
    )

    error_message = "Terraform is running against an AWS account that is not present in allowed_aws_account_ids."
  }
}


# ==============================================================================
# Production Region Guard
#
# Production can optionally be restricted to approved regions.
# ==============================================================================

check "production_region" {
  assert {
    condition = (
      var.environment != "production" ||
      length(var.production_allowed_regions) == 0 ||
      contains(
        var.production_allowed_regions,
        var.aws_region
      )
    )

    error_message = "Production deployment attempted in a region that is not included in production_allowed_regions."
  }
}


# ==============================================================================
# Primary / Secondary Region Validation
# ==============================================================================

check "different_secondary_region" {
  assert {
    condition = (
      var.enable_cross_region_replication == false ||
      var.secondary_aws_region != var.aws_region
    )

    error_message = "secondary_aws_region must differ from aws_region when cross-region replication is enabled."
  }
}