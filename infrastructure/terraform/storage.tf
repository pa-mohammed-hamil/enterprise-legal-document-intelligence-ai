# ==============================================================================
# Enterprise Legal Document Intelligence AI
# Storage Infrastructure
#
# File:
#   infrastructure/terraform/storage.tf
#
# Provides:
#   - S3 legal-document storage
#   - S3 AI/ML model artifact storage
#   - S3 backup storage
#   - S3 encryption/versioning/lifecycle/public-access protection
#   - KMS customer-managed encryption key
#   - Secrets Manager application/database/Redis secrets
#   - Optional access logging
# ==============================================================================


# ==============================================================================
# KMS
# ==============================================================================

resource "aws_kms_key" "storage" {
  description = "KMS key for ${local.application_name} encrypted data."

  enable_key_rotation = true

  deletion_window_in_days = var.kms_deletion_window_days

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-storage-kms"
      Component = "Security"
    }
  )
}


resource "aws_kms_alias" "storage" {
  name = "alias/${local.name_prefix}-storage"

  target_key_id = aws_kms_key.storage.key_id
}


# ==============================================================================
# S3 Access Logs Bucket
# ==============================================================================

resource "aws_s3_bucket" "access_logs" {
  count = var.enable_s3_access_logging ? 1 : 0

  bucket = "${local.name_prefix}-access-logs-${local.deployment_suffix}"

  force_destroy = var.environment != "production"

  tags = merge(
    local.common_tags,
    {
      Name               = "${local.name_prefix}-access-logs"
      Component          = "Storage"
      DataClassification = "SecuritySensitive"
    }
  )
}


resource "aws_s3_bucket_public_access_block" "access_logs" {
  count = var.enable_s3_access_logging ? 1 : 0

  bucket = aws_s3_bucket.access_logs[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}


resource "aws_s3_bucket_ownership_controls" "access_logs" {
  count = var.enable_s3_access_logging ? 1 : 0

  bucket = aws_s3_bucket.access_logs[0].id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}


resource "aws_s3_bucket_versioning" "access_logs" {
  count = var.enable_s3_access_logging ? 1 : 0

  bucket = aws_s3_bucket.access_logs[0].id

  versioning_configuration {
    status = "Enabled"
  }
}


resource "aws_s3_bucket_server_side_encryption_configuration" "access_logs" {
  count = var.enable_s3_access_logging ? 1 : 0

  bucket = aws_s3_bucket.access_logs[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}


resource "aws_s3_bucket_lifecycle_configuration" "access_logs" {
  count = var.enable_s3_access_logging ? 1 : 0

  bucket = aws_s3_bucket.access_logs[0].id

  rule {
    id     = "expire-access-logs"
    status = "Enabled"

    filter {}

    expiration {
      days = var.s3_access_log_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = var.s3_access_log_retention_days
    }
  }

  depends_on = [
    aws_s3_bucket_versioning.access_logs
  ]
}


# ==============================================================================
# Legal Documents Bucket
#
# Stores:
#   documents/uploads
#   documents/processed
#   documents/versions
#   documents/extracted-text
#   documents/ocr
#   documents/tables
#   rag/embeddings
#   rag/indexes
#   audit exports
# ==============================================================================

resource "aws_s3_bucket" "documents" {
  bucket = local.resource_names.documents_bucket

  force_destroy = var.environment != "production"

  tags = merge(
    local.common_tags,
    {
      Name               = local.resource_names.documents_bucket
      Component          = "LegalDocuments"
      DataClassification = "Confidential"
    }
  )
}


resource "aws_s3_bucket_ownership_controls" "documents" {
  bucket = aws_s3_bucket.documents.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}


resource "aws_s3_bucket_public_access_block" "documents" {
  bucket = aws_s3_bucket.documents.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}


resource "aws_s3_bucket_versioning" "documents" {
  bucket = aws_s3_bucket.documents.id

  versioning_configuration {
    status = "Enabled"
  }
}


resource "aws_s3_bucket_server_side_encryption_configuration" "documents" {
  bucket = aws_s3_bucket.documents.id

  rule {
    bucket_key_enabled = true

    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.storage.arn
      sse_algorithm     = "aws:kms"
    }
  }
}


resource "aws_s3_bucket_logging" "documents" {
  count = var.enable_s3_access_logging ? 1 : 0

  bucket = aws_s3_bucket.documents.id

  target_bucket = aws_s3_bucket.access_logs[0].id
  target_prefix = "documents/"
}


# ==============================================================================
# Documents Lifecycle
# ==============================================================================

resource "aws_s3_bucket_lifecycle_configuration" "documents" {
  bucket = aws_s3_bucket.documents.id

  rule {
    id     = "document-storage-lifecycle"
    status = "Enabled"

    filter {}

    transition {
      days          = var.document_standard_ia_after_days
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = var.document_glacier_after_days
      storage_class = "GLACIER_IR"
    }

    noncurrent_version_transition {
      noncurrent_days = var.document_noncurrent_ia_after_days
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_transition {
      noncurrent_days = var.document_noncurrent_glacier_after_days
      storage_class   = "GLACIER_IR"
    }

    noncurrent_version_expiration {
      noncurrent_days = var.document_noncurrent_expiration_days
    }
  }

  rule {
    id     = "abort-incomplete-multipart-uploads"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  depends_on = [
    aws_s3_bucket_versioning.documents
  ]
}


# ==============================================================================
# Documents HTTPS-only Bucket Policy
# ==============================================================================

data "aws_iam_policy_document" "documents" {
  statement {
    sid    = "DenyInsecureTransport"
    effect = "Deny"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    actions = [
      "s3:*"
    ]

    resources = [
      aws_s3_bucket.documents.arn,
      "${aws_s3_bucket.documents.arn}/*"
    ]

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}


resource "aws_s3_bucket_policy" "documents" {
  bucket = aws_s3_bucket.documents.id
  policy = data.aws_iam_policy_document.documents.json

  depends_on = [
    aws_s3_bucket_public_access_block.documents
  ]
}


# ==============================================================================
# AI/ML Models Bucket
#
# Stores:
#   base model metadata
#   fine-tuned models
#   LoRA adapters
#   classifier artifacts
#   embeddings artifacts
#   evaluation outputs
# ==============================================================================

resource "aws_s3_bucket" "models" {
  bucket = local.resource_names.models_bucket

  force_destroy = var.environment != "production"

  tags = merge(
    local.common_tags,
    {
      Name               = local.resource_names.models_bucket
      Component          = "AIModels"
      DataClassification = "Internal"
    }
  )
}


resource "aws_s3_bucket_ownership_controls" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}


resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}


resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id

  versioning_configuration {
    status = "Enabled"
  }
}


resource "aws_s3_bucket_server_side_encryption_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    bucket_key_enabled = true

    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.storage.arn
      sse_algorithm     = "aws:kms"
    }
  }
}


resource "aws_s3_bucket_logging" "models" {
  count = var.enable_s3_access_logging ? 1 : 0

  bucket = aws_s3_bucket.models.id

  target_bucket = aws_s3_bucket.access_logs[0].id
  target_prefix = "models/"
}


resource "aws_s3_bucket_lifecycle_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    id     = "model-artifact-lifecycle"
    status = "Enabled"

    filter {}

    noncurrent_version_transition {
      noncurrent_days = var.model_noncurrent_ia_after_days
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_transition {
      noncurrent_days = var.model_noncurrent_glacier_after_days
      storage_class   = "GLACIER_IR"
    }

    noncurrent_version_expiration {
      noncurrent_days = var.model_noncurrent_expiration_days
    }
  }

  rule {
    id     = "abort-incomplete-model-uploads"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  depends_on = [
    aws_s3_bucket_versioning.models
  ]
}


data "aws_iam_policy_document" "models" {
  statement {
    sid    = "DenyInsecureTransport"
    effect = "Deny"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    actions = [
      "s3:*"
    ]

    resources = [
      aws_s3_bucket.models.arn,
      "${aws_s3_bucket.models.arn}/*"
    ]

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}


resource "aws_s3_bucket_policy" "models" {
  bucket = aws_s3_bucket.models.id
  policy = data.aws_iam_policy_document.models.json

  depends_on = [
    aws_s3_bucket_public_access_block.models
  ]
}


# ==============================================================================
# Backup Bucket
# ==============================================================================

resource "aws_s3_bucket" "backups" {
  bucket = local.resource_names.backups_bucket

  # Backups should not be accidentally deleted.
  force_destroy = false

  tags = merge(
    local.common_tags,
    {
      Name               = local.resource_names.backups_bucket
      Component          = "Backup"
      DataClassification = "Confidential"
    }
  )
}


resource "aws_s3_bucket_ownership_controls" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}


resource "aws_s3_bucket_public_access_block" "backups" {
  bucket = aws_s3_bucket.backups.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}


resource "aws_s3_bucket_versioning" "backups" {
  bucket = aws_s3_bucket.backups.id

  versioning_configuration {
    status = "Enabled"
  }
}


resource "aws_s3_bucket_server_side_encryption_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    bucket_key_enabled = true

    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.storage.arn
      sse_algorithm     = "aws:kms"
    }
  }
}


resource "aws_s3_bucket_logging" "backups" {
  count = var.enable_s3_access_logging ? 1 : 0

  bucket = aws_s3_bucket.backups.id

  target_bucket = aws_s3_bucket.access_logs[0].id
  target_prefix = "backups/"
}


# ==============================================================================
# Backup Lifecycle
# ==============================================================================

resource "aws_s3_bucket_lifecycle_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    id     = "backup-retention"
    status = "Enabled"

    filter {}

    transition {
      days          = var.backup_standard_ia_after_days
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = var.backup_glacier_after_days
      storage_class = "GLACIER"
    }

    expiration {
      days = var.backup_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = var.backup_noncurrent_expiration_days
    }
  }

  rule {
    id     = "abort-incomplete-backup-uploads"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  depends_on = [
    aws_s3_bucket_versioning.backups
  ]
}


data "aws_iam_policy_document" "backups" {
  statement {
    sid    = "DenyInsecureTransport"
    effect = "Deny"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    actions = [
      "s3:*"
    ]

    resources = [
      aws_s3_bucket.backups.arn,
      "${aws_s3_bucket.backups.arn}/*"
    ]

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}


resource "aws_s3_bucket_policy" "backups" {
  bucket = aws_s3_bucket.backups.id
  policy = data.aws_iam_policy_document.backups.json

  depends_on = [
    aws_s3_bucket_public_access_block.backups
  ]
}


# ==============================================================================
# Application Secrets
# ==============================================================================

resource "aws_secretsmanager_secret" "application" {
  name = "${local.name_prefix}/application"

  description = "Application secrets for ${local.application_name}."

  kms_key_id = aws_kms_key.storage.arn

  recovery_window_in_days = var.secret_recovery_window_days

  tags = merge(
    local.common_tags,
    {
      Name               = "${local.name_prefix}-application-secret"
      Component          = "Secrets"
      DataClassification = "Secret"
    }
  )
}


resource "aws_secretsmanager_secret_version" "application" {
  secret_id = aws_secretsmanager_secret.application.id

  secret_string = jsonencode({
    application_secret = random_password.application_secret.result

    environment = var.environment
  })
}


# ==============================================================================
# Database Secret
#
# database.tf should use the same generated password:
#
# random_password.database_password.result
# ==============================================================================

resource "aws_secretsmanager_secret" "database" {
  name = "${local.name_prefix}/database"

  description = "PostgreSQL credentials for the legal intelligence platform."

  kms_key_id = aws_kms_key.storage.arn

  recovery_window_in_days = var.secret_recovery_window_days

  tags = merge(
    local.common_tags,
    {
      Name               = "${local.name_prefix}-database-secret"
      Component          = "Database"
      DataClassification = "Secret"
    }
  )
}


resource "aws_secretsmanager_secret_version" "database" {
  secret_id = aws_secretsmanager_secret.database.id

  secret_string = jsonencode({
    username = var.database_username

    password = random_password.database_password.result

    database = var.database_name

    port = 5432
  })
}


# ==============================================================================
# Redis Secret
# ==============================================================================

resource "aws_secretsmanager_secret" "redis" {
  name = "${local.name_prefix}/redis"

  description = "Redis authentication credentials."

  kms_key_id = aws_kms_key.storage.arn

  recovery_window_in_days = var.secret_recovery_window_days

  tags = merge(
    local.common_tags,
    {
      Name               = "${local.name_prefix}-redis-secret"
      Component          = "Redis"
      DataClassification = "Secret"
    }
  )
}


resource "aws_secretsmanager_secret_version" "redis" {
  secret_id = aws_secretsmanager_secret.redis.id

  secret_string = jsonencode({
    auth_token = random_password.redis_auth_token.result

    port = 6379

    tls = true
  })
}


# ==============================================================================
# Optional S3 Cross-Region Replication
#
# The secondary bucket is only created when replication is enabled.
# ==============================================================================

resource "aws_s3_bucket" "documents_replica" {
  count = var.enable_cross_region_replication ? 1 : 0

  provider = aws.secondary

  bucket = "${local.name_prefix}-documents-replica-${local.deployment_suffix}"

  force_destroy = false

  tags = merge(
    local.common_tags,
    {
      Name               = "${local.name_prefix}-documents-replica"
      Component          = "DisasterRecovery"
      RegionRole         = "Secondary"
      DataClassification = "Confidential"
    }
  )
}


resource "aws_s3_bucket_public_access_block" "documents_replica" {
  count = var.enable_cross_region_replication ? 1 : 0

  provider = aws.secondary

  bucket = aws_s3_bucket.documents_replica[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}


resource "aws_s3_bucket_ownership_controls" "documents_replica" {
  count = var.enable_cross_region_replication ? 1 : 0

  provider = aws.secondary

  bucket = aws_s3_bucket.documents_replica[0].id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}


resource "aws_s3_bucket_versioning" "documents_replica" {
  count = var.enable_cross_region_replication ? 1 : 0

  provider = aws.secondary

  bucket = aws_s3_bucket.documents_replica[0].id

  versioning_configuration {
    status = "Enabled"
  }
}


resource "aws_s3_bucket_server_side_encryption_configuration" "documents_replica" {
  count = var.enable_cross_region_replication ? 1 : 0

  provider = aws.secondary

  bucket = aws_s3_bucket.documents_replica[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}


# ==============================================================================
# IAM Role for Cross-Region Replication
# ==============================================================================

resource "aws_iam_role" "s3_replication" {
  count = var.enable_cross_region_replication ? 1 : 0

  name = "${local.name_prefix}-s3-replication-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Effect = "Allow"

        Principal = {
          Service = "s3.amazonaws.com"
        }

        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = local.common_tags
}


data "aws_iam_policy_document" "s3_replication" {
  count = var.enable_cross_region_replication ? 1 : 0

  statement {
    sid = "ReadSourceBucket"

    actions = [
      "s3:GetReplicationConfiguration",
      "s3:ListBucket"
    ]

    resources = [
      aws_s3_bucket.documents.arn
    ]
  }

  statement {
    sid = "ReadSourceObjects"

    actions = [
      "s3:GetObjectVersionForReplication",
      "s3:GetObjectVersionAcl",
      "s3:GetObjectVersionTagging"
    ]

    resources = [
      "${aws_s3_bucket.documents.arn}/*"
    ]
  }

  statement {
    sid = "ReplicateObjects"

    actions = [
      "s3:ReplicateObject",
      "s3:ReplicateDelete",
      "s3:ReplicateTags"
    ]

    resources = [
      "${aws_s3_bucket.documents_replica[0].arn}/*"
    ]
  }

  statement {
    sid = "DecryptSourceObjects"

    actions = [
      "kms:Decrypt"
    ]

    resources = [
      aws_kms_key.storage.arn
    ]
  }
}


resource "aws_iam_role_policy" "s3_replication" {
  count = var.enable_cross_region_replication ? 1 : 0

  name = "${local.name_prefix}-s3-replication-policy"

  role = aws_iam_role.s3_replication[0].id

  policy = data.aws_iam_policy_document.s3_replication[0].json
}


# ==============================================================================
# Replication Configuration
# ==============================================================================

resource "aws_s3_bucket_replication_configuration" "documents" {
  count = var.enable_cross_region_replication ? 1 : 0

  bucket = aws_s3_bucket.documents.id

  role = aws_iam_role.s3_replication[0].arn

  rule {
    id     = "legal-document-cross-region-replication"
    status = "Enabled"

    filter {}

    delete_marker_replication {
      status = "Enabled"
    }

    source_selection_criteria {
      sse_kms_encrypted_objects {
        status = "Enabled"
      }
    }

    destination {
      bucket = aws_s3_bucket.documents_replica[0].arn

      storage_class = "STANDARD_IA"
    }
  }

  depends_on = [
    aws_s3_bucket_versioning.documents,
    aws_s3_bucket_versioning.documents_replica,
    aws_iam_role_policy.s3_replication
  ]
}


# ==============================================================================
# Storage Locals
# ==============================================================================

locals {
  storage_buckets = {
    documents = aws_s3_bucket.documents.bucket
    models    = aws_s3_bucket.models.bucket
    backups   = aws_s3_bucket.backups.bucket
  }

  storage_bucket_arns = {
    documents = aws_s3_bucket.documents.arn
    models    = aws_s3_bucket.models.arn
    backups   = aws_s3_bucket.backups.arn
  }

  storage_secrets = {
    application = aws_secretsmanager_secret.application.arn
    database    = aws_secretsmanager_secret.database.arn
    redis       = aws_secretsmanager_secret.redis.arn
  }
}


# ==============================================================================
# Validation
# ==============================================================================

check "production_storage_protection" {
  assert {
    condition = (
      var.environment != "production" ||
      (
        aws_s3_bucket.documents.force_destroy == false &&
        aws_s3_bucket.models.force_destroy == false &&
        aws_s3_bucket.backups.force_destroy == false
      )
    )

    error_message = "Production S3 buckets must not permit force destruction."
  }
}


check "document_lifecycle_order" {
  assert {
    condition = (
      var.document_glacier_after_days >
      var.document_standard_ia_after_days
    )

    error_message = "document_glacier_after_days must be greater than document_standard_ia_after_days."
  }
}


check "backup_lifecycle_order" {
  assert {
    condition = (
      var.backup_glacier_after_days >
      var.backup_standard_ia_after_days
    )

    error_message = "backup_glacier_after_days must be greater than backup_standard_ia_after_days."
  }
}