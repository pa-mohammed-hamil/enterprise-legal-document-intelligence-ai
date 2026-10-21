# ==============================================================================
# Enterprise Legal Document Intelligence AI
# AWS Networking Infrastructure
#
# File:
#   infrastructure/terraform/networking.tf
#
# Architecture:
#
# Internet
#    │
#    ▼
# Internet Gateway
#    │
#    ▼
# Public Subnets
#    │
#    ├── Application Load Balancer
#    └── NAT Gateway(s)
#             │
#             ▼
#       Private App Subnets
#             │
#             ├── Backend ECS
#             ├── Frontend ECS
#             ├── Worker ECS
#             └── Qdrant
#                    │
#                    ▼
#          Private Data Subnets
#                    │
#                    ├── PostgreSQL RDS
#                    └── Redis
#
# Security:
#   - No direct public access to backend/data services
#   - ALB is the primary public entry point
#   - NAT provides controlled outbound connectivity
#   - Dedicated app/data subnet tiers
#   - VPC endpoints reduce public AWS API traffic
#   - Flow Logs provide network audit visibility
# ==============================================================================


# ==============================================================================
# VPC
# ==============================================================================

resource "aws_vpc" "main" {
  cidr_block = var.vpc_cidr

  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = merge(
    local.common_tags,
    {
      Name      = local.resource_names.vpc
      Component = "Networking"
    }
  )
}


# ==============================================================================
# Internet Gateway
# ==============================================================================

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-igw"
    }
  )
}


# ==============================================================================
# Public Subnets
#
# Used for:
#   - Internet-facing ALB
#   - NAT gateways
# ==============================================================================

resource "aws_subnet" "public" {
  count = length(local.availability_zones)

  vpc_id = aws_vpc.main.id

  cidr_block = cidrsubnet(
    var.vpc_cidr,
    4,
    count.index
  )

  availability_zone = local.availability_zones[count.index]

  map_public_ip_on_launch = true

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-public-${count.index + 1}"

      Tier = "Public"

      AvailabilityZone = local.availability_zones[count.index]

      "kubernetes.io/role/elb" = "1"
    }
  )
}


# ==============================================================================
# Private Application Subnets
#
# Used for:
#   - Backend
#   - Frontend
#   - Celery/AI workers
#   - Qdrant
# ==============================================================================

resource "aws_subnet" "private_app" {
  count = length(local.availability_zones)

  vpc_id = aws_vpc.main.id

  cidr_block = cidrsubnet(
    var.vpc_cidr,
    4,
    count.index + 4
  )

  availability_zone = local.availability_zones[count.index]

  map_public_ip_on_launch = false

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-private-app-${count.index + 1}"

      Tier = "Application"

      AvailabilityZone = local.availability_zones[count.index]

      "kubernetes.io/role/internal-elb" = "1"
    }
  )
}


# ==============================================================================
# Private Data Subnets
#
# Used for:
#   - PostgreSQL
#   - Redis
#
# These subnets intentionally have no direct internet route.
# ==============================================================================

resource "aws_subnet" "private_data" {
  count = length(local.availability_zones)

  vpc_id = aws_vpc.main.id

  cidr_block = cidrsubnet(
    var.vpc_cidr,
    4,
    count.index + 8
  )

  availability_zone = local.availability_zones[count.index]

  map_public_ip_on_launch = false

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-private-data-${count.index + 1}"

      Tier = "Data"

      AvailabilityZone = local.availability_zones[count.index]
    }
  )
}


# ==============================================================================
# Elastic IPs for NAT
# ==============================================================================

resource "aws_eip" "nat" {
  count = var.single_nat_gateway ? 1 : length(local.availability_zones)

  domain = "vpc"

  depends_on = [
    aws_internet_gateway.main
  ]

  tags = merge(
    local.common_tags,
    {
      Name = var.single_nat_gateway
        ? "${local.name_prefix}-nat-eip"
        : "${local.name_prefix}-nat-eip-${count.index + 1}"
    }
  )
}


# ==============================================================================
# NAT Gateways
# ==============================================================================

resource "aws_nat_gateway" "main" {
  count = var.single_nat_gateway ? 1 : length(local.availability_zones)

  allocation_id = aws_eip.nat[count.index].id

  subnet_id = var.single_nat_gateway
    ? aws_subnet.public[0].id
    : aws_subnet.public[count.index].id

  depends_on = [
    aws_internet_gateway.main
  ]

  tags = merge(
    local.common_tags,
    {
      Name = var.single_nat_gateway
        ? "${local.name_prefix}-nat"
        : "${local.name_prefix}-nat-${count.index + 1}"
    }
  )
}


# ==============================================================================
# Public Route Table
# ==============================================================================

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"

    gateway_id = aws_internet_gateway.main.id
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-public-rt"
      Tier = "Public"
    }
  )
}


# ==============================================================================
# Public Route Table Associations
# ==============================================================================

resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)

  subnet_id = aws_subnet.public[count.index].id

  route_table_id = aws_route_table.public.id
}


# ==============================================================================
# Private Application Route Tables
# ==============================================================================

resource "aws_route_table" "private_app" {
  count = length(local.availability_zones)

  vpc_id = aws_vpc.main.id

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-private-app-rt-${count.index + 1}"
      Tier = "Application"
    }
  )
}


# ==============================================================================
# Private Application Internet Routes
#
# Application services need outbound connectivity for:
#   - Hugging Face model downloads
#   - package registries
#   - external LLM APIs
#   - email/integration services
# ==============================================================================

resource "aws_route" "private_app_nat" {
  count = length(local.availability_zones)

  route_table_id = aws_route_table.private_app[count.index].id

  destination_cidr_block = "0.0.0.0/0"

  nat_gateway_id = var.single_nat_gateway
    ? aws_nat_gateway.main[0].id
    : aws_nat_gateway.main[count.index].id
}


resource "aws_route_table_association" "private_app" {
  count = length(aws_subnet.private_app)

  subnet_id = aws_subnet.private_app[count.index].id

  route_table_id = aws_route_table.private_app[count.index].id
}


# ==============================================================================
# Private Data Route Table
#
# No internet gateway or NAT default route.
# ==============================================================================

resource "aws_route_table" "private_data" {
  vpc_id = aws_vpc.main.id

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-private-data-rt"
      Tier = "Data"
    }
  )
}


resource "aws_route_table_association" "private_data" {
  count = length(aws_subnet.private_data)

  subnet_id = aws_subnet.private_data[count.index].id

  route_table_id = aws_route_table.private_data.id
}


# ==============================================================================
# ALB Security Group
# ==============================================================================

resource "aws_security_group" "alb" {
  name        = "${local.name_prefix}-alb-sg"
  description = "Security group for public application load balancer."

  vpc_id = aws_vpc.main.id

  ingress {
    description = "HTTP"

    from_port = 80
    to_port   = 80
    protocol  = "tcp"

    cidr_blocks = var.allowed_http_cidr_blocks
  }

  ingress {
    description = "HTTPS"

    from_port = 443
    to_port   = 443
    protocol  = "tcp"

    cidr_blocks = var.allowed_https_cidr_blocks
  }

  egress {
    description = "Traffic to application services"

    from_port = 0
    to_port   = 0
    protocol  = "-1"

    cidr_blocks = [
      var.vpc_cidr
    ]
  }

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-alb-sg"
      Component = "LoadBalancer"
    }
  )
}


# ==============================================================================
# Backend Security Group
# ==============================================================================

resource "aws_security_group" "backend" {
  name        = "${local.name_prefix}-backend-sg"
  description = "Security group for FastAPI backend services."

  vpc_id = aws_vpc.main.id

  ingress {
    description = "Backend traffic from ALB"

    from_port = var.backend_port
    to_port   = var.backend_port
    protocol  = "tcp"

    security_groups = [
      aws_security_group.alb.id
    ]
  }

  egress {
    description = "Backend outbound traffic"

    from_port = 0
    to_port   = 0
    protocol  = "-1"

    cidr_blocks = [
      "0.0.0.0/0"
    ]
  }

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-backend-sg"
      Component = "Backend"
    }
  )
}


# ==============================================================================
# Frontend Security Group
# ==============================================================================

resource "aws_security_group" "frontend" {
  name        = "${local.name_prefix}-frontend-sg"
  description = "Security group for frontend container services."

  vpc_id = aws_vpc.main.id

  ingress {
    description = "Frontend traffic from ALB"

    from_port = var.frontend_port
    to_port   = var.frontend_port
    protocol  = "tcp"

    security_groups = [
      aws_security_group.alb.id
    ]
  }

  egress {
    description = "Frontend outbound traffic"

    from_port = 0
    to_port   = 0
    protocol  = "-1"

    cidr_blocks = [
      "0.0.0.0/0"
    ]
  }

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-frontend-sg"
      Component = "Frontend"
    }
  )
}


# ==============================================================================
# Worker Security Group
# ==============================================================================

resource "aws_security_group" "worker" {
  name        = "${local.name_prefix}-worker-sg"
  description = "Security group for document processing and AI workers."

  vpc_id = aws_vpc.main.id

  egress {
    description = "Worker outbound traffic"

    from_port = 0
    to_port   = 0
    protocol  = "-1"

    cidr_blocks = [
      "0.0.0.0/0"
    ]
  }

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-worker-sg"
      Component = "Worker"
    }
  )
}


# ==============================================================================
# PostgreSQL Security Group
# ==============================================================================

resource "aws_security_group" "postgres" {
  name        = "${local.name_prefix}-postgres-sg"
  description = "Security group for PostgreSQL database."

  vpc_id = aws_vpc.main.id

  ingress {
    description = "PostgreSQL from backend"

    from_port = 5432
    to_port   = 5432
    protocol  = "tcp"

    security_groups = [
      aws_security_group.backend.id
    ]
  }

  ingress {
    description = "PostgreSQL from workers"

    from_port = 5432
    to_port   = 5432
    protocol  = "tcp"

    security_groups = [
      aws_security_group.worker.id
    ]
  }

  egress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"

    cidr_blocks = [
      "0.0.0.0/0"
    ]
  }

  tags = merge(
    local.common_tags,
    {
      Name               = "${local.name_prefix}-postgres-sg"
      Component          = "Database"
      DataClassification = "Confidential"
    }
  )
}


# ==============================================================================
# Redis Security Group
# ==============================================================================

resource "aws_security_group" "redis" {
  name        = "${local.name_prefix}-redis-sg"
  description = "Security group for Redis/ElastiCache."

  vpc_id = aws_vpc.main.id

  ingress {
    description = "Redis from backend"

    from_port = 6379
    to_port   = 6379
    protocol  = "tcp"

    security_groups = [
      aws_security_group.backend.id
    ]
  }

  ingress {
    description = "Redis from workers"

    from_port = 6379
    to_port   = 6379
    protocol  = "tcp"

    security_groups = [
      aws_security_group.worker.id
    ]
  }

  egress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"

    cidr_blocks = [
      "0.0.0.0/0"
    ]
  }

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-redis-sg"
      Component = "Cache"
    }
  )
}


# ==============================================================================
# Qdrant Security Group
# ==============================================================================

resource "aws_security_group" "qdrant" {
  name        = "${local.name_prefix}-qdrant-sg"
  description = "Security group for Qdrant vector database."

  vpc_id = aws_vpc.main.id

  ingress {
    description = "Qdrant HTTP from backend"

    from_port = local.ports.qdrant_http
    to_port   = local.ports.qdrant_http
    protocol  = "tcp"

    security_groups = [
      aws_security_group.backend.id
    ]
  }

  ingress {
    description = "Qdrant HTTP from workers"

    from_port = local.ports.qdrant_http
    to_port   = local.ports.qdrant_http
    protocol  = "tcp"

    security_groups = [
      aws_security_group.worker.id
    ]
  }

  ingress {
    description = "Qdrant gRPC from backend"

    from_port = local.ports.qdrant_grpc
    to_port   = local.ports.qdrant_grpc
    protocol  = "tcp"

    security_groups = [
      aws_security_group.backend.id
    ]
  }

  ingress {
    description = "Qdrant gRPC from workers"

    from_port = local.ports.qdrant_grpc
    to_port   = local.ports.qdrant_grpc
    protocol  = "tcp"

    security_groups = [
      aws_security_group.worker.id
    ]
  }

  egress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"

    cidr_blocks = [
      "0.0.0.0/0"
    ]
  }

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-qdrant-sg"
      Component = "VectorDatabase"
    }
  )
}


# ==============================================================================
# VPC Endpoint Security Group
# ==============================================================================

resource "aws_security_group" "vpc_endpoints" {
  name        = "${local.name_prefix}-vpc-endpoints-sg"
  description = "Security group for private AWS interface endpoints."

  vpc_id = aws_vpc.main.id

  ingress {
    description = "HTTPS from application VPC"

    from_port = 443
    to_port   = 443
    protocol  = "tcp"

    cidr_blocks = [
      var.vpc_cidr
    ]
  }

  egress {
    from_port = 0
    to_port   = 0
    protocol  = "-1"

    cidr_blocks = [
      "0.0.0.0/0"
    ]
  }

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-vpc-endpoints-sg"
      Component = "VPCEndpoints"
    }
  )
}


# ==============================================================================
# S3 Gateway Endpoint
#
# Legal documents and model artifacts can be accessed without traversing NAT.
# ==============================================================================

resource "aws_vpc_endpoint" "s3" {
  count = var.enable_vpc_endpoints ? 1 : 0

  vpc_id = aws_vpc.main.id

  service_name = "com.amazonaws.${local.aws_region}.s3"

  vpc_endpoint_type = "Gateway"

  route_table_ids = concat(
    [
      for route_table in aws_route_table.private_app :
      route_table.id
    ],
    [
      aws_route_table.private_data.id
    ]
  )

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-s3-endpoint"
    }
  )
}


# ==============================================================================
# ECR API Endpoint
# ==============================================================================

resource "aws_vpc_endpoint" "ecr_api" {
  count = var.enable_vpc_endpoints ? 1 : 0

  vpc_id = aws_vpc.main.id

  service_name = "com.amazonaws.${local.aws_region}.ecr.api"

  vpc_endpoint_type = "Interface"

  subnet_ids = aws_subnet.private_app[*].id

  security_group_ids = [
    aws_security_group.vpc_endpoints.id
  ]

  private_dns_enabled = true

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-ecr-api-endpoint"
    }
  )
}


# ==============================================================================
# ECR Docker Endpoint
# ==============================================================================

resource "aws_vpc_endpoint" "ecr_dkr" {
  count = var.enable_vpc_endpoints ? 1 : 0

  vpc_id = aws_vpc.main.id

  service_name = "com.amazonaws.${local.aws_region}.ecr.dkr"

  vpc_endpoint_type = "Interface"

  subnet_ids = aws_subnet.private_app[*].id

  security_group_ids = [
    aws_security_group.vpc_endpoints.id
  ]

  private_dns_enabled = true

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-ecr-dkr-endpoint"
    }
  )
}


# ==============================================================================
# CloudWatch Logs Endpoint
# ==============================================================================

resource "aws_vpc_endpoint" "logs" {
  count = var.enable_vpc_endpoints ? 1 : 0

  vpc_id = aws_vpc.main.id

  service_name = "com.amazonaws.${local.aws_region}.logs"

  vpc_endpoint_type = "Interface"

  subnet_ids = aws_subnet.private_app[*].id

  security_group_ids = [
    aws_security_group.vpc_endpoints.id
  ]

  private_dns_enabled = true

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-logs-endpoint"
    }
  )
}


# ==============================================================================
# Secrets Manager Endpoint
# ==============================================================================

resource "aws_vpc_endpoint" "secretsmanager" {
  count = var.enable_vpc_endpoints ? 1 : 0

  vpc_id = aws_vpc.main.id

  service_name = "com.amazonaws.${local.aws_region}.secretsmanager"

  vpc_endpoint_type = "Interface"

  subnet_ids = aws_subnet.private_app[*].id

  security_group_ids = [
    aws_security_group.vpc_endpoints.id
  ]

  private_dns_enabled = true

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-secretsmanager-endpoint"
    }
  )
}


# ==============================================================================
# Systems Manager Endpoint
# ==============================================================================

resource "aws_vpc_endpoint" "ssm" {
  count = var.enable_vpc_endpoints ? 1 : 0

  vpc_id = aws_vpc.main.id

  service_name = "com.amazonaws.${local.aws_region}.ssm"

  vpc_endpoint_type = "Interface"

  subnet_ids = aws_subnet.private_app[*].id

  security_group_ids = [
    aws_security_group.vpc_endpoints.id
  ]

  private_dns_enabled = true

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-ssm-endpoint"
    }
  )
}


# ==============================================================================
# KMS Endpoint
# ==============================================================================

resource "aws_vpc_endpoint" "kms" {
  count = var.enable_vpc_endpoints ? 1 : 0

  vpc_id = aws_vpc.main.id

  service_name = "com.amazonaws.${local.aws_region}.kms"

  vpc_endpoint_type = "Interface"

  subnet_ids = aws_subnet.private_app[*].id

  security_group_ids = [
    aws_security_group.vpc_endpoints.id
  ]

  private_dns_enabled = true

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-kms-endpoint"
    }
  )
}


# ==============================================================================
# STS Endpoint
# ==============================================================================

resource "aws_vpc_endpoint" "sts" {
  count = var.enable_vpc_endpoints ? 1 : 0

  vpc_id = aws_vpc.main.id

  service_name = "com.amazonaws.${local.aws_region}.sts"

  vpc_endpoint_type = "Interface"

  subnet_ids = aws_subnet.private_app[*].id

  security_group_ids = [
    aws_security_group.vpc_endpoints.id
  ]

  private_dns_enabled = true

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name_prefix}-sts-endpoint"
    }
  )
}


# ==============================================================================
# VPC Flow Logs
# ==============================================================================

resource "aws_cloudwatch_log_group" "vpc_flow_logs" {
  count = var.enable_vpc_flow_logs ? 1 : 0

  name = "${local.resource_names.logs}/network/vpc-flow"

  retention_in_days = var.vpc_flow_log_retention_days

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-vpc-flow-logs"
      Component = "NetworkSecurity"
    }
  )
}


# ==============================================================================
# IAM Role for VPC Flow Logs
# ==============================================================================

resource "aws_iam_role" "vpc_flow_logs" {
  count = var.enable_vpc_flow_logs ? 1 : 0

  name = "${local.name_prefix}-vpc-flow-logs-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Effect = "Allow"

        Principal = {
          Service = "vpc-flow-logs.amazonaws.com"
        }

        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = local.common_tags
}


resource "aws_iam_role_policy" "vpc_flow_logs" {
  count = var.enable_vpc_flow_logs ? 1 : 0

  name = "${local.name_prefix}-vpc-flow-logs-policy"

  role = aws_iam_role.vpc_flow_logs[0].id

  policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Effect = "Allow"

        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]

        Resource = "*"
      }
    ]
  })
}


# ==============================================================================
# VPC Flow Log
# ==============================================================================

resource "aws_flow_log" "main" {
  count = var.enable_vpc_flow_logs ? 1 : 0

  vpc_id = aws_vpc.main.id

  traffic_type = "ALL"

  log_destination_type = "cloud-watch-logs"

  log_destination = aws_cloudwatch_log_group.vpc_flow_logs[0].arn

  iam_role_arn = aws_iam_role.vpc_flow_logs[0].arn

  max_aggregation_interval = 60

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-vpc-flow-log"
      Component = "NetworkSecurity"
    }
  )
}


# ==============================================================================
# Database Subnet Group
#
# database.tf can reference:
#
# aws_db_subnet_group.database.name
# ==============================================================================

resource "aws_db_subnet_group" "database" {
  name = "${local.name_prefix}-database-subnet-group"

  subnet_ids = aws_subnet.private_data[*].id

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-database-subnet-group"
      Component = "Database"
    }
  )
}


# ==============================================================================
# ElastiCache Subnet Group
#
# database.tf can reference:
#
# aws_elasticache_subnet_group.redis.name
# ==============================================================================

resource "aws_elasticache_subnet_group" "redis" {
  name = "${local.name_prefix}-redis-subnet-group"

  subnet_ids = aws_subnet.private_data[*].id

  tags = merge(
    local.common_tags,
    {
      Name      = "${local.name_prefix}-redis-subnet-group"
      Component = "Redis"
    }
  )
}


# ==============================================================================
# Network Locals
# ==============================================================================

locals {
  public_subnet_ids = aws_subnet.public[*].id

  private_app_subnet_ids = aws_subnet.private_app[*].id

  private_data_subnet_ids = aws_subnet.private_data[*].id


  network_security_groups = {
    alb = aws_security_group.alb.id

    backend = aws_security_group.backend.id

    frontend = aws_security_group.frontend.id

    worker = aws_security_group.worker.id

    postgres = aws_security_group.postgres.id

    redis = aws_security_group.redis.id

    qdrant = aws_security_group.qdrant.id

    vpc_endpoints = aws_security_group.vpc_endpoints.id
  }
}


# ==============================================================================
# Network Validation
# ==============================================================================

check "vpc_cidr_size" {
  assert {
    condition = can(cidrsubnet(var.vpc_cidr, 4, 8))

    error_message = "vpc_cidr must provide enough address space for public, application and data subnet tiers."
  }
}


check "production_nat_availability" {
  assert {
    condition = (
      var.environment != "production" ||
      var.single_nat_gateway == false
    )

    error_message = "Production should use one NAT gateway per availability zone instead of single_nat_gateway."
  }
}


# ==============================================================================
# Outputs
# ==============================================================================

output "vpc_id" {
  description = "ID of the application VPC."

  value = aws_vpc.main.id
}


output "vpc_cidr" {
  description = "CIDR range of the application VPC."

  value = aws_vpc.main.cidr_block
}


output "public_subnet_ids" {
  description = "Public subnet IDs used by the internet-facing load balancer."

  value = aws_subnet.public[*].id
}


output "private_app_subnet_ids" {
  description = "Private application subnet IDs."

  value = aws_subnet.private_app[*].id
}


output "private_data_subnet_ids" {
  description = "Private database/cache subnet IDs."

  value = aws_subnet.private_data[*].id
}


output "alb_security_group_id" {
  description = "ALB security group ID."

  value = aws_security_group.alb.id
}


output "backend_security_group_id" {
  description = "Backend ECS security group ID."

  value = aws_security_group.backend.id
}


output "frontend_security_group_id" {
  description = "Frontend ECS security group ID."

  value = aws_security_group.frontend.id
}


output "worker_security_group_id" {
  description = "Worker ECS security group ID."

  value = aws_security_group.worker.id
}


output "postgres_security_group_id" {
  description = "PostgreSQL security group ID."

  value = aws_security_group.postgres.id
}


output "redis_security_group_id" {
  description = "Redis security group ID."

  value = aws_security_group.redis.id
}


output "qdrant_security_group_id" {
  description = "Qdrant security group ID."

  value = aws_security_group.qdrant.id
}


output "database_subnet_group_name" {
  description = "RDS subnet group."

  value = aws_db_subnet_group.database.name
}


output "redis_subnet_group_name" {
  description = "ElastiCache subnet group."

  value = aws_elasticache_subnet_group.redis.name
}


output "nat_gateway_ids" {
  description = "NAT gateway IDs."

  value = aws_nat_gateway.main[*].id
}