# deployment/terraform/modules/compute/compute.tf

###############################################################################
# DATA SOURCES
###############################################################################

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

###############################################################################
# LOCALS
###############################################################################

locals {
  name = "${var.project_name}-${var.environment}"

  common_tags = merge(
    {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "Terraform"
      Service     = "compute"
    },
    var.tags
  )
}

###############################################################################
# EKS CLUSTER IAM ROLE
###############################################################################

resource "aws_iam_role" "eks_cluster" {
  name = "${local.name}-eks-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Effect = "Allow"

        Principal = {
          Service = "eks.amazonaws.com"
        }

        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  role = aws_iam_role.eks_cluster.name

  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

###############################################################################
# EKS NODE IAM ROLE
###############################################################################

resource "aws_iam_role" "eks_nodes" {
  name = "${local.name}-eks-node-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Effect = "Allow"

        Principal = {
          Service = "ec2.amazonaws.com"
        }

        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "worker_node" {
  role = aws_iam_role.eks_nodes.name

  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "container_registry" {
  role = aws_iam_role.eks_nodes.name

  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy_attachment" "cni" {
  role = aws_iam_role.eks_nodes.name

  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

###############################################################################
# CLUSTER SECURITY GROUP
###############################################################################

resource "aws_security_group" "eks_cluster" {
  name        = "${local.name}-eks-cluster-sg"
  description = "Security group for ${local.name} EKS control plane"
  vpc_id      = var.vpc_id

  egress {
    description = "Allow outbound traffic"

    from_port = 0
    to_port   = 0
    protocol  = "-1"

    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name}-eks-cluster-sg"
    }
  )
}

###############################################################################
# NODE SECURITY GROUP
###############################################################################

resource "aws_security_group" "eks_nodes" {
  name        = "${local.name}-eks-node-sg"
  description = "Security group for ${local.name} EKS worker nodes"
  vpc_id      = var.vpc_id

  egress {
    description = "Allow outbound traffic"

    from_port = 0
    to_port   = 0
    protocol  = "-1"

    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name}-eks-node-sg"
    }
  )
}

###############################################################################
# NODE-TO-NODE TRAFFIC
###############################################################################

resource "aws_security_group_rule" "nodes_internal" {
  description = "Allow node-to-node communication"

  type = "ingress"

  from_port = 0
  to_port   = 65535
  protocol  = "-1"

  security_group_id = aws_security_group.eks_nodes.id

  source_security_group_id = aws_security_group.eks_nodes.id
}

###############################################################################
# CONTROL PLANE -> NODES
###############################################################################

resource "aws_security_group_rule" "cluster_to_nodes" {
  description = "Allow EKS control plane to communicate with worker nodes"

  type = "ingress"

  from_port = 1025
  to_port   = 65535
  protocol  = "tcp"

  security_group_id = aws_security_group.eks_nodes.id

  source_security_group_id = aws_security_group.eks_cluster.id
}

###############################################################################
# NODES -> CONTROL PLANE
###############################################################################

resource "aws_security_group_rule" "nodes_to_cluster_https" {
  description = "Allow worker nodes to communicate with Kubernetes API"

  type = "ingress"

  from_port = 443
  to_port   = 443
  protocol  = "tcp"

  security_group_id = aws_security_group.eks_cluster.id

  source_security_group_id = aws_security_group.eks_nodes.id
}

###############################################################################
# EKS CLUSTER
###############################################################################

resource "aws_eks_cluster" "main" {
  name = "${local.name}-eks"

  role_arn = aws_iam_role.eks_cluster.arn

  version = var.kubernetes_version

  enabled_cluster_log_types = [
    "api",
    "audit",
    "authenticator",
    "controllerManager",
    "scheduler"
  ]

  vpc_config {
    subnet_ids = var.private_subnet_ids

    security_group_ids = [
      aws_security_group.eks_cluster.id
    ]

    endpoint_private_access = true

    endpoint_public_access = var.eks_public_endpoint

    public_access_cidrs = (
      var.eks_public_endpoint
      ? var.eks_public_access_cidrs
      : []
    )
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy
  ]

  tags = local.common_tags
}

###############################################################################
# CLOUDWATCH LOG GROUP
###############################################################################

resource "aws_cloudwatch_log_group" "eks" {
  name = "/aws/eks/${local.name}-eks/cluster"

  retention_in_days = var.log_retention_days

  tags = local.common_tags
}

###############################################################################
# CPU NODE GROUP
###############################################################################

resource "aws_eks_node_group" "cpu" {
  cluster_name = aws_eks_cluster.main.name

  node_group_name = "${local.name}-cpu"

  node_role_arn = aws_iam_role.eks_nodes.arn

  subnet_ids = var.private_subnet_ids

  instance_types = var.cpu_instance_types

  capacity_type = var.cpu_capacity_type

  disk_size = var.cpu_disk_size

  scaling_config {
    desired_size = var.cpu_desired_size
    min_size     = var.cpu_min_size
    max_size     = var.cpu_max_size
  }

  update_config {
    max_unavailable_percentage = 25
  }

  labels = {
    workload = "general"
    compute  = "cpu"
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name}-cpu-node"
    }
  )

  depends_on = [
    aws_iam_role_policy_attachment.worker_node,
    aws_iam_role_policy_attachment.container_registry,
    aws_iam_role_policy_attachment.cni
  ]
}

###############################################################################
# GPU NODE GROUP
###############################################################################

resource "aws_eks_node_group" "gpu" {
  count = var.enable_gpu_nodes ? 1 : 0

  cluster_name = aws_eks_cluster.main.name

  node_group_name = "${local.name}-gpu"

  node_role_arn = aws_iam_role.eks_nodes.arn

  subnet_ids = var.private_subnet_ids

  instance_types = var.gpu_instance_types

  capacity_type = var.gpu_capacity_type

  disk_size = var.gpu_disk_size

  scaling_config {
    desired_size = var.gpu_desired_size
    min_size     = var.gpu_min_size
    max_size     = var.gpu_max_size
  }

  update_config {
    max_unavailable = 1
  }

  labels = {
    workload   = "ai"
    compute    = "gpu"
    accelerator = "nvidia"
  }

  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name}-gpu-node"
    }
  )

  depends_on = [
    aws_iam_role_policy_attachment.worker_node,
    aws_iam_role_policy_attachment.container_registry,
    aws_iam_role_policy_attachment.cni
  ]
}

###############################################################################
# OPTIONAL SPOT NODE GROUP
###############################################################################

resource "aws_eks_node_group" "spot" {
  count = var.enable_spot_nodes ? 1 : 0

  cluster_name = aws_eks_cluster.main.name

  node_group_name = "${local.name}-spot"

  node_role_arn = aws_iam_role.eks_nodes.arn

  subnet_ids = var.private_subnet_ids

  instance_types = var.spot_instance_types

  capacity_type = "SPOT"

  disk_size = var.spot_disk_size

  scaling_config {
    desired_size = var.spot_desired_size
    min_size     = var.spot_min_size
    max_size     = var.spot_max_size
  }

  update_config {
    max_unavailable_percentage = 25
  }

  labels = {
    workload = "batch"
    compute  = "spot"
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${local.name}-spot-node"
    }
  )

  depends_on = [
    aws_iam_role_policy_attachment.worker_node,
    aws_iam_role_policy_attachment.container_registry,
    aws_iam_role_policy_attachment.cni
  ]
}

###############################################################################
# OIDC PROVIDER
###############################################################################

data "tls_certificate" "eks" {
  url = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

resource "aws_iam_openid_connect_provider" "eks" {
  client_id_list = [
    "sts.amazonaws.com"
  ]

  thumbprint_list = [
    data.tls_certificate.eks.certificates[0].sha1_fingerprint
  ]

  url = aws_eks_cluster.main.identity[0].oidc[0].issuer

  tags = local.common_tags
}

###############################################################################
# EKS ADDON - VPC CNI
###############################################################################

resource "aws_eks_addon" "vpc_cni" {
  cluster_name = aws_eks_cluster.main.name

  addon_name = "vpc-cni"

  resolve_conflicts_on_update = "PRESERVE"

  tags = local.common_tags

  depends_on = [
    aws_eks_node_group.cpu
  ]
}

###############################################################################
# EKS ADDON - COREDNS
###############################################################################

resource "aws_eks_addon" "coredns" {
  cluster_name = aws_eks_cluster.main.name

  addon_name = "coredns"

  resolve_conflicts_on_update = "PRESERVE"

  tags = local.common_tags

  depends_on = [
    aws_eks_node_group.cpu
  ]
}

###############################################################################
# EKS ADDON - KUBE PROXY
###############################################################################

resource "aws_eks_addon" "kube_proxy" {
  cluster_name = aws_eks_cluster.main.name

  addon_name = "kube-proxy"

  resolve_conflicts_on_update = "PRESERVE"

  tags = local.common_tags

  depends_on = [
    aws_eks_node_group.cpu
  ]
}

###############################################################################
# EKS ADDON - EBS CSI
###############################################################################

resource "aws_eks_addon" "ebs_csi" {
  count = var.enable_ebs_csi ? 1 : 0

  cluster_name = aws_eks_cluster.main.name

  addon_name = "aws-ebs-csi-driver"

  resolve_conflicts_on_update = "PRESERVE"

  tags = local.common_tags
}

###############################################################################
# OUTPUTS
###############################################################################

output "cluster_id" {
  description = "EKS cluster ID"
  value       = aws_eks_cluster.main.id
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.main.name
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = aws_eks_cluster.main.arn
}

output "cluster_endpoint" {
  description = "Kubernetes API endpoint"
  value       = aws_eks_cluster.main.endpoint
}

output "cluster_version" {
  description = "Kubernetes version"
  value       = aws_eks_cluster.main.version
}

output "cluster_certificate_authority_data" {
  description = "EKS CA certificate"
  value       = aws_eks_cluster.main.certificate_authority[0].data
  sensitive   = true
}

output "cluster_security_group_id" {
  value = aws_security_group.eks_cluster.id
}

output "node_security_group_id" {
  value = aws_security_group.eks_nodes.id
}

output "node_role_arn" {
  value = aws_iam_role.eks_nodes.arn
}

output "cluster_role_arn" {
  value = aws_iam_role.eks_cluster.arn
}

output "oidc_provider_arn" {
  value = aws_iam_openid_connect_provider.eks.arn
}

output "oidc_issuer_url" {
  value = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

output "cpu_node_group_name" {
  value = aws_eks_node_group.cpu.node_group_name
}

output "gpu_node_group_name" {
  value = (
    var.enable_gpu_nodes
    ? aws_eks_node_group.gpu[0].node_group_name
    : null
  )
}

output "spot_node_group_name" {
  value = (
    var.enable_spot_nodes
    ? aws_eks_node_group.spot[0].node_group_name
    : null
  )
}