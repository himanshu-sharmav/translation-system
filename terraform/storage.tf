# S3 Buckets and Storage Configuration

# S3 Bucket for ML Models
resource "aws_s3_bucket" "models" {
  bucket = "${var.project_name}-models-${random_id.bucket_suffix.hex}"

  tags = {
    Name        = "${var.project_name}-models-bucket"
    Purpose     = "ML model storage"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "models" {
  bucket = aws_s3_bucket.models.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "models" {
  bucket = aws_s3_bucket.models.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "models" {
  bucket = aws_s3_bucket.models.id

  rule {
    id     = "model_lifecycle"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = 90
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# S3 Bucket for Translation Results
resource "aws_s3_bucket" "translations" {
  bucket = "${var.project_name}-translations-${random_id.bucket_suffix.hex}"

  tags = {
    Name        = "${var.project_name}-translations-bucket"
    Purpose     = "Translation results storage"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "translations" {
  bucket = aws_s3_bucket.translations.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "translations" {
  bucket = aws_s3_bucket.translations.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "translations" {
  bucket = aws_s3_bucket.translations.id

  rule {
    id     = "translation_lifecycle"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    transition {
      days          = 365
      storage_class = "DEEP_ARCHIVE"
    }

    expiration {
      days = 2555  # 7 years
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# S3 Bucket for Backups
resource "aws_s3_bucket" "backups" {
  bucket = "${var.project_name}-backups-${random_id.bucket_suffix.hex}"

  tags = {
    Name        = "${var.project_name}-backups-bucket"
    Purpose     = "Database and system backups"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "backups" {
  bucket = aws_s3_bucket.backups.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "backups" {
  bucket = aws_s3_bucket.backups.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        kms_master_key_id = aws_kms_key.main.arn
        sse_algorithm     = "aws:kms"
      }
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  rule {
    id     = "backup_lifecycle"
    status = "Enabled"

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = var.backup_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = 7
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 1
    }
  }
}

# Block public access for all buckets
resource "aws_s3_bucket_public_access_block" "models" {
  bucket = aws_s3_bucket.models.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "translations" {
  bucket = aws_s3_bucket.translations.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "backups" {
  bucket = aws_s3_bucket.backups.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Random ID for bucket naming
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# KMS Key for encryption
resource "aws_kms_key" "main" {
  description             = "KMS key for ${var.project_name}"
  deletion_window_in_days = 7

  tags = {
    Name = "${var.project_name}-kms-key"
  }
}

resource "aws_kms_alias" "main" {
  name          = "alias/${var.project_name}-key"
  target_key_id = aws_kms_key.main.key_id
}

# EFS for shared model storage (optional)
resource "aws_efs_file_system" "models" {
  count = var.environment == "prod" ? 1 : 0

  creation_token   = "${var.project_name}-models-efs"
  performance_mode = "generalPurpose"
  throughput_mode  = "provisioned"
  provisioned_throughput_in_mibps = 100

  encrypted = true
  kms_key_id = aws_kms_key.main.arn

  tags = {
    Name = "${var.project_name}-models-efs"
  }
}

resource "aws_efs_mount_target" "models" {
  count = var.environment == "prod" ? length(aws_subnet.private) : 0

  file_system_id  = aws_efs_file_system.models[0].id
  subnet_id       = aws_subnet.private[count.index].id
  security_groups = [aws_security_group.efs[0].id]
}

resource "aws_security_group" "efs" {
  count = var.environment == "prod" ? 1 : 0

  name_prefix = "${var.project_name}-efs-"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "NFS"
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id, aws_security_group.gpu_instances.id]
  }

  tags = {
    Name = "${var.project_name}-efs-sg"
  }
}