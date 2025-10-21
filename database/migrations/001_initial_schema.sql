-- Initial schema for machine translation backend system
-- Migration: 001_initial_schema.sql

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create translation_jobs table
CREATE TABLE translation_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    source_language VARCHAR(10) NOT NULL,
    target_language VARCHAR(10) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    word_count INTEGER NOT NULL CHECK (word_count > 0),
    priority VARCHAR(20) DEFAULT 'normal' CHECK (priority IN ('normal', 'high', 'critical')),
    status VARCHAR(20) DEFAULT 'queued' CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
    progress DECIMAL(5,2) DEFAULT 0.00 CHECK (progress >= 0.00 AND progress <= 100.00),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    estimated_completion TIMESTAMP WITH TIME ZONE,
    callback_url VARCHAR(500),
    result_url VARCHAR(500),
    error_message TEXT,
    compute_instance_id VARCHAR(100),
    processing_time_ms INTEGER CHECK (processing_time_ms >= 0),
    
    -- Constraints
    CONSTRAINT valid_completion_time CHECK (
        (status = 'completed' AND completed_at IS NOT NULL) OR 
        (status != 'completed')
    ),
    CONSTRAINT valid_start_time CHECK (
        started_at IS NULL OR started_at >= created_at
    ),
    CONSTRAINT valid_processing_time CHECK (
        (status = 'completed' AND processing_time_ms IS NOT NULL) OR
        (status != 'completed')
    )
);

-- Create indexes for translation_jobs
CREATE INDEX idx_translation_jobs_status_priority ON translation_jobs (status, priority);
CREATE INDEX idx_translation_jobs_user_created ON translation_jobs (user_id, created_at DESC);
CREATE INDEX idx_translation_jobs_content_hash ON translation_jobs (content_hash);
CREATE INDEX idx_translation_jobs_created_at ON translation_jobs (created_at DESC);
CREATE INDEX idx_translation_jobs_status ON translation_jobs (status);
CREATE INDEX idx_translation_jobs_priority ON translation_jobs (priority);
CREATE INDEX idx_translation_jobs_compute_instance ON translation_jobs (compute_instance_id);

-- Create translation_cache table
CREATE TABLE translation_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_hash VARCHAR(64) NOT NULL,
    source_language VARCHAR(10) NOT NULL,
    target_language VARCHAR(10) NOT NULL,
    source_content TEXT NOT NULL,
    translated_content TEXT NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    confidence_score DECIMAL(5,2) CHECK (confidence_score >= 0.00 AND confidence_score <= 1.00),
    access_count INTEGER DEFAULT 1 CHECK (access_count >= 0),
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Unique constraint for cache entries
    CONSTRAINT unique_cache_entry UNIQUE (content_hash, source_language, target_language, model_version)
);

-- Create indexes for translation_cache
CREATE INDEX idx_translation_cache_hash_lang ON translation_cache (content_hash, source_language, target_language);
CREATE INDEX idx_translation_cache_last_accessed ON translation_cache (last_accessed DESC);
CREATE INDEX idx_translation_cache_access_count ON translation_cache (access_count DESC);
CREATE INDEX idx_translation_cache_model_version ON translation_cache (model_version);
CREATE INDEX idx_translation_cache_created_at ON translation_cache (created_at DESC);

-- Create system_metrics table
CREATE TABLE system_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,4) NOT NULL,
    instance_id VARCHAR(100),
    tags JSONB,
    
    -- Constraints
    CONSTRAINT valid_metric_name CHECK (LENGTH(metric_name) > 0)
);

-- Create indexes for system_metrics
CREATE INDEX idx_system_metrics_timestamp_metric ON system_metrics (timestamp DESC, metric_name);
CREATE INDEX idx_system_metrics_instance_timestamp ON system_metrics (instance_id, timestamp DESC);
CREATE INDEX idx_system_metrics_metric_name ON system_metrics (metric_name);
CREATE INDEX idx_system_metrics_timestamp ON system_metrics (timestamp DESC);

-- Create GIN index for JSONB tags
CREATE INDEX idx_system_metrics_tags ON system_metrics USING GIN (tags);

-- Create compute_instances table
CREATE TABLE compute_instances (
    id VARCHAR(100) PRIMARY KEY,
    instance_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'starting' CHECK (status IN ('starting', 'running', 'stopping', 'stopped', 'failed')),
    gpu_utilization DECIMAL(5,2) DEFAULT 0.00 CHECK (gpu_utilization >= 0.00 AND gpu_utilization <= 100.00),
    memory_usage DECIMAL(5,2) DEFAULT 0.00 CHECK (memory_usage >= 0.00 AND memory_usage <= 100.00),
    cpu_utilization DECIMAL(5,2) DEFAULT 0.00 CHECK (cpu_utilization >= 0.00 AND cpu_utilization <= 100.00),
    active_jobs INTEGER DEFAULT 0 CHECK (active_jobs >= 0),
    max_concurrent_jobs INTEGER DEFAULT 1 CHECK (max_concurrent_jobs > 0),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    terminated_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT valid_termination_time CHECK (
        (status = 'stopped' AND terminated_at IS NOT NULL) OR 
        (status != 'stopped')
    )
);

-- Create indexes for compute_instances
CREATE INDEX idx_compute_instances_status ON compute_instances (status);
CREATE INDEX idx_compute_instances_last_heartbeat ON compute_instances (last_heartbeat DESC);
CREATE INDEX idx_compute_instances_instance_type ON compute_instances (instance_type);
CREATE INDEX idx_compute_instances_active_jobs ON compute_instances (active_jobs);

-- Create users table for authentication
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    api_key VARCHAR(64) UNIQUE NOT NULL,
    is_active BOOLEAN DEFAULT true,
    rate_limit_per_minute INTEGER DEFAULT 100 CHECK (rate_limit_per_minute > 0),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT valid_user_id CHECK (LENGTH(user_id) > 0),
    CONSTRAINT valid_api_key CHECK (LENGTH(api_key) >= 32)
);

-- Create indexes for users
CREATE INDEX idx_users_user_id ON users (user_id);
CREATE INDEX idx_users_api_key ON users (api_key);
CREATE INDEX idx_users_email ON users (email);
CREATE INDEX idx_users_is_active ON users (is_active);

-- Create rate_limits table for tracking API usage
CREATE TABLE rate_limits (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    endpoint VARCHAR(100) NOT NULL,
    request_count INTEGER DEFAULT 1 CHECK (request_count > 0),
    window_start TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    window_end TIMESTAMP WITH TIME ZONE DEFAULT NOW() + INTERVAL '1 minute',
    
    -- Unique constraint for rate limiting windows
    CONSTRAINT unique_rate_limit_window UNIQUE (user_id, endpoint, window_start)
);

-- Create indexes for rate_limits
CREATE INDEX idx_rate_limits_user_window ON rate_limits (user_id, window_end DESC);
CREATE INDEX idx_rate_limits_window_end ON rate_limits (window_end);

-- Create audit_logs table for security and compliance
CREATE TABLE audit_logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    request_data JSONB,
    response_status INTEGER,
    
    -- Constraints
    CONSTRAINT valid_action CHECK (LENGTH(action) > 0)
);

-- Create indexes for audit_logs
CREATE INDEX idx_audit_logs_timestamp ON audit_logs (timestamp DESC);
CREATE INDEX idx_audit_logs_user_id ON audit_logs (user_id, timestamp DESC);
CREATE INDEX idx_audit_logs_action ON audit_logs (action);
CREATE INDEX idx_audit_logs_resource ON audit_logs (resource_type, resource_id);

-- Create GIN index for JSONB request_data
CREATE INDEX idx_audit_logs_request_data ON audit_logs USING GIN (request_data);

-- Create queue_metrics table for queue performance tracking
CREATE TABLE queue_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    priority VARCHAR(20) NOT NULL CHECK (priority IN ('normal', 'high', 'critical')),
    queue_depth INTEGER NOT NULL CHECK (queue_depth >= 0),
    avg_wait_time_seconds DECIMAL(10,2) CHECK (avg_wait_time_seconds >= 0),
    processing_rate_per_minute DECIMAL(10,2) CHECK (processing_rate_per_minute >= 0),
    
    -- Unique constraint to prevent duplicate entries for same timestamp/priority
    CONSTRAINT unique_queue_metric UNIQUE (timestamp, priority)
);

-- Create indexes for queue_metrics
CREATE INDEX idx_queue_metrics_timestamp ON queue_metrics (timestamp DESC);
CREATE INDEX idx_queue_metrics_priority ON queue_metrics (priority, timestamp DESC);

-- Create cost_tracking table for cost monitoring
CREATE TABLE cost_tracking (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    date DATE DEFAULT CURRENT_DATE,
    compute_cost_usd DECIMAL(10,4) DEFAULT 0.0000 CHECK (compute_cost_usd >= 0),
    storage_cost_usd DECIMAL(10,4) DEFAULT 0.0000 CHECK (storage_cost_usd >= 0),
    network_cost_usd DECIMAL(10,4) DEFAULT 0.0000 CHECK (network_cost_usd >= 0),
    total_cost_usd DECIMAL(10,4) GENERATED ALWAYS AS (compute_cost_usd + storage_cost_usd + network_cost_usd) STORED,
    words_processed INTEGER DEFAULT 0 CHECK (words_processed >= 0),
    jobs_completed INTEGER DEFAULT 0 CHECK (jobs_completed >= 0),
    instance_hours DECIMAL(10,2) DEFAULT 0.00 CHECK (instance_hours >= 0),
    
    -- Unique constraint for daily cost tracking
    CONSTRAINT unique_daily_cost UNIQUE (date)
);

-- Create indexes for cost_tracking
CREATE INDEX idx_cost_tracking_date ON cost_tracking (date DESC);
CREATE INDEX idx_cost_tracking_timestamp ON cost_tracking (timestamp DESC);

-- Create views for common queries

-- View for active jobs with queue position
CREATE VIEW active_jobs_with_position AS
SELECT 
    j.*,
    ROW_NUMBER() OVER (
        PARTITION BY j.priority 
        ORDER BY 
            CASE j.priority 
                WHEN 'critical' THEN 1 
                WHEN 'high' THEN 2 
                WHEN 'normal' THEN 3 
            END,
            j.created_at
    ) as queue_position
FROM translation_jobs j
WHERE j.status IN ('queued', 'processing');

-- View for job statistics by user
CREATE VIEW user_job_stats AS
SELECT 
    user_id,
    COUNT(*) as total_jobs,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_jobs,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_jobs,
    COUNT(CASE WHEN status IN ('queued', 'processing') THEN 1 END) as active_jobs,
    AVG(CASE WHEN status = 'completed' THEN processing_time_ms END) as avg_processing_time_ms,
    SUM(word_count) as total_words_processed,
    MIN(created_at) as first_job_date,
    MAX(created_at) as last_job_date
FROM translation_jobs
GROUP BY user_id;

-- View for cache performance metrics
CREATE VIEW cache_performance AS
SELECT 
    DATE(created_at) as date,
    COUNT(*) as cache_entries,
    SUM(access_count) as total_accesses,
    AVG(access_count) as avg_accesses_per_entry,
    COUNT(DISTINCT source_language || '-' || target_language) as language_pairs,
    AVG(confidence_score) as avg_confidence_score
FROM translation_cache
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- View for system performance summary
CREATE VIEW system_performance_summary AS
SELECT 
    DATE(timestamp) as date,
    AVG(CASE WHEN metric_name = 'gpu_utilization_percent' THEN metric_value END) as avg_gpu_utilization,
    AVG(CASE WHEN metric_name = 'memory_usage_percent' THEN metric_value END) as avg_memory_usage,
    AVG(CASE WHEN metric_name = 'translation_latency_seconds' THEN metric_value END) as avg_translation_latency,
    MAX(CASE WHEN metric_name = 'queue_depth_total' THEN metric_value END) as max_queue_depth,
    AVG(CASE WHEN metric_name = 'cache_hit_ratio' THEN metric_value END) as avg_cache_hit_ratio
FROM system_metrics
WHERE DATE(timestamp) >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- Create functions for common operations

-- Function to calculate queue position for a job
CREATE OR REPLACE FUNCTION get_queue_position(job_uuid UUID)
RETURNS INTEGER AS $$
DECLARE
    job_priority VARCHAR(20);
    job_created_at TIMESTAMP WITH TIME ZONE;
    position INTEGER;
BEGIN
    -- Get job details
    SELECT priority, created_at INTO job_priority, job_created_at
    FROM translation_jobs
    WHERE id = job_uuid AND status = 'queued';
    
    IF NOT FOUND THEN
        RETURN NULL;
    END IF;
    
    -- Calculate position in queue
    SELECT COUNT(*) + 1 INTO position
    FROM translation_jobs
    WHERE status = 'queued'
    AND (
        (priority = 'critical' AND job_priority != 'critical') OR
        (priority = 'high' AND job_priority = 'normal') OR
        (priority = job_priority AND created_at < job_created_at)
    );
    
    RETURN position;
END;
$$ LANGUAGE plpgsql;

-- Function to clean up old cache entries
CREATE OR REPLACE FUNCTION cleanup_old_cache_entries(retention_hours INTEGER DEFAULT 168)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM translation_cache
    WHERE last_accessed < NOW() - (retention_hours || ' hours')::INTERVAL
    AND access_count < 2;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to update cache access statistics
CREATE OR REPLACE FUNCTION update_cache_access(cache_uuid UUID)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE translation_cache
    SET access_count = access_count + 1,
        last_accessed = NOW()
    WHERE id = cache_uuid;
    
    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_accessed = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add comments for documentation
COMMENT ON TABLE translation_jobs IS 'Stores all translation job requests and their status';
COMMENT ON TABLE translation_cache IS 'Caches completed translations for reuse';
COMMENT ON TABLE system_metrics IS 'Stores system performance and monitoring metrics';
COMMENT ON TABLE compute_instances IS 'Tracks compute instances and their resource usage';
COMMENT ON TABLE users IS 'User accounts and API key management';
COMMENT ON TABLE rate_limits IS 'API rate limiting tracking';
COMMENT ON TABLE audit_logs IS 'Security and compliance audit trail';
COMMENT ON TABLE queue_metrics IS 'Queue performance metrics over time';
COMMENT ON TABLE cost_tracking IS 'Daily cost tracking and optimization metrics';

-- Insert initial data for development
INSERT INTO users (user_id, email, api_key, rate_limit_per_minute) VALUES
('dev-user', 'dev@example.com', 'dev-api-key-12345678901234567890123456789012', 1000),
('test-user', 'test@example.com', 'test-api-key-12345678901234567890123456789012', 100);

-- Create initial system metrics for monitoring setup
INSERT INTO system_metrics (metric_name, metric_value, instance_id, tags) VALUES
('system_startup', 1.0, 'init', '{"event": "database_initialized", "version": "1.0.0"}');

COMMIT;