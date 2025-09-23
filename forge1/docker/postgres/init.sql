-- forge1/docker/postgres/init.sql
-- PostgreSQL initialization script for Forge 1

-- Create database if not exists
SELECT 'CREATE DATABASE forge1'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'forge1')\gexec

-- Connect to forge1 database
\c forge1;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS forge1_core;
CREATE SCHEMA IF NOT EXISTS forge1_agents;
CREATE SCHEMA IF NOT EXISTS forge1_memory;
CREATE SCHEMA IF NOT EXISTS forge1_audit;

-- Set search path
SET search_path TO forge1_core, public;

-- Create basic tables for Forge 1
CREATE TABLE IF NOT EXISTS forge1_core.ai_employees (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    capabilities JSONB,
    performance_metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS forge1_core.task_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    employee_id UUID REFERENCES forge1_core.ai_employees(id),
    task_definition JSONB NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    performance_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Enhanced memory contexts table with full feature set
CREATE TABLE IF NOT EXISTS forge1_memory.memory_contexts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    employee_id UUID NOT NULL,
    session_id UUID,
    memory_type VARCHAR(50) NOT NULL,
    content JSONB NOT NULL,
    summary TEXT,
    keywords TEXT[],
    
    -- Vector embeddings (using pgvector extension if available)
    embeddings VECTOR(1536),
    embedding_model VARCHAR(100),
    
    -- Relevance scoring
    semantic_similarity FLOAT DEFAULT 0.0,
    temporal_relevance FLOAT DEFAULT 0.0,
    context_relevance FLOAT DEFAULT 0.0,
    usage_frequency FLOAT DEFAULT 0.0,
    overall_relevance_score FLOAT DEFAULT 0.0,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Security and permissions
    security_level VARCHAR(20) DEFAULT 'internal',
    owner_id VARCHAR(255) NOT NULL,
    shared_with TEXT[],
    
    -- Metadata
    source VARCHAR(255),
    tags TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Relationships
    parent_memory_id UUID,
    related_memory_ids UUID[]
);

-- Memory conflicts table
CREATE TABLE IF NOT EXISTS forge1_memory.memory_conflicts (
    conflict_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    memory_ids UUID[] NOT NULL,
    conflict_type VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    confidence_score FLOAT DEFAULT 0.0,
    resolution_strategy TEXT,
    resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- Memory sharing table
CREATE TABLE IF NOT EXISTS forge1_memory.memory_shares (
    share_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    memory_id UUID NOT NULL REFERENCES forge1_memory.memory_contexts(id),
    from_employee_id UUID NOT NULL,
    to_employee_id UUID NOT NULL,
    share_type VARCHAR(20) NOT NULL,
    permissions TEXT[],
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    accessed_at TIMESTAMP WITH TIME ZONE
);

-- Memory pruning rules table
CREATE TABLE IF NOT EXISTS forge1_memory.memory_pruning_rules (
    rule_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    conditions JSONB NOT NULL,
    action VARCHAR(20) NOT NULL,
    priority INTEGER DEFAULT 5,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS forge1_audit.audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    user_id VARCHAR(255),
    employee_id UUID,
    event_data JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_ai_employees_type ON forge1_core.ai_employees(type);
CREATE INDEX IF NOT EXISTS idx_ai_employees_created_at ON forge1_core.ai_employees(created_at);
CREATE INDEX IF NOT EXISTS idx_task_executions_employee_id ON forge1_core.task_executions(employee_id);
CREATE INDEX IF NOT EXISTS idx_task_executions_status ON forge1_core.task_executions(status);
CREATE INDEX IF NOT EXISTS idx_task_executions_created_at ON forge1_core.task_executions(created_at);
-- Enhanced indexes for memory system performance
CREATE INDEX IF NOT EXISTS idx_memory_contexts_employee_id ON forge1_memory.memory_contexts(employee_id);
CREATE INDEX IF NOT EXISTS idx_memory_contexts_session_id ON forge1_memory.memory_contexts(session_id);
CREATE INDEX IF NOT EXISTS idx_memory_contexts_type ON forge1_memory.memory_contexts(memory_type);
CREATE INDEX IF NOT EXISTS idx_memory_contexts_owner ON forge1_memory.memory_contexts(owner_id);
CREATE INDEX IF NOT EXISTS idx_memory_contexts_created_at ON forge1_memory.memory_contexts(created_at);
CREATE INDEX IF NOT EXISTS idx_memory_contexts_relevance ON forge1_memory.memory_contexts(overall_relevance_score);
CREATE INDEX IF NOT EXISTS idx_memory_contexts_keywords ON forge1_memory.memory_contexts USING GIN(keywords);
CREATE INDEX IF NOT EXISTS idx_memory_contexts_tags ON forge1_memory.memory_contexts USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_memory_contexts_security ON forge1_memory.memory_contexts(security_level);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_memory_contexts_content_search ON forge1_memory.memory_contexts USING GIN(to_tsvector('english', content::text));
CREATE INDEX IF NOT EXISTS idx_memory_contexts_summary_search ON forge1_memory.memory_contexts USING GIN(to_tsvector('english', summary));
CREATE INDEX IF NOT EXISTS idx_audit_log_event_type ON forge1_audit.audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON forge1_audit.audit_log(created_at);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_ai_employees_updated_at 
    BEFORE UPDATE ON forge1_core.ai_employees 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_memory_contexts_updated_at 
    BEFORE UPDATE ON forge1_memory.memory_contexts 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA forge1_core TO forge1_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA forge1_agents TO forge1_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA forge1_memory TO forge1_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA forge1_audit TO forge1_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA forge1_core TO forge1_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA forge1_agents TO forge1_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA forge1_memory TO forge1_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA forge1_audit TO forge1_user;