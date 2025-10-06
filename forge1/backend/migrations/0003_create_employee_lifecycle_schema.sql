-- Migration: Create AI Employee Lifecycle Schema
-- Description: Creates tables and indexes for complete AI employee lifecycle management
-- Requirements: 1.1, 1.3, 2.1, 2.2, 8.2

-- Create employee lifecycle schema
CREATE SCHEMA IF NOT EXISTS forge1_employees;

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Clients table for multi-tenant client management
CREATE TABLE IF NOT EXISTS forge1_employees.clients (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    tier VARCHAR(50) NOT NULL DEFAULT 'standard',
    configuration JSONB NOT NULL DEFAULT '{}',
    max_employees INTEGER NOT NULL DEFAULT 10,
    allowed_models TEXT[] NOT NULL DEFAULT ARRAY['gpt-4', 'gpt-3.5-turbo'],
    security_level VARCHAR(50) NOT NULL DEFAULT 'standard',
    compliance_requirements TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    
    -- Constraints
    CONSTRAINT valid_tier CHECK (tier IN ('standard', 'professional', 'enterprise')),
    CONSTRAINT valid_status CHECK (status IN ('active', 'inactive', 'suspended')),
    CONSTRAINT valid_security_level CHECK (security_level IN ('basic', 'standard', 'high', 'maximum'))
);

-- AI Employees table for individual employee configurations
CREATE TABLE IF NOT EXISTS forge1_employees.employees (
    id VARCHAR(255) PRIMARY KEY,
    client_id VARCHAR(255) NOT NULL REFERENCES forge1_employees.clients(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    role VARCHAR(255) NOT NULL,
    
    -- Personality configuration
    personality JSONB NOT NULL DEFAULT '{}',
    communication_style VARCHAR(50) NOT NULL DEFAULT 'professional',
    formality_level VARCHAR(50) NOT NULL DEFAULT 'formal',
    expertise_level VARCHAR(50) NOT NULL DEFAULT 'expert',
    response_length VARCHAR(50) NOT NULL DEFAULT 'detailed',
    creativity_level FLOAT NOT NULL DEFAULT 0.7 CHECK (creativity_level >= 0.0 AND creativity_level <= 1.0),
    empathy_level FLOAT NOT NULL DEFAULT 0.7 CHECK (empathy_level >= 0.0 AND empathy_level <= 1.0),
    
    -- Model preferences
    model_preferences JSONB NOT NULL DEFAULT '{}',
    primary_model VARCHAR(100) NOT NULL DEFAULT 'gpt-4',
    fallback_models TEXT[] NOT NULL DEFAULT ARRAY['gpt-3.5-turbo'],
    temperature FLOAT NOT NULL DEFAULT 0.7 CHECK (temperature >= 0.0 AND temperature <= 2.0),
    max_tokens INTEGER NOT NULL DEFAULT 2000 CHECK (max_tokens > 0),
    specialized_models JSONB NOT NULL DEFAULT '{}',
    
    -- Access and capabilities
    tool_access TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    knowledge_sources TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_interaction_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    
    -- Constraints
    CONSTRAINT valid_employee_status CHECK (status IN ('active', 'inactive', 'training', 'archived')),
    CONSTRAINT valid_communication_style CHECK (communication_style IN ('professional', 'friendly', 'technical', 'casual')),
    CONSTRAINT valid_formality_level CHECK (formality_level IN ('formal', 'casual', 'adaptive')),
    CONSTRAINT valid_expertise_level CHECK (expertise_level IN ('expert', 'intermediate', 'beginner')),
    CONSTRAINT valid_response_length CHECK (response_length IN ('concise', 'detailed', 'adaptive'))
);

-- Employee interactions table for conversation history and memory
CREATE TABLE IF NOT EXISTS forge1_employees.employee_interactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id VARCHAR(255) NOT NULL,
    employee_id VARCHAR(255) NOT NULL REFERENCES forge1_employees.employees(id) ON DELETE CASCADE,
    interaction_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    
    -- Interaction content
    message TEXT NOT NULL,
    response TEXT NOT NULL,
    context JSONB DEFAULT '{}',
    
    -- Memory and embeddings
    embedding VECTOR(1536), -- OpenAI embedding dimension (text-embedding-3-small: 1536, text-embedding-3-large: 3072)
    memory_type VARCHAR(50) NOT NULL DEFAULT 'conversation',
    importance_score FLOAT DEFAULT 0.5 CHECK (importance_score >= 0.0 AND importance_score <= 1.0),
    
    -- Metadata
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    processing_time_ms FLOAT,
    model_used VARCHAR(100),
    tokens_used INTEGER,
    cost DECIMAL(10,6),
    
    -- Constraints
    CONSTRAINT valid_memory_type CHECK (memory_type IN ('conversation', 'task', 'knowledge', 'feedback', 'system')),
    CONSTRAINT fk_client_employee FOREIGN KEY (client_id, employee_id) 
        REFERENCES forge1_employees.employees(client_id, id) ON DELETE CASCADE
);

-- Employee memory summaries for long-term context
CREATE TABLE IF NOT EXISTS forge1_employees.employee_memory_summaries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id VARCHAR(255) NOT NULL,
    employee_id VARCHAR(255) NOT NULL REFERENCES forge1_employees.employees(id) ON DELETE CASCADE,
    
    -- Summary content
    summary_text TEXT NOT NULL,
    summary_type VARCHAR(50) NOT NULL DEFAULT 'conversation',
    time_period_start TIMESTAMP WITH TIME ZONE NOT NULL,
    time_period_end TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Embeddings and metadata
    embedding VECTOR(1536), -- OpenAI embedding dimension
    interaction_count INTEGER NOT NULL DEFAULT 0,
    importance_score FLOAT DEFAULT 0.5 CHECK (importance_score >= 0.0 AND importance_score <= 1.0),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_summary_type CHECK (summary_type IN ('conversation', 'daily', 'weekly', 'project', 'relationship')),
    CONSTRAINT valid_time_period CHECK (time_period_end > time_period_start),
    CONSTRAINT fk_summary_client_employee FOREIGN KEY (client_id, employee_id) 
        REFERENCES forge1_employees.employees(client_id, id) ON DELETE CASCADE
);

-- Employee knowledge base for custom knowledge sources
CREATE TABLE IF NOT EXISTS forge1_employees.employee_knowledge (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id VARCHAR(255) NOT NULL,
    employee_id VARCHAR(255) NOT NULL REFERENCES forge1_employees.employees(id) ON DELETE CASCADE,
    
    -- Knowledge content
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    source_type VARCHAR(50) NOT NULL DEFAULT 'document',
    source_url VARCHAR(1000),
    
    -- Embeddings and search
    embedding VECTOR(1536), -- OpenAI embedding dimension
    keywords TEXT[] DEFAULT ARRAY[]::TEXT[],
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed_at TIMESTAMP WITH TIME ZONE,
    access_count INTEGER DEFAULT 0,
    
    -- Constraints
    CONSTRAINT valid_source_type CHECK (source_type IN ('document', 'url', 'manual', 'training', 'api')),
    CONSTRAINT fk_knowledge_client_employee FOREIGN KEY (client_id, employee_id) 
        REFERENCES forge1_employees.employees(client_id, id) ON DELETE CASCADE
);

-- Performance indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_clients_status ON forge1_employees.clients(status);
CREATE INDEX IF NOT EXISTS idx_clients_tier ON forge1_employees.clients(tier);
CREATE INDEX IF NOT EXISTS idx_clients_created_at ON forge1_employees.clients(created_at);

CREATE INDEX IF NOT EXISTS idx_employees_client_id ON forge1_employees.employees(client_id);
CREATE INDEX IF NOT EXISTS idx_employees_status ON forge1_employees.employees(status);
CREATE INDEX IF NOT EXISTS idx_employees_role ON forge1_employees.employees(role);
CREATE INDEX IF NOT EXISTS idx_employees_created_at ON forge1_employees.employees(created_at);
CREATE INDEX IF NOT EXISTS idx_employees_last_interaction ON forge1_employees.employees(last_interaction_at);

CREATE INDEX IF NOT EXISTS idx_interactions_employee_id ON forge1_employees.employee_interactions(employee_id);
CREATE INDEX IF NOT EXISTS idx_interactions_client_id ON forge1_employees.employee_interactions(client_id);
CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON forge1_employees.employee_interactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_interactions_session_id ON forge1_employees.employee_interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_interactions_memory_type ON forge1_employees.employee_interactions(memory_type);
CREATE INDEX IF NOT EXISTS idx_interactions_client_employee ON forge1_employees.employee_interactions(client_id, employee_id);

CREATE INDEX IF NOT EXISTS idx_memory_summaries_employee_id ON forge1_employees.employee_memory_summaries(employee_id);
CREATE INDEX IF NOT EXISTS idx_memory_summaries_time_period ON forge1_employees.employee_memory_summaries(time_period_start, time_period_end);
CREATE INDEX IF NOT EXISTS idx_memory_summaries_type ON forge1_employees.employee_memory_summaries(summary_type);

CREATE INDEX IF NOT EXISTS idx_knowledge_employee_id ON forge1_employees.employee_knowledge(employee_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_source_type ON forge1_employees.employee_knowledge(source_type);
CREATE INDEX IF NOT EXISTS idx_knowledge_keywords ON forge1_employees.employee_knowledge USING GIN(keywords);
CREATE INDEX IF NOT EXISTS idx_knowledge_tags ON forge1_employees.employee_knowledge USING GIN(tags);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_interactions_message_search ON forge1_employees.employee_interactions 
    USING GIN(to_tsvector('english', message));
CREATE INDEX IF NOT EXISTS idx_interactions_response_search ON forge1_employees.employee_interactions 
    USING GIN(to_tsvector('english', response));
CREATE INDEX IF NOT EXISTS idx_knowledge_content_search ON forge1_employees.employee_knowledge 
    USING GIN(to_tsvector('english', content));

-- Row Level Security for tenant isolation
ALTER TABLE forge1_employees.clients ENABLE ROW LEVEL SECURITY;
ALTER TABLE forge1_employees.employees ENABLE ROW LEVEL SECURITY;
ALTER TABLE forge1_employees.employee_interactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE forge1_employees.employee_memory_summaries ENABLE ROW LEVEL SECURITY;
ALTER TABLE forge1_employees.employee_knowledge ENABLE ROW LEVEL SECURITY;

-- RLS Policies for complete tenant isolation
-- Note: These policies assume current_setting('app.current_client_id') is set by the application

-- Clients can only see their own record
CREATE POLICY client_isolation ON forge1_employees.clients
    USING (id = current_setting('app.current_client_id', true));

-- Employees belong to specific clients
CREATE POLICY employee_isolation ON forge1_employees.employees
    USING (client_id = current_setting('app.current_client_id', true));

-- Interactions are isolated by client_id
CREATE POLICY interaction_isolation ON forge1_employees.employee_interactions
    USING (client_id = current_setting('app.current_client_id', true));

-- Memory summaries are isolated by client_id
CREATE POLICY memory_summary_isolation ON forge1_employees.employee_memory_summaries
    USING (client_id = current_setting('app.current_client_id', true));

-- Knowledge is isolated by client_id
CREATE POLICY knowledge_isolation ON forge1_employees.employee_knowledge
    USING (client_id = current_setting('app.current_client_id', true));

-- Triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_clients_updated_at 
    BEFORE UPDATE ON forge1_employees.clients 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_employees_updated_at 
    BEFORE UPDATE ON forge1_employees.employees 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_knowledge_updated_at 
    BEFORE UPDATE ON forge1_employees.employee_knowledge 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions to forge1_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA forge1_employees TO forge1_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA forge1_employees TO forge1_user;
GRANT USAGE ON SCHEMA forge1_employees TO forge1_user;

-- Insert sample data for testing
INSERT INTO forge1_employees.clients (id, name, industry, tier, max_employees) VALUES
('client_demo_001', 'Demo Law Firm', 'legal', 'professional', 50),
('client_demo_002', 'Tech Startup Inc', 'technology', 'standard', 20)
ON CONFLICT (id) DO NOTHING;