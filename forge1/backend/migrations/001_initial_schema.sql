-- Migration 001: Initial Employee Lifecycle Schema
-- Creates the foundational database schema for the Employee Lifecycle System
-- Requirements: 5.3, 5.4, 8.5

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create enum types
CREATE TYPE client_tier AS ENUM ('basic', 'professional', 'enterprise');
CREATE TYPE client_status AS ENUM ('active', 'inactive', 'suspended');
CREATE TYPE employee_status AS ENUM ('active', 'inactive', 'archived');
CREATE TYPE communication_style AS ENUM ('friendly', 'professional', 'technical', 'casual');
CREATE TYPE formality_level AS ENUM ('very_casual', 'casual', 'neutral', 'formal', 'very_formal');
CREATE TYPE expertise_level AS ENUM ('beginner', 'intermediate', 'advanced', 'expert');
CREATE TYPE response_length AS ENUM ('brief', 'moderate', 'detailed', 'comprehensive');
CREATE TYPE memory_type AS ENUM ('interaction', 'knowledge', 'context', 'system');
CREATE TYPE security_level AS ENUM ('standard', 'high', 'maximum');

-- Clients table
CREATE TABLE clients (
    id VARCHAR(50) PRIMARY KEY DEFAULT ('client_' || encode(gen_random_bytes(16), 'hex')),
    tenant_id VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    industry VARCHAR(100),
    tier client_tier NOT NULL DEFAULT 'basic',
    status client_status NOT NULL DEFAULT 'active',
    max_employees INTEGER NOT NULL DEFAULT 10,
    current_employees INTEGER NOT NULL DEFAULT 0,
    allowed_models TEXT[] DEFAULT ARRAY['gpt-3.5-turbo'],
    security_level security_level NOT NULL DEFAULT 'standard',
    compliance_requirements TEXT[] DEFAULT ARRAY[]::TEXT[],
    configuration JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(50),
    updated_by VARCHAR(50)
);

-- Employees table
CREATE TABLE employees (
    id VARCHAR(50) PRIMARY KEY DEFAULT ('emp_' || encode(gen_random_bytes(16), 'hex')),
    client_id VARCHAR(50) NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    tenant_id VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    role VARCHAR(255) NOT NULL,
    status employee_status NOT NULL DEFAULT 'active',
    industry VARCHAR(100),
    expertise_areas TEXT[] DEFAULT ARRAY[]::TEXT[],
    communication_style communication_style NOT NULL DEFAULT 'friendly',
    tools_needed TEXT[] DEFAULT ARRAY[]::TEXT[],
    knowledge_domains TEXT[] DEFAULT ARRAY[]::TEXT[],
    personality JSONB NOT NULL DEFAULT '{}',
    model_preferences JSONB NOT NULL DEFAULT '{}',
    tool_access TEXT[] DEFAULT ARRAY[]::TEXT[],
    knowledge_sources TEXT[] DEFAULT ARRAY[]::TEXT[],
    configuration JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_interaction_at TIMESTAMP WITH TIME ZONE,
    created_by VARCHAR(50),
    updated_by VARCHAR(50)
);

-- Interactions table
CREATE TABLE interactions (
    id VARCHAR(50) PRIMARY KEY DEFAULT ('int_' || encode(gen_random_bytes(16), 'hex')),
    employee_id VARCHAR(50) NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    tenant_id VARCHAR(50) NOT NULL,
    session_id VARCHAR(100),
    message TEXT NOT NULL,
    response TEXT NOT NULL,
    context JSONB DEFAULT '{}',
    processing_time_ms INTEGER,
    tokens_used INTEGER,
    cost DECIMAL(10,6),
    model_used VARCHAR(100),
    confidence_score DECIMAL(3,2),
    feedback_rating INTEGER CHECK (feedback_rating >= 1 AND feedback_rating <= 5),
    feedback_text TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(50)
);

-- Memories table
CREATE TABLE memories (
    id VARCHAR(50) PRIMARY KEY DEFAULT ('mem_' || encode(gen_random_bytes(16), 'hex')),
    employee_id VARCHAR(50) NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    tenant_id VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    memory_type memory_type NOT NULL DEFAULT 'interaction',
    importance_score DECIMAL(3,2) NOT NULL DEFAULT 0.5 CHECK (importance_score >= 0 AND importance_score <= 1),
    context JSONB DEFAULT '{}',
    embedding_vector VECTOR(1536), -- For vector similarity search
    source_interaction_id VARCHAR(50) REFERENCES interactions(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    created_by VARCHAR(50)
);

-- Knowledge sources table
CREATE TABLE knowledge_sources (
    id VARCHAR(50) PRIMARY KEY DEFAULT ('kb_' || encode(gen_random_bytes(16), 'hex')),
    employee_id VARCHAR(50) NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    tenant_id VARCHAR(50) NOT NULL,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    source_type VARCHAR(50) NOT NULL,
    source_url TEXT,
    keywords TEXT[] DEFAULT ARRAY[]::TEXT[],
    metadata JSONB DEFAULT '{}',
    status VARCHAR(20) NOT NULL DEFAULT 'processing',
    chunks_created INTEGER DEFAULT 0,
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(50)
);

-- Analytics events table
CREATE TABLE analytics_events (
    id VARCHAR(50) PRIMARY KEY DEFAULT ('evt_' || encode(gen_random_bytes(16), 'hex')),
    tenant_id VARCHAR(50) NOT NULL,
    client_id VARCHAR(50) REFERENCES clients(id) ON DELETE CASCADE,
    employee_id VARCHAR(50) REFERENCES employees(id) ON DELETE CASCADE,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    session_id VARCHAR(100),
    user_id VARCHAR(50)
);

-- Performance metrics table
CREATE TABLE performance_metrics (
    id VARCHAR(50) PRIMARY KEY DEFAULT ('met_' || encode(gen_random_bytes(16), 'hex')),
    tenant_id VARCHAR(50) NOT NULL,
    client_id VARCHAR(50) REFERENCES clients(id) ON DELETE CASCADE,
    employee_id VARCHAR(50) REFERENCES employees(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    metric_unit VARCHAR(20),
    dimensions JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Audit log table
CREATE TABLE audit_logs (
    id VARCHAR(50) PRIMARY KEY DEFAULT ('aud_' || encode(gen_random_bytes(16), 'hex')),
    tenant_id VARCHAR(50) NOT NULL,
    table_name VARCHAR(100) NOT NULL,
    record_id VARCHAR(50) NOT NULL,
    operation VARCHAR(20) NOT NULL, -- INSERT, UPDATE, DELETE
    old_values JSONB,
    new_values JSONB,
    changed_by VARCHAR(50),
    changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT
);

-- Create indexes for performance
CREATE INDEX idx_clients_tenant_status ON clients(tenant_id, status);
CREATE INDEX idx_clients_tier ON clients(tier);
CREATE INDEX idx_clients_created_at ON clients(created_at DESC);

CREATE INDEX idx_employees_client_status ON employees(client_id, status);
CREATE INDEX idx_employees_tenant_status ON employees(tenant_id, status);
CREATE INDEX idx_employees_role ON employees(role);
CREATE INDEX idx_employees_created_at ON employees(created_at DESC);
CREATE INDEX idx_employees_last_interaction ON employees(last_interaction_at DESC);

CREATE INDEX idx_interactions_employee_created ON interactions(employee_id, created_at DESC);
CREATE INDEX idx_interactions_tenant_created ON interactions(tenant_id, created_at DESC);
CREATE INDEX idx_interactions_session ON interactions(session_id);
CREATE INDEX idx_interactions_model_used ON interactions(model_used);

CREATE INDEX idx_memories_employee_type ON memories(employee_id, memory_type);
CREATE INDEX idx_memories_tenant_created ON memories(tenant_id, created_at DESC);
CREATE INDEX idx_memories_importance ON memories(importance_score DESC);
CREATE INDEX idx_memories_expires ON memories(expires_at) WHERE expires_at IS NOT NULL;

CREATE INDEX idx_knowledge_sources_employee ON knowledge_sources(employee_id);
CREATE INDEX idx_knowledge_sources_tenant ON knowledge_sources(tenant_id);
CREATE INDEX idx_knowledge_sources_status ON knowledge_sources(status);

CREATE INDEX idx_analytics_events_tenant_type ON analytics_events(tenant_id, event_type);
CREATE INDEX idx_analytics_events_employee_timestamp ON analytics_events(employee_id, timestamp DESC);
CREATE INDEX idx_analytics_events_client_timestamp ON analytics_events(client_id, timestamp DESC);

CREATE INDEX idx_performance_metrics_employee_name ON performance_metrics(employee_id, metric_name);
CREATE INDEX idx_performance_metrics_client_timestamp ON performance_metrics(client_id, timestamp DESC);

CREATE INDEX idx_audit_logs_tenant_table ON audit_logs(tenant_id, table_name);
CREATE INDEX idx_audit_logs_record ON audit_logs(table_name, record_id);
CREATE INDEX idx_audit_logs_timestamp ON audit_logs(changed_at DESC);

-- Full-text search indexes
CREATE INDEX idx_employees_search ON employees USING gin(to_tsvector('english', name || ' ' || role));
CREATE INDEX idx_memories_search ON memories USING gin(to_tsvector('english', content));
CREATE INDEX idx_knowledge_sources_search ON knowledge_sources USING gin(to_tsvector('english', title || ' ' || content));

-- Row Level Security (RLS) policies
ALTER TABLE clients ENABLE ROW LEVEL SECURITY;
ALTER TABLE employees ENABLE ROW LEVEL SECURITY;
ALTER TABLE interactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE analytics_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE performance_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- RLS policies for tenant isolation
CREATE POLICY tenant_isolation_clients ON clients
    FOR ALL TO application_role
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY tenant_isolation_employees ON employees
    FOR ALL TO application_role
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY tenant_isolation_interactions ON interactions
    FOR ALL TO application_role
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY tenant_isolation_memories ON memories
    FOR ALL TO application_role
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY tenant_isolation_knowledge_sources ON knowledge_sources
    FOR ALL TO application_role
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY tenant_isolation_analytics_events ON analytics_events
    FOR ALL TO application_role
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY tenant_isolation_performance_metrics ON performance_metrics
    FOR ALL TO application_role
    USING (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY tenant_isolation_audit_logs ON audit_logs
    FOR ALL TO application_role
    USING (tenant_id = current_setting('app.current_tenant_id', true));

-- Create application role
CREATE ROLE application_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO application_role;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO application_role;

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_clients_updated_at BEFORE UPDATE ON clients
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_employees_updated_at BEFORE UPDATE ON employees
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_knowledge_sources_updated_at BEFORE UPDATE ON knowledge_sources
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create trigger for maintaining client employee count
CREATE OR REPLACE FUNCTION update_client_employee_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE clients 
        SET current_employees = current_employees + 1 
        WHERE id = NEW.client_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE clients 
        SET current_employees = current_employees - 1 
        WHERE id = OLD.client_id;
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        -- Handle status changes
        IF OLD.status = 'active' AND NEW.status != 'active' THEN
            UPDATE clients 
            SET current_employees = current_employees - 1 
            WHERE id = NEW.client_id;
        ELSIF OLD.status != 'active' AND NEW.status = 'active' THEN
            UPDATE clients 
            SET current_employees = current_employees + 1 
            WHERE id = NEW.client_id;
        END IF;
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ language 'plpgsql';

CREATE TRIGGER maintain_client_employee_count
    AFTER INSERT OR UPDATE OR DELETE ON employees
    FOR EACH ROW EXECUTE FUNCTION update_client_employee_count();

-- Create audit trigger function
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_logs (
            tenant_id, table_name, record_id, operation, 
            new_values, changed_by, ip_address, user_agent
        ) VALUES (
            NEW.tenant_id, TG_TABLE_NAME, NEW.id, TG_OP,
            to_jsonb(NEW), current_setting('app.current_user_id', true),
            inet(current_setting('app.client_ip', true)),
            current_setting('app.user_agent', true)
        );
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_logs (
            tenant_id, table_name, record_id, operation,
            old_values, new_values, changed_by, ip_address, user_agent
        ) VALUES (
            NEW.tenant_id, TG_TABLE_NAME, NEW.id, TG_OP,
            to_jsonb(OLD), to_jsonb(NEW), current_setting('app.current_user_id', true),
            inet(current_setting('app.client_ip', true)),
            current_setting('app.user_agent', true)
        );
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_logs (
            tenant_id, table_name, record_id, operation,
            old_values, changed_by, ip_address, user_agent
        ) VALUES (
            OLD.tenant_id, TG_TABLE_NAME, OLD.id, TG_OP,
            to_jsonb(OLD), current_setting('app.current_user_id', true),
            inet(current_setting('app.client_ip', true)),
            current_setting('app.user_agent', true)
        );
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ language 'plpgsql';

-- Create audit triggers for main tables
CREATE TRIGGER audit_clients_trigger
    AFTER INSERT OR UPDATE OR DELETE ON clients
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_employees_trigger
    AFTER INSERT OR UPDATE OR DELETE ON employees
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER audit_interactions_trigger
    AFTER INSERT OR UPDATE OR DELETE ON interactions
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

-- Create views for common queries
CREATE VIEW active_employees AS
SELECT e.*, c.name as client_name, c.tier as client_tier
FROM employees e
JOIN clients c ON e.client_id = c.id
WHERE e.status = 'active' AND c.status = 'active';

CREATE VIEW employee_stats AS
SELECT 
    e.id,
    e.name,
    e.role,
    COUNT(i.id) as total_interactions,
    AVG(i.processing_time_ms) as avg_processing_time,
    AVG(i.tokens_used) as avg_tokens_used,
    SUM(i.cost) as total_cost,
    AVG(i.feedback_rating) as avg_rating,
    MAX(i.created_at) as last_interaction_at
FROM employees e
LEFT JOIN interactions i ON e.id = i.employee_id
WHERE e.status = 'active'
GROUP BY e.id, e.name, e.role;

CREATE VIEW client_usage_summary AS
SELECT 
    c.id,
    c.name,
    c.tier,
    c.current_employees,
    c.max_employees,
    COUNT(DISTINCT e.id) as active_employees,
    COUNT(i.id) as total_interactions,
    SUM(i.cost) as total_cost,
    AVG(i.feedback_rating) as avg_satisfaction
FROM clients c
LEFT JOIN employees e ON c.id = e.client_id AND e.status = 'active'
LEFT JOIN interactions i ON e.id = i.employee_id
WHERE c.status = 'active'
GROUP BY c.id, c.name, c.tier, c.current_employees, c.max_employees;

-- Create functions for common operations
CREATE OR REPLACE FUNCTION get_employee_memory_context(
    p_employee_id VARCHAR(50),
    p_limit INTEGER DEFAULT 10
) RETURNS TABLE(
    memory_id VARCHAR(50),
    content TEXT,
    importance_score DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT m.id, m.content, m.importance_score, m.created_at
    FROM memories m
    WHERE m.employee_id = p_employee_id
      AND (m.expires_at IS NULL OR m.expires_at > NOW())
    ORDER BY m.importance_score DESC, m.created_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION cleanup_expired_memories()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM memories 
    WHERE expires_at IS NOT NULL AND expires_at < NOW();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create maintenance procedures
CREATE OR REPLACE FUNCTION maintenance_cleanup()
RETURNS TEXT AS $$
DECLARE
    result TEXT := '';
    expired_memories INTEGER;
    old_analytics INTEGER;
    old_audit_logs INTEGER;
BEGIN
    -- Clean up expired memories
    SELECT cleanup_expired_memories() INTO expired_memories;
    result := result || 'Cleaned up ' || expired_memories || ' expired memories. ';
    
    -- Clean up old analytics events (older than 90 days)
    DELETE FROM analytics_events 
    WHERE timestamp < NOW() - INTERVAL '90 days';
    GET DIAGNOSTICS old_analytics = ROW_COUNT;
    result := result || 'Cleaned up ' || old_analytics || ' old analytics events. ';
    
    -- Clean up old audit logs (older than 1 year)
    DELETE FROM audit_logs 
    WHERE changed_at < NOW() - INTERVAL '1 year';
    GET DIAGNOSTICS old_audit_logs = ROW_COUNT;
    result := result || 'Cleaned up ' || old_audit_logs || ' old audit logs. ';
    
    -- Update table statistics
    ANALYZE;
    result := result || 'Updated table statistics.';
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT EXECUTE ON FUNCTION get_employee_memory_context(VARCHAR(50), INTEGER) TO application_role;
GRANT EXECUTE ON FUNCTION cleanup_expired_memories() TO application_role;
GRANT EXECUTE ON FUNCTION maintenance_cleanup() TO application_role;

-- Insert initial configuration data
INSERT INTO clients (tenant_id, name, industry, tier, max_employees, allowed_models, security_level) VALUES
('system', 'System Client', 'Technology', 'enterprise', 1000, ARRAY['gpt-4', 'gpt-3.5-turbo'], 'maximum');

-- Migration completion
INSERT INTO schema_migrations (version, applied_at) VALUES ('001', NOW())
ON CONFLICT (version) DO NOTHING;