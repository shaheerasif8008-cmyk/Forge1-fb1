-- Schema migrations tracking table
-- This table tracks which migrations have been applied to the database

CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(20) PRIMARY KEY,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    applied_by VARCHAR(100) DEFAULT current_user,
    description TEXT,
    checksum VARCHAR(64)
);

-- Insert initial migration tracking
INSERT INTO schema_migrations (version, description) VALUES 
('000', 'Schema migrations tracking table created')
ON CONFLICT (version) DO NOTHING;