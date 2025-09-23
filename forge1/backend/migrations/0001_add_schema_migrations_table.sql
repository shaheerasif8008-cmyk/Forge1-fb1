-- Create schema_migrations tracking table
CREATE SCHEMA IF NOT EXISTS forge1_migrations;
CREATE TABLE IF NOT EXISTS forge1_migrations.schema_migrations (
    id SERIAL PRIMARY KEY,
    filename TEXT UNIQUE NOT NULL,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

