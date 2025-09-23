-- Add tenant tag GIN index to memory_contexts.tags for faster tenant filtering
CREATE INDEX IF NOT EXISTS idx_memory_contexts_tags_gin ON forge1_memory.memory_contexts USING GIN(tags);

