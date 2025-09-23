# forge1/backend/forge1/core/memory_models.py
"""
Forge 1 Memory System Data Models

Defines the data structures for the advanced memory system including:
- Memory contexts and embeddings
- Semantic relationships
- Agent memory sharing
- Performance tracking
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any, Union
from enum import Enum
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, validator
import numpy as np

class MemoryType(str, Enum):
    """Types of memory contexts"""
    CONVERSATION = "conversation"
    TASK_EXECUTION = "task_execution"
    KNOWLEDGE = "knowledge"
    EXPERIENCE = "experience"
    SKILL = "skill"
    PREFERENCE = "preference"
    ERROR = "error"
    SUCCESS = "success"

class SecurityLevel(str, Enum):
    """Security levels for memory access"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class RelevanceScore(BaseModel):
    """Relevance scoring for memory retrieval"""
    semantic_similarity: float = Field(ge=0.0, le=1.0)
    temporal_relevance: float = Field(ge=0.0, le=1.0)
    context_relevance: float = Field(ge=0.0, le=1.0)
    usage_frequency: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    
    @validator('overall_score', always=True)
    def calculate_overall_score(cls, v, values):
        """Calculate weighted overall relevance score"""
        weights = {
            'semantic_similarity': 0.4,
            'temporal_relevance': 0.2,
            'context_relevance': 0.3,
            'usage_frequency': 0.1
        }
        
        score = sum(
            values.get(key, 0.0) * weight 
            for key, weight in weights.items()
        )
        return min(max(score, 0.0), 1.0)

class MemoryContext(BaseModel):
    """Core memory context model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    employee_id: str
    session_id: Optional[str] = None
    memory_type: MemoryType
    content: Dict[str, Any]
    summary: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    
    # Vector embeddings
    embeddings: Optional[List[float]] = None
    embedding_model: Optional[str] = "text-embedding-ada-002"
    
    # Relevance and scoring
    relevance_score: Optional[RelevanceScore] = None
    access_count: int = 0
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Security and permissions
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    owner_id: str
    shared_with: List[str] = Field(default_factory=list)
    
    # Metadata
    source: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    
    # Relationships
    parent_memory_id: Optional[str] = None
    related_memory_ids: List[str] = Field(default_factory=list)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def update_access(self):
        """Update access tracking"""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
    
    def is_expired(self) -> bool:
        """Check if memory has expired"""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def can_access(self, user_id: str) -> bool:
        """Check if user can access this memory"""
        return (
            self.owner_id == user_id or 
            user_id in self.shared_with or
            self.security_level == SecurityLevel.PUBLIC
        )

class MemoryQuery(BaseModel):
    """Query model for memory retrieval"""
    query_text: Optional[str] = None
    query_embedding: Optional[List[float]] = None
    memory_types: Optional[List[MemoryType]] = None
    employee_ids: Optional[List[str]] = None
    session_ids: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    
    # Filtering
    min_relevance_score: float = 0.0
    max_age_days: Optional[int] = None
    security_levels: Optional[List[SecurityLevel]] = None
    
    # Pagination and limits
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    
    # Sorting
    sort_by: str = "relevance_score"  # relevance_score, created_at, last_accessed
    sort_order: str = "desc"  # asc, desc

class MemorySearchResult(BaseModel):
    """Result model for memory search"""
    memory: MemoryContext
    similarity_score: float = Field(ge=0.0, le=1.0)
    rank: int
    explanation: Optional[str] = None

class MemorySearchResponse(BaseModel):
    """Response model for memory search"""
    results: List[MemorySearchResult]
    total_count: int
    query_time_ms: float
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class MemoryStats(BaseModel):
    """Memory statistics model"""
    total_memories: int = 0
    memories_by_type: Dict[MemoryType, int] = Field(default_factory=dict)
    memories_by_security_level: Dict[SecurityLevel, int] = Field(default_factory=dict)
    average_relevance_score: float = 0.0
    most_accessed_memory_id: Optional[str] = None
    oldest_memory_date: Optional[datetime] = None
    newest_memory_date: Optional[datetime] = None
    total_storage_mb: float = 0.0

class MemoryConflict(BaseModel):
    """Model for memory conflicts"""
    conflict_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_ids: List[str]
    conflict_type: str  # "contradiction", "duplication", "inconsistency"
    description: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    resolution_strategy: Optional[str] = None
    resolved: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None

class MemoryPruningRule(BaseModel):
    """Rules for memory pruning"""
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    conditions: Dict[str, Any]  # Conditions for pruning
    action: str  # "delete", "archive", "compress"
    priority: int = Field(ge=1, le=10)  # 1 = highest priority
    enabled: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class MemoryShare(BaseModel):
    """Model for memory sharing between agents"""
    share_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_id: str
    from_employee_id: str
    to_employee_id: str
    share_type: str  # "copy", "reference", "temporary"
    permissions: List[str] = Field(default_factory=lambda: ["read"])  # read, write, delete
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: Optional[datetime] = None

@dataclass
class EmbeddingRequest:
    """Request for generating embeddings"""
    text: str
    model: str = "text-embedding-ada-002"
    user_id: Optional[str] = None
    cache_key: Optional[str] = None

@dataclass
class EmbeddingResponse:
    """Response from embedding generation"""
    embedding: List[float]
    model: str
    token_count: int
    processing_time_ms: float
    cached: bool = False

class MemoryOptimizationResult(BaseModel):
    """Result of memory optimization operations"""
    operation: str  # "pruning", "compression", "deduplication"
    memories_processed: int
    memories_removed: int
    memories_compressed: int
    storage_saved_mb: float
    processing_time_ms: float
    details: Dict[str, Any] = Field(default_factory=dict)

class MemoryBackup(BaseModel):
    """Model for memory backup operations"""
    backup_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    employee_id: str
    backup_type: str  # "full", "incremental", "selective"
    memory_count: int
    file_path: str
    file_size_mb: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

# Database table schemas for PostgreSQL
MEMORY_TABLES_SQL = """
-- Enhanced memory contexts table
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

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_memory_contexts_employee_id ON forge1_memory.memory_contexts(employee_id);
CREATE INDEX IF NOT EXISTS idx_memory_contexts_session_id ON forge1_memory.memory_contexts(session_id);
CREATE INDEX IF NOT EXISTS idx_memory_contexts_type ON forge1_memory.memory_contexts(memory_type);
CREATE INDEX IF NOT EXISTS idx_memory_contexts_owner ON forge1_memory.memory_contexts(owner_id);
CREATE INDEX IF NOT EXISTS idx_memory_contexts_created_at ON forge1_memory.memory_contexts(created_at);
CREATE INDEX IF NOT EXISTS idx_memory_contexts_relevance ON forge1_memory.memory_contexts(overall_relevance_score);
CREATE INDEX IF NOT EXISTS idx_memory_contexts_keywords ON forge1_memory.memory_contexts USING GIN(keywords);
CREATE INDEX IF NOT EXISTS idx_memory_contexts_tags ON forge1_memory.memory_contexts USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_memory_contexts_security ON forge1_memory.memory_contexts(security_level);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_memory_contexts_content_search ON forge1_memory.memory_contexts USING GIN(to_tsvector('english', content::text));
CREATE INDEX IF NOT EXISTS idx_memory_contexts_summary_search ON forge1_memory.memory_contexts USING GIN(to_tsvector('english', summary));

-- Triggers for updated_at
CREATE OR REPLACE FUNCTION update_memory_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_memory_contexts_updated_at 
    BEFORE UPDATE ON forge1_memory.memory_contexts 
    FOR EACH ROW EXECUTE FUNCTION update_memory_updated_at();
"""

# Export all models and constants
__all__ = [
    "MemoryType",
    "SecurityLevel", 
    "RelevanceScore",
    "MemoryContext",
    "MemoryQuery",
    "MemorySearchResult",
    "MemorySearchResponse",
    "MemoryStats",
    "MemoryConflict",
    "MemoryPruningRule",
    "MemoryShare",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "MemoryOptimizationResult",
    "MemoryBackup",
    "MEMORY_TABLES_SQL"
]