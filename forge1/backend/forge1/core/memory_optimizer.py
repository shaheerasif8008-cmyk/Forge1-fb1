# forge1/backend/forge1/core/memory_optimizer.py
"""
Forge 1 Memory Optimization System

Implements intelligent memory optimization including:
- Memory pruning algorithms based on relevance and age
- Memory sharing mechanisms between agents
- Conflict resolution for inconsistent memories
- Memory compression and deduplication
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

import numpy as np

from forge1.core.database_config import DatabaseManager, get_database_manager
from forge1.core.memory_models import (
    MemoryContext, MemoryConflict, MemoryPruningRule, MemoryShare,
    MemoryOptimizationResult, MemoryType, SecurityLevel
)
from forge1.core.memory_manager import MemoryManager, get_memory_manager

logger = logging.getLogger(__name__)

class PruningStrategy(str, Enum):
    """Memory pruning strategies"""
    AGE_BASED = "age_based"
    RELEVANCE_BASED = "relevance_based"
    ACCESS_BASED = "access_based"
    SIZE_BASED = "size_based"
    HYBRID = "hybrid"

class ConflictType(str, Enum):
    """Types of memory conflicts"""
    CONTRADICTION = "contradiction"
    DUPLICATION = "duplication"
    INCONSISTENCY = "inconsistency"
    OUTDATED = "outdated"

class ShareType(str, Enum):
    """Types of memory sharing"""
    COPY = "copy"
    REFERENCE = "reference"
    TEMPORARY = "temporary"
    SYNCHRONIZED = "synchronized"

@dataclass
class PruningCriteria:
    """Criteria for memory pruning decisions"""
    max_age_days: Optional[int] = None
    min_relevance_score: Optional[float] = None
    min_access_count: Optional[int] = None
    max_storage_mb: Optional[float] = None
    preserve_types: List[MemoryType] = None
    preserve_security_levels: List[SecurityLevel] = None

@dataclass
class ConflictResolution:
    """Resolution for memory conflicts"""
    conflict_id: str
    resolution_type: str  # "merge", "keep_newest", "keep_highest_score", "manual"
    winning_memory_id: Optional[str] = None
    merged_memory_id: Optional[str] = None
    confidence: float = 0.0

class MemoryOptimizer:
    """Intelligent memory optimization and management"""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None, memory_manager: Optional[MemoryManager] = None):
        self.db_manager = db_manager
        self.memory_manager = memory_manager
        self._initialized = False
        
        # Default pruning rules
        self.default_pruning_rules = [
            MemoryPruningRule(
                name="Old Low-Relevance Cleanup",
                description="Remove memories older than 90 days with low relevance",
                conditions={
                    "max_age_days": 90,
                    "max_relevance_score": 0.3,
                    "exclude_types": ["knowledge", "skill"]
                },
                action="delete",
                priority=3
            ),
            MemoryPruningRule(
                name="Unused Memory Cleanup", 
                description="Remove memories not accessed in 30 days with low access count",
                conditions={
                    "max_age_days": 30,
                    "max_access_count": 2,
                    "exclude_security_levels": ["confidential", "restricted"]
                },
                action="archive",
                priority=5
            ),
            MemoryPruningRule(
                name="Duplicate Conversation Cleanup",
                description="Compress duplicate conversation memories",
                conditions={
                    "memory_type": "conversation",
                    "similarity_threshold": 0.95
                },
                action="compress",
                priority=7
            )
        ]
    
    async def initialize(self):
        """Initialize the memory optimizer"""
        if self._initialized:
            return
        
        if not self.db_manager:
            self.db_manager = await get_database_manager()
        
        if not self.memory_manager:
            self.memory_manager = await get_memory_manager()
        
        # Install default pruning rules
        await self._install_default_pruning_rules()
        
        self._initialized = True
        logger.info("Memory optimizer initialized")
    
    async def optimize_memories(self, employee_id: Optional[str] = None, user_id: Optional[str] = None) -> MemoryOptimizationResult:
        """Run comprehensive memory optimization"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        total_processed = 0
        total_removed = 0
        total_compressed = 0
        storage_saved = 0.0
        
        try:
            logger.info(f"Starting memory optimization for employee: {employee_id}, user: {user_id}")
            
            # Step 1: Detect and resolve conflicts
            conflicts_resolved = await self.detect_and_resolve_conflicts(employee_id, user_id)
            logger.info(f"Resolved {len(conflicts_resolved)} memory conflicts")
            
            # Step 2: Run pruning algorithms
            pruning_result = await self.prune_memories(employee_id, user_id)
            total_processed += pruning_result.memories_processed
            total_removed += pruning_result.memories_removed
            storage_saved += pruning_result.storage_saved_mb
            
            # Step 3: Compress similar memories
            compression_result = await self.compress_similar_memories(employee_id, user_id)
            total_processed += compression_result.memories_processed
            total_compressed += compression_result.memories_compressed
            storage_saved += compression_result.storage_saved_mb
            
            # Step 4: Deduplicate exact matches
            dedup_result = await self.deduplicate_memories(employee_id, user_id)
            total_processed += dedup_result.memories_processed
            total_removed += dedup_result.memories_removed
            storage_saved += dedup_result.storage_saved_mb
            
            processing_time = (time.time() - start_time) * 1000
            
            result = MemoryOptimizationResult(
                operation="comprehensive_optimization",
                memories_processed=total_processed,
                memories_removed=total_removed,
                memories_compressed=total_compressed,
                storage_saved_mb=storage_saved,
                processing_time_ms=processing_time,
                details={
                    "conflicts_resolved": len(conflicts_resolved),
                    "pruning_result": pruning_result.dict(),
                    "compression_result": compression_result.dict(),
                    "deduplication_result": dedup_result.dict()
                }
            )
            
            logger.info(f"Memory optimization completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return MemoryOptimizationResult(
                operation="comprehensive_optimization",
                memories_processed=0,
                memories_removed=0,
                memories_compressed=0,
                storage_saved_mb=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)}
            )
    
    async def prune_memories(self, employee_id: Optional[str] = None, user_id: Optional[str] = None) -> MemoryOptimizationResult:
        """Prune memories based on configured rules"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        memories_processed = 0
        memories_removed = 0
        storage_saved = 0.0
        
        try:
            # Get active pruning rules
            pruning_rules = await self._get_active_pruning_rules()
            
            for rule in sorted(pruning_rules, key=lambda r: r.priority):
                logger.info(f"Applying pruning rule: {rule.name}")
                
                # Find memories matching rule conditions
                candidate_memories = await self._find_memories_for_pruning(rule, employee_id, user_id)
                
                for memory in candidate_memories:
                    memories_processed += 1
                    
                    if rule.action == "delete":
                        success = await self.memory_manager.delete_memory(memory.id, memory.owner_id)
                        if success:
                            memories_removed += 1
                            storage_saved += self._estimate_memory_size(memory)
                            logger.debug(f"Deleted memory {memory.id} via rule {rule.name}")
                    
                    elif rule.action == "archive":
                        # Move to archive (could be separate table or mark as archived)
                        await self._archive_memory(memory)
                        memories_removed += 1
                        storage_saved += self._estimate_memory_size(memory) * 0.5  # Assume 50% compression
                        logger.debug(f"Archived memory {memory.id} via rule {rule.name}")
                    
                    elif rule.action == "compress":
                        # Compress memory content
                        compressed_size = await self._compress_memory(memory)
                        if compressed_size > 0:
                            storage_saved += compressed_size
                            logger.debug(f"Compressed memory {memory.id} via rule {rule.name}")
            
            processing_time = (time.time() - start_time) * 1000
            
            return MemoryOptimizationResult(
                operation="pruning",
                memories_processed=memories_processed,
                memories_removed=memories_removed,
                memories_compressed=0,
                storage_saved_mb=storage_saved,
                processing_time_ms=processing_time,
                details={"rules_applied": len(pruning_rules)}
            )
            
        except Exception as e:
            logger.error(f"Memory pruning failed: {e}")
            return MemoryOptimizationResult(
                operation="pruning",
                memories_processed=0,
                memories_removed=0,
                memories_compressed=0,
                storage_saved_mb=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)}
            )
    
    async def detect_and_resolve_conflicts(self, employee_id: Optional[str] = None, user_id: Optional[str] = None) -> List[ConflictResolution]:
        """Detect and resolve memory conflicts"""
        if not self._initialized:
            await self.initialize()
        
        resolutions = []
        
        try:
            # Detect contradictions
            contradictions = await self._detect_contradictions(employee_id, user_id)
            for conflict in contradictions:
                resolution = await self._resolve_contradiction(conflict)
                if resolution:
                    resolutions.append(resolution)
            
            # Detect duplications
            duplications = await self._detect_duplications(employee_id, user_id)
            for conflict in duplications:
                resolution = await self._resolve_duplication(conflict)
                if resolution:
                    resolutions.append(resolution)
            
            # Detect inconsistencies
            inconsistencies = await self._detect_inconsistencies(employee_id, user_id)
            for conflict in inconsistencies:
                resolution = await self._resolve_inconsistency(conflict)
                if resolution:
                    resolutions.append(resolution)
            
            logger.info(f"Resolved {len(resolutions)} memory conflicts")
            return resolutions
            
        except Exception as e:
            logger.error(f"Conflict detection and resolution failed: {e}")
            return []
    
    async def share_memory(self, memory_id: str, from_employee_id: str, to_employee_id: str, 
                          share_type: ShareType, permissions: List[str], user_id: str,
                          expires_at: Optional[datetime] = None) -> Optional[str]:
        """Share memory between agents"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Verify source memory exists and user has permission
            source_memory = await self.memory_manager.retrieve_memory(memory_id, user_id)
            if not source_memory:
                logger.warning(f"Cannot share memory {memory_id}: not found or no access")
                return None
            
            # Check if sharing is allowed based on security level
            if source_memory.security_level == SecurityLevel.RESTRICTED:
                logger.warning(f"Cannot share restricted memory {memory_id}")
                return None
            
            # Create memory share record
            share = MemoryShare(
                memory_id=memory_id,
                from_employee_id=from_employee_id,
                to_employee_id=to_employee_id,
                share_type=share_type.value,
                permissions=permissions,
                expires_at=expires_at
            )
            
            # Store share record
            async with self.db_manager.postgres.acquire() as conn:
                share_id = await conn.fetchval("""
                    INSERT INTO forge1_memory.memory_shares (
                        share_id, memory_id, from_employee_id, to_employee_id,
                        share_type, permissions, expires_at, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING share_id
                """, share.share_id, share.memory_id, share.from_employee_id,
                    share.to_employee_id, share.share_type, share.permissions,
                    share.expires_at, share.created_at)
            
            # Handle different share types
            if share_type == ShareType.COPY:
                # Create a copy of the memory for the target agent
                copied_memory = source_memory.copy()
                copied_memory.id = str(uuid.uuid4())
                copied_memory.employee_id = to_employee_id
                copied_memory.parent_memory_id = memory_id
                copied_memory.source = f"shared_from_{from_employee_id}"
                
                await self.memory_manager.store_memory(copied_memory)
                logger.info(f"Copied memory {memory_id} to employee {to_employee_id}")
            
            elif share_type == ShareType.REFERENCE:
                # Add target employee to shared_with list
                await self.memory_manager.update_memory(
                    memory_id, 
                    {"shared_with": source_memory.shared_with + [to_employee_id]},
                    user_id
                )
                logger.info(f"Shared memory {memory_id} reference with employee {to_employee_id}")
            
            elif share_type == ShareType.SYNCHRONIZED:
                # Create bidirectional sharing (both agents can modify)
                source_memory.shared_with.append(to_employee_id)
                await self.memory_manager.update_memory(
                    memory_id,
                    {"shared_with": source_memory.shared_with},
                    user_id
                )
                logger.info(f"Created synchronized sharing for memory {memory_id}")
            
            return share_id
            
        except Exception as e:
            logger.error(f"Memory sharing failed: {e}")
            return None
    
    async def compress_similar_memories(self, employee_id: Optional[str] = None, user_id: Optional[str] = None) -> MemoryOptimizationResult:
        """Compress similar memories to save storage"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        memories_processed = 0
        memories_compressed = 0
        storage_saved = 0.0
        
        try:
            # Find groups of similar memories
            similarity_groups = await self._find_similar_memory_groups(employee_id, user_id, threshold=0.85)
            
            for group in similarity_groups:
                if len(group) < 2:
                    continue
                
                memories_processed += len(group)
                
                # Compress group into a single representative memory
                compressed_memory = await self._compress_memory_group(group)
                if compressed_memory:
                    # Remove original memories except the representative
                    for memory in group[1:]:  # Keep first as representative
                        success = await self.memory_manager.delete_memory(memory.id, memory.owner_id)
                        if success:
                            memories_compressed += 1
                            storage_saved += self._estimate_memory_size(memory)
                    
                    logger.info(f"Compressed {len(group)} similar memories into one")
            
            processing_time = (time.time() - start_time) * 1000
            
            return MemoryOptimizationResult(
                operation="compression",
                memories_processed=memories_processed,
                memories_removed=0,
                memories_compressed=memories_compressed,
                storage_saved_mb=storage_saved,
                processing_time_ms=processing_time,
                details={"similarity_groups": len(similarity_groups)}
            )
            
        except Exception as e:
            logger.error(f"Memory compression failed: {e}")
            return MemoryOptimizationResult(
                operation="compression",
                memories_processed=0,
                memories_removed=0,
                memories_compressed=0,
                storage_saved_mb=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)}
            )
    
    async def deduplicate_memories(self, employee_id: Optional[str] = None, user_id: Optional[str] = None) -> MemoryOptimizationResult:
        """Remove exact duplicate memories"""
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        memories_processed = 0
        memories_removed = 0
        storage_saved = 0.0
        
        try:
            # Find exact duplicates based on content hash
            duplicates = await self._find_duplicate_memories(employee_id, user_id)
            
            for duplicate_group in duplicates:
                if len(duplicate_group) < 2:
                    continue
                
                memories_processed += len(duplicate_group)
                
                # Keep the most recent or most accessed memory
                keeper = max(duplicate_group, key=lambda m: (m.access_count, m.created_at))
                
                # Remove duplicates
                for memory in duplicate_group:
                    if memory.id != keeper.id:
                        success = await self.memory_manager.delete_memory(memory.id, memory.owner_id)
                        if success:
                            memories_removed += 1
                            storage_saved += self._estimate_memory_size(memory)
                
                logger.info(f"Removed {len(duplicate_group) - 1} duplicates, kept {keeper.id}")
            
            processing_time = (time.time() - start_time) * 1000
            
            return MemoryOptimizationResult(
                operation="deduplication",
                memories_processed=memories_processed,
                memories_removed=memories_removed,
                memories_compressed=0,
                storage_saved_mb=storage_saved,
                processing_time_ms=processing_time,
                details={"duplicate_groups": len(duplicates)}
            )
            
        except Exception as e:
            logger.error(f"Memory deduplication failed: {e}")
            return MemoryOptimizationResult(
                operation="deduplication",
                memories_processed=0,
                memories_removed=0,
                memories_compressed=0,
                storage_saved_mb=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)}
            )
    
    # Helper methods
    
    async def _install_default_pruning_rules(self):
        """Install default pruning rules if they don't exist"""
        try:
            async with self.db_manager.postgres.acquire() as conn:
                for rule in self.default_pruning_rules:
                    # Check if rule already exists
                    exists = await conn.fetchval("""
                        SELECT EXISTS(
                            SELECT 1 FROM forge1_memory.memory_pruning_rules 
                            WHERE name = $1
                        )
                    """, rule.name)
                    
                    if not exists:
                        await conn.execute("""
                            INSERT INTO forge1_memory.memory_pruning_rules (
                                rule_id, name, description, conditions, action, priority, enabled
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """, rule.rule_id, rule.name, rule.description,
                            json.dumps(rule.conditions), rule.action, rule.priority, rule.enabled)
                        
                        logger.info(f"Installed pruning rule: {rule.name}")
                
        except Exception as e:
            logger.error(f"Failed to install default pruning rules: {e}")
    
    async def _get_active_pruning_rules(self) -> List[MemoryPruningRule]:
        """Get active pruning rules from database"""
        try:
            async with self.db_manager.postgres.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM forge1_memory.memory_pruning_rules 
                    WHERE enabled = true 
                    ORDER BY priority ASC
                """)
                
                rules = []
                for row in rows:
                    rule = MemoryPruningRule(
                        rule_id=str(row['rule_id']),
                        name=row['name'],
                        description=row['description'],
                        conditions=row['conditions'],
                        action=row['action'],
                        priority=row['priority'],
                        enabled=row['enabled'],
                        created_at=row['created_at']
                    )
                    rules.append(rule)
                
                return rules
                
        except Exception as e:
            logger.error(f"Failed to get pruning rules: {e}")
            return []
    
    async def _find_memories_for_pruning(self, rule: MemoryPruningRule, employee_id: Optional[str], user_id: Optional[str]) -> List[MemoryContext]:
        """Find memories that match pruning rule conditions"""
        try:
            conditions = rule.conditions
            sql_conditions = ["1=1"]
            sql_params = []
            param_count = 0
            
            # Employee filter
            if employee_id:
                param_count += 1
                sql_conditions.append(f"employee_id = ${param_count}")
                sql_params.append(employee_id)
            
            # User filter
            if user_id:
                param_count += 1
                sql_conditions.append(f"owner_id = ${param_count}")
                sql_params.append(user_id)
            
            # Age condition
            if "max_age_days" in conditions:
                param_count += 1
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=conditions["max_age_days"])
                sql_conditions.append(f"created_at < ${param_count}")
                sql_params.append(cutoff_date)
            
            # Relevance condition
            if "max_relevance_score" in conditions:
                param_count += 1
                sql_conditions.append(f"overall_relevance_score <= ${param_count}")
                sql_params.append(conditions["max_relevance_score"])
            
            # Access count condition
            if "max_access_count" in conditions:
                param_count += 1
                sql_conditions.append(f"access_count <= ${param_count}")
                sql_params.append(conditions["max_access_count"])
            
            # Memory type exclusions
            if "exclude_types" in conditions:
                param_count += 1
                sql_conditions.append(f"memory_type != ALL(${param_count})")
                sql_params.append(conditions["exclude_types"])
            
            # Security level exclusions
            if "exclude_security_levels" in conditions:
                param_count += 1
                sql_conditions.append(f"security_level != ALL(${param_count})")
                sql_params.append(conditions["exclude_security_levels"])
            
            where_clause = " AND ".join(sql_conditions)
            
            async with self.db_manager.postgres.acquire() as conn:
                rows = await conn.fetch(f"""
                    SELECT * FROM forge1_memory.memory_contexts 
                    WHERE {where_clause}
                    LIMIT 1000
                """, *sql_params)
                
                memories = []
                for row in rows:
                    memory = self.memory_manager._row_to_memory_context(row)
                    memories.append(memory)
                
                return memories
                
        except Exception as e:
            logger.error(f"Failed to find memories for pruning: {e}")
            return []
    
    async def _detect_contradictions(self, employee_id: Optional[str], user_id: Optional[str]) -> List[MemoryConflict]:
        """Detect contradictory memories"""
        # This would implement semantic analysis to find contradictions
        # For now, return empty list as placeholder
        return []
    
    async def _detect_duplications(self, employee_id: Optional[str], user_id: Optional[str]) -> List[MemoryConflict]:
        """Detect duplicate memories"""
        # This would find memories with high semantic similarity
        # For now, return empty list as placeholder
        return []
    
    async def _detect_inconsistencies(self, employee_id: Optional[str], user_id: Optional[str]) -> List[MemoryConflict]:
        """Detect inconsistent memories"""
        # This would find memories with conflicting information
        # For now, return empty list as placeholder
        return []
    
    async def _resolve_contradiction(self, conflict: MemoryConflict) -> Optional[ConflictResolution]:
        """Resolve a contradiction conflict"""
        # Placeholder implementation
        return None
    
    async def _resolve_duplication(self, conflict: MemoryConflict) -> Optional[ConflictResolution]:
        """Resolve a duplication conflict"""
        # Placeholder implementation
        return None
    
    async def _resolve_inconsistency(self, conflict: MemoryConflict) -> Optional[ConflictResolution]:
        """Resolve an inconsistency conflict"""
        # Placeholder implementation
        return None
    
    async def _find_similar_memory_groups(self, employee_id: Optional[str], user_id: Optional[str], threshold: float = 0.85) -> List[List[MemoryContext]]:
        """Find groups of similar memories"""
        # This would use embeddings to find similar memories
        # For now, return empty list as placeholder
        return []
    
    async def _find_duplicate_memories(self, employee_id: Optional[str], user_id: Optional[str]) -> List[List[MemoryContext]]:
        """Find exact duplicate memories"""
        # This would find memories with identical content hashes
        # For now, return empty list as placeholder
        return []
    
    async def _compress_memory_group(self, memories: List[MemoryContext]) -> Optional[MemoryContext]:
        """Compress a group of similar memories into one"""
        # This would merge similar memories into a single representative
        # For now, return the first memory as placeholder
        return memories[0] if memories else None
    
    async def _archive_memory(self, memory: MemoryContext):
        """Archive a memory (move to cold storage)"""
        # This would move memory to archive table or mark as archived
        # For now, just log the action
        logger.info(f"Archiving memory {memory.id}")
    
    async def _compress_memory(self, memory: MemoryContext) -> float:
        """Compress memory content and return storage saved"""
        # This would compress the memory content
        # For now, return estimated savings
        return self._estimate_memory_size(memory) * 0.3  # Assume 30% compression
    
    def _estimate_memory_size(self, memory: MemoryContext) -> float:
        """Estimate memory storage size in MB"""
        # Rough estimation based on content size
        content_size = len(json.dumps(memory.content).encode('utf-8'))
        embedding_size = len(memory.embeddings) * 4 if memory.embeddings else 0  # 4 bytes per float
        metadata_size = 1024  # Rough estimate for metadata
        
        total_bytes = content_size + embedding_size + metadata_size
        return total_bytes / (1024 * 1024)  # Convert to MB

# Global memory optimizer instance
_memory_optimizer: Optional[MemoryOptimizer] = None

async def get_memory_optimizer() -> MemoryOptimizer:
    """Get or create the global memory optimizer instance"""
    global _memory_optimizer
    
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
        await _memory_optimizer.initialize()
    
    return _memory_optimizer

# Export main classes
__all__ = [
    "MemoryOptimizer",
    "PruningStrategy",
    "ConflictType", 
    "ShareType",
    "PruningCriteria",
    "ConflictResolution",
    "get_memory_optimizer"
]