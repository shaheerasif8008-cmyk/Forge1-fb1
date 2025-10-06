"""
Weaviate Vector Database Integration Adapter

Provides tenant-aware Weaviate client with schema management, vector operations,
and comprehensive tenant isolation for Forge1 memory management.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum

import weaviate
from weaviate.client import Client
from weaviate.exceptions import WeaviateException

from forge1.integrations.base_adapter import BaseAdapter, HealthCheckResult, AdapterStatus, ExecutionContext, TenantContext
from forge1.config.integration_settings import IntegrationType, settings_manager
from forge1.core.tenancy import get_current_tenant
from forge1.core.dlp import redact_payload

logger = logging.getLogger(__name__)

class VectorOperation(Enum):
    """Vector operation types"""
    CREATE_SCHEMA = "create_schema"
    STORE_VECTOR = "store_vector"
    SEARCH_VECTORS = "search_vectors"
    DELETE_VECTOR = "delete_vector"
    UPDATE_VECTOR = "update_vector"
    BATCH_IMPORT = "batch_import"

@dataclass
class VectorData:
    """Vector data structure"""
    id: str
    vector: List[float]
    properties: Dict[str, Any]
    class_name: str
    tenant_id: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class VectorQuery:
    """Vector search query"""
    vector: Optional[List[float]] = None
    class_name: str = "Memory"
    where_filter: Optional[Dict[str, Any]] = None
    limit: int = 10
    offset: int = 0
    certainty: float = 0.7
    distance: Optional[float] = None
    tenant_id: Optional[str] = None
    additional_properties: List[str] = None
    
    def __post_init__(self):
        if self.additional_properties is None:
            self.additional_properties = []

@dataclass
class VectorResult:
    """Vector search result"""
    id: str
    properties: Dict[str, Any]
    certainty: float
    distance: Optional[float]
    vector: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class SchemaDefinition:
    """Weaviate schema definition"""
    class_name: str
    description: str
    properties: List[Dict[str, Any]]
    vectorizer: str = "none"
    vector_index_type: str = "hnsw"
    vector_index_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.vector_index_config is None:
            self.vector_index_config = {
                "ef": 64,
                "efConstruction": 128,
                "maxConnections": 64
            }

class WeaviateAdapter(BaseAdapter):
    """Weaviate vector database adapter with tenant isolation"""
    
    def __init__(self):
        config = settings_manager.get_config(IntegrationType.VECTOR)
        super().__init__("weaviate", config)
        
        self.weaviate_config = config
        self.client: Optional[Client] = None
        self._tenant_schemas: Dict[str, List[str]] = {}  # tenant_id -> [class_names]
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
        self._connection_retries = 0
        self._max_retries = 3
    
    async def initialize(self) -> bool:
        """Initialize Weaviate client and connection"""
        try:
            # Create Weaviate client
            auth_config = None
            if self.weaviate_config.auth_client_secret:
                auth_config = weaviate.AuthApiKey(api_key=self.weaviate_config.auth_client_secret)
            
            self.client = weaviate.Client(
                url=f"{self.weaviate_config.scheme}://{self.weaviate_config.host}:{self.weaviate_config.port}",
                auth_client_secret=auth_config,
                timeout_config=self.weaviate_config.timeout_config,
                additional_headers=self.weaviate_config.additional_headers,
                startup_period=self.weaviate_config.startup_period
            )
            
            # Test connection
            if not self.client.is_ready():
                raise ConnectionError("Weaviate is not ready")
            
            # Initialize base schemas
            await self._initialize_base_schemas()
            
            # Load existing tenant schemas
            await self._load_existing_schemas()
            
            logger.info("Weaviate adapter initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate adapter: {e}")
            return False
    
    async def health_check(self) -> HealthCheckResult:
        """Perform health check of Weaviate"""
        start_time = time.time()
        
        try:
            if not self.client:
                return HealthCheckResult(
                    status=AdapterStatus.UNHEALTHY,
                    message="Weaviate client not initialized",
                    details={},
                    timestamp=time.time(),
                    response_time_ms=0
                )
            
            # Check if Weaviate is ready
            is_ready = self.client.is_ready()
            
            if not is_ready:
                return HealthCheckResult(
                    status=AdapterStatus.UNHEALTHY,
                    message="Weaviate is not ready",
                    details={"ready": False},
                    timestamp=time.time(),
                    response_time_ms=(time.time() - start_time) * 1000
                )
            
            # Get cluster metadata
            meta = self.client.get_meta()
            
            # Get schema information
            schema = self.client.schema.get()
            class_count = len(schema.get("classes", []))
            
            # Check node status
            nodes_status = self.client.cluster.get_nodes_status()
            healthy_nodes = sum(1 for node in nodes_status if node.get("status") == "HEALTHY")
            total_nodes = len(nodes_status)
            
            # Determine status
            if healthy_nodes == 0:
                status = AdapterStatus.UNHEALTHY
                message = "No healthy Weaviate nodes"
            elif healthy_nodes < total_nodes:
                status = AdapterStatus.DEGRADED
                message = f"Weaviate partially healthy ({healthy_nodes}/{total_nodes} nodes)"
            else:
                status = AdapterStatus.HEALTHY
                message = f"Weaviate healthy ({healthy_nodes} nodes)"
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                status=status,
                message=message,
                details={
                    "version": meta.get("version", "unknown"),
                    "hostname": meta.get("hostname", "unknown"),
                    "class_count": class_count,
                    "healthy_nodes": healthy_nodes,
                    "total_nodes": total_nodes,
                    "tenant_schemas": len(self._tenant_schemas),
                    "connection_retries": self._connection_retries
                },
                timestamp=time.time(),
                response_time_ms=response_time
            )
            
        except Exception as e:
            self._connection_retries += 1
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                status=AdapterStatus.UNHEALTHY,
                message=f"Weaviate health check failed: {str(e)}",
                details={
                    "error": str(e),
                    "connection_retries": self._connection_retries
                },
                timestamp=time.time(),
                response_time_ms=response_time
            )
    
    async def cleanup(self) -> bool:
        """Clean up Weaviate resources"""
        try:
            if self.client:
                # Weaviate client doesn't have explicit cleanup
                self.client = None
            
            logger.info("Weaviate adapter cleaned up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup Weaviate adapter: {e}")
            return False
    
    async def create_tenant_schema(self, tenant_id: str, schema_definitions: Optional[List[SchemaDefinition]] = None) -> bool:
        """Create tenant-specific schema classes"""
        
        if not await self.ensure_initialized():
            raise RuntimeError("Weaviate adapter not initialized")
        
        try:
            if not schema_definitions:
                # Use default memory schema
                schema_definitions = [self._get_default_memory_schema(tenant_id)]
            
            created_classes = []
            
            for schema_def in schema_definitions:
                tenant_class_name = self._get_tenant_class_name(schema_def.class_name, tenant_id)
                
                # Check if class already exists
                if self.client.schema.exists(tenant_class_name):
                    logger.info(f"Schema class {tenant_class_name} already exists")
                    created_classes.append(tenant_class_name)
                    continue
                
                # Create class definition
                class_definition = {
                    "class": tenant_class_name,
                    "description": f"{schema_def.description} (Tenant: {tenant_id})",
                    "properties": schema_def.properties,
                    "vectorizer": schema_def.vectorizer,
                    "vectorIndexType": schema_def.vector_index_type,
                    "vectorIndexConfig": schema_def.vector_index_config
                }
                
                # Create the class
                self.client.schema.create_class(class_definition)
                created_classes.append(tenant_class_name)
                
                logger.info(f"Created schema class {tenant_class_name}")
            
            # Update tenant schema tracking
            if tenant_id not in self._tenant_schemas:
                self._tenant_schemas[tenant_id] = []
            
            self._tenant_schemas[tenant_id].extend(created_classes)
            
            # Cache schema definitions
            for class_name in created_classes:
                self._schema_cache[class_name] = self.client.schema.get(class_name)
            
            logger.info(f"Created {len(created_classes)} schema classes for tenant {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create tenant schema for {tenant_id}: {e}")
            return False
    
    async def store_vector(self, vector_data: VectorData, context: Optional[ExecutionContext] = None) -> str:
        """Store vector data with tenant isolation"""
        
        if not await self.ensure_initialized():
            raise RuntimeError("Weaviate adapter not initialized")
        
        if not context:
            context = self.create_execution_context("store_vector")
        
        tenant_class_name = self._get_tenant_class_name(vector_data.class_name, vector_data.tenant_id)
        
        try:
            start_time = time.time()
            
            # Ensure tenant schema exists
            if not self.client.schema.exists(tenant_class_name):
                await self.create_tenant_schema(vector_data.tenant_id)
            
            # Apply DLP redaction to properties
            safe_properties, violations = redact_payload(vector_data.properties)
            
            # Add tenant metadata
            safe_properties["tenant_id"] = vector_data.tenant_id
            safe_properties["created_at"] = time.time()
            
            if violations:
                safe_properties["dlp_violations"] = len(violations)
                context.metadata["dlp_violations"] = len(violations)
            
            # Store the vector
            with self.client.batch as batch:
                batch.add_data_object(
                    data_object=safe_properties,
                    class_name=tenant_class_name,
                    uuid=vector_data.id,
                    vector=vector_data.vector
                )
            
            execution_time = (time.time() - start_time) * 1000
            self.record_operation(VectorOperation.STORE_VECTOR.value, execution_time, True)
            
            # Log operation
            self.log_operation(
                "store_vector",
                context,
                duration_ms=execution_time,
                success=True
            )
            
            logger.info(f"Stored vector {vector_data.id} in class {tenant_class_name}")
            return vector_data.id
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.record_operation(VectorOperation.STORE_VECTOR.value, execution_time, False)
            
            self.log_operation(
                "store_vector",
                context,
                duration_ms=execution_time,
                success=False,
                error=str(e)
            )
            
            logger.error(f"Failed to store vector {vector_data.id}: {e}")
            raise
    
    async def search_vectors(self, query: VectorQuery, context: Optional[ExecutionContext] = None) -> List[VectorResult]:
        """Search vectors with tenant isolation"""
        
        if not await self.ensure_initialized():
            raise RuntimeError("Weaviate adapter not initialized")
        
        if not context:
            context = self.create_execution_context("search_vectors")
        
        # Ensure tenant context
        if not query.tenant_id:
            query.tenant_id = context.tenant_context.tenant_id
        
        tenant_class_name = self._get_tenant_class_name(query.class_name, query.tenant_id)
        
        try:
            start_time = time.time()
            
            # Build search query
            search_builder = self.client.query.get(tenant_class_name, query.additional_properties or ["*"])
            
            # Add vector search if provided
            if query.vector:
                if query.certainty:
                    search_builder = search_builder.with_near_vector({
                        "vector": query.vector,
                        "certainty": query.certainty
                    })
                elif query.distance:
                    search_builder = search_builder.with_near_vector({
                        "vector": query.vector,
                        "distance": query.distance
                    })
            
            # Add where filter with tenant isolation
            where_filter = query.where_filter or {}
            where_filter["tenant_id"] = {"equal": query.tenant_id}
            
            search_builder = search_builder.with_where(where_filter)
            
            # Add pagination
            search_builder = search_builder.with_limit(query.limit)
            if query.offset > 0:
                search_builder = search_builder.with_offset(query.offset)
            
            # Add additional metadata
            search_builder = search_builder.with_additional(["certainty", "distance", "id"])
            
            # Execute search
            result = search_builder.do()
            
            # Process results
            results = []
            if "data" in result and "Get" in result["data"] and tenant_class_name in result["data"]["Get"]:
                for item in result["data"]["Get"][tenant_class_name]:
                    additional = item.get("_additional", {})
                    
                    vector_result = VectorResult(
                        id=additional.get("id", ""),
                        properties={k: v for k, v in item.items() if not k.startswith("_")},
                        certainty=additional.get("certainty", 0.0),
                        distance=additional.get("distance"),
                        metadata={
                            "tenant_id": query.tenant_id,
                            "class_name": tenant_class_name
                        }
                    )
                    results.append(vector_result)
            
            execution_time = (time.time() - start_time) * 1000
            self.record_operation(VectorOperation.SEARCH_VECTORS.value, execution_time, True)
            
            # Log operation
            self.log_operation(
                "search_vectors",
                context,
                duration_ms=execution_time,
                success=True
            )
            
            logger.info(f"Vector search returned {len(results)} results from {tenant_class_name}")
            return results
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.record_operation(VectorOperation.SEARCH_VECTORS.value, execution_time, False)
            
            self.log_operation(
                "search_vectors",
                context,
                duration_ms=execution_time,
                success=False,
                error=str(e)
            )
            
            logger.error(f"Vector search failed in {tenant_class_name}: {e}")
            raise
    
    async def delete_vector(self, vector_id: str, class_name: str, tenant_id: str, 
                          context: Optional[ExecutionContext] = None) -> bool:
        """Delete vector with tenant isolation"""
        
        if not await self.ensure_initialized():
            raise RuntimeError("Weaviate adapter not initialized")
        
        if not context:
            context = self.create_execution_context("delete_vector")
        
        tenant_class_name = self._get_tenant_class_name(class_name, tenant_id)
        
        try:
            start_time = time.time()
            
            # Delete the vector
            self.client.data_object.delete(
                uuid=vector_id,
                class_name=tenant_class_name
            )
            
            execution_time = (time.time() - start_time) * 1000
            self.record_operation(VectorOperation.DELETE_VECTOR.value, execution_time, True)
            
            # Log operation
            self.log_operation(
                "delete_vector",
                context,
                duration_ms=execution_time,
                success=True
            )
            
            logger.info(f"Deleted vector {vector_id} from {tenant_class_name}")
            return True
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.record_operation(VectorOperation.DELETE_VECTOR.value, execution_time, False)
            
            self.log_operation(
                "delete_vector",
                context,
                duration_ms=execution_time,
                success=False,
                error=str(e)
            )
            
            logger.error(f"Failed to delete vector {vector_id}: {e}")
            return False
    
    async def batch_import_vectors(self, vectors: List[VectorData], context: Optional[ExecutionContext] = None) -> Dict[str, Any]:
        """Batch import vectors with tenant isolation"""
        
        if not await self.ensure_initialized():
            raise RuntimeError("Weaviate adapter not initialized")
        
        if not context:
            context = self.create_execution_context("batch_import")
        
        try:
            start_time = time.time()
            
            # Group vectors by tenant and class
            tenant_groups = {}
            for vector in vectors:
                tenant_class = self._get_tenant_class_name(vector.class_name, vector.tenant_id)
                if tenant_class not in tenant_groups:
                    tenant_groups[tenant_class] = []
                tenant_groups[tenant_class].append(vector)
            
            # Ensure all schemas exist
            for vector in vectors:
                tenant_class_name = self._get_tenant_class_name(vector.class_name, vector.tenant_id)
                if not self.client.schema.exists(tenant_class_name):
                    await self.create_tenant_schema(vector.tenant_id)
            
            imported_count = 0
            failed_count = 0
            
            # Batch import by tenant class
            with self.client.batch as batch:
                batch.batch_size = 100
                batch.dynamic = True
                
                for tenant_class, class_vectors in tenant_groups.items():
                    for vector in class_vectors:
                        try:
                            # Apply DLP redaction
                            safe_properties, violations = redact_payload(vector.properties)
                            
                            # Add tenant metadata
                            safe_properties["tenant_id"] = vector.tenant_id
                            safe_properties["created_at"] = time.time()
                            
                            if violations:
                                safe_properties["dlp_violations"] = len(violations)
                            
                            batch.add_data_object(
                                data_object=safe_properties,
                                class_name=tenant_class,
                                uuid=vector.id,
                                vector=vector.vector
                            )
                            
                            imported_count += 1
                            
                        except Exception as e:
                            failed_count += 1
                            logger.warning(f"Failed to add vector {vector.id} to batch: {e}")
            
            execution_time = (time.time() - start_time) * 1000
            self.record_operation(VectorOperation.BATCH_IMPORT.value, execution_time, True)
            
            result = {
                "total_vectors": len(vectors),
                "imported_count": imported_count,
                "failed_count": failed_count,
                "tenant_classes": len(tenant_groups),
                "execution_time_ms": execution_time
            }
            
            # Log operation
            self.log_operation(
                "batch_import",
                context,
                duration_ms=execution_time,
                success=True
            )
            
            logger.info(f"Batch imported {imported_count} vectors, {failed_count} failed")
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.record_operation(VectorOperation.BATCH_IMPORT.value, execution_time, False)
            
            self.log_operation(
                "batch_import",
                context,
                duration_ms=execution_time,
                success=False,
                error=str(e)
            )
            
            logger.error(f"Batch import failed: {e}")
            raise
    
    def _get_tenant_class_name(self, base_class: str, tenant_id: str) -> str:
        """Generate tenant-scoped class name"""
        # Use a consistent naming convention for tenant isolation
        return f"{base_class}_{tenant_id.replace('-', '_')}"
    
    def _get_default_memory_schema(self, tenant_id: str) -> SchemaDefinition:
        """Get default memory schema for tenant"""
        
        properties = [
            {
                "name": "content",
                "dataType": ["text"],
                "description": "Memory content"
            },
            {
                "name": "summary",
                "dataType": ["text"],
                "description": "Memory summary"
            },
            {
                "name": "memory_type",
                "dataType": ["string"],
                "description": "Type of memory"
            },
            {
                "name": "employee_id",
                "dataType": ["string"],
                "description": "Employee ID"
            },
            {
                "name": "session_id",
                "dataType": ["string"],
                "description": "Session ID"
            },
            {
                "name": "tenant_id",
                "dataType": ["string"],
                "description": "Tenant ID"
            },
            {
                "name": "created_at",
                "dataType": ["number"],
                "description": "Creation timestamp"
            },
            {
                "name": "security_level",
                "dataType": ["string"],
                "description": "Security level"
            },
            {
                "name": "dlp_violations",
                "dataType": ["int"],
                "description": "Number of DLP violations"
            }
        ]
        
        return SchemaDefinition(
            class_name="Memory",
            description=f"Memory storage for tenant {tenant_id}",
            properties=properties,
            vectorizer="none",  # We provide our own vectors
            vector_index_type="hnsw",
            vector_index_config={
                "ef": 64,
                "efConstruction": 128,
                "maxConnections": 64,
                "vectorCacheMaxObjects": 500000
            }
        )
    
    async def _initialize_base_schemas(self):
        """Initialize base schema templates"""
        # This would set up any global schema templates
        pass
    
    async def _load_existing_schemas(self):
        """Load existing tenant schemas from Weaviate"""
        try:
            schema = self.client.schema.get()
            
            for class_def in schema.get("classes", []):
                class_name = class_def["class"]
                
                # Check if this is a tenant-specific class
                if "_" in class_name:
                    parts = class_name.split("_")
                    if len(parts) >= 2:
                        base_class = parts[0]
                        tenant_part = "_".join(parts[1:])
                        
                        # Extract tenant ID (reverse the replacement of - with _)
                        tenant_id = tenant_part.replace("_", "-")
                        
                        if tenant_id not in self._tenant_schemas:
                            self._tenant_schemas[tenant_id] = []
                        
                        self._tenant_schemas[tenant_id].append(class_name)
                        self._schema_cache[class_name] = class_def
            
            logger.info(f"Loaded {len(self._tenant_schemas)} tenant schemas")
            
        except Exception as e:
            logger.warning(f"Failed to load existing schemas: {e}")
    
    def get_tenant_schemas(self, tenant_id: str) -> List[str]:
        """Get schema classes for a tenant"""
        return self._tenant_schemas.get(tenant_id, [])
    
    def get_all_tenant_schemas(self) -> Dict[str, List[str]]:
        """Get all tenant schemas"""
        return self._tenant_schemas.copy()
    
    async def clear_tenant_data(self, tenant_id: str, context: Optional[ExecutionContext] = None) -> Dict[str, Any]:
        """Clear all data for a tenant (use with caution)"""
        
        if not await self.ensure_initialized():
            raise RuntimeError("Weaviate adapter not initialized")
        
        if not context:
            context = self.create_execution_context("clear_tenant_data")
        
        try:
            tenant_classes = self.get_tenant_schemas(tenant_id)
            deleted_objects = 0
            
            for class_name in tenant_classes:
                # Delete all objects in the class
                result = self.client.batch.delete_objects(
                    class_name=class_name,
                    where={
                        "path": ["tenant_id"],
                        "operator": "Equal",
                        "valueString": tenant_id
                    }
                )
                
                if result and "results" in result:
                    deleted_objects += len(result["results"]["successful"])
            
            logger.warning(f"Cleared {deleted_objects} objects for tenant {tenant_id}")
            
            return {
                "tenant_id": tenant_id,
                "deleted_objects": deleted_objects,
                "cleared_classes": len(tenant_classes),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to clear tenant data for {tenant_id}: {e}")
            raise

# Global Weaviate adapter instance
weaviate_adapter = WeaviateAdapter()