# forge1/backend/forge1/core/database_config.py
"""
Forge 1 Database Configuration

Hybrid database architecture supporting:
- PostgreSQL for structured data storage
- Vector database (Pinecone/Weaviate) for embeddings
- Redis for high-speed caching layer
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import asyncpg
import redis.asyncio as redis
import pinecone
import weaviate
from weaviate.classes.init import Auth
from pydantic import BaseSettings, Field

logger = logging.getLogger(__name__)

class VectorDBType(Enum):
    """Supported vector database types"""
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"

@dataclass
class DatabaseHealth:
    """Database health status"""
    postgres: bool = False
    redis: bool = False
    vector_db: bool = False
    overall: bool = False

class DatabaseSettings(BaseSettings):
    """Database configuration settings"""
    
    # PostgreSQL settings
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_db: str = Field(default="forge1", env="POSTGRES_DB")
    postgres_user: str = Field(default="forge1_user", env="POSTGRES_USER")
    postgres_password: str = Field(default="forge1_db_pass", env="POSTGRES_PASSWORD")
    postgres_max_connections: int = Field(default=20, env="POSTGRES_MAX_CONNECTIONS")
    postgres_min_connections: int = Field(default=5, env="POSTGRES_MIN_CONNECTIONS")
    
    # Redis settings
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(default="forge1_redis_pass", env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_max_connections: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")
    
    # Vector database settings
    vector_db_type: VectorDBType = Field(default=VectorDBType.PINECONE, env="VECTOR_DB_TYPE")
    
    # Pinecone settings
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_environment: str = Field(default="us-east-1-aws", env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(default="forge1-memory", env="PINECONE_INDEX_NAME")
    
    # Weaviate settings
    weaviate_url: str = Field(default="http://localhost:8080", env="WEAVIATE_URL")
    weaviate_api_key: Optional[str] = Field(default=None, env="WEAVIATE_API_KEY")
    weaviate_class_name: str = Field(default="Forge1Memory", env="WEAVIATE_CLASS_NAME")
    
    # Embedding settings
    embedding_dimension: int = Field(default=1536, env="EMBEDDING_DIMENSION")  # OpenAI default
    embedding_model: str = Field(default="text-embedding-ada-002", env="EMBEDDING_MODEL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

class DatabaseManager:
    """Manages all database connections and operations"""
    
    def __init__(self, settings: Optional[DatabaseSettings] = None):
        self.settings = settings or DatabaseSettings()
        self._postgres_pool: Optional[asyncpg.Pool] = None
        self._redis_client: Optional[redis.Redis] = None
        self._vector_client: Optional[Any] = None
        self._initialized = False
        
        logger.info(f"Database manager initialized with vector DB: {self.settings.vector_db_type.value}")
    
    async def initialize(self) -> None:
        """Initialize all database connections"""
        if self._initialized:
            logger.warning("Database manager already initialized")
            return
        
        try:
            # Initialize PostgreSQL
            await self._init_postgres()
            
            # Initialize Redis
            await self._init_redis()
            
            # Initialize Vector Database
            await self._init_vector_db()
            
            self._initialized = True
            logger.info("All database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            await self.cleanup()
            raise
    
    async def _init_postgres(self) -> None:
        """Initialize PostgreSQL connection pool"""
        try:
            dsn = (
                f"postgresql://{self.settings.postgres_user}:{self.settings.postgres_password}"
                f"@{self.settings.postgres_host}:{self.settings.postgres_port}/{self.settings.postgres_db}"
            )
            
            self._postgres_pool = await asyncpg.create_pool(
                dsn,
                min_size=self.settings.postgres_min_connections,
                max_size=self.settings.postgres_max_connections,
                command_timeout=30,
                server_settings={
                    'application_name': 'forge1-backend',
                    'search_path': 'forge1_core,forge1_memory,public'
                }
            )
            
            # Test connection
            async with self._postgres_pool.acquire() as conn:
                await conn.execute("SELECT 1")
            
            logger.info("PostgreSQL connection pool initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise
    
    async def _init_redis(self) -> None:
        """Initialize Redis connection"""
        try:
            self._redis_client = redis.Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                password=self.settings.redis_password,
                db=self.settings.redis_db,
                max_connections=self.settings.redis_max_connections,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Test connection
            await self._redis_client.ping()
            
            logger.info("Redis connection initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    async def _init_vector_db(self) -> None:
        """Initialize vector database connection"""
        try:
            if self.settings.vector_db_type == VectorDBType.PINECONE:
                await self._init_pinecone()
            elif self.settings.vector_db_type == VectorDBType.WEAVIATE:
                await self._init_weaviate()
            else:
                raise ValueError(f"Unsupported vector DB type: {self.settings.vector_db_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            raise
    
    async def _init_pinecone(self) -> None:
        """Initialize Pinecone vector database"""
        if not self.settings.pinecone_api_key:
            logger.warning("Pinecone API key not provided, using mock client")
            self._vector_client = MockPineconeClient()
            return
        
        try:
            # Initialize Pinecone
            pinecone.init(
                api_key=self.settings.pinecone_api_key,
                environment=self.settings.pinecone_environment
            )
            
            # Check if index exists, create if not
            if self.settings.pinecone_index_name not in pinecone.list_indexes():
                logger.info(f"Creating Pinecone index: {self.settings.pinecone_index_name}")
                pinecone.create_index(
                    name=self.settings.pinecone_index_name,
                    dimension=self.settings.embedding_dimension,
                    metric="cosine"
                )
            
            self._vector_client = pinecone.Index(self.settings.pinecone_index_name)
            
            # Test connection
            stats = self._vector_client.describe_index_stats()
            logger.info(f"Pinecone initialized - vectors: {stats.get('total_vector_count', 0)}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            # Fallback to mock client for development
            self._vector_client = MockPineconeClient()
    
    async def _init_weaviate(self) -> None:
        """Initialize Weaviate vector database"""
        try:
            auth_config = None
            if self.settings.weaviate_api_key:
                auth_config = Auth.api_key(self.settings.weaviate_api_key)
            
            self._vector_client = weaviate.connect_to_custom(
                http_host=self.settings.weaviate_url.replace("http://", "").replace("https://", ""),
                http_port=8080,
                http_secure=False,
                auth_credentials=auth_config
            )
            
            # Test connection
            if self._vector_client.is_ready():
                logger.info("Weaviate connection initialized")
            else:
                raise Exception("Weaviate not ready")
                
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate: {e}")
            # Fallback to mock client for development
            self._vector_client = MockWeaviateClient()
    
    async def health_check(self) -> DatabaseHealth:
        """Check health of all database connections"""
        health = DatabaseHealth()
        
        # Check PostgreSQL
        try:
            if self._postgres_pool:
                async with self._postgres_pool.acquire() as conn:
                    await conn.execute("SELECT 1")
                health.postgres = True
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
        
        # Check Redis
        try:
            if self._redis_client:
                await self._redis_client.ping()
                health.redis = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
        
        # Check Vector DB
        try:
            if self._vector_client:
                if hasattr(self._vector_client, 'describe_index_stats'):
                    # Pinecone
                    self._vector_client.describe_index_stats()
                elif hasattr(self._vector_client, 'is_ready'):
                    # Weaviate
                    self._vector_client.is_ready()
                health.vector_db = True
        except Exception as e:
            logger.error(f"Vector DB health check failed: {e}")
        
        health.overall = health.postgres and health.redis and health.vector_db
        return health
    
    async def cleanup(self) -> None:
        """Clean up all database connections"""
        try:
            if self._postgres_pool:
                await self._postgres_pool.close()
                self._postgres_pool = None
            
            if self._redis_client:
                await self._redis_client.close()
                self._redis_client = None
            
            if self._vector_client:
                if hasattr(self._vector_client, 'close'):
                    await self._vector_client.close()
                self._vector_client = None
            
            self._initialized = False
            logger.info("Database connections cleaned up")
            
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")
    
    @property
    def postgres(self) -> Optional[asyncpg.Pool]:
        """Get PostgreSQL connection pool"""
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
        return self._postgres_pool
    
    @property
    def redis(self) -> Optional[redis.Redis]:
        """Get Redis client"""
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
        return self._redis_client
    
    @property
    def vector_db(self) -> Optional[Any]:
        """Get vector database client"""
        if not self._initialized:
            raise RuntimeError("Database manager not initialized")
        return self._vector_client

class MockPineconeClient:
    """Mock Pinecone client for development/testing"""
    
    def __init__(self):
        self._vectors = {}
        logger.info("Using mock Pinecone client")
    
    def describe_index_stats(self):
        return {"total_vector_count": len(self._vectors)}
    
    def upsert(self, vectors, namespace=""):
        for vector in vectors:
            self._vectors[vector["id"]] = vector
        return {"upserted_count": len(vectors)}
    
    def query(self, vector, top_k=10, namespace="", include_metadata=True):
        # Simple mock query - returns random results
        import random
        results = list(self._vectors.values())
        random.shuffle(results)
        return {
            "matches": [
                {
                    "id": v["id"],
                    "score": random.uniform(0.7, 0.95),
                    "metadata": v.get("metadata", {})
                }
                for v in results[:top_k]
            ]
        }
    
    def delete(self, ids, namespace=""):
        for id in ids:
            self._vectors.pop(id, None)
        return {"deleted_count": len(ids)}

class MockWeaviateClient:
    """Mock Weaviate client for development/testing"""
    
    def __init__(self):
        self._objects = {}
        logger.info("Using mock Weaviate client")
    
    def is_ready(self):
        return True
    
    def close(self):
        pass

# Global database manager instance
_db_manager: Optional[DatabaseManager] = None

async def get_database_manager() -> DatabaseManager:
    """Get or create the global database manager instance"""
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
        await _db_manager.initialize()
    
    return _db_manager

async def cleanup_database_manager():
    """Cleanup the global database manager"""
    global _db_manager
    
    if _db_manager:
        await _db_manager.cleanup()
        _db_manager = None

# Export main classes and functions
__all__ = [
    "DatabaseManager",
    "DatabaseSettings", 
    "DatabaseHealth",
    "VectorDBType",
    "get_database_manager",
    "cleanup_database_manager"
]