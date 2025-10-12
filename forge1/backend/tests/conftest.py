# forge1/backend/tests/conftest.py
"""
Pytest configuration and shared fixtures for Employee Lifecycle System tests

Provides common test fixtures, database setup, and test utilities
for unit, integration, and end-to-end tests.

Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3
"""

import pytest
import asyncio
import os
import tempfile
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import AsyncMock

try:
    import asyncpg
    HAS_ASYNCPG = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    asyncpg = None  # type: ignore[assignment]
    HAS_ASYNCPG = False
try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    TestClient = None  # type: ignore[assignment]
    HAS_FASTAPI = False

try:
    from forge1.main import app
    HAS_APP = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    app = None  # type: ignore[assignment]
    HAS_APP = False

try:
    from forge1.core.database_config import DatabaseManager
    HAS_DB_MANAGER = True
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    DatabaseManager = None  # type: ignore[assignment]
    HAS_DB_MANAGER = False
try:  # Phase 1: allow foundational tests to run without optional heavy deps
    from forge1.services.employee_manager import EmployeeManager
    from forge1.services.employee_memory_manager import EmployeeMemoryManager
    from forge1.services.client_manager import ClientManager
    from forge1.models.employee_models import (
        Employee, EmployeeStatus, PersonalityConfig, ModelPreferences,
        CommunicationStyle, FormalityLevel, ExpertiseLevel, ResponseLength
    )
except ModuleNotFoundError:  # pragma: no cover - legacy fixtures guard
    EmployeeManager = None  # type: ignore[assignment]
    EmployeeMemoryManager = None  # type: ignore[assignment]
    ClientManager = None  # type: ignore[assignment]
    Employee = EmployeeStatus = PersonalityConfig = ModelPreferences = None  # type: ignore
    CommunicationStyle = FormalityLevel = ExpertiseLevel = ResponseLength = None  # type: ignore


# Test configuration
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "postgresql://test:test@localhost:5432/test_employee_db")
TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/1")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_database():
    """Set up test database for the session"""
    # Create test database connection
    if not HAS_ASYNCPG:
        pytest.skip("asyncpg not available")

    try:
        conn = await asyncpg.connect(TEST_DATABASE_URL)
        
        # Create test tables if they don't exist
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS test_clients (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                industry VARCHAR,
                tier VARCHAR,
                max_employees INTEGER,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS test_employees (
                id VARCHAR PRIMARY KEY,
                client_id VARCHAR NOT NULL,
                name VARCHAR NOT NULL,
                role VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                personality JSONB,
                model_preferences JSONB,
                tool_access TEXT[],
                knowledge_sources TEXT[],
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                last_interaction_at TIMESTAMP WITH TIME ZONE
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS test_interactions (
                id VARCHAR PRIMARY KEY,
                employee_id VARCHAR NOT NULL,
                session_id VARCHAR,
                message TEXT NOT NULL,
                response TEXT NOT NULL,
                context JSONB,
                processing_time_ms INTEGER,
                tokens_used INTEGER,
                cost DECIMAL(10,6),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS test_memories (
                id VARCHAR PRIMARY KEY,
                employee_id VARCHAR NOT NULL,
                content TEXT NOT NULL,
                memory_type VARCHAR NOT NULL,
                importance_score DECIMAL(3,2),
                context JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        
        yield conn
        
        # Cleanup
        await conn.execute("DROP TABLE IF EXISTS test_memories")
        await conn.execute("DROP TABLE IF EXISTS test_interactions")
        await conn.execute("DROP TABLE IF EXISTS test_employees")
        await conn.execute("DROP TABLE IF EXISTS test_clients")
        
        await conn.close()
        
    except Exception as e:
        pytest.skip(f"Test database not available: {e}")


@pytest.fixture
async def db_manager(test_database):
    """Mock database manager for testing"""
    if not HAS_DB_MANAGER:
        pytest.skip("database manager not available")

    manager = AsyncMock(spec=DatabaseManager)
    
    # Mock connection context manager
    mock_conn = AsyncMock()
    manager.get_connection.return_value.__aenter__.return_value = mock_conn
    manager.get_connection.return_value.__aexit__.return_value = None
    
    return manager, mock_conn


@pytest.fixture
def test_client():
    """FastAPI test client"""
    if not (HAS_FASTAPI and HAS_APP):
        pytest.skip("fastapi not available")
    return TestClient(app)


@pytest.fixture
def sample_client_data():
    """Sample client data for testing"""
    return {
        "name": "Test Corporation",
        "industry": "Technology",
        "tier": "enterprise",
        "max_employees": 25,
        "allowed_models": ["gpt-4", "gpt-3.5-turbo"],
        "security_level": "high",
        "compliance_requirements": ["SOC2", "GDPR"]
    }


@pytest.fixture
def sample_employee_requirements():
    """Sample employee requirements for testing"""
    return {
        "role": "Customer Support Specialist",
        "industry": "Technology",
        "expertise_areas": ["customer_service", "technical_support"],
        "communication_style": "friendly",
        "tools_needed": ["email", "chat"],
        "knowledge_domains": ["product_documentation"],
        "personality_traits": {
            "empathy_level": 0.9,
            "patience_level": 0.95
        },
        "model_preferences": {
            "primary_model": "gpt-4",
            "temperature": 0.7
        }
    }


@pytest.fixture
def sample_employee():
    """Sample employee instance for testing"""
    return Employee(
        id="test_emp_123",
        client_id="test_client_123",
        name="Test Employee",
        role="Customer Support Specialist",
        status=EmployeeStatus.ACTIVE,
        personality=PersonalityConfig(
            communication_style=CommunicationStyle.FRIENDLY,
            formality_level=FormalityLevel.CASUAL,
            expertise_level=ExpertiseLevel.INTERMEDIATE,
            response_length=ResponseLength.DETAILED,
            creativity_level=0.7,
            empathy_level=0.9,
            custom_traits={"patience_level": 0.95}
        ),
        model_preferences=ModelPreferences(
            primary_model="gpt-4",
            fallback_models=["gpt-3.5-turbo"],
            temperature=0.7,
            max_tokens=2000,
            specialized_models={}
        ),
        tool_access=["email", "chat"],
        knowledge_sources=["kb_001"],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def test_headers():
    """Standard test headers for API requests"""
    return {
        "Content-Type": "application/json",
        "Authorization": "Bearer test_token",
        "X-Tenant-ID": "test_client_001",
        "X-Client-ID": "test_client_001",
        "X-User-ID": "test_user_001"
    }


@pytest.fixture
async def employee_manager(db_manager):
    """Employee manager with mocked dependencies"""
    db_mgr, mock_conn = db_manager
    
    manager = EmployeeManager(db_mgr)
    
    # Mock other dependencies
    manager.client_manager = AsyncMock()
    manager.memory_manager = AsyncMock()
    manager.personality_manager = AsyncMock()
    manager.interaction_processor = AsyncMock()
    
    # Mock initialization
    manager._initialized = True
    
    return manager


@pytest.fixture
async def memory_manager(db_manager):
    """Memory manager with mocked dependencies"""
    db_mgr, mock_conn = db_manager
    
    manager = EmployeeMemoryManager(db_mgr)
    
    # Mock vector store
    manager.vector_store = AsyncMock()
    manager._initialized = True
    
    return manager


@pytest.fixture
async def client_manager(db_manager):
    """Client manager with mocked dependencies"""
    db_mgr, mock_conn = db_manager
    
    manager = ClientManager(db_mgr)
    manager._initialized = True
    
    return manager


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing"""
    return {
        "message": "Hello! How can I help you today?",
        "processing_time_ms": 1200,
        "tokens_used": 50,
        "cost": 0.002,
        "model_used": "gpt-4",
        "confidence_score": 0.95
    }


@pytest.fixture
def mock_memory_entries():
    """Mock memory entries for testing"""
    return [
        {
            "id": "mem_001",
            "content": "Customer asked about billing procedures",
            "memory_type": "interaction",
            "importance_score": 0.8,
            "context": {"topic": "billing", "sentiment": "neutral"},
            "created_at": datetime.now(timezone.utc)
        },
        {
            "id": "mem_002",
            "content": "Resolved technical issue with API integration",
            "memory_type": "resolution",
            "importance_score": 0.9,
            "context": {"topic": "technical", "resolution": "successful"},
            "created_at": datetime.now(timezone.utc)
        }
    ]


@pytest.fixture
def temp_file():
    """Create a temporary file for testing"""
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


# Test utilities
class TestDataFactory:
    """Factory for creating test data"""
    
    @staticmethod
    def create_client_data(name: str = "Test Client", **kwargs) -> Dict[str, Any]:
        """Create client data for testing"""
        default_data = {
            "name": name,
            "industry": "Technology",
            "tier": "enterprise",
            "max_employees": 25,
            "allowed_models": ["gpt-4", "gpt-3.5-turbo"],
            "security_level": "high"
        }
        default_data.update(kwargs)
        return default_data
    
    @staticmethod
    def create_employee_requirements(role: str = "Test Role", **kwargs) -> Dict[str, Any]:
        """Create employee requirements for testing"""
        default_requirements = {
            "role": role,
            "industry": "Technology",
            "expertise_areas": ["general"],
            "communication_style": "friendly",
            "tools_needed": ["email"],
            "knowledge_domains": ["general"],
            "personality_traits": {"empathy_level": 0.8},
            "model_preferences": {"primary_model": "gpt-4", "temperature": 0.7}
        }
        default_requirements.update(kwargs)
        return default_requirements
    
    @staticmethod
    def create_interaction_data(message: str = "Test message", **kwargs) -> Dict[str, Any]:
        """Create interaction data for testing"""
        default_data = {
            "message": message,
            "session_id": "test_session",
            "context": {"user_type": "test"},
            "include_memory": True,
            "memory_limit": 5
        }
        default_data.update(kwargs)
        return default_data


@pytest.fixture
def test_data_factory():
    """Test data factory fixture"""
    return TestDataFactory


# Async test helpers
async def wait_for_condition(condition_func, timeout: float = 5.0, interval: float = 0.1):
    """Wait for a condition to become true"""
    import time
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if await condition_func():
            return True
        await asyncio.sleep(interval)
    
    return False


@pytest.fixture
def wait_for():
    """Wait for condition helper"""
    return wait_for_condition


# Performance testing helpers
class PerformanceTracker:
    """Track performance metrics during tests"""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, name: str):
        """Start timing an operation"""
        import time
        self.metrics[name] = {"start": time.time()}
    
    def end_timer(self, name: str):
        """End timing an operation"""
        import time
        if name in self.metrics:
            self.metrics[name]["end"] = time.time()
            self.metrics[name]["duration"] = self.metrics[name]["end"] - self.metrics[name]["start"]
    
    def get_duration(self, name: str) -> float:
        """Get duration of an operation"""
        return self.metrics.get(name, {}).get("duration", 0.0)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all performance metrics"""
        return self.metrics


@pytest.fixture
def performance_tracker():
    """Performance tracking fixture"""
    return PerformanceTracker()


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.e2e = pytest.mark.e2e
pytest.mark.performance = pytest.mark.performance
pytest.mark.security = pytest.mark.security


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "security: Security tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file paths"""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add performance marker for performance tests
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        
        # Add security marker for security tests
        if "security" in item.name.lower() or "auth" in item.name.lower():
            item.add_marker(pytest.mark.security)