# Employee Lifecycle System - Test Suite

Comprehensive test suite for the Employee Lifecycle System covering unit tests, integration tests, end-to-end tests, and performance validation.

## Test Structure

```
tests/
├── unit/                           # Unit tests
│   ├── test_employee_manager.py    # Employee management unit tests
│   ├── test_employee_memory_manager.py  # Memory management unit tests
│   ├── test_client_manager.py      # Client management unit tests
│   └── test_models.py              # Data model unit tests
├── integration/                    # Integration tests
│   ├── test_employee_lifecycle_integration.py  # Full lifecycle integration
│   ├── test_api_integration.py     # API endpoint integration
│   └── test_database_integration.py  # Database integration
├── e2e/                           # End-to-end tests
│   ├── test_employee_system_e2e.py  # Complete system E2E tests
│   └── test_user_workflows_e2e.py   # User workflow E2E tests
├── performance/                    # Performance tests
│   ├── test_load_performance.py    # Load testing
│   └── test_memory_performance.py  # Memory usage tests
├── conftest.py                    # Pytest configuration and fixtures
└── README.md                      # This file
```

## Requirements Coverage

### Requirement 1.1: Employee Creation and Management
- **Unit Tests**: `test_employee_manager.py::test_create_employee_success`
- **Integration Tests**: `test_employee_lifecycle_integration.py::test_complete_employee_lifecycle`
- **E2E Tests**: `test_employee_system_e2e.py::test_complete_customer_support_workflow`

### Requirement 1.2: Employee Configuration
- **Unit Tests**: `test_employee_manager.py::test_update_employee_success`
- **Integration Tests**: `test_employee_lifecycle_integration.py::test_employee_configuration_management`
- **E2E Tests**: `test_employee_system_e2e.py::test_complete_customer_support_workflow`

### Requirement 1.3: Employee Deletion and Archival
- **Unit Tests**: `test_employee_manager.py::test_delete_employee_success`
- **Integration Tests**: `test_employee_lifecycle_integration.py::test_complete_employee_lifecycle`
- **E2E Tests**: `test_employee_system_e2e.py::test_complete_customer_support_workflow`

### Requirement 2.1: Memory Storage and Retrieval
- **Unit Tests**: `test_employee_memory_manager.py::test_store_memory_success`
- **Integration Tests**: `test_employee_lifecycle_integration.py::test_employee_memory_management`
- **E2E Tests**: `test_employee_system_e2e.py::test_complete_customer_support_workflow`

### Requirement 2.2: Memory Search and Context
- **Unit Tests**: `test_employee_memory_manager.py::test_search_memories_success`
- **Integration Tests**: `test_employee_lifecycle_integration.py::test_employee_memory_management`
- **E2E Tests**: `test_employee_system_e2e.py::test_complete_customer_support_workflow`

### Requirement 2.3: Memory Management
- **Unit Tests**: `test_employee_memory_manager.py::test_cleanup_old_memories`
- **Integration Tests**: `test_employee_lifecycle_integration.py::test_employee_memory_management`
- **E2E Tests**: `test_employee_system_e2e.py::test_complete_customer_support_workflow`

### Requirement 3.1: Analytics and Monitoring
- **Integration Tests**: `test_employee_lifecycle_integration.py::test_analytics_integration`
- **E2E Tests**: `test_employee_system_e2e.py::test_complete_customer_support_workflow`

### Requirement 3.2: Performance Optimization
- **Integration Tests**: `test_employee_lifecycle_integration.py::test_performance_optimization_integration`
- **E2E Tests**: `test_employee_system_e2e.py::test_system_resilience_and_recovery`
- **Performance Tests**: `test_load_performance.py`, `test_memory_performance.py`

## Running Tests

### Prerequisites

1. **Install Dependencies**:
   ```bash
   python run_tests.py setup
   ```

2. **Environment Setup**:
   ```bash
   export TEST_DATABASE_URL="postgresql://test:test@localhost:5432/test_employee_db"
   export TEST_REDIS_URL="redis://localhost:6379/1"
   export TESTING=1
   ```

### Test Commands

#### Run All Tests
```bash
python run_tests.py all
```

#### Run Specific Test Types
```bash
# Unit tests only
python run_tests.py unit

# Integration tests only
python run_tests.py integration

# End-to-end tests only
python run_tests.py e2e

# Performance tests only
python run_tests.py performance

# Security tests only
python run_tests.py security
```

#### Run Specific Tests
```bash
# Run specific test file
python run_tests.py run tests/unit/test_employee_manager.py

# Run specific test function
python run_tests.py run tests/unit/test_employee_manager.py::TestEmployeeManager::test_create_employee_success

# Run with verbose output
python run_tests.py unit --verbose
```

#### Coverage Reports
```bash
# Generate coverage report
python run_tests.py coverage

# Run tests without coverage
python run_tests.py all --no-coverage
```

#### Code Quality
```bash
# Run linting and formatting checks
python run_tests.py lint

# Clean test artifacts
python run_tests.py clean
```

### Using Pytest Directly

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=forge1 --cov-report=html

# Run specific markers
pytest -m unit
pytest -m integration
pytest -m e2e
pytest -m performance
pytest -m security

# Run with specific options
pytest -v -s --tb=short
pytest --maxfail=1
pytest --lf  # Run last failed tests
```

## Test Configuration

### Pytest Configuration (`pytest.ini`)
- Test discovery patterns
- Coverage settings
- Markers for test categorization
- Output formatting
- Asyncio support

### Fixtures (`conftest.py`)
- Database setup and teardown
- Mock services and dependencies
- Test data factories
- Performance tracking utilities
- Common test utilities

## Test Data and Mocking

### Test Data Factory
The `TestDataFactory` class provides standardized test data:

```python
# Create test client data
client_data = TestDataFactory.create_client_data(name="Test Corp")

# Create test employee requirements
requirements = TestDataFactory.create_employee_requirements(role="Support Agent")

# Create test interaction data
interaction = TestDataFactory.create_interaction_data(message="Hello")
```

### Mocking Strategy
- **Database**: Mocked using `AsyncMock` for unit tests, real database for integration
- **External APIs**: Mocked responses for consistent testing
- **LLM Services**: Mocked to avoid API costs and ensure deterministic results
- **Cache/Redis**: Mocked for unit tests, optional real Redis for integration

## Performance Testing

### Load Testing
- Concurrent user simulation
- API endpoint stress testing
- Database performance under load
- Memory usage monitoring

### Benchmarking
- Response time measurements
- Throughput analysis
- Resource utilization tracking
- Performance regression detection

## Security Testing

### Authentication and Authorization
- Token validation testing
- Permission boundary testing
- Tenant isolation verification

### Input Validation
- SQL injection prevention
- XSS protection testing
- Input sanitization verification

### Data Privacy
- PII handling verification
- Data encryption testing
- Audit trail validation

## Continuous Integration

### GitHub Actions Integration
```yaml
# Example CI configuration
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: python run_tests.py setup
      - name: Run tests
        run: python run_tests.py all --skip-e2e
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

### Test Environments
- **Development**: Full test suite with real services
- **CI/CD**: Unit and integration tests with mocked services
- **Staging**: E2E tests against staging environment
- **Production**: Health checks and smoke tests

## Test Maintenance

### Adding New Tests
1. Follow naming conventions (`test_*.py`)
2. Use appropriate markers (`@pytest.mark.unit`, etc.)
3. Include requirement references in docstrings
4. Update this README with coverage information

### Test Data Management
- Use factories for consistent test data
- Clean up test data after each test
- Avoid hardcoded values in tests
- Use fixtures for reusable test setup

### Performance Considerations
- Mock external dependencies in unit tests
- Use database transactions for test isolation
- Implement proper cleanup to prevent test interference
- Monitor test execution time and optimize slow tests

## Troubleshooting

### Common Issues

#### Database Connection Errors
```bash
# Check database is running
docker ps | grep postgres

# Verify connection string
echo $TEST_DATABASE_URL
```

#### Import Errors
```bash
# Install in development mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

#### Async Test Issues
```bash
# Ensure pytest-asyncio is installed
pip install pytest-asyncio

# Check asyncio mode in pytest.ini
grep asyncio_mode pytest.ini
```

### Debug Mode
```bash
# Run with debug output
pytest -v -s --tb=long --log-cli-level=DEBUG

# Run single test with debugging
pytest -v -s tests/unit/test_employee_manager.py::test_create_employee_success --pdb
```

## Test Metrics and Reporting

### Coverage Goals
- **Overall Coverage**: ≥ 80%
- **Critical Paths**: ≥ 95%
- **New Code**: 100%

### Quality Metrics
- **Test Execution Time**: < 5 minutes for full suite
- **Test Reliability**: > 99% pass rate
- **Code Quality**: All linting checks pass

### Reporting
- HTML coverage reports in `htmlcov/`
- XML coverage reports for CI integration
- Performance benchmarks in test output
- Security scan results in CI logs

## Contributing

When adding new functionality:

1. **Write tests first** (TDD approach)
2. **Ensure requirement coverage** for all new features
3. **Add appropriate markers** for test categorization
4. **Update documentation** including this README
5. **Verify CI passes** before merging

For questions or issues with the test suite, please refer to the project documentation or create an issue in the project repository.