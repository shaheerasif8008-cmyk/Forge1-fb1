# forge1/backend/forge1/tests/test_tool_registry.py
"""
Tests for Tool Registry System

Comprehensive tests for the tool registry, management, and execution system.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock

from forge1.tools.tool_registry import (
    ToolRegistry,
    BaseTool,
    ToolCategory,
    ToolStatus,
    AuthenticationType
)
from forge1.tools.example_tools import (
    EmailTool,
    WebScrapingTool,
    DataAnalysisTool,
    SlackTool,
    CalendarTool
)


class TestBaseTool:
    """Test cases for BaseTool base class"""
    
    class MockTool(BaseTool):
        """Mock tool for testing"""
        
        def __init__(self):
            super().__init__(
                name="mock_tool",
                description="A mock tool for testing",
                category=ToolCategory.CUSTOM,
                version="1.0.0",
                authentication_type=AuthenticationType.NONE
            )
        
        async def execute(self, message: str = "test", **kwargs):
            return {"result": f"Mock execution: {message}"}
        
        def validate_parameters(self, parameters):
            return "message" in parameters
    
    @pytest.fixture
    def mock_tool(self):
        """Create mock tool for testing"""
        return self.MockTool()
    
    def test_tool_initialization(self, mock_tool):
        """Test tool initialization"""
        
        assert mock_tool.name == "mock_tool"
        assert mock_tool.description == "A mock tool for testing"
        assert mock_tool.category == ToolCategory.CUSTOM
        assert mock_tool.version == "1.0.0"
        assert mock_tool.authentication_type == AuthenticationType.NONE
        assert mock_tool.tool_id.startswith("tool_")
        
        # Check metrics initialization
        assert mock_tool.metrics["executions"] == 0
        assert mock_tool.metrics["successes"] == 0
        assert mock_tool.metrics["failures"] == 0
    
    @pytest.mark.asyncio
    async def test_tool_execution(self, mock_tool):
        """Test tool execution"""
        
        result = await mock_tool.execute(message="hello world")
        
        assert result["result"] == "Mock execution: hello world"
    
    def test_parameter_validation(self, mock_tool):
        """Test parameter validation"""
        
        # Valid parameters
        assert mock_tool.validate_parameters({"message": "test"}) == True
        
        # Invalid parameters
        assert mock_tool.validate_parameters({}) == False
        assert mock_tool.validate_parameters({"other": "value"}) == False
    
    def test_get_schema(self, mock_tool):
        """Test getting tool schema"""
        
        schema = mock_tool.get_schema()
        
        assert schema["name"] == "mock_tool"
        assert schema["description"] == "A mock tool for testing"
        assert schema["category"] == "custom"
        assert schema["version"] == "1.0.0"
        assert schema["authentication_type"] == "none"
        assert "parameters" in schema
        assert "returns" in schema
    
    def test_get_metadata(self, mock_tool):
        """Test getting tool metadata"""
        
        metadata = mock_tool.get_metadata()
        
        assert metadata["tool_id"] == mock_tool.tool_id
        assert metadata["name"] == "mock_tool"
        assert metadata["category"] == "custom"
        assert metadata["version"] == "1.0.0"
        assert "created_at" in metadata
    
    @pytest.mark.asyncio
    async def test_metrics_update(self, mock_tool):
        """Test metrics update"""
        
        initial_executions = mock_tool.metrics["executions"]
        
        # Update metrics with successful execution
        await mock_tool._update_metrics(1.5, True)
        
        assert mock_tool.metrics["executions"] == initial_executions + 1
        assert mock_tool.metrics["successes"] == 1
        assert mock_tool.metrics["failures"] == 0
        assert mock_tool.metrics["average_execution_time"] == 1.5
        
        # Update metrics with failed execution
        await mock_tool._update_metrics(2.0, False)
        
        assert mock_tool.metrics["executions"] == initial_executions + 2
        assert mock_tool.metrics["successes"] == 1
        assert mock_tool.metrics["failures"] == 1
        assert mock_tool.metrics["average_execution_time"] == 1.75  # (1.5 + 2.0) / 2


class TestToolRegistry:
    """Test cases for ToolRegistry"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for tool registry"""
        return {
            "security_manager": Mock(),
            "performance_monitor": Mock(),
            "memory_manager": Mock()
        }
    
    @pytest.fixture
    def tool_registry(self, mock_dependencies):
        """Create tool registry for testing"""
        mock_dependencies["security_manager"].validate_tool_registration = AsyncMock()
        mock_dependencies["security_manager"].validate_tool_execution = AsyncMock()
        mock_dependencies["performance_monitor"].track_tool_execution = AsyncMock()
        mock_dependencies["memory_manager"].store_memory = AsyncMock()
        
        return ToolRegistry(**mock_dependencies)
    
    @pytest.fixture
    def sample_tool(self):
        """Create sample tool for testing"""
        return EmailTool()
    
    def test_registry_initialization(self, tool_registry):
        """Test tool registry initialization"""
        
        assert tool_registry.registry_id.startswith("registry_")
        assert len(tool_registry.tools) == 0
        assert len(tool_registry.tool_categories) == len(ToolCategory)
        assert tool_registry.registry_metrics["tools_registered"] == 0
    
    @pytest.mark.asyncio
    async def test_register_tool(self, tool_registry, sample_tool, mock_dependencies):
        """Test tool registration"""
        
        tool_id = await tool_registry.register_tool(sample_tool)
        
        assert tool_id == sample_tool.tool_id
        assert tool_id in tool_registry.tools
        assert sample_tool.tool_id in tool_registry.tool_categories[sample_tool.category]
        assert sample_tool.name in tool_registry.tool_versions
        
        # Verify security validation was called
        mock_dependencies["security_manager"].validate_tool_registration.assert_called_once()
        
        # Verify memory storage was called
        mock_dependencies["memory_manager"].store_memory.assert_called_once()
        
        # Check metrics updated
        assert tool_registry.registry_metrics["tools_registered"] == 1
    
    @pytest.mark.asyncio
    async def test_register_duplicate_tool(self, tool_registry, sample_tool):
        """Test registering duplicate tool"""
        
        # Register tool first time
        await tool_registry.register_tool(sample_tool)
        
        # Try to register same tool again
        duplicate_tool = EmailTool()
        
        with pytest.raises(ValueError, match="already exists"):
            await tool_registry.register_tool(duplicate_tool)
        
        # Should succeed with replace_existing=True
        new_tool_id = await tool_registry.register_tool(duplicate_tool, replace_existing=True)
        assert new_tool_id == duplicate_tool.tool_id
    
    @pytest.mark.asyncio
    async def test_unregister_tool(self, tool_registry, sample_tool):
        """Test tool unregistration"""
        
        # Register tool first
        tool_id = await tool_registry.register_tool(sample_tool)
        
        # Verify tool is registered
        assert tool_id in tool_registry.tools
        
        # Unregister tool
        result = await tool_registry.unregister_tool(tool_id)
        
        assert result == True
        assert tool_id not in tool_registry.tools
        assert tool_id not in tool_registry.tool_categories[sample_tool.category]
        
        # Try to unregister non-existent tool
        result = await tool_registry.unregister_tool("non_existent")
        assert result == False
    
    def test_get_tool(self, tool_registry, sample_tool):
        """Test getting tool by ID"""
        
        # Tool not registered yet
        assert tool_registry.get_tool(sample_tool.tool_id) is None
        
        # Register tool
        asyncio.run(tool_registry.register_tool(sample_tool))
        
        # Get tool
        retrieved_tool = tool_registry.get_tool(sample_tool.tool_id)
        assert retrieved_tool == sample_tool
    
    def test_find_tool_by_name(self, tool_registry, sample_tool):
        """Test finding tool by name"""
        
        # Tool not registered yet
        assert tool_registry.find_tool_by_name(sample_tool.name) is None
        
        # Register tool
        asyncio.run(tool_registry.register_tool(sample_tool))
        
        # Find tool by name
        found_tool = tool_registry.find_tool_by_name(sample_tool.name)
        assert found_tool == sample_tool
        
        # Find tool by name and version
        found_tool = tool_registry.find_tool_by_name(sample_tool.name, sample_tool.version)
        assert found_tool == sample_tool
        
        # Find tool with wrong version
        found_tool = tool_registry.find_tool_by_name(sample_tool.name, "2.0.0")
        assert found_tool is None
    
    @pytest.mark.asyncio
    async def test_get_tools_by_category(self, tool_registry):
        """Test getting tools by category"""
        
        # Register tools in different categories
        email_tool = EmailTool()
        scraper_tool = WebScrapingTool()
        
        await tool_registry.register_tool(email_tool)
        await tool_registry.register_tool(scraper_tool)
        
        # Get communication tools
        comm_tools = tool_registry.get_tools_by_category(ToolCategory.COMMUNICATION)
        assert len(comm_tools) == 1
        assert comm_tools[0] == email_tool
        
        # Get data analysis tools
        data_tools = tool_registry.get_tools_by_category(ToolCategory.DATA_ANALYSIS)
        assert len(data_tools) == 1
        assert data_tools[0] == scraper_tool
        
        # Get tools from empty category
        empty_tools = tool_registry.get_tools_by_category(ToolCategory.SECURITY)
        assert len(empty_tools) == 0
    
    @pytest.mark.asyncio
    async def test_search_tools(self, tool_registry):
        """Test searching tools"""
        
        # Register multiple tools
        email_tool = EmailTool()
        scraper_tool = WebScrapingTool()
        slack_tool = SlackTool()
        
        await tool_registry.register_tool(email_tool)
        await tool_registry.register_tool(scraper_tool)
        await tool_registry.register_tool(slack_tool)
        
        # Search by query
        results = tool_registry.search_tools(query="email")
        assert len(results) == 1
        assert results[0] == email_tool
        
        # Search by category
        results = tool_registry.search_tools(category=ToolCategory.COMMUNICATION)
        assert len(results) == 2  # email and slack
        
        # Search by authentication type
        results = tool_registry.search_tools(authentication_type=AuthenticationType.BEARER_TOKEN)
        assert len(results) == 1
        assert results[0] == slack_tool
        
        # Combined search
        results = tool_registry.search_tools(
            category=ToolCategory.COMMUNICATION,
            authentication_type=AuthenticationType.BASIC_AUTH
        )
        assert len(results) == 1
        assert results[0] == email_tool
    
    @pytest.mark.asyncio
    async def test_execute_tool(self, tool_registry, sample_tool, mock_dependencies):
        """Test tool execution through registry"""
        
        # Register tool
        tool_id = await tool_registry.register_tool(sample_tool)
        
        # Execute tool with valid parameters
        parameters = {
            "to": "test@example.com",
            "subject": "Test Subject",
            "body": "Test Body"
        }
        
        result = await tool_registry.execute_tool(tool_id, parameters)
        
        assert result["success"] == True
        assert result["tool_id"] == tool_id
        assert result["tool_name"] == sample_tool.name
        assert "execution_id" in result
        assert "execution_time" in result
        
        # Verify security validation was called
        mock_dependencies["security_manager"].validate_tool_execution.assert_called_once()
        
        # Verify performance tracking was called
        mock_dependencies["performance_monitor"].track_tool_execution.assert_called_once()
        
        # Check registry metrics updated
        assert tool_registry.registry_metrics["total_executions"] == 1
        assert tool_registry.registry_metrics["successful_executions"] == 1
    
    @pytest.mark.asyncio
    async def test_execute_tool_invalid_parameters(self, tool_registry, sample_tool):
        """Test tool execution with invalid parameters"""
        
        # Register tool
        tool_id = await tool_registry.register_tool(sample_tool)
        
        # Execute tool with invalid parameters
        parameters = {"invalid": "params"}
        
        result = await tool_registry.execute_tool(tool_id, parameters)
        
        assert result["success"] == False
        assert "Invalid parameters" in result["error"]
        assert result["tool_id"] == tool_id
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self, tool_registry):
        """Test executing non-existent tool"""
        
        result = await tool_registry.execute_tool("non_existent", {})
        
        assert result["success"] == False
        assert "not found" in result["error"]
        assert result["tool_id"] == "non_existent"
    
    @pytest.mark.asyncio
    async def test_tool_dependencies(self, tool_registry, sample_tool):
        """Test tool dependency management"""
        
        # Register tool with dependencies
        dependency_tool = WebScrapingTool()
        dep_tool_id = await tool_registry.register_tool(dependency_tool)
        
        tool_id = await tool_registry.register_tool(sample_tool, dependencies=[dep_tool_id])
        
        # Check dependencies
        dependencies = tool_registry.get_tool_dependencies(tool_id)
        assert dependencies == [dep_tool_id]
        
        # Check empty dependencies
        empty_deps = tool_registry.get_tool_dependencies(dep_tool_id)
        assert empty_deps == []
    
    @pytest.mark.asyncio
    async def test_tool_versions(self, tool_registry):
        """Test tool version management"""
        
        # Register multiple versions of same tool
        tool_v1 = EmailTool()
        tool_v1.version = "1.0.0"
        
        tool_v2 = EmailTool()
        tool_v2.version = "2.0.0"
        
        await tool_registry.register_tool(tool_v1)
        await tool_registry.register_tool(tool_v2, replace_existing=True)
        
        # Check versions
        versions = tool_registry.get_tool_versions(tool_v1.name)
        assert "1.0.0" in versions
        assert "2.0.0" in versions
    
    def test_get_registry_info(self, tool_registry):
        """Test getting registry information"""
        
        info = tool_registry.get_registry_info()
        
        assert info["registry_id"] == tool_registry.registry_id
        assert "created_at" in info
        assert info["total_tools"] == 0
        assert "categories" in info
        assert "metrics" in info
        assert info["tool_versions_count"] == 0
        assert info["dependencies_count"] == 0
    
    @pytest.mark.asyncio
    async def test_validate_tool_compatibility(self, tool_registry):
        """Test tool compatibility validation"""
        
        # Register multiple tools
        email_tool = EmailTool()
        scraper_tool = WebScrapingTool()
        
        email_id = await tool_registry.register_tool(email_tool)
        scraper_id = await tool_registry.register_tool(scraper_tool)
        
        # Test compatibility
        compatibility = await tool_registry.validate_tool_compatibility([email_id, scraper_id])
        
        assert "compatible" in compatibility
        assert "tool_analysis" in compatibility
        assert compatibility["tool_analysis"]["total_tools"] == 2
        
        # Test with non-existent tool
        compatibility = await tool_registry.validate_tool_compatibility([email_id, "non_existent"])
        
        assert compatibility["compatible"] == False
        assert len(compatibility["issues"]) > 0
    
    @pytest.mark.asyncio
    async def test_export_import_registry(self, tool_registry, sample_tool):
        """Test registry export and import"""
        
        # Register tool
        await tool_registry.register_tool(sample_tool)
        
        # Export registry
        export_data = await tool_registry.export_registry(include_metrics=True)
        
        assert "registry_info" in export_data
        assert "tools" in export_data
        assert "categories" in export_data
        assert "export_timestamp" in export_data
        assert len(export_data["tools"]) == 1
        
        # Create new registry and import
        new_registry = ToolRegistry(
            security_manager=Mock(),
            performance_monitor=Mock(),
            memory_manager=Mock()
        )
        
        new_registry.memory_manager.store_memory = AsyncMock()
        
        import_result = await new_registry.import_registry(export_data)
        
        assert import_result["success"] == True
        assert import_result["imported_tools"] == 1
        assert import_result["skipped_tools"] == 0


class TestExampleTools:
    """Test cases for example tools"""
    
    @pytest.mark.asyncio
    async def test_email_tool(self):
        """Test email tool functionality"""
        
        email_tool = EmailTool()
        
        # Test valid execution
        result = await email_tool.execute(
            to="test@example.com",
            subject="Test Subject",
            body="Test Body"
        )
        
        assert result["success"] == True
        assert result["recipient"] == "test@example.com"
        assert "message_id" in result
        assert "sent_at" in result
        
        # Test parameter validation
        assert email_tool.validate_parameters({
            "to": "test@example.com",
            "subject": "Test",
            "body": "Body"
        }) == True
        
        assert email_tool.validate_parameters({
            "to": "invalid-email",
            "subject": "Test",
            "body": "Body"
        }) == False
    
    @pytest.mark.asyncio
    async def test_web_scraping_tool(self):
        """Test web scraping tool functionality"""
        
        scraper_tool = WebScrapingTool()
        
        # Test valid execution
        result = await scraper_tool.execute(
            url="https://example.com",
            selector=".content",
            format="text"
        )
        
        assert result["success"] == True
        assert result["url"] == "https://example.com"
        assert "data" in result
        assert "scraped_at" in result
        
        # Test parameter validation
        assert scraper_tool.validate_parameters({
            "url": "https://example.com"
        }) == True
        
        assert scraper_tool.validate_parameters({
            "url": "invalid-url"
        }) == False
    
    @pytest.mark.asyncio
    async def test_data_analysis_tool(self):
        """Test data analysis tool functionality"""
        
        analysis_tool = DataAnalysisTool()
        
        # Test summary operation
        test_data = [
            {"name": "John", "age": 30, "salary": 50000},
            {"name": "Jane", "age": 25, "salary": 60000}
        ]
        
        result = await analysis_tool.execute(
            data=test_data,
            operation="summary"
        )
        
        assert result["success"] == True
        assert result["operation"] == "summary"
        assert result["records_processed"] == 2
        assert "result" in result
        
        # Test parameter validation
        assert analysis_tool.validate_parameters({
            "data": test_data,
            "operation": "summary"
        }) == True
        
        assert analysis_tool.validate_parameters({
            "data": "invalid",
            "operation": "summary"
        }) == False
    
    @pytest.mark.asyncio
    async def test_slack_tool(self):
        """Test Slack tool functionality"""
        
        slack_tool = SlackTool()
        
        # Test valid execution
        result = await slack_tool.execute(
            channel="#general",
            message="Hello, team!"
        )
        
        assert result["success"] == True
        assert result["channel"] == "#general"
        assert "message_ts" in result
        assert "sent_at" in result
        
        # Test parameter validation
        assert slack_tool.validate_parameters({
            "channel": "#general",
            "message": "Hello"
        }) == True
        
        assert slack_tool.validate_parameters({
            "channel": "invalid",
            "message": "Hello"
        }) == False
    
    @pytest.mark.asyncio
    async def test_calendar_tool(self):
        """Test calendar tool functionality"""
        
        calendar_tool = CalendarTool()
        
        # Test create event
        result = await calendar_tool.execute(
            action="create_event",
            title="Team Meeting",
            start_time="2024-01-15T10:00:00Z",
            end_time="2024-01-15T11:00:00Z"
        )
        
        assert result["success"] == True
        assert result["action"] == "create_event"
        assert "result" in result
        
        # Test list events
        result = await calendar_tool.execute(action="list_events")
        
        assert result["success"] == True
        assert result["action"] == "list_events"
        assert isinstance(result["result"], list)
        
        # Test parameter validation
        assert calendar_tool.validate_parameters({
            "action": "create_event",
            "title": "Meeting",
            "start_time": "2024-01-15T10:00:00Z",
            "end_time": "2024-01-15T11:00:00Z"
        }) == True
        
        assert calendar_tool.validate_parameters({
            "action": "invalid_action"
        }) == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])