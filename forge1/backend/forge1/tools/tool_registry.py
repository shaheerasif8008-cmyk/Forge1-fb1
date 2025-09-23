# forge1/backend/forge1/tools/tool_registry.py
"""
Tool Registry and Management System for Forge 1

Comprehensive tool integration system providing:
- Centralized tool discovery and registration
- Standardized tool integration framework
- Tool versioning and compatibility management
- Enterprise security and authentication
- Performance monitoring and optimization
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union, Callable, Type
from enum import Enum
from abc import ABC, abstractmethod
import json
import uuid
import importlib
import inspect

from forge1.core.security_manager import SecurityManager
from forge1.core.performance_monitor import PerformanceMonitor
from forge1.core.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class ToolCategory(Enum):
    """Categories of tools available in the registry"""
    COMMUNICATION = "communication"
    PRODUCTIVITY = "productivity"
    DATA_ANALYSIS = "data_analysis"
    AUTOMATION = "automation"
    INTEGRATION = "integration"
    SECURITY = "security"
    MONITORING = "monitoring"
    CUSTOM = "custom"

class ToolStatus(Enum):
    """Status of tools in the registry"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    MAINTENANCE = "maintenance"
    BETA = "beta"

class AuthenticationType(Enum):
    """Types of authentication supported by tools"""
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BASIC_AUTH = "basic_auth"
    BEARER_TOKEN = "bearer_token"
    CERTIFICATE = "certificate"
    NONE = "none"

class BaseTool(ABC):
    """Base class for all Forge 1 tools"""
    
    def __init__(
        self,
        name: str,
        description: str,
        category: ToolCategory,
        version: str = "1.0.0",
        authentication_type: AuthenticationType = AuthenticationType.NONE,
        **kwargs
    ):
        """Initialize base tool
        
        Args:
            name: Tool name
            description: Tool description
            category: Tool category
            version: Tool version
            authentication_type: Type of authentication required
            **kwargs: Additional tool configuration
        """
        self.name = name
        self.description = description
        self.category = category
        self.version = version
        self.authentication_type = authentication_type
        self.tool_id = f"tool_{uuid.uuid4().hex[:8]}"
        self.created_at = datetime.now(timezone.utc)
        self.config = kwargs
        
        # Tool metadata
        self.metadata = {
            "tool_id": self.tool_id,
            "name": name,
            "description": description,
            "category": category.value,
            "version": version,
            "authentication_type": authentication_type.value,
            "created_at": self.created_at.isoformat(),
            "config": kwargs
        }
        
        # Performance metrics
        self.metrics = {
            "executions": 0,
            "successes": 0,
            "failures": 0,
            "average_execution_time": 0.0,
            "last_execution": None,
            "total_execution_time": 0.0
        }
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters
        
        Returns:
            Tool execution result
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate tool parameters
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            True if parameters are valid
        """
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for parameter validation
        
        Returns:
            Tool schema definition
        """
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "version": self.version,
            "authentication_type": self.authentication_type.value,
            "parameters": self._get_parameter_schema(),
            "returns": self._get_return_schema()
        }
    
    def _get_parameter_schema(self) -> Dict[str, Any]:
        """Get parameter schema - to be overridden by subclasses"""
        return {"type": "object", "properties": {}}
    
    def _get_return_schema(self) -> Dict[str, Any]:
        """Get return schema - to be overridden by subclasses"""
        return {"type": "object", "properties": {"result": {"type": "string"}}}
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get tool metadata"""
        return self.metadata.copy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get tool performance metrics"""
        return self.metrics.copy()
    
    async def _update_metrics(self, execution_time: float, success: bool) -> None:
        """Update tool performance metrics"""
        self.metrics["executions"] += 1
        self.metrics["last_execution"] = datetime.now(timezone.utc).isoformat()
        self.metrics["total_execution_time"] += execution_time
        
        if success:
            self.metrics["successes"] += 1
        else:
            self.metrics["failures"] += 1
        
        # Update average execution time
        self.metrics["average_execution_time"] = (
            self.metrics["total_execution_time"] / self.metrics["executions"]
        )

class ToolRegistry:
    """Centralized tool registry and management system"""
    
    def __init__(
        self,
        security_manager: SecurityManager,
        performance_monitor: PerformanceMonitor,
        memory_manager: MemoryManager
    ):
        """Initialize tool registry
        
        Args:
            security_manager: Security management system
            performance_monitor: Performance monitoring system
            memory_manager: Memory management system
        """
        self.security_manager = security_manager
        self.performance_monitor = performance_monitor
        self.memory_manager = memory_manager
        
        # Registry state
        self.tools: Dict[str, BaseTool] = {}
        self.tool_categories: Dict[ToolCategory, List[str]] = {
            category: [] for category in ToolCategory
        }
        self.tool_versions: Dict[str, List[str]] = {}  # tool_name -> [versions]
        self.tool_dependencies: Dict[str, List[str]] = {}  # tool_id -> [dependency_ids]
        
        # Registry metadata
        self.registry_id = f"registry_{uuid.uuid4().hex[:8]}"
        self.created_at = datetime.now(timezone.utc)
        
        # Performance metrics
        self.registry_metrics = {
            "tools_registered": 0,
            "tools_executed": 0,
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "categories_used": 0
        }
        
        logger.info(f"Tool registry {self.registry_id} initialized")
    
    async def register_tool(
        self,
        tool: BaseTool,
        dependencies: List[str] = None,
        replace_existing: bool = False
    ) -> str:
        """Register a tool in the registry
        
        Args:
            tool: Tool instance to register
            dependencies: List of tool IDs this tool depends on
            replace_existing: Whether to replace existing tool with same name
            
        Returns:
            Tool ID
        """
        
        # Check if tool already exists
        existing_tool_id = self._find_tool_by_name(tool.name)
        if existing_tool_id and not replace_existing:
            raise ValueError(f"Tool {tool.name} already exists. Use replace_existing=True to replace.")
        
        # Security validation
        await self.security_manager.validate_tool_registration(
            tool_name=tool.name,
            tool_category=tool.category.value,
            authentication_type=tool.authentication_type.value
        )
        
        # Register tool
        if existing_tool_id and replace_existing:
            # Remove old version
            await self.unregister_tool(existing_tool_id)
        
        self.tools[tool.tool_id] = tool
        
        # Update category mapping
        if tool.tool_id not in self.tool_categories[tool.category]:
            self.tool_categories[tool.category].append(tool.tool_id)
        
        # Update version tracking
        if tool.name not in self.tool_versions:
            self.tool_versions[tool.name] = []
        if tool.version not in self.tool_versions[tool.name]:
            self.tool_versions[tool.name].append(tool.version)
        
        # Store dependencies
        if dependencies:
            self.tool_dependencies[tool.tool_id] = dependencies
        
        # Store tool metadata in memory
        await self.memory_manager.store_memory(
            content=json.dumps(tool.get_metadata()),
            memory_type="tool_metadata",
            metadata={
                "tool_id": tool.tool_id,
                "tool_name": tool.name,
                "category": tool.category.value,
                "version": tool.version
            }
        )
        
        # Update metrics
        self.registry_metrics["tools_registered"] += 1
        if tool.category not in [cat for cat, tools in self.tool_categories.items() if tools]:
            self.registry_metrics["categories_used"] += 1
        
        logger.info(f"Registered tool {tool.name} (v{tool.version}) with ID {tool.tool_id}")
        return tool.tool_id
    
    async def unregister_tool(self, tool_id: str) -> bool:
        """Unregister a tool from the registry
        
        Args:
            tool_id: ID of tool to unregister
            
        Returns:
            True if tool was unregistered successfully
        """
        
        if tool_id not in self.tools:
            return False
        
        tool = self.tools[tool_id]
        
        # Remove from category mapping
        if tool_id in self.tool_categories[tool.category]:
            self.tool_categories[tool.category].remove(tool_id)
        
        # Remove dependencies
        if tool_id in self.tool_dependencies:
            del self.tool_dependencies[tool_id]
        
        # Remove from tools
        del self.tools[tool_id]
        
        logger.info(f"Unregistered tool {tool.name} with ID {tool_id}")
        return True
    
    def get_tool(self, tool_id: str) -> Optional[BaseTool]:
        """Get tool by ID
        
        Args:
            tool_id: Tool ID
            
        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(tool_id)
    
    def find_tool_by_name(self, tool_name: str, version: str = None) -> Optional[BaseTool]:
        """Find tool by name and optionally version
        
        Args:
            tool_name: Tool name
            version: Tool version (optional)
            
        Returns:
            Tool instance or None if not found
        """
        
        for tool in self.tools.values():
            if tool.name == tool_name:
                if version is None or tool.version == version:
                    return tool
        
        return None
    
    def _find_tool_by_name(self, tool_name: str) -> Optional[str]:
        """Find tool ID by name
        
        Args:
            tool_name: Tool name
            
        Returns:
            Tool ID or None if not found
        """
        
        for tool_id, tool in self.tools.items():
            if tool.name == tool_name:
                return tool_id
        
        return None
    
    def get_tools_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """Get all tools in a category
        
        Args:
            category: Tool category
            
        Returns:
            List of tools in the category
        """
        
        tool_ids = self.tool_categories.get(category, [])
        return [self.tools[tool_id] for tool_id in tool_ids if tool_id in self.tools]
    
    def search_tools(
        self,
        query: str = None,
        category: ToolCategory = None,
        authentication_type: AuthenticationType = None,
        status: ToolStatus = None
    ) -> List[BaseTool]:
        """Search tools based on criteria
        
        Args:
            query: Search query (searches name and description)
            category: Filter by category
            authentication_type: Filter by authentication type
            status: Filter by status
            
        Returns:
            List of matching tools
        """
        
        results = []
        
        for tool in self.tools.values():
            # Category filter
            if category and tool.category != category:
                continue
            
            # Authentication type filter
            if authentication_type and tool.authentication_type != authentication_type:
                continue
            
            # Query filter
            if query:
                query_lower = query.lower()
                if (query_lower not in tool.name.lower() and 
                    query_lower not in tool.description.lower()):
                    continue
            
            results.append(tool)
        
        return results
    
    async def execute_tool(
        self,
        tool_id: str,
        parameters: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute a tool with given parameters
        
        Args:
            tool_id: Tool ID
            parameters: Tool parameters
            context: Execution context
            
        Returns:
            Tool execution result
        """
        
        if tool_id not in self.tools:
            return {
                "success": False,
                "error": f"Tool {tool_id} not found",
                "tool_id": tool_id
            }
        
        tool = self.tools[tool_id]
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now(timezone.utc)
        
        try:
            # Validate parameters
            if not tool.validate_parameters(parameters):
                return {
                    "success": False,
                    "error": "Invalid parameters",
                    "tool_id": tool_id,
                    "execution_id": execution_id
                }
            
            # Security validation
            await self.security_manager.validate_tool_execution(
                tool_id=tool_id,
                tool_name=tool.name,
                parameters=parameters,
                context=context or {}
            )
            
            # Execute tool
            result = await tool.execute(**parameters)
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Update tool metrics
            await tool._update_metrics(execution_time, True)
            
            # Update registry metrics
            await self._update_registry_metrics(execution_time, True)
            
            # Track performance
            await self.performance_monitor.track_tool_execution(
                tool_id=tool_id,
                tool_name=tool.name,
                execution_id=execution_id,
                execution_time=execution_time,
                success=True,
                parameters=parameters
            )
            
            return {
                "success": True,
                "result": result,
                "tool_id": tool_id,
                "execution_id": execution_id,
                "execution_time": execution_time,
                "tool_name": tool.name
            }
            
        except Exception as e:
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Update tool metrics
            await tool._update_metrics(execution_time, False)
            
            # Update registry metrics
            await self._update_registry_metrics(execution_time, False)
            
            # Track performance
            await self.performance_monitor.track_tool_execution(
                tool_id=tool_id,
                tool_name=tool.name,
                execution_id=execution_id,
                execution_time=execution_time,
                success=False,
                error=str(e),
                parameters=parameters
            )
            
            logger.error(f"Tool execution failed for {tool_id}: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "tool_id": tool_id,
                "execution_id": execution_id,
                "execution_time": execution_time,
                "tool_name": tool.name
            }
    
    async def _update_registry_metrics(self, execution_time: float, success: bool) -> None:
        """Update registry performance metrics"""
        
        self.registry_metrics["total_executions"] += 1
        
        if success:
            self.registry_metrics["successful_executions"] += 1
        else:
            self.registry_metrics["failed_executions"] += 1
        
        # Update average execution time
        current_avg = self.registry_metrics["average_execution_time"]
        total_executions = self.registry_metrics["total_executions"]
        self.registry_metrics["average_execution_time"] = (
            (current_avg * (total_executions - 1) + execution_time) / total_executions
        )
    
    def get_tool_dependencies(self, tool_id: str) -> List[str]:
        """Get tool dependencies
        
        Args:
            tool_id: Tool ID
            
        Returns:
            List of dependency tool IDs
        """
        return self.tool_dependencies.get(tool_id, [])
    
    def get_tool_versions(self, tool_name: str) -> List[str]:
        """Get all versions of a tool
        
        Args:
            tool_name: Tool name
            
        Returns:
            List of versions
        """
        return self.tool_versions.get(tool_name, [])
    
    def get_registry_info(self) -> Dict[str, Any]:
        """Get registry information and statistics
        
        Returns:
            Registry information
        """
        
        category_stats = {}
        for category, tool_ids in self.tool_categories.items():
            category_stats[category.value] = {
                "count": len(tool_ids),
                "tools": [self.tools[tid].name for tid in tool_ids if tid in self.tools]
            }
        
        return {
            "registry_id": self.registry_id,
            "created_at": self.created_at.isoformat(),
            "total_tools": len(self.tools),
            "categories": category_stats,
            "metrics": self.registry_metrics.copy(),
            "tool_versions_count": len(self.tool_versions),
            "dependencies_count": len(self.tool_dependencies)
        }
    
    def get_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered tools
        
        Returns:
            Dictionary of tool information
        """
        
        return {
            tool_id: {
                "metadata": tool.get_metadata(),
                "metrics": tool.get_metrics(),
                "schema": tool.get_schema(),
                "dependencies": self.get_tool_dependencies(tool_id)
            }
            for tool_id, tool in self.tools.items()
        }
    
    async def validate_tool_compatibility(self, tool_ids: List[str]) -> Dict[str, Any]:
        """Validate compatibility between multiple tools
        
        Args:
            tool_ids: List of tool IDs to check compatibility
            
        Returns:
            Compatibility analysis
        """
        
        compatibility_report = {
            "compatible": True,
            "issues": [],
            "warnings": [],
            "tool_analysis": {}
        }
        
        tools = [self.tools[tid] for tid in tool_ids if tid in self.tools]
        
        if len(tools) != len(tool_ids):
            missing_tools = [tid for tid in tool_ids if tid not in self.tools]
            compatibility_report["issues"].append(f"Missing tools: {missing_tools}")
            compatibility_report["compatible"] = False
        
        # Check authentication compatibility
        auth_types = set(tool.authentication_type for tool in tools)
        if len(auth_types) > 3:  # Arbitrary threshold
            compatibility_report["warnings"].append(
                f"Multiple authentication types required: {[at.value for at in auth_types]}"
            )
        
        # Check category distribution
        categories = [tool.category for tool in tools]
        category_counts = {}
        for cat in categories:
            category_counts[cat.value] = category_counts.get(cat.value, 0) + 1
        
        compatibility_report["tool_analysis"] = {
            "total_tools": len(tools),
            "authentication_types": [at.value for at in auth_types],
            "category_distribution": category_counts,
            "version_info": {tool.name: tool.version for tool in tools}
        }
        
        return compatibility_report
    
    async def export_registry(self, include_metrics: bool = True) -> Dict[str, Any]:
        """Export registry configuration and data
        
        Args:
            include_metrics: Whether to include performance metrics
            
        Returns:
            Registry export data
        """
        
        export_data = {
            "registry_info": self.get_registry_info(),
            "tools": {},
            "categories": {cat.value: tool_ids for cat, tool_ids in self.tool_categories.items()},
            "dependencies": self.tool_dependencies.copy(),
            "versions": self.tool_versions.copy(),
            "export_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Export tool data
        for tool_id, tool in self.tools.items():
            tool_data = {
                "metadata": tool.get_metadata(),
                "schema": tool.get_schema()
            }
            
            if include_metrics:
                tool_data["metrics"] = tool.get_metrics()
            
            export_data["tools"][tool_id] = tool_data
        
        return export_data
    
    async def import_registry(self, import_data: Dict[str, Any], merge: bool = False) -> Dict[str, Any]:
        """Import registry configuration and data
        
        Args:
            import_data: Registry import data
            merge: Whether to merge with existing registry or replace
            
        Returns:
            Import result
        """
        
        import_result = {
            "success": True,
            "imported_tools": 0,
            "skipped_tools": 0,
            "errors": []
        }
        
        try:
            if not merge:
                # Clear existing registry
                self.tools.clear()
                self.tool_categories = {category: [] for category in ToolCategory}
                self.tool_versions.clear()
                self.tool_dependencies.clear()
            
            # Import tools (metadata only - actual tool instances need to be recreated)
            tools_data = import_data.get("tools", {})
            for tool_id, tool_data in tools_data.items():
                try:
                    # Store tool metadata for reference
                    metadata = tool_data["metadata"]
                    await self.memory_manager.store_memory(
                        content=json.dumps(metadata),
                        memory_type="imported_tool_metadata",
                        metadata={
                            "tool_id": tool_id,
                            "tool_name": metadata["name"],
                            "import_timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    )
                    
                    import_result["imported_tools"] += 1
                    
                except Exception as e:
                    import_result["errors"].append(f"Failed to import tool {tool_id}: {e}")
                    import_result["skipped_tools"] += 1
            
            # Import dependencies
            if "dependencies" in import_data:
                self.tool_dependencies.update(import_data["dependencies"])
            
            # Import versions
            if "versions" in import_data:
                self.tool_versions.update(import_data["versions"])
            
            logger.info(f"Registry import completed: {import_result['imported_tools']} tools imported")
            
        except Exception as e:
            import_result["success"] = False
            import_result["errors"].append(f"Import failed: {e}")
            logger.error(f"Registry import failed: {e}")
        
        return import_result