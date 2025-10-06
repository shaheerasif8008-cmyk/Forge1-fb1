"""
Tool Reliability and Planning Excellence System
Comprehensive tool success tracking, benchmarking, and self-healing capabilities
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import json
import logging
from collections import defaultdict, deque
import statistics

from forge1.backend.forge1.core.monitoring import MetricsCollector


class ToolStatus(Enum):
    """Tool execution status"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    RETRY = "retry"
    DEGRADED = "degraded"


class FrameworkType(Enum):
    """Supported AI frameworks"""
    LANGCHAIN = "langchain"
    CREWAI = "crewai"
    AUTOGEN = "autogen"
    CUSTOM = "custom"


@dataclass
class ToolExecution:
    """Record of a tool execution attempt"""
    tool_name: str
    framework: FrameworkType
    start_time: datetime
    end_time: Optional[datetime] = None
    status: ToolStatus = ToolStatus.SUCCESS
    latency_ms: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolBenchmark:
    """Benchmark metrics for a tool"""
    tool_name: str
    framework: FrameworkType
    success_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    timeout_rate: float
    total_executions: int
    last_updated: datetime
    performance_trend: str  # improving, stable, degrading


@dataclass
class PlanningTask:
    """A planning task with execution tracking"""
    task_id: str
    description: str
    framework: FrameworkType
    tools_required: List[str]
    estimated_duration: timedelta
    priority: int
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_count: int = 0
    retry_count: int = 0


class ToolReliabilityManager:
    """
    Manages tool reliability tracking, benchmarking, and self-healing
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.logger = logging.getLogger("tool_reliability")
        
        # Execution tracking
        self.executions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.benchmarks: Dict[str, ToolBenchmark] = {}
        
        # Reliability thresholds
        self.success_threshold = 0.99  # 99% success rate target
        self.latency_threshold_ms = 5000  # 5 second max latency
        self.error_rate_threshold = 0.01  # 1% max error rate
        
        # Self-healing configuration
        self.max_retries = 3
        self.retry_backoff_base = 1.0  # seconds
        self.circuit_breaker_threshold = 0.5  # 50% failure rate triggers circuit breaker
        self.circuit_breaker_timeout = 300  # 5 minutes
        
        # Circuit breaker state
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Tool registry
        self.registered_tools: Dict[str, Dict[str, Any]] = {}
    
    def register_tool(
        self, 
        tool_name: str, 
        framework: FrameworkType,
        expected_latency_ms: float = 1000,
        timeout_ms: float = 30000,
        retry_strategy: str = "exponential_backoff"
    ) -> None:
        """Register a tool with expected performance characteristics"""
        
        self.registered_tools[tool_name] = {
            "framework": framework,
            "expected_latency_ms": expected_latency_ms,
            "timeout_ms": timeout_ms,
            "retry_strategy": retry_strategy,
            "registered_at": datetime.utcnow()
        }
        
        self.logger.info(f"Registered tool: {tool_name} ({framework.value})")
    
    async def execute_with_tracking(
        self,
        tool_name: str,
        tool_function: Callable,
        *args,
        **kwargs
    ) -> Tuple[Any, ToolExecution]:
        """
        Execute a tool with comprehensive tracking and reliability features
        """
        if tool_name not in self.registered_tools:
            raise ValueError(f"Tool {tool_name} not registered")
        
        tool_config = self.registered_tools[tool_name]
        framework = tool_config["framework"]
        
        # Check circuit breaker
        if self._is_circuit_breaker_open(tool_name):
            raise Exception(f"Circuit breaker open for tool: {tool_name}")
        
        execution = ToolExecution(
            tool_name=tool_name,
            framework=framework,
            start_time=datetime.utcnow()
        )
        
        try:
            # Execute with timeout
            timeout_seconds = tool_config["timeout_ms"] / 1000
            result = await asyncio.wait_for(
                tool_function(*args, **kwargs),
                timeout=timeout_seconds
            )
            
            # Record successful execution
            execution.end_time = datetime.utcnow()
            execution.status = ToolStatus.SUCCESS
            execution.latency_ms = (execution.end_time - execution.start_time).total_seconds() * 1000
            
            # Track input/output sizes if possible
            execution.input_size = self._estimate_size(args, kwargs)
            execution.output_size = self._estimate_size(result)
            
            self._record_execution(execution)
            self._update_circuit_breaker(tool_name, True)
            
            return result, execution
            
        except asyncio.TimeoutError:
            execution.end_time = datetime.utcnow()
            execution.status = ToolStatus.TIMEOUT
            execution.error_message = f"Tool execution timed out after {timeout_seconds}s"
            
            self._record_execution(execution)
            self._update_circuit_breaker(tool_name, False)
            
            raise Exception(execution.error_message)
            
        except Exception as e:
            execution.end_time = datetime.utcnow()
            execution.status = ToolStatus.FAILURE
            execution.error_message = str(e)
            
            self._record_execution(execution)
            self._update_circuit_breaker(tool_name, False)
            
            raise e
    
    async def execute_with_retry(
        self,
        tool_name: str,
        tool_function: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a tool with automatic retry and backoff
        """
        tool_config = self.registered_tools[tool_name]
        retry_strategy = tool_config.get("retry_strategy", "exponential_backoff")
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result, execution = await self.execute_with_tracking(
                    tool_name, tool_function, *args, **kwargs
                )
                
                if attempt > 0:
                    self.logger.info(f"Tool {tool_name} succeeded on retry {attempt}")
                    self.metrics.increment(f"tool_retry_success_{tool_name}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    # Calculate backoff delay
                    if retry_strategy == "exponential_backoff":
                        delay = self.retry_backoff_base * (2 ** attempt)
                    else:
                        delay = self.retry_backoff_base
                    
                    self.logger.warning(
                        f"Tool {tool_name} failed on attempt {attempt + 1}, "
                        f"retrying in {delay}s: {str(e)}"
                    )
                    
                    self.metrics.increment(f"tool_retry_attempt_{tool_name}")
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(
                        f"Tool {tool_name} failed after {self.max_retries} retries: {str(e)}"
                    )
                    self.metrics.increment(f"tool_retry_exhausted_{tool_name}")
        
        raise last_exception
    
    def get_tool_benchmark(self, tool_name: str) -> Optional[ToolBenchmark]:
        """Get current benchmark metrics for a tool"""
        return self.benchmarks.get(tool_name)
    
    def get_framework_benchmarks(self, framework: FrameworkType) -> List[ToolBenchmark]:
        """Get benchmarks for all tools in a framework"""
        return [
            benchmark for benchmark in self.benchmarks.values()
            if benchmark.framework == framework
        ]
    
    def update_benchmarks(self) -> None:
        """Update benchmark metrics for all tools"""
        for tool_name, executions in self.executions.items():
            if not executions:
                continue
            
            # Calculate metrics from recent executions
            recent_executions = list(executions)[-100:]  # Last 100 executions
            
            if not recent_executions:
                continue
            
            successful_executions = [
                ex for ex in recent_executions 
                if ex.status == ToolStatus.SUCCESS
            ]
            
            failed_executions = [
                ex for ex in recent_executions 
                if ex.status == ToolStatus.FAILURE
            ]
            
            timeout_executions = [
                ex for ex in recent_executions 
                if ex.status == ToolStatus.TIMEOUT
            ]
            
            # Calculate success rate
            success_rate = len(successful_executions) / len(recent_executions)
            
            # Calculate latency metrics
            latencies = [
                ex.latency_ms for ex in successful_executions 
                if ex.latency_ms is not None
            ]
            
            if latencies:
                avg_latency = statistics.mean(latencies)
                p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
                p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
            else:
                avg_latency = p95_latency = p99_latency = 0.0
            
            # Calculate error rates
            error_rate = len(failed_executions) / len(recent_executions)
            timeout_rate = len(timeout_executions) / len(recent_executions)
            
            # Determine performance trend
            if len(recent_executions) >= 20:
                first_half = recent_executions[:len(recent_executions)//2]
                second_half = recent_executions[len(recent_executions)//2:]
                
                first_half_success = sum(1 for ex in first_half if ex.status == ToolStatus.SUCCESS) / len(first_half)
                second_half_success = sum(1 for ex in second_half if ex.status == ToolStatus.SUCCESS) / len(second_half)
                
                if second_half_success > first_half_success + 0.05:
                    trend = "improving"
                elif second_half_success < first_half_success - 0.05:
                    trend = "degrading"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"
            
            # Get framework from first execution
            framework = recent_executions[0].framework
            
            # Update benchmark
            self.benchmarks[tool_name] = ToolBenchmark(
                tool_name=tool_name,
                framework=framework,
                success_rate=success_rate,
                avg_latency_ms=avg_latency,
                p95_latency_ms=p95_latency,
                p99_latency_ms=p99_latency,
                error_rate=error_rate,
                timeout_rate=timeout_rate,
                total_executions=len(recent_executions),
                last_updated=datetime.utcnow(),
                performance_trend=trend
            )
            
            # Record metrics
            self.metrics.record_metric(f"tool_success_rate_{tool_name}", success_rate)
            self.metrics.record_metric(f"tool_avg_latency_{tool_name}", avg_latency)
            self.metrics.record_metric(f"tool_error_rate_{tool_name}", error_rate)
    
    def get_reliability_report(self) -> Dict[str, Any]:
        """Generate comprehensive reliability report"""
        self.update_benchmarks()
        
        # Overall statistics
        all_benchmarks = list(self.benchmarks.values())
        
        if not all_benchmarks:
            return {"status": "no_data", "tools": []}
        
        overall_success_rate = statistics.mean([b.success_rate for b in all_benchmarks])
        overall_avg_latency = statistics.mean([b.avg_latency_ms for b in all_benchmarks])
        overall_error_rate = statistics.mean([b.error_rate for b in all_benchmarks])
        
        # Framework breakdown
        framework_stats = {}
        for framework in FrameworkType:
            framework_benchmarks = self.get_framework_benchmarks(framework)
            if framework_benchmarks:
                framework_stats[framework.value] = {
                    "tool_count": len(framework_benchmarks),
                    "avg_success_rate": statistics.mean([b.success_rate for b in framework_benchmarks]),
                    "avg_latency_ms": statistics.mean([b.avg_latency_ms for b in framework_benchmarks]),
                    "avg_error_rate": statistics.mean([b.error_rate for b in framework_benchmarks])
                }
        
        # Identify problematic tools
        problematic_tools = [
            b.tool_name for b in all_benchmarks
            if (b.success_rate < self.success_threshold or 
                b.avg_latency_ms > self.latency_threshold_ms or
                b.error_rate > self.error_rate_threshold)
        ]
        
        # Performance trends
        improving_tools = [b.tool_name for b in all_benchmarks if b.performance_trend == "improving"]
        degrading_tools = [b.tool_name for b in all_benchmarks if b.performance_trend == "degrading"]
        
        return {
            "status": "healthy" if overall_success_rate >= self.success_threshold else "degraded",
            "overall_metrics": {
                "success_rate": overall_success_rate,
                "avg_latency_ms": overall_avg_latency,
                "error_rate": overall_error_rate,
                "total_tools": len(all_benchmarks)
            },
            "framework_breakdown": framework_stats,
            "problematic_tools": problematic_tools,
            "performance_trends": {
                "improving": improving_tools,
                "degrading": degrading_tools
            },
            "circuit_breakers": {
                tool: state for tool, state in self.circuit_breakers.items()
                if state.get("is_open", False)
            },
            "generated_at": datetime.utcnow().isoformat()
        }
    
    # Private methods
    def _record_execution(self, execution: ToolExecution) -> None:
        """Record a tool execution"""
        self.executions[execution.tool_name].append(execution)
        
        # Record metrics
        self.metrics.increment(f"tool_execution_{execution.tool_name}")
        self.metrics.increment(f"tool_status_{execution.status.value}_{execution.tool_name}")
        
        if execution.latency_ms:
            self.metrics.record_metric(f"tool_latency_{execution.tool_name}", execution.latency_ms)
    
    def _is_circuit_breaker_open(self, tool_name: str) -> bool:
        """Check if circuit breaker is open for a tool"""
        if tool_name not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[tool_name]
        
        if not breaker.get("is_open", False):
            return False
        
        # Check if timeout has passed
        if datetime.utcnow() > breaker.get("open_until", datetime.utcnow()):
            breaker["is_open"] = False
            self.logger.info(f"Circuit breaker closed for tool: {tool_name}")
            return False
        
        return True
    
    def _update_circuit_breaker(self, tool_name: str, success: bool) -> None:
        """Update circuit breaker state based on execution result"""
        if tool_name not in self.circuit_breakers:
            self.circuit_breakers[tool_name] = {
                "failure_count": 0,
                "success_count": 0,
                "is_open": False
            }
        
        breaker = self.circuit_breakers[tool_name]
        
        if success:
            breaker["success_count"] += 1
            breaker["failure_count"] = max(0, breaker["failure_count"] - 1)
        else:
            breaker["failure_count"] += 1
        
        # Calculate recent failure rate
        total_recent = breaker["success_count"] + breaker["failure_count"]
        if total_recent >= 10:  # Minimum sample size
            failure_rate = breaker["failure_count"] / total_recent
            
            if failure_rate >= self.circuit_breaker_threshold and not breaker["is_open"]:
                # Open circuit breaker
                breaker["is_open"] = True
                breaker["open_until"] = datetime.utcnow() + timedelta(seconds=self.circuit_breaker_timeout)
                
                self.logger.warning(
                    f"Circuit breaker opened for tool: {tool_name} "
                    f"(failure rate: {failure_rate:.2%})"
                )
                self.metrics.increment(f"circuit_breaker_opened_{tool_name}")
    
    def _estimate_size(self, *objects) -> int:
        """Estimate size of objects in bytes"""
        try:
            return len(json.dumps(objects, default=str))
        except:
            return 0


class PlanningReliabilityManager:
    """
    Manages planning task reliability and execution optimization
    """
    
    def __init__(self, tool_reliability: ToolReliabilityManager, metrics_collector: MetricsCollector):
        self.tool_reliability = tool_reliability
        self.metrics = metrics_collector
        self.logger = logging.getLogger("planning_reliability")
        
        # Task tracking
        self.active_tasks: Dict[str, PlanningTask] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        
        # Planning performance targets
        self.planning_success_rate_target = 0.95  # 95%
        self.planning_error_threshold = 0.05  # 5%
    
    def create_planning_task(
        self,
        task_id: str,
        description: str,
        framework: FrameworkType,
        tools_required: List[str],
        estimated_duration: timedelta,
        priority: int = 1,
        dependencies: List[str] = None
    ) -> PlanningTask:
        """Create a new planning task"""
        
        task = PlanningTask(
            task_id=task_id,
            description=description,
            framework=framework,
            tools_required=tools_required,
            estimated_duration=estimated_duration,
            priority=priority,
            dependencies=dependencies or []
        )
        
        self.active_tasks[task_id] = task
        self.logger.info(f"Created planning task: {task_id}")
        
        return task
    
    async def execute_planning_task(self, task_id: str) -> Dict[str, Any]:
        """Execute a planning task with reliability tracking"""
        
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task = self.active_tasks[task_id]
        task.start_time = datetime.utcnow()
        task.status = "running"
        
        try:
            # Check dependencies
            for dep_id in task.dependencies:
                if dep_id in self.active_tasks:
                    raise Exception(f"Dependency {dep_id} not completed")
            
            # Validate tool availability
            for tool_name in task.tools_required:
                if self.tool_reliability._is_circuit_breaker_open(tool_name):
                    raise Exception(f"Required tool {tool_name} unavailable (circuit breaker open)")
            
            # Execute task (placeholder - would contain actual planning logic)
            result = await self._execute_task_logic(task)
            
            # Mark as completed
            task.end_time = datetime.utcnow()
            task.status = "completed"
            
            # Move to completed tasks
            self.completed_tasks.append(task)
            del self.active_tasks[task_id]
            
            self.metrics.increment("planning_task_success")
            self.logger.info(f"Planning task completed: {task_id}")
            
            return {
                "task_id": task_id,
                "status": "completed",
                "result": result,
                "duration_seconds": (task.end_time - task.start_time).total_seconds(),
                "error_count": task.error_count,
                "retry_count": task.retry_count
            }
            
        except Exception as e:
            task.error_count += 1
            task.status = "failed"
            
            self.metrics.increment("planning_task_failure")
            self.logger.error(f"Planning task failed: {task_id} - {str(e)}")
            
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
                "error_count": task.error_count
            }
    
    def get_planning_metrics(self) -> Dict[str, Any]:
        """Get planning performance metrics"""
        
        if not self.completed_tasks:
            return {"status": "no_data"}
        
        recent_tasks = list(self.completed_tasks)[-100:]  # Last 100 tasks
        
        successful_tasks = [t for t in recent_tasks if t.status == "completed"]
        failed_tasks = [t for t in recent_tasks if t.status == "failed"]
        
        success_rate = len(successful_tasks) / len(recent_tasks)
        error_rate = len(failed_tasks) / len(recent_tasks)
        
        # Calculate average duration
        durations = [
            (t.end_time - t.start_time).total_seconds()
            for t in successful_tasks
            if t.start_time and t.end_time
        ]
        
        avg_duration = statistics.mean(durations) if durations else 0
        
        # Framework breakdown
        framework_stats = {}
        for framework in FrameworkType:
            framework_tasks = [t for t in recent_tasks if t.framework == framework]
            if framework_tasks:
                framework_success = [t for t in framework_tasks if t.status == "completed"]
                framework_stats[framework.value] = {
                    "total_tasks": len(framework_tasks),
                    "success_rate": len(framework_success) / len(framework_tasks),
                    "avg_duration": statistics.mean([
                        (t.end_time - t.start_time).total_seconds()
                        for t in framework_success
                        if t.start_time and t.end_time
                    ]) if framework_success else 0
                }
        
        return {
            "overall_metrics": {
                "success_rate": success_rate,
                "error_rate": error_rate,
                "avg_duration_seconds": avg_duration,
                "total_tasks": len(recent_tasks)
            },
            "framework_breakdown": framework_stats,
            "active_tasks": len(self.active_tasks),
            "meets_targets": {
                "success_rate": success_rate >= self.planning_success_rate_target,
                "error_rate": error_rate <= self.planning_error_threshold
            }
        }
    
    async def _execute_task_logic(self, task: PlanningTask) -> Dict[str, Any]:
        """Execute the actual planning task logic"""
        # Placeholder for actual planning implementation
        # Would integrate with LangChain, CrewAI, AutoGen based on task.framework
        
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return {
            "framework": task.framework.value,
            "tools_used": task.tools_required,
            "planning_result": "Task planning completed successfully"
        }