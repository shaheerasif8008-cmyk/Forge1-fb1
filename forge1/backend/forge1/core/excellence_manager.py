"""
Multi-Agent Excellence Manager
Integrates tool reliability, code sandboxes, and adversarial evaluation for comprehensive agent excellence
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import logging

from forge1.backend.forge1.core.monitoring import MetricsCollector
from forge1.backend.forge1.core.models import ModelRouter
from forge1.backend.forge1.core.memory import MemoryManager
from forge1.backend.forge1.core.security import SecretManager
from forge1.backend.forge1.core.tool_reliability import (
    ToolReliabilityManager, PlanningReliabilityManager, FrameworkType
)
from forge1.backend.forge1.core.code_sandbox import (
    CodeSandboxManager, SandboxType, ResourceLimits
)
from forge1.backend.forge1.core.adversarial_evaluation import (
    AdversarialEvaluationManager, AdversarialTaskType
)


@dataclass
class ExcellenceMetrics:
    """Comprehensive excellence metrics for AI agents"""
    tool_reliability_score: float
    planning_success_rate: float
    code_execution_safety: float
    adversarial_resilience: float
    overall_excellence_score: float
    last_updated: datetime


class MultiAgentExcellenceManager:
    """
    Comprehensive manager for multi-agent excellence across all reliability dimensions
    
    Integrates:
    - Tool reliability and planning excellence
    - Secure code execution capabilities
    - Adversarial evaluation and stress testing
    - Performance monitoring and optimization
    """
    
    def __init__(
        self,
        model_router: ModelRouter,
        memory_manager: MemoryManager,
        metrics_collector: MetricsCollector,
        secret_manager: SecretManager
    ):
        self.model_router = model_router
        self.memory_manager = memory_manager
        self.metrics = metrics_collector
        self.secret_manager = secret_manager
        self.logger = logging.getLogger("excellence_manager")
        
        # Initialize core components
        self.tool_reliability = ToolReliabilityManager(metrics_collector)
        self.planning_reliability = PlanningReliabilityManager(
            self.tool_reliability, metrics_collector
        )
        self.code_sandbox = CodeSandboxManager(metrics_collector, secret_manager)
        self.adversarial_evaluator = AdversarialEvaluationManager(
            model_router, memory_manager, metrics_collector
        )
        
        # Excellence tracking
        self.agent_excellence_scores: Dict[str, ExcellenceMetrics] = {}
        
        # Excellence targets
        self.excellence_targets = {
            "tool_reliability_score": 0.99,  # 99% tool success rate
            "planning_success_rate": 0.95,   # 95% planning success
            "code_execution_safety": 0.999,  # 99.9% safe code execution
            "adversarial_resilience": 0.85,  # 85% adversarial test pass rate
            "overall_excellence_score": 0.90 # 90% overall excellence
        }
    
    async def initialize_agent_excellence(self, agent_id: str) -> None:
        """Initialize excellence tracking for an agent"""
        
        self.logger.info(f"Initializing excellence tracking for agent: {agent_id}")
        
        # Register common tools for the agent
        await self._register_agent_tools(agent_id)
        
        # Initialize baseline metrics
        self.agent_excellence_scores[agent_id] = ExcellenceMetrics(
            tool_reliability_score=0.0,
            planning_success_rate=0.0,
            code_execution_safety=0.0,
            adversarial_resilience=0.0,
            overall_excellence_score=0.0,
            last_updated=datetime.utcnow()
        )
    
    async def evaluate_agent_excellence(
        self,
        agent_id: str,
        comprehensive: bool = True
    ) -> ExcellenceMetrics:
        """
        Perform comprehensive excellence evaluation for an agent
        
        Args:
            agent_id: ID of agent to evaluate
            comprehensive: Whether to run full evaluation suite
        
        Returns:
            Updated excellence metrics
        """
        
        self.logger.info(f"Evaluating excellence for agent: {agent_id}")
        
        # Ensure agent is initialized
        if agent_id not in self.agent_excellence_scores:
            await self.initialize_agent_excellence(agent_id)
        
        # Evaluate tool reliability
        tool_score = await self._evaluate_tool_reliability(agent_id)
        
        # Evaluate planning capabilities
        planning_score = await self._evaluate_planning_reliability(agent_id)
        
        # Evaluate code execution safety
        code_safety_score = await self._evaluate_code_execution_safety(agent_id)
        
        # Evaluate adversarial resilience
        if comprehensive:
            adversarial_score = await self._evaluate_adversarial_resilience(agent_id)
        else:
            # Use cached score for quick evaluation
            adversarial_score = self.agent_excellence_scores[agent_id].adversarial_resilience
        
        # Calculate overall excellence score
        overall_score = self._calculate_overall_excellence(
            tool_score, planning_score, code_safety_score, adversarial_score
        )
        
        # Update metrics
        excellence_metrics = ExcellenceMetrics(
            tool_reliability_score=tool_score,
            planning_success_rate=planning_score,
            code_execution_safety=code_safety_score,
            adversarial_resilience=adversarial_score,
            overall_excellence_score=overall_score,
            last_updated=datetime.utcnow()
        )
        
        self.agent_excellence_scores[agent_id] = excellence_metrics
        
        # Record metrics
        self._record_excellence_metrics(agent_id, excellence_metrics)
        
        # Store in memory for historical tracking
        await self._store_excellence_evaluation(agent_id, excellence_metrics)
        
        self.logger.info(
            f"Excellence evaluation completed for {agent_id}: "
            f"Overall score {overall_score:.3f}"
        )
        
        return excellence_metrics
    
    async def run_continuous_monitoring(
        self,
        agent_ids: List[str],
        monitoring_interval_minutes: int = 30
    ) -> None:
        """
        Run continuous excellence monitoring for multiple agents
        
        Args:
            agent_ids: List of agent IDs to monitor
            monitoring_interval_minutes: How often to run evaluations
        """
        
        self.logger.info(f"Starting continuous monitoring for {len(agent_ids)} agents")
        
        while True:
            try:
                # Evaluate each agent
                for agent_id in agent_ids:
                    try:
                        # Run quick evaluation (not comprehensive to avoid overhead)
                        await self.evaluate_agent_excellence(agent_id, comprehensive=False)
                        
                        # Check if intervention is needed
                        await self._check_excellence_thresholds(agent_id)
                        
                    except Exception as e:
                        self.logger.error(f"Error monitoring agent {agent_id}: {e}")
                
                # Wait for next monitoring cycle
                await asyncio.sleep(monitoring_interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(60)  # Brief pause before retrying
    
    async def run_stress_test_suite(
        self,
        agent_id: str,
        duration_minutes: int = 60,
        intensity_level: int = 5
    ) -> Dict[str, Any]:
        """
        Run comprehensive stress test suite on an agent
        
        Args:
            agent_id: ID of agent to stress test
            duration_minutes: Duration of stress test
            intensity_level: Intensity level (1-10)
        
        Returns:
            Comprehensive stress test report
        """
        
        self.logger.info(f"Starting stress test suite for agent: {agent_id}")
        
        # Generate stress test tasks
        stress_tasks = await self.adversarial_evaluator.task_generator.generate_task_suite(
            task_types=[
                AdversarialTaskType.STRESS_TEST,
                AdversarialTaskType.PERFORMANCE_DEGRADATION,
                AdversarialTaskType.RESOURCE_EXHAUSTION
            ],
            difficulty_range=(intensity_level, 10),
            tasks_per_type=10
        )
        
        # Run adversarial stress test
        adversarial_report = await self.adversarial_evaluator.run_stress_test_suite(
            agent_id, duration_minutes, concurrent_tasks=intensity_level * 2
        )
        
        # Run tool reliability stress test
        tool_stress_report = await self._run_tool_stress_test(agent_id, duration_minutes)
        
        # Run code execution stress test
        code_stress_report = await self._run_code_execution_stress_test(
            agent_id, duration_minutes, intensity_level
        )
        
        # Compile comprehensive report
        stress_report = {
            "agent_id": agent_id,
            "stress_test_summary": {
                "duration_minutes": duration_minutes,
                "intensity_level": intensity_level,
                "start_time": datetime.utcnow().isoformat(),
                "test_components": ["adversarial", "tool_reliability", "code_execution"]
            },
            "adversarial_results": adversarial_report,
            "tool_reliability_results": tool_stress_report,
            "code_execution_results": code_stress_report,
            "overall_assessment": self._assess_stress_test_results(
                adversarial_report, tool_stress_report, code_stress_report
            ),
            "recommendations": self._generate_stress_test_recommendations(
                adversarial_report, tool_stress_report, code_stress_report
            )
        }
        
        # Store stress test results
        await self._store_stress_test_results(agent_id, stress_report)
        
        self.logger.info(f"Stress test suite completed for agent: {agent_id}")
        
        return stress_report
    
    def get_excellence_dashboard(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get excellence dashboard for an agent or system overview
        
        Args:
            agent_id: Optional agent ID for specific dashboard
        
        Returns:
            Excellence dashboard data
        """
        
        if agent_id:
            # Agent-specific dashboard
            if agent_id not in self.agent_excellence_scores:
                return {"status": "agent_not_found", "agent_id": agent_id}
            
            metrics = self.agent_excellence_scores[agent_id]
            
            # Get detailed component metrics
            tool_reliability_report = self.tool_reliability.get_reliability_report()
            planning_metrics = self.planning_reliability.get_planning_metrics()
            sandbox_metrics = self.code_sandbox.get_sandbox_metrics()
            adversarial_metrics = self.adversarial_evaluator.get_evaluation_metrics(agent_id)
            
            return {
                "agent_id": agent_id,
                "excellence_summary": {
                    "overall_score": metrics.overall_excellence_score,
                    "tool_reliability": metrics.tool_reliability_score,
                    "planning_success": metrics.planning_success_rate,
                    "code_safety": metrics.code_execution_safety,
                    "adversarial_resilience": metrics.adversarial_resilience,
                    "last_updated": metrics.last_updated.isoformat()
                },
                "meets_targets": {
                    target: getattr(metrics, target) >= threshold
                    for target, threshold in self.excellence_targets.items()
                },
                "detailed_metrics": {
                    "tool_reliability": tool_reliability_report,
                    "planning_metrics": planning_metrics,
                    "sandbox_metrics": sandbox_metrics,
                    "adversarial_metrics": adversarial_metrics
                }
            }
        else:
            # System overview dashboard
            all_agents = list(self.agent_excellence_scores.keys())
            
            if not all_agents:
                return {"status": "no_agents_tracked"}
            
            # Calculate system-wide metrics
            system_metrics = {}
            for metric_name in ["tool_reliability_score", "planning_success_rate", 
                               "code_execution_safety", "adversarial_resilience", 
                               "overall_excellence_score"]:
                values = [getattr(metrics, metric_name) for metrics in self.agent_excellence_scores.values()]
                system_metrics[metric_name] = {
                    "average": sum(values) / len(values),
                    "minimum": min(values),
                    "maximum": max(values)
                }
            
            # Identify top and bottom performers
            sorted_agents = sorted(
                self.agent_excellence_scores.items(),
                key=lambda x: x[1].overall_excellence_score,
                reverse=True
            )
            
            return {
                "system_overview": {
                    "total_agents": len(all_agents),
                    "system_metrics": system_metrics,
                    "top_performers": [
                        {"agent_id": agent_id, "score": metrics.overall_excellence_score}
                        for agent_id, metrics in sorted_agents[:5]
                    ],
                    "needs_attention": [
                        {"agent_id": agent_id, "score": metrics.overall_excellence_score}
                        for agent_id, metrics in sorted_agents[-5:]
                        if metrics.overall_excellence_score < self.excellence_targets["overall_excellence_score"]
                    ]
                }
            }
    
    # Private methods
    async def _register_agent_tools(self, agent_id: str) -> None:
        """Register common tools for an agent"""
        
        # Register framework-specific tools
        common_tools = [
            ("langchain_agent", FrameworkType.LANGCHAIN, 1000, 30000),
            ("crewai_coordinator", FrameworkType.CREWAI, 2000, 45000),
            ("autogen_conversation", FrameworkType.AUTOGEN, 1500, 60000),
            ("memory_retrieval", FrameworkType.CUSTOM, 500, 10000),
            ("model_routing", FrameworkType.CUSTOM, 800, 15000)
        ]
        
        for tool_name, framework, expected_latency, timeout in common_tools:
            self.tool_reliability.register_tool(
                tool_name=f"{agent_id}_{tool_name}",
                framework=framework,
                expected_latency_ms=expected_latency,
                timeout_ms=timeout
            )
    
    async def _evaluate_tool_reliability(self, agent_id: str) -> float:
        """Evaluate tool reliability for an agent"""
        
        # Update benchmarks
        self.tool_reliability.update_benchmarks()
        
        # Get reliability report
        report = self.tool_reliability.get_reliability_report()
        
        if report["status"] == "no_data":
            return 0.0
        
        # Calculate weighted reliability score
        overall_metrics = report["overall_metrics"]
        
        # Weight success rate heavily, with latency and error rate as factors
        reliability_score = (
            overall_metrics["success_rate"] * 0.7 +
            min(1.0, 1000 / max(overall_metrics["avg_latency_ms"], 1)) * 0.2 +
            (1.0 - overall_metrics["error_rate"]) * 0.1
        )
        
        return min(1.0, reliability_score)
    
    async def _evaluate_planning_reliability(self, agent_id: str) -> float:
        """Evaluate planning reliability for an agent"""
        
        planning_metrics = self.planning_reliability.get_planning_metrics()
        
        if planning_metrics.get("status") == "no_data":
            return 0.0
        
        overall_metrics = planning_metrics["overall_metrics"]
        return overall_metrics["success_rate"]
    
    async def _evaluate_code_execution_safety(self, agent_id: str) -> float:
        """Evaluate code execution safety for an agent"""
        
        sandbox_metrics = self.code_sandbox.get_sandbox_metrics()
        
        if sandbox_metrics.get("status") == "no_data":
            return 1.0  # Default to safe if no code execution
        
        # Calculate safety score based on success rate and security violations
        total_executions = sandbox_metrics["total_executions"]
        security_violations = sandbox_metrics["security_violations"]
        
        if total_executions == 0:
            return 1.0
        
        # Safety score considers both execution success and security
        violation_rate = security_violations / total_executions
        safety_score = 1.0 - violation_rate
        
        # Factor in sandbox success rates
        sandbox_breakdown = sandbox_metrics.get("sandbox_breakdown", {})
        if sandbox_breakdown:
            avg_success_rate = sum(
                stats["success_rate"] for stats in sandbox_breakdown.values()
            ) / len(sandbox_breakdown)
            safety_score = (safety_score + avg_success_rate) / 2
        
        return max(0.0, min(1.0, safety_score))
    
    async def _evaluate_adversarial_resilience(self, agent_id: str) -> float:
        """Evaluate adversarial resilience for an agent"""
        
        # Generate and run adversarial evaluation suite
        adversarial_tasks = await self.adversarial_evaluator.task_generator.generate_task_suite(
            task_types=[
                AdversarialTaskType.STRESS_TEST,
                AdversarialTaskType.EDGE_CASE_HANDLING,
                AdversarialTaskType.SECURITY_PROBING
            ],
            difficulty_range=(3, 8),
            tasks_per_type=3
        )
        
        # Run evaluation
        evaluation_report = await self.adversarial_evaluator.run_adversarial_evaluation(
            agent_id, adversarial_tasks, parallel_execution=True
        )
        
        # Extract resilience score
        if "evaluation_summary" in evaluation_report:
            return evaluation_report["evaluation_summary"]["pass_rate"]
        
        return 0.0
    
    def _calculate_overall_excellence(
        self,
        tool_score: float,
        planning_score: float,
        code_safety_score: float,
        adversarial_score: float
    ) -> float:
        """Calculate weighted overall excellence score"""
        
        # Weighted average with emphasis on reliability and safety
        weights = {
            "tool_reliability": 0.3,
            "planning_success": 0.25,
            "code_safety": 0.25,
            "adversarial_resilience": 0.2
        }
        
        overall_score = (
            tool_score * weights["tool_reliability"] +
            planning_score * weights["planning_success"] +
            code_safety_score * weights["code_safety"] +
            adversarial_score * weights["adversarial_resilience"]
        )
        
        return min(1.0, max(0.0, overall_score))
    
    def _record_excellence_metrics(self, agent_id: str, metrics: ExcellenceMetrics) -> None:
        """Record excellence metrics"""
        
        self.metrics.record_metric(f"excellence_overall_{agent_id}", metrics.overall_excellence_score)
        self.metrics.record_metric(f"excellence_tool_reliability_{agent_id}", metrics.tool_reliability_score)
        self.metrics.record_metric(f"excellence_planning_{agent_id}", metrics.planning_success_rate)
        self.metrics.record_metric(f"excellence_code_safety_{agent_id}", metrics.code_execution_safety)
        self.metrics.record_metric(f"excellence_adversarial_{agent_id}", metrics.adversarial_resilience)
    
    async def _store_excellence_evaluation(self, agent_id: str, metrics: ExcellenceMetrics) -> None:
        """Store excellence evaluation in memory"""
        
        evaluation_data = {
            "agent_id": agent_id,
            "overall_excellence_score": metrics.overall_excellence_score,
            "tool_reliability_score": metrics.tool_reliability_score,
            "planning_success_rate": metrics.planning_success_rate,
            "code_execution_safety": metrics.code_execution_safety,
            "adversarial_resilience": metrics.adversarial_resilience,
            "evaluation_timestamp": metrics.last_updated.isoformat()
        }
        
        await self.memory_manager.store_context(
            context_type="excellence_evaluation",
            content=evaluation_data,
            metadata={
                "agent_id": agent_id,
                "evaluation_type": "excellence",
                "timestamp": metrics.last_updated.isoformat()
            }
        )
    
    async def _check_excellence_thresholds(self, agent_id: str) -> None:
        """Check if agent meets excellence thresholds and trigger interventions"""
        
        if agent_id not in self.agent_excellence_scores:
            return
        
        metrics = self.agent_excellence_scores[agent_id]
        
        # Check each threshold
        for metric_name, threshold in self.excellence_targets.items():
            current_value = getattr(metrics, metric_name)
            
            if current_value < threshold:
                self.logger.warning(
                    f"Agent {agent_id} below threshold for {metric_name}: "
                    f"{current_value:.3f} < {threshold:.3f}"
                )
                
                # Record threshold violation
                self.metrics.increment(f"excellence_threshold_violation_{metric_name}_{agent_id}")
                
                # Trigger intervention if overall score is critically low
                if metric_name == "overall_excellence_score" and current_value < 0.7:
                    await self._trigger_excellence_intervention(agent_id, metrics)
    
    async def _trigger_excellence_intervention(self, agent_id: str, metrics: ExcellenceMetrics) -> None:
        """Trigger intervention for critically low excellence scores"""
        
        self.logger.critical(f"Triggering excellence intervention for agent: {agent_id}")
        
        # Log intervention
        self.metrics.increment(f"excellence_intervention_triggered_{agent_id}")
        
        # Store intervention record
        intervention_data = {
            "agent_id": agent_id,
            "intervention_reason": "critical_excellence_score",
            "excellence_metrics": {
                "overall_score": metrics.overall_excellence_score,
                "tool_reliability": metrics.tool_reliability_score,
                "planning_success": metrics.planning_success_rate,
                "code_safety": metrics.code_execution_safety,
                "adversarial_resilience": metrics.adversarial_resilience
            },
            "intervention_timestamp": datetime.utcnow().isoformat()
        }
        
        await self.memory_manager.store_context(
            context_type="excellence_intervention",
            content=intervention_data,
            metadata={
                "agent_id": agent_id,
                "intervention_type": "excellence",
                "severity": "critical"
            }
        )
    
    async def _run_tool_stress_test(self, agent_id: str, duration_minutes: int) -> Dict[str, Any]:
        """Run tool reliability stress test"""
        
        # Placeholder for tool stress testing
        return {
            "duration_minutes": duration_minutes,
            "tool_executions": 100,
            "success_rate": 0.95,
            "avg_latency_ms": 1200,
            "circuit_breaker_activations": 2
        }
    
    async def _run_code_execution_stress_test(
        self,
        agent_id: str,
        duration_minutes: int,
        intensity_level: int
    ) -> Dict[str, Any]:
        """Run code execution stress test"""
        
        # Generate stress test code samples
        stress_codes = [
            "print('Hello World')",
            "for i in range(1000): print(i)",
            "import time; time.sleep(1); print('Done')"
        ]
        
        executions = []
        
        # Run multiple code executions
        for _ in range(intensity_level * 5):
            code = stress_codes[_ % len(stress_codes)]
            
            try:
                execution = await self.code_sandbox.execute_code(
                    code=code,
                    language="python",
                    sandbox_type=SandboxType.KUBERNETES_JOB,
                    resource_limits=ResourceLimits(
                        cpu_cores=0.1,
                        memory_mb=64,
                        execution_timeout_seconds=30
                    )
                )
                executions.append(execution)
            except Exception as e:
                self.logger.warning(f"Code execution failed during stress test: {e}")
        
        # Analyze results
        successful_executions = len([e for e in executions if e.status.value == "completed"])
        success_rate = successful_executions / len(executions) if executions else 0
        
        return {
            "duration_minutes": duration_minutes,
            "total_executions": len(executions),
            "successful_executions": successful_executions,
            "success_rate": success_rate,
            "security_violations": sum(len(e.security_violations) for e in executions)
        }
    
    def _assess_stress_test_results(
        self,
        adversarial_report: Dict[str, Any],
        tool_report: Dict[str, Any],
        code_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess overall stress test results"""
        
        # Extract key metrics
        adversarial_pass_rate = adversarial_report.get("stress_test_summary", {}).get("overall_pass_rate", 0)
        tool_success_rate = tool_report.get("success_rate", 0)
        code_success_rate = code_report.get("success_rate", 0)
        
        # Calculate overall stress resilience
        overall_resilience = (adversarial_pass_rate + tool_success_rate + code_success_rate) / 3
        
        return {
            "overall_stress_resilience": overall_resilience,
            "component_scores": {
                "adversarial_resilience": adversarial_pass_rate,
                "tool_reliability": tool_success_rate,
                "code_execution_safety": code_success_rate
            },
            "stress_test_passed": overall_resilience >= 0.8,
            "critical_issues": overall_resilience < 0.6
        }
    
    def _generate_stress_test_recommendations(
        self,
        adversarial_report: Dict[str, Any],
        tool_report: Dict[str, Any],
        code_report: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on stress test results"""
        
        recommendations = []
        
        # Analyze adversarial performance
        adversarial_pass_rate = adversarial_report.get("stress_test_summary", {}).get("overall_pass_rate", 0)
        if adversarial_pass_rate < 0.8:
            recommendations.append("Improve adversarial resilience through enhanced error handling and recovery mechanisms")
        
        # Analyze tool reliability
        tool_success_rate = tool_report.get("success_rate", 0)
        if tool_success_rate < 0.95:
            recommendations.append("Optimize tool reliability with better retry mechanisms and circuit breakers")
        
        # Analyze code execution safety
        code_success_rate = code_report.get("success_rate", 0)
        security_violations = code_report.get("security_violations", 0)
        
        if code_success_rate < 0.9:
            recommendations.append("Enhance code execution sandbox stability and resource management")
        
        if security_violations > 0:
            recommendations.append("Strengthen code security validation to prevent security violations")
        
        return recommendations
    
    async def _store_stress_test_results(self, agent_id: str, report: Dict[str, Any]) -> None:
        """Store stress test results in memory"""
        
        await self.memory_manager.store_context(
            context_type="stress_test_report",
            content=report,
            metadata={
                "agent_id": agent_id,
                "test_type": "comprehensive_stress_test",
                "timestamp": datetime.utcnow().isoformat()
            }
        )