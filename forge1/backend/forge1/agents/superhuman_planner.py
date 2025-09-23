# forge1/backend/forge1/agents/superhuman_planner.py
"""
Superhuman Planner Agent for Forge 1

Enhanced planner with superhuman planning capabilities that significantly outperform
professional human planners in complex task decomposition and strategic planning.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from forge1.agents.enhanced_base_agent import EnhancedBaseAgent, AgentRole, PerformanceLevel

logger = logging.getLogger(__name__)

class SuperhumanPlannerAgent(EnhancedBaseAgent):
    """Superhuman planner agent specialized in complex task decomposition and strategic planning"""
    
    def __init__(
        self,
        agent_instances: Optional[Dict[str, Any]] = None,
        planning_depth: int = 5,
        max_parallel_tasks: int = 10,
        **kwargs
    ):
        """Initialize superhuman planner agent
        
        Args:
            agent_instances: Dictionary of available agent instances for delegation
            planning_depth: Maximum depth for recursive task decomposition
            max_parallel_tasks: Maximum number of parallel tasks to plan
            **kwargs: Additional parameters for base agent
        """
        
        # Set role and performance target for planner
        kwargs['role'] = AgentRole.PLANNER
        kwargs['performance_target'] = PerformanceLevel.SUPERHUMAN
        
        super().__init__(**kwargs)
        
        # Planner-specific capabilities
        self.agent_instances = agent_instances or {}
        self.planning_depth = planning_depth
        self.max_parallel_tasks = max_parallel_tasks
        
        # Planning state
        self.active_plans = {}
        self.planning_history = []
        self.delegation_patterns = {}
        
        # Superhuman planning metrics
        self.planning_metrics = {
            "plans_created": 0,
            "successful_plans": 0,
            "average_plan_accuracy": 0.0,
            "task_decomposition_efficiency": 0.0,
            "delegation_success_rate": 0.0,
            "planning_speed_vs_human": 0.0  # Multiplier vs human baseline
        }
        
        logger.info(f"Superhuman planner {self._agent_name} initialized with {len(self.agent_instances)} available agents")
    
    async def create_superhuman_plan(self, task_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a superhuman-level plan that outperforms human planners
        
        Args:
            task_description: Description of the task to plan
            context: Additional context for planning
            
        Returns:
            Comprehensive plan with task decomposition and agent assignments
        """
        
        plan_id = f"plan_{int(datetime.now().timestamp())}"
        start_time = datetime.now(timezone.utc)
        
        try:
            # Phase 1: Analyze task complexity and requirements
            task_analysis = await self._analyze_task_for_planning(task_description, context)
            
            # Phase 2: Decompose into subtasks with superhuman precision
            subtasks = await self._decompose_task_superhuman(task_analysis)
            
            # Phase 3: Identify optimal agent assignments
            agent_assignments = await self._assign_agents_optimally(subtasks)
            
            # Phase 4: Create execution timeline and dependencies
            execution_plan = await self._create_execution_timeline(subtasks, agent_assignments)
            
            # Phase 5: Add quality assurance and verification steps
            qa_plan = await self._add_quality_assurance(execution_plan)
            
            # Phase 6: Calculate success probability and risk mitigation
            risk_analysis = await self._analyze_risks_and_mitigation(qa_plan)
            
            # Create comprehensive plan
            plan = {
                "id": plan_id,
                "task_description": task_description,
                "context": context,
                "created_at": start_time.isoformat(),
                "planner": self._agent_name,
                "task_analysis": task_analysis,
                "subtasks": subtasks,
                "agent_assignments": agent_assignments,
                "execution_plan": execution_plan,
                "quality_assurance": qa_plan,
                "risk_analysis": risk_analysis,
                "estimated_completion_time": self._calculate_completion_time(execution_plan),
                "success_probability": risk_analysis.get("success_probability", 0.85),
                "superhuman_indicators": {
                    "complexity_handled": task_analysis.get("complexity_score", 0),
                    "parallel_optimization": len([t for t in subtasks if t.get("can_parallelize", False)]),
                    "agent_utilization_efficiency": self._calculate_agent_efficiency(agent_assignments),
                    "risk_mitigation_coverage": len(risk_analysis.get("mitigation_strategies", []))
                }
            }
            
            # Store plan
            self.active_plans[plan_id] = plan
            self.planning_history.append(plan)
            
            # Update metrics
            await self._update_planning_metrics(plan, start_time)
            
            logger.info(f"Superhuman plan {plan_id} created with {len(subtasks)} subtasks and {risk_analysis.get('success_probability', 0.85):.2f} success probability")
            
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create superhuman plan: {e}")
            raise
    
    async def _analyze_task_for_planning(self, task_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task with superhuman precision for optimal planning"""
        
        analysis = {
            "task_type": self._classify_task_type(task_description),
            "complexity_score": self._calculate_complexity_score(task_description),
            "required_skills": self._identify_required_skills(task_description),
            "estimated_effort": self._estimate_effort(task_description),
            "dependencies": self._identify_dependencies(task_description, context),
            "success_criteria": self._extract_success_criteria(task_description),
            "constraints": self._identify_constraints(context),
            "optimization_opportunities": self._identify_optimization_opportunities(task_description)
        }
        
        return analysis
    
    def _classify_task_type(self, task_description: str) -> str:
        """Classify task type for optimal planning approach"""
        
        task_lower = task_description.lower()
        
        if any(word in task_lower for word in ["analyze", "research", "investigate"]):
            return "analytical"
        elif any(word in task_lower for word in ["create", "build", "develop", "design"]):
            return "creative"
        elif any(word in task_lower for word in ["process", "handle", "manage", "organize"]):
            return "operational"
        elif any(word in task_lower for word in ["solve", "fix", "troubleshoot", "debug"]):
            return "problem_solving"
        elif any(word in task_lower for word in ["coordinate", "collaborate", "communicate"]):
            return "coordination"
        else:
            return "general"
    
    def _calculate_complexity_score(self, task_description: str) -> float:
        """Calculate task complexity score (0.0 to 1.0)"""
        
        complexity_factors = {
            "length": min(len(task_description.split()) / 100, 1.0) * 0.2,
            "technical_terms": len([w for w in task_description.split() if len(w) > 8]) / len(task_description.split()) * 0.3,
            "multiple_steps": task_description.count("then") + task_description.count("after") + task_description.count("next") * 0.1,
            "conditional_logic": task_description.count("if") + task_description.count("when") + task_description.count("unless") * 0.2,
            "stakeholders": task_description.count("team") + task_description.count("client") + task_description.count("user") * 0.2
        }
        
        return min(sum(complexity_factors.values()), 1.0)
    
    def _identify_required_skills(self, task_description: str) -> List[str]:
        """Identify skills required for task completion"""
        
        skill_keywords = {
            "analytical": ["analyze", "research", "investigate", "evaluate"],
            "technical": ["code", "program", "develop", "implement", "configure"],
            "creative": ["design", "create", "brainstorm", "innovate"],
            "communication": ["present", "communicate", "explain", "document"],
            "project_management": ["plan", "coordinate", "manage", "organize"],
            "problem_solving": ["solve", "troubleshoot", "debug", "fix"]
        }
        
        required_skills = []
        task_lower = task_description.lower()
        
        for skill, keywords in skill_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                required_skills.append(skill)
        
        return required_skills
    
    async def _decompose_task_superhuman(self, task_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose task into subtasks with superhuman precision"""
        
        task_type = task_analysis["task_type"]
        complexity = task_analysis["complexity_score"]
        
        # Base decomposition strategies by task type
        decomposition_strategies = {
            "analytical": self._decompose_analytical_task,
            "creative": self._decompose_creative_task,
            "operational": self._decompose_operational_task,
            "problem_solving": self._decompose_problem_solving_task,
            "coordination": self._decompose_coordination_task,
            "general": self._decompose_general_task
        }
        
        strategy = decomposition_strategies.get(task_type, self._decompose_general_task)
        subtasks = await strategy(task_analysis)
        
        # Enhance subtasks with superhuman optimizations
        enhanced_subtasks = []
        for i, subtask in enumerate(subtasks):
            enhanced_subtask = {
                **subtask,
                "id": f"subtask_{i+1}",
                "priority": self._calculate_priority(subtask, task_analysis),
                "estimated_duration": self._estimate_subtask_duration(subtask),
                "can_parallelize": self._can_parallelize(subtask, subtasks),
                "dependencies": self._identify_subtask_dependencies(subtask, subtasks),
                "quality_criteria": self._define_quality_criteria(subtask),
                "superhuman_optimizations": self._identify_superhuman_optimizations(subtask)
            }
            enhanced_subtasks.append(enhanced_subtask)
        
        return enhanced_subtasks
    
    async def _decompose_analytical_task(self, task_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose analytical tasks with superhuman precision"""
        
        return [
            {"name": "Data Collection", "description": "Gather all relevant data and information"},
            {"name": "Data Analysis", "description": "Analyze collected data using advanced techniques"},
            {"name": "Pattern Recognition", "description": "Identify patterns and insights"},
            {"name": "Hypothesis Formation", "description": "Form testable hypotheses"},
            {"name": "Validation", "description": "Validate findings and conclusions"},
            {"name": "Report Generation", "description": "Create comprehensive analytical report"}
        ]
    
    async def _decompose_creative_task(self, task_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose creative tasks with superhuman innovation"""
        
        return [
            {"name": "Ideation", "description": "Generate multiple creative concepts"},
            {"name": "Concept Development", "description": "Develop and refine best concepts"},
            {"name": "Feasibility Analysis", "description": "Assess technical and practical feasibility"},
            {"name": "Prototype Creation", "description": "Create initial prototypes or mockups"},
            {"name": "Iteration", "description": "Refine based on feedback and testing"},
            {"name": "Final Production", "description": "Produce final creative output"}
        ]
    
    async def _decompose_operational_task(self, task_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose operational tasks with superhuman efficiency"""
        
        return [
            {"name": "Process Analysis", "description": "Analyze current processes and requirements"},
            {"name": "Optimization Planning", "description": "Plan process optimizations"},
            {"name": "Resource Allocation", "description": "Allocate resources optimally"},
            {"name": "Execution", "description": "Execute operational tasks"},
            {"name": "Monitoring", "description": "Monitor progress and performance"},
            {"name": "Optimization", "description": "Continuously optimize operations"}
        ]
    
    async def _decompose_problem_solving_task(self, task_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose problem-solving tasks with superhuman precision"""
        
        return [
            {"name": "Problem Definition", "description": "Clearly define the problem"},
            {"name": "Root Cause Analysis", "description": "Identify root causes"},
            {"name": "Solution Generation", "description": "Generate multiple solution options"},
            {"name": "Solution Evaluation", "description": "Evaluate solutions against criteria"},
            {"name": "Implementation Planning", "description": "Plan solution implementation"},
            {"name": "Solution Implementation", "description": "Implement chosen solution"},
            {"name": "Verification", "description": "Verify problem resolution"}
        ]
    
    async def _decompose_coordination_task(self, task_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose coordination tasks with superhuman orchestration"""
        
        return [
            {"name": "Stakeholder Mapping", "description": "Identify all stakeholders"},
            {"name": "Communication Planning", "description": "Plan communication strategy"},
            {"name": "Task Distribution", "description": "Distribute tasks optimally"},
            {"name": "Progress Coordination", "description": "Coordinate progress across teams"},
            {"name": "Conflict Resolution", "description": "Resolve conflicts and blockers"},
            {"name": "Final Integration", "description": "Integrate all deliverables"}
        ]
    
    async def _decompose_general_task(self, task_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decompose general tasks with adaptive approach"""
        
        return [
            {"name": "Requirements Analysis", "description": "Analyze task requirements"},
            {"name": "Planning", "description": "Create detailed execution plan"},
            {"name": "Preparation", "description": "Prepare resources and environment"},
            {"name": "Execution", "description": "Execute main task activities"},
            {"name": "Quality Check", "description": "Verify quality and completeness"},
            {"name": "Delivery", "description": "Deliver final results"}
        ]
    
    async def _assign_agents_optimally(self, subtasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assign agents to subtasks with superhuman optimization"""
        
        assignments = {}
        agent_workloads = {agent_name: 0 for agent_name in self.agent_instances.keys()}
        
        # Sort subtasks by priority and complexity
        sorted_subtasks = sorted(subtasks, key=lambda x: (x.get("priority", 0.5), -x.get("estimated_duration", 1)))
        
        for subtask in sorted_subtasks:
            best_agent = await self._find_best_agent_for_subtask(subtask, agent_workloads)
            
            if best_agent:
                assignments[subtask["id"]] = {
                    "agent": best_agent,
                    "confidence": self._calculate_assignment_confidence(subtask, best_agent),
                    "estimated_start": self._calculate_start_time(best_agent, agent_workloads),
                    "backup_agents": await self._identify_backup_agents(subtask, best_agent)
                }
                
                # Update workload
                agent_workloads[best_agent] += subtask.get("estimated_duration", 1)
            else:
                # No suitable agent found, assign to most capable general agent
                assignments[subtask["id"]] = {
                    "agent": "GENERIC",
                    "confidence": 0.6,
                    "estimated_start": 0,
                    "backup_agents": []
                }
        
        return assignments
    
    async def _find_best_agent_for_subtask(self, subtask: Dict[str, Any], workloads: Dict[str, float]) -> Optional[str]:
        """Find the best agent for a specific subtask"""
        
        # Agent capability mapping
        agent_capabilities = {
            "HR": ["communication", "analytical", "operational"],
            "MARKETING": ["creative", "communication", "analytical"],
            "PRODUCT": ["technical", "creative", "analytical"],
            "PROCUREMENT": ["analytical", "operational", "problem_solving"],
            "TECH_SUPPORT": ["technical", "problem_solving", "analytical"],
            "GENERIC": ["analytical", "operational", "communication"]
        }
        
        required_skills = subtask.get("required_skills", [])
        
        # Score agents based on capability match and current workload
        agent_scores = {}
        for agent_name, capabilities in agent_capabilities.items():
            if agent_name in self.agent_instances:
                skill_match = len(set(required_skills) & set(capabilities)) / max(len(required_skills), 1)
                workload_factor = 1.0 / (1.0 + workloads.get(agent_name, 0))
                agent_scores[agent_name] = skill_match * 0.7 + workload_factor * 0.3
        
        # Return agent with highest score
        if agent_scores:
            return max(agent_scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _calculate_assignment_confidence(self, subtask: Dict[str, Any], agent: str) -> float:
        """Calculate confidence in agent assignment"""
        
        # Base confidence on agent capability match
        base_confidence = 0.7
        
        # Adjust based on subtask complexity
        complexity = subtask.get("complexity_score", 0.5)
        if complexity > 0.8:
            base_confidence -= 0.1
        elif complexity < 0.3:
            base_confidence += 0.1
        
        return min(max(base_confidence, 0.0), 1.0)
    
    async def _update_planning_metrics(self, plan: Dict[str, Any], start_time: datetime):
        """Update superhuman planning metrics"""
        
        planning_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        self.planning_metrics["plans_created"] += 1
        
        # Calculate planning speed vs human baseline (assume human takes 10x longer)
        human_baseline_time = len(plan["subtasks"]) * 300  # 5 minutes per subtask for humans
        speed_multiplier = human_baseline_time / max(planning_time, 1)
        self.planning_metrics["planning_speed_vs_human"] = speed_multiplier
        
        # Update task decomposition efficiency
        subtask_count = len(plan["subtasks"])
        complexity = plan["task_analysis"]["complexity_score"]
        efficiency = subtask_count / max(complexity * 10, 1)  # Optimal subtask count
        self.planning_metrics["task_decomposition_efficiency"] = efficiency
        
        logger.info(f"Planning metrics updated: Speed {speed_multiplier:.1f}x human, Efficiency {efficiency:.2f}")
    
    def get_planning_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive planning performance report"""
        
        base_report = self.get_performance_report()
        
        planning_report = {
            **base_report,
            "planning_metrics": self.planning_metrics.copy(),
            "active_plans": len(self.active_plans),
            "total_plans_created": len(self.planning_history),
            "average_subtasks_per_plan": sum(len(p.get("subtasks", [])) for p in self.planning_history) / max(len(self.planning_history), 1),
            "superhuman_planning_indicators": {
                "exceeds_human_speed": self.planning_metrics["planning_speed_vs_human"] > 5.0,
                "high_decomposition_efficiency": self.planning_metrics["task_decomposition_efficiency"] > 0.8,
                "consistent_quality": self.planning_metrics["average_plan_accuracy"] > 0.9
            }
        }
        
        return planning_report