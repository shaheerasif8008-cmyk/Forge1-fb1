# forge1/backend/forge1/tests/test_orchestrator_qa_integration.py
"""
Integration Tests for Agent Orchestrator with Quality Assurance and Conflict Resolution

Tests the complete integration of quality assurance, conflict resolution, and escalation
management within the agent orchestration system.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

from forge1.agents.agent_orchestrator import AgentOrchestrator, TaskPriority, CoordinationStrategy
from forge1.agents.enhanced_base_agent import EnhancedBaseAgent, AgentRole, PerformanceLevel
from forge1.core.memory_manager import MemoryManager
from forge1.core.quality_assurance import QualityLevel
from forge1.core.conflict_resolution import ConflictType
from forge1.core.escalation_manager import EscalationTrigger, EscalationPriority


class MockEnhancedAgent(EnhancedBaseAgent):
    """Mock enhanced agent for testing"""
    
    def __init__(self, agent_name: str, role: AgentRole, response_content: str, confidence: float = 0.9):
        super().__init__(
            agent_name=agent_name,
            role=role,
            performance_target=PerformanceLevel.SUPERHUMAN
        )
        self.response_content = response_content
        self.response_confidence = confidence
    
    async def execute_task_superhuman(self, task: dict, context: dict = None) -> dict:
        """Mock task execution"""
        return {
            "content": self.response_content,
            "confidence": self.response_confidence,
            "agent": self._agent_name,
            "execution_time": 30,
            "validation_passed": True
        }


class TestOrchestratorQAIntegration:
    """Integration tests for orchestrator with QA systems"""
    
    @pytest.fixture
    async def memory_manager(self):
        """Create mock memory manager"""
        memory_manager = Mock(spec=MemoryManager)
        memory_manager.store_memory = AsyncMock(return_value=True)
        memory_manager.retrieve_memories = AsyncMock(return_value=[])
        memory_manager.get_relevant_context = AsyncMock(return_value={})
        return memory_manager
    
    @pytest.fixture
    async def orchestrator(self, memory_manager):
        """Create orchestrator with QA systems enabled"""
        return AgentOrchestrator(
            session_id="test_session",
            user_id="test_user",
            memory_manager=memory_manager,
            enable_quality_assurance=True,
            enable_conflict_resolution=True
        )
    
    @pytest.fixture
    def sample_agents(self):
        """Create sample agents for testing"""
        return [
            MockEnhancedAgent(
                "expert_analyst",
                AgentRole.SPECIALIST,
                "Comprehensive market analysis shows 25% growth in technology sector with strong supporting evidence from multiple data sources and expert validation.",
                0.95
            ),
            MockEnhancedAgent(
                "junior_analyst", 
                AgentRole.EXECUTOR,
                "Tech stocks are good. They go up.",
                0.6
            ),
            MockEnhancedAgent(
                "conservative_advisor",
                AgentRole.SPECIALIST,
                "Recommend conservative investment approach with bonds and stable dividend stocks to minimize risk.",
                0.85
            )
        ]
    
    @pytest.mark.asyncio
    async def test_orchestrator_quality_review_integration(self, orchestrator, sample_agents):
        """Test orchestrator integration with quality review system"""
        
        # Register agents
        for agent in sample_agents:
            await orchestrator.register_agent(agent)
        
        # Test quality review of high-quality result
        high_quality_result = {
            "content": "Exceptional comprehensive analysis with detailed methodology, extensive research, clear conclusions, and actionable recommendations backed by solid evidence.",
            "confidence": 0.98,
            "validation_passed": True
        }
        
        context = {
            "task_type": "market_analysis",
            "requirements": ["comprehensive analysis", "evidence-based conclusions"],
            "business_impact": 0.8
        }
        
        review_report = await orchestrator.conduct_quality_review(
            high_quality_result, context, ["expert_analyst"], QualityLevel.SUPERHUMAN
        )
        
        assert review_report["quality_decision"]["approved"] == True
        assert review_report["quality_level"] == "superhuman"
        assert "superhuman_validation" in review_report
        
        # Check orchestrator metrics update
        status = orchestrator.get_orchestration_status()
        assert status["metrics"]["quality_reviews_conducted"] >= 1
        assert status["metrics"]["quality_approvals"] >= 1
    
    @pytest.mark.asyncio
    async def test_orchestrator_conflict_resolution_integration(self, orchestrator, sample_agents):
        """Test orchestrator integration with conflict resolution system"""
        
        # Register agents
        for agent in sample_agents:
            await orchestrator.register_agent(agent)
        
        # Create conflicting results
        conflicting_results = [
            {
                "content": "Strongly recommend aggressive growth strategy with 100% technology stocks for maximum returns.",
                "confidence": 0.9,
                "agent": "expert_analyst"
            },
            {
                "content": "Strongly recommend conservative strategy with 100% bonds for safety and stability.",
                "confidence": 0.85,
                "agent": "conservative_advisor"
            }
        ]
        
        context = {
            "task_type": "investment_strategy",
            "requirements": ["risk assessment", "return optimization"],
            "business_impact": 0.9
        }
        
        resolution_report = await orchestrator.resolve_agent_conflict(
            ["expert_analyst", "conservative_advisor"],
            conflicting_results,
            context,
            ConflictType.APPROACH_CONFLICT
        )
        
        assert "conflict_analysis" in resolution_report
        assert "final_result" in resolution_report
        
        # Check if conflict was resolved or escalated
        final_result = resolution_report["final_result"]
        assert "resolved" in final_result
        
        # Check orchestrator metrics update
        status = orchestrator.get_orchestration_status()
        assert status["metrics"]["conflicts_detected"] >= 1
    
    @pytest.mark.asyncio
    async def test_result_validation_with_quality_and_conflicts(self, orchestrator, sample_agents):
        """Test comprehensive result validation with both quality and conflict detection"""
        
        # Register agents
        for agent in sample_agents:
            await orchestrator.register_agent(agent)
        
        # Create mixed quality results with potential conflicts
        mixed_results = [
            {
                "content": "Excellent comprehensive market analysis with detailed research methodology, extensive data analysis, and well-supported conclusions showing 25% technology sector growth.",
                "confidence": 0.95,
                "validation_passed": True
            },
            {
                "content": "Bad analysis. Wrong conclusions. No evidence.",
                "confidence": 0.3,
                "validation_passed": False
            },
            {
                "content": "Thorough market research indicates technology sector decline of 15% with supporting evidence from industry reports and expert interviews.",
                "confidence": 0.9,
                "validation_passed": True
            }
        ]
        
        agents_involved = ["expert_analyst", "junior_analyst", "conservative_advisor"]
        
        context = {
            "task_type": "market_analysis",
            "requirements": ["comprehensive research", "evidence-based conclusions", "accurate predictions"],
            "business_impact": 0.8
        }
        
        validation_report = await orchestrator.validate_agent_results(
            mixed_results, agents_involved, context, enable_conflict_detection=True
        )
        
        assert validation_report["validation_id"].startswith("val_")
        assert validation_report["results_count"] == 3
        assert len(validation_report["quality_reviews"]) == 3
        
        # Should detect conflicts between contradictory results
        if len(validation_report["conflicts_detected"]) > 0:
            conflict = validation_report["conflicts_detected"][0]
            assert "conflict" in conflict
            assert "resolution" in conflict
        
        # Should have final results with approved quality
        approved_results = [
            result for result in validation_report["final_results"]
            if result["quality_score"] >= 0.8
        ]
        assert len(approved_results) >= 1
        
        # Validation status should reflect the outcome
        assert validation_report["validation_status"] in ["approved", "partially_approved", "rejected"]
    
    @pytest.mark.asyncio
    async def test_quality_failure_escalation(self, orchestrator, sample_agents):
        """Test escalation when quality assurance fails"""
        
        # Register agents
        for agent in sample_agents:
            await orchestrator.register_agent(agent)
        
        # Create very low quality result that should trigger escalation
        poor_quality_result = {
            "content": "bad",
            "confidence": 0.1,
            "validation_passed": False
        }
        
        context = {
            "task_type": "critical_analysis",
            "requirements": ["high quality", "comprehensive analysis"],
            "business_impact": 0.9,
            "timeline_critical": True
        }
        
        # Mock the escalation manager to track escalation calls
        with patch.object(orchestrator.escalation_manager, 'trigger_escalation', new_callable=AsyncMock) as mock_escalation:
            mock_escalation.return_value = {
                "escalation_id": "esc_test123",
                "status": "initiated",
                "initial_level": 2
            }
            
            review_report = await orchestrator.conduct_quality_review(
                poor_quality_result, context, ["junior_analyst"], QualityLevel.SUPERHUMAN
            )
            
            # Should not approve poor quality
            assert review_report["quality_decision"]["approved"] == False
            
            # Should trigger escalation for quality failure
            mock_escalation.assert_called_once()
            escalation_call = mock_escalation.call_args
            assert escalation_call[0][0] == EscalationTrigger.QUALITY_FAILURE
            assert escalation_call[0][2] in [EscalationPriority.HIGH, EscalationPriority.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_unresolved_conflict_escalation(self, orchestrator, sample_agents):
        """Test escalation when conflicts cannot be resolved"""
        
        # Register agents
        for agent in sample_agents:
            await orchestrator.register_agent(agent)
        
        # Create highly conflicting results that are difficult to resolve
        difficult_conflict_results = [
            {
                "content": "Absolutely certain that cryptocurrency will increase 1000% this year based on my analysis.",
                "confidence": 0.99,
                "agent": "crypto_bull"
            },
            {
                "content": "Absolutely certain that cryptocurrency will crash to zero this year based on my analysis.",
                "confidence": 0.99,
                "agent": "crypto_bear"
            }
        ]
        
        context = {
            "task_type": "investment_prediction",
            "business_impact": 0.9,
            "conflict_severity": 0.9
        }
        
        # Mock the escalation manager and conflict resolution to simulate unresolved conflict
        with patch.object(orchestrator.escalation_manager, 'trigger_escalation', new_callable=AsyncMock) as mock_escalation:
            with patch.object(orchestrator.conflict_resolution, 'detect_and_resolve_conflict', new_callable=AsyncMock) as mock_conflict:
                
                # Mock unresolved conflict
                mock_conflict.return_value = {
                    "id": "conflict_test123",
                    "final_result": {
                        "resolved": False,
                        "escalated": True,
                        "escalation_level": "supervisor"
                    },
                    "conflict_analysis": {
                        "severity": 0.9
                    }
                }
                
                mock_escalation.return_value = {
                    "escalation_id": "esc_conflict123",
                    "status": "initiated",
                    "initial_level": 2
                }
                
                resolution_report = await orchestrator.resolve_agent_conflict(
                    ["crypto_bull", "crypto_bear"],
                    difficult_conflict_results,
                    context,
                    ConflictType.RESULT_DISAGREEMENT
                )
                
                # Should detect unresolved conflict
                assert resolution_report["final_result"]["resolved"] == False
                
                # Should trigger escalation for unresolved conflict
                mock_escalation.assert_called_once()
                escalation_call = mock_escalation.call_args
                assert escalation_call[0][0] == EscalationTrigger.CONFLICT_UNRESOLVED
    
    @pytest.mark.asyncio
    async def test_superhuman_performance_validation_integration(self, orchestrator, sample_agents):
        """Test superhuman performance validation in orchestrator context"""
        
        # Register agents
        for agent in sample_agents:
            await orchestrator.register_agent(agent)
        
        # Create superhuman performance result
        superhuman_result = {
            "content": "Exceptional analysis demonstrating superior methodology, comprehensive data analysis covering 500+ companies, innovative predictive modeling with 99.2% accuracy, and strategic recommendations that significantly exceed professional human capabilities in depth, accuracy, and actionable value. Analysis completed in 30 seconds vs typical human time of 40 hours.",
            "confidence": 0.99,
            "execution_time": 30,
            "validation_passed": True,
            "accuracy_score": 0.992,
            "innovation_score": 0.95
        }
        
        context = {
            "task_type": "comprehensive_market_analysis",
            "human_baseline": {
                "accuracy": 0.85,
                "speed": 1.0,
                "quality": 0.8,
                "consistency": 0.75
            },
            "expected_time": 144000,  # 40 hours in seconds
            "speed_multiplier": 4800,  # 4800x faster than human
            "requirements": ["superhuman accuracy", "comprehensive coverage", "innovative insights"]
        }
        
        review_report = await orchestrator.conduct_quality_review(
            superhuman_result, context, ["expert_analyst"], QualityLevel.SUPERHUMAN
        )
        
        # Should approve superhuman performance
        assert review_report["quality_decision"]["approved"] == True
        
        # Check superhuman validation
        superhuman_validation = review_report["superhuman_validation"]
        assert superhuman_validation["achieves_superhuman"] == True
        assert superhuman_validation["overall_superhuman_ratio"] >= 1.2
        assert superhuman_validation["performance_tier"] in [
            "Superhuman", "Advanced Superhuman", "Extreme Superhuman"
        ]
        
        # Quality score should be very high
        assert review_report["quality_assessment"]["overall_score"] >= 0.9
        assert review_report["quality_assessment"]["quality_grade"] in ["A+", "A++"]
    
    @pytest.mark.asyncio
    async def test_quality_metrics_tracking(self, orchestrator, sample_agents):
        """Test quality metrics tracking across multiple operations"""
        
        # Register agents
        for agent in sample_agents:
            await orchestrator.register_agent(agent)
        
        # Perform multiple quality reviews
        test_results = [
            {
                "content": "Excellent high-quality analysis with comprehensive methodology.",
                "confidence": 0.95,
                "validation_passed": True
            },
            {
                "content": "Poor quality analysis with no evidence.",
                "confidence": 0.3,
                "validation_passed": False
            },
            {
                "content": "Good quality analysis with solid methodology and conclusions.",
                "confidence": 0.85,
                "validation_passed": True
            }
        ]
        
        context = {
            "task_type": "analysis",
            "requirements": ["quality analysis"],
            "business_impact": 0.7
        }
        
        # Conduct multiple reviews
        for i, result in enumerate(test_results):
            await orchestrator.conduct_quality_review(
                result, context, [f"agent_{i}"], QualityLevel.STANDARD
            )
        
        # Check quality metrics
        quality_metrics = orchestrator.get_quality_metrics()
        
        assert "orchestrator_metrics" in quality_metrics
        assert "quality_assurance_metrics" in quality_metrics
        assert "escalation_metrics" in quality_metrics
        
        orchestrator_metrics = quality_metrics["orchestrator_metrics"]
        assert orchestrator_metrics["quality_reviews_conducted"] >= 3
        assert orchestrator_metrics["quality_approvals"] >= 1  # At least one should be approved
        
        qa_metrics = quality_metrics["quality_assurance_metrics"]
        assert qa_metrics["reviews_conducted"] >= 3
        assert qa_metrics["quality_improvements"] >= 1
    
    @pytest.mark.asyncio
    async def test_active_quality_issues_tracking(self, orchestrator, sample_agents):
        """Test tracking of active quality issues and conflicts"""
        
        # Register agents
        for agent in sample_agents:
            await orchestrator.register_agent(agent)
        
        # Create scenario with quality issues
        problematic_result = {
            "content": "Analysis contains PII: john.doe@company.com, SSN: 123-45-6789",
            "confidence": 0.4,
            "validation_passed": False
        }
        
        context = {
            "task_type": "sensitive_analysis",
            "requirements": ["privacy compliance", "high quality"],
            "business_impact": 0.9
        }
        
        # This should create quality issues and potentially escalations
        await orchestrator.conduct_quality_review(
            problematic_result, context, ["problematic_agent"], QualityLevel.COMPREHENSIVE
        )
        
        # Check active quality issues
        active_issues = orchestrator.get_active_quality_issues()
        
        assert "active_escalations" in active_issues
        assert "recent_quality_reviews" in active_issues
        
        # Should have recent quality reviews
        assert len(active_issues["recent_quality_reviews"]) >= 1
        
        # Check if escalations were triggered for compliance violations
        if active_issues["active_escalations"]:
            escalation = list(active_issues["active_escalations"].values())[0]
            assert escalation["trigger"] in ["quality_failure", "compliance_violation"]
    
    @pytest.mark.asyncio
    async def test_end_to_end_quality_workflow(self, orchestrator, sample_agents):
        """Test complete end-to-end quality workflow with all systems"""
        
        # Register agents
        for agent in sample_agents:
            await orchestrator.register_agent(agent)
        
        # Simulate complex multi-agent task with various quality levels
        agent_results = [
            {
                "content": "Exceptional comprehensive market analysis with detailed methodology, extensive research covering 200+ companies, innovative predictive modeling, and strategic recommendations backed by solid evidence and expert validation.",
                "confidence": 0.98,
                "validation_passed": True,
                "execution_time": 45
            },
            {
                "content": "Basic analysis shows market trends are mixed with some growth potential.",
                "confidence": 0.7,
                "validation_passed": True,
                "execution_time": 300
            },
            {
                "content": "Comprehensive analysis indicates strong bearish trends with 30% market decline expected based on economic indicators and expert consensus.",
                "confidence": 0.92,
                "validation_passed": True,
                "execution_time": 60
            }
        ]
        
        agents_involved = ["expert_analyst", "junior_analyst", "conservative_advisor"]
        
        context = {
            "task_type": "strategic_market_analysis",
            "requirements": ["comprehensive analysis", "evidence-based conclusions", "strategic recommendations"],
            "business_impact": 0.9,
            "human_baseline": {
                "accuracy": 0.85,
                "speed": 1.0,
                "quality": 0.8,
                "consistency": 0.75
            },
            "expected_time": 7200  # 2 hours human baseline
        }
        
        # Execute complete validation workflow
        validation_report = await orchestrator.validate_agent_results(
            agent_results, agents_involved, context, enable_conflict_detection=True
        )
        
        # Verify comprehensive validation
        assert validation_report["results_count"] == 3
        assert len(validation_report["quality_reviews"]) == 3
        
        # Check quality review outcomes
        high_quality_reviews = [
            review for review in validation_report["quality_reviews"]
            if review["review"]["quality_decision"]["approved"]
        ]
        assert len(high_quality_reviews) >= 2  # At least 2 should pass quality review
        
        # Check conflict detection between contradictory results
        if len(validation_report["conflicts_detected"]) > 0:
            conflict = validation_report["conflicts_detected"][0]
            assert "resolution" in conflict
            resolution = conflict["resolution"]
            assert "final_result" in resolution
        
        # Final validation should have approved results
        assert len(validation_report["final_results"]) >= 1
        
        # Check overall validation status
        assert validation_report["validation_status"] in ["approved", "partially_approved"]
        
        # Verify metrics were updated
        final_metrics = orchestrator.get_quality_metrics()
        assert final_metrics["orchestrator_metrics"]["quality_reviews_conducted"] >= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])