# forge1/backend/forge1/tests/test_quality_assurance.py
"""
Tests for Quality Assurance System

Comprehensive tests for quality assurance, conflict resolution, and escalation management.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock

from forge1.core.quality_assurance import QualityAssuranceSystem, QualityLevel
from forge1.core.conflict_resolution import ConflictResolutionSystem, ConflictType, ResolutionStrategy
from forge1.core.escalation_manager import EscalationManager, EscalationTrigger, EscalationPriority


class TestQualityAssuranceSystem:
    """Test cases for Quality Assurance System"""
    
    @pytest.fixture
    def qa_system(self):
        """Create QA system for testing"""
        return QualityAssuranceSystem(
            quality_threshold=0.95,
            superhuman_standards=True
        )
    
    @pytest.fixture
    def sample_result(self):
        """Sample result for testing"""
        return {
            "content": "This is a comprehensive analysis of the market trends showing significant growth in the technology sector.",
            "confidence": 0.9,
            "validation_passed": True
        }
    
    @pytest.fixture
    def sample_context(self):
        """Sample context for testing"""
        return {
            "task_type": "analysis",
            "requirements": ["market analysis", "growth trends", "technology sector"],
            "expected_format": "text",
            "business_impact": 0.8,
            "human_baseline": {
                "accuracy": 0.85,
                "speed": 1.0,
                "quality": 0.8,
                "consistency": 0.75
            }
        }
    
    @pytest.mark.asyncio
    async def test_quality_review_superhuman_standards(self, qa_system, sample_result, sample_context):
        """Test quality review with superhuman standards"""
        
        agents_involved = ["analyst_agent_1"]
        
        review_report = await qa_system.conduct_quality_review(
            sample_result, sample_context, agents_involved, QualityLevel.SUPERHUMAN
        )
        
        assert review_report["id"].startswith("qa_")
        assert review_report["quality_level"] == "superhuman"
        assert "quality_assessment" in review_report
        assert "compliance_check" in review_report
        assert "superhuman_validation" in review_report
        assert "quality_decision" in review_report
        
        # Check quality assessment structure
        qa = review_report["quality_assessment"]
        assert "dimensions" in qa
        assert "overall_score" in qa
        assert "quality_grade" in qa
        
        # Check superhuman validation
        sv = review_report["superhuman_validation"]
        assert "superhuman_ratios" in sv
        assert "overall_superhuman_ratio" in sv
        assert "performance_tier" in sv
    
    @pytest.mark.asyncio
    async def test_quality_dimensions_assessment(self, qa_system, sample_context):
        """Test individual quality dimensions assessment"""
        
        high_quality_result = {
            "content": "Comprehensive market analysis reveals 25% growth in technology sector with strong evidence from multiple data sources including quarterly reports, industry surveys, and expert interviews. The analysis demonstrates clear methodology, accurate data interpretation, and actionable insights for strategic decision-making.",
            "confidence": 0.95,
            "validation_passed": True
        }
        
        review_report = await qa_system.conduct_quality_review(
            high_quality_result, sample_context, ["expert_agent"], QualityLevel.COMPREHENSIVE
        )
        
        dimensions = review_report["quality_assessment"]["dimensions"]
        
        # All dimensions should be present
        expected_dimensions = ["accuracy", "completeness", "relevance", "clarity", "efficiency", "innovation", "reliability"]
        for dimension in expected_dimensions:
            assert dimension in dimensions
            assert 0.0 <= dimensions[dimension] <= 1.0
        
        # High quality content should score well
        assert review_report["quality_assessment"]["overall_score"] > 0.8
    
    @pytest.mark.asyncio
    async def test_compliance_verification(self, qa_system, sample_context):
        """Test compliance verification functionality"""
        
        # Test result with potential compliance issues
        problematic_result = {
            "content": "Analysis shows john.doe@company.com has SSN 123-45-6789 and phone 555-123-4567. Password is secret123.",
            "confidence": 0.8
        }
        
        review_report = await qa_system.conduct_quality_review(
            problematic_result, sample_context, ["agent_1"], QualityLevel.STANDARD
        )
        
        compliance_check = review_report["compliance_check"]
        
        assert "checks" in compliance_check
        assert "overall_compliance" in compliance_check
        assert "violations" in compliance_check
        
        # Should detect privacy and security violations
        assert compliance_check["checks"]["data_privacy"] < 0.8  # Should detect PII
        assert compliance_check["checks"]["security_standards"] < 0.8  # Should detect exposed credentials
        assert not compliance_check["compliant"]
        assert len(compliance_check["violations"]) > 0
    
    @pytest.mark.asyncio
    async def test_improvement_recommendations(self, qa_system, sample_context):
        """Test improvement recommendations generation"""
        
        low_quality_result = {
            "content": "bad analysis. not good. wrong.",
            "confidence": 0.3,
            "validation_passed": False
        }
        
        review_report = await qa_system.conduct_quality_review(
            low_quality_result, sample_context, ["weak_agent"], QualityLevel.STANDARD
        )
        
        improvements = review_report["improvements"]
        
        assert "recommendations" in improvements
        assert "total_recommendations" in improvements
        assert "critical_issues" in improvements
        assert "improvement_potential" in improvements
        
        # Should have multiple recommendations for low quality content
        assert improvements["total_recommendations"] > 0
        assert len(improvements["recommendations"]) > 0
        
        # Check recommendation structure
        for rec in improvements["recommendations"]:
            assert "type" in rec
            assert "priority" in rec
            assert "suggestion" in rec
    
    @pytest.mark.asyncio
    async def test_quality_decision_making(self, qa_system, sample_context):
        """Test quality decision making logic"""
        
        # Test high quality result
        high_quality_result = {
            "content": "Excellent comprehensive analysis with detailed methodology, accurate data, clear conclusions, and actionable recommendations based on thorough research and expert validation.",
            "confidence": 0.98,
            "validation_passed": True
        }
        
        review_report = await qa_system.conduct_quality_review(
            high_quality_result, sample_context, ["expert_agent"], QualityLevel.SUPERHUMAN
        )
        
        decision = review_report["quality_decision"]
        
        assert decision["approved"] == True
        assert decision["overall_score"] >= qa_system.quality_threshold
        assert decision["confidence"] > 0.8
        assert "decision_rationale" in decision
        assert "next_steps" in decision
        
        # Test low quality result
        low_quality_result = {
            "content": "bad",
            "confidence": 0.2,
            "validation_passed": False
        }
        
        review_report_low = await qa_system.conduct_quality_review(
            low_quality_result, sample_context, ["weak_agent"], QualityLevel.STANDARD
        )
        
        decision_low = review_report_low["quality_decision"]
        
        assert decision_low["approved"] == False
        assert decision_low["overall_score"] < qa_system.quality_threshold
        assert "Address quality issues" in decision_low["next_steps"]


class TestConflictResolutionSystem:
    """Test cases for Conflict Resolution System"""
    
    @pytest.fixture
    def conflict_system(self):
        """Create conflict resolution system for testing"""
        return ConflictResolutionSystem(
            resolution_timeout=300,
            enable_auto_resolution=True,
            consensus_threshold=0.7
        )
    
    @pytest.fixture
    def conflicting_results(self):
        """Sample conflicting results for testing"""
        return [
            {
                "content": "Recommend investing in technology stocks for maximum growth potential.",
                "confidence": 0.9,
                "agent": "growth_agent"
            },
            {
                "content": "Recommend conservative bonds for stable returns and risk mitigation.",
                "confidence": 0.85,
                "agent": "conservative_agent"
            },
            {
                "content": "Recommend diversified portfolio with 60% stocks and 40% bonds.",
                "confidence": 0.8,
                "agent": "balanced_agent"
            }
        ]
    
    @pytest.fixture
    def conflict_context(self):
        """Sample conflict context for testing"""
        return {
            "task_type": "investment_recommendation",
            "requirements": ["risk assessment", "return optimization", "diversification"],
            "business_impact": 0.8,
            "deadline": (datetime.now(timezone.utc)).isoformat(),
            "stakeholder_impact": 0.7
        }
    
    @pytest.mark.asyncio
    async def test_conflict_detection_and_resolution(self, conflict_system, conflicting_results, conflict_context):
        """Test complete conflict detection and resolution process"""
        
        agents_involved = ["growth_agent", "conservative_agent", "balanced_agent"]
        
        resolution_report = await conflict_system.detect_and_resolve_conflict(
            agents_involved, conflicting_results, conflict_context
        )
        
        assert resolution_report["id"].startswith("conflict_")
        assert "conflict_analysis" in resolution_report
        assert "resolution_strategy" in resolution_report
        assert "final_result" in resolution_report
        
        # Check conflict analysis
        analysis = resolution_report["conflict_analysis"]
        assert "conflict_type" in analysis
        assert "severity" in analysis
        assert "agent_positions" in analysis
        assert len(analysis["agent_positions"]) == 3
        
        # Check final result
        final_result = resolution_report["final_result"]
        assert "resolved" in final_result
        assert "resolution" in final_result or "escalated" in final_result
    
    @pytest.mark.asyncio
    async def test_evidence_based_resolution(self, conflict_system, conflict_context):
        """Test evidence-based conflict resolution strategy"""
        
        # Create results with different evidence strengths
        results_with_evidence = [
            {
                "content": "Investment analysis based on comprehensive data from 50 companies, research studies, and expert analysis shows technology sector growth of 25% with supporting evidence from quarterly reports.",
                "confidence": 0.95
            },
            {
                "content": "I think technology is good.",
                "confidence": 0.6
            }
        ]
        
        agents = ["expert_agent", "basic_agent"]
        
        resolution_report = await conflict_system.detect_and_resolve_conflict(
            agents, results_with_evidence, conflict_context, ConflictType.RESULT_DISAGREEMENT
        )
        
        # Should select the result with stronger evidence
        if resolution_report["final_result"]["resolved"]:
            selected_agent = resolution_report["final_result"]["resolution"].get("selected_agent")
            # The expert agent with stronger evidence should be selected
            assert selected_agent == "expert_agent" or resolution_report["resolution_strategy"] == "evidence_based"
    
    @pytest.mark.asyncio
    async def test_consensus_building_resolution(self, conflict_system, conflict_context):
        """Test consensus building resolution strategy"""
        
        # Create results with some common elements
        consensus_results = [
            {
                "content": "Recommend diversified investment portfolio with technology focus and risk management.",
                "confidence": 0.8
            },
            {
                "content": "Suggest diversified approach with technology investments and proper risk assessment.",
                "confidence": 0.85
            },
            {
                "content": "Propose balanced diversified strategy including technology sector with risk controls.",
                "confidence": 0.9
            }
        ]
        
        agents = ["agent_1", "agent_2", "agent_3"]
        
        resolution_report = await conflict_system.detect_and_resolve_conflict(
            agents, consensus_results, conflict_context, ConflictType.APPROACH_CONFLICT
        )
        
        if resolution_report["resolution_strategy"] == "consensus_building":
            result = resolution_report["resolution_result"]["result"]
            assert "consensus_result" in result
            assert "common_elements" in result
            assert len(result["common_elements"]) > 0
    
    @pytest.mark.asyncio
    async def test_conflict_escalation(self, conflict_system, conflict_context):
        """Test conflict escalation when resolution fails"""
        
        # Create highly conflicting results that are hard to resolve
        difficult_results = [
            {
                "content": "Absolutely must invest everything in cryptocurrency immediately.",
                "confidence": 0.95
            },
            {
                "content": "Never invest in cryptocurrency under any circumstances.",
                "confidence": 0.95
            }
        ]
        
        agents = ["crypto_bull", "crypto_bear"]
        
        resolution_report = await conflict_system.detect_and_resolve_conflict(
            agents, difficult_results, conflict_context, ConflictType.RESULT_DISAGREEMENT
        )
        
        # Check if escalation was triggered
        if not resolution_report["final_result"]["resolved"]:
            assert resolution_report["final_result"]["escalated"] == True
            assert "escalation_level" in resolution_report["final_result"]
    
    @pytest.mark.asyncio
    async def test_automated_resolution_strategy(self, conflict_system, conflict_context):
        """Test automated resolution for simple conflicts"""
        
        simple_results = [
            {
                "content": "Result A with good quality.",
                "confidence": 0.9
            },
            {
                "content": "Result B with lower quality.",
                "confidence": 0.7
            }
        ]
        
        agents = ["agent_a", "agent_b"]
        
        # Force automated resolution by setting low complexity
        conflict_context["complexity"] = 0.2
        
        resolution_report = await conflict_system.detect_and_resolve_conflict(
            agents, simple_results, conflict_context
        )
        
        if resolution_report["resolution_strategy"] == "automated":
            # Should select the result with higher confidence
            result = resolution_report["resolution_result"]["result"]
            assert result["confidence"] >= 0.9


class TestEscalationManager:
    """Test cases for Escalation Manager"""
    
    @pytest.fixture
    def escalation_manager(self):
        """Create escalation manager for testing"""
        return EscalationManager(
            max_escalation_levels=4,
            enable_auto_fallback=True,
            emergency_contacts=["admin@company.com", "cto@company.com"]
        )
    
    @pytest.fixture
    def escalation_context(self):
        """Sample escalation context for testing"""
        return {
            "task_id": "task_123",
            "agents_involved": ["agent_1", "agent_2"],
            "failure_reason": "Quality standards not met",
            "business_impact": 0.8,
            "timeline_critical": True
        }
    
    @pytest.mark.asyncio
    async def test_escalation_trigger(self, escalation_manager, escalation_context):
        """Test escalation triggering and initial response"""
        
        escalation_result = await escalation_manager.trigger_escalation(
            EscalationTrigger.QUALITY_FAILURE,
            escalation_context,
            EscalationPriority.HIGH
        )
        
        assert escalation_result["status"] == "initiated"
        assert "escalation_id" in escalation_result
        assert escalation_result["initial_level"] >= 1
        assert "estimated_resolution_time" in escalation_result
        
        # Check if escalation is tracked
        escalation_id = escalation_result["escalation_id"]
        status = escalation_manager.get_escalation_status(escalation_id)
        assert status is not None
        assert status["trigger"] == "quality_failure"
        assert status["priority"] == "high"
    
    @pytest.mark.asyncio
    async def test_escalation_levels(self, escalation_manager, escalation_context):
        """Test different escalation levels"""
        
        # Test low priority escalation (should start at level 1)
        low_priority_result = await escalation_manager.trigger_escalation(
            EscalationTrigger.PERFORMANCE_DEGRADATION,
            escalation_context,
            EscalationPriority.LOW
        )
        assert low_priority_result["initial_level"] == 1
        
        # Test critical escalation (should start at higher level)
        critical_result = await escalation_manager.trigger_escalation(
            EscalationTrigger.COMPLIANCE_VIOLATION,
            escalation_context,
            EscalationPriority.CRITICAL
        )
        assert critical_result["initial_level"] >= 3
        
        # Test emergency escalation (should start at highest level)
        emergency_result = await escalation_manager.trigger_escalation(
            EscalationTrigger.SYSTEM_ERROR,
            escalation_context,
            EscalationPriority.EMERGENCY
        )
        assert emergency_result["initial_level"] == 4
    
    @pytest.mark.asyncio
    async def test_fallback_mechanisms(self, escalation_manager, escalation_context):
        """Test fallback mechanism execution"""
        
        # Trigger escalation that should activate fallback
        escalation_result = await escalation_manager.trigger_escalation(
            EscalationTrigger.TIMEOUT_EXCEEDED,
            escalation_context,
            EscalationPriority.MEDIUM
        )
        
        # Check if fallback was executed
        if escalation_result.get("fallback_executed"):
            escalation_id = escalation_result["escalation_id"]
            status = escalation_manager.get_escalation_status(escalation_id)
            assert status["fallback_executed"] == True
            assert "fallback_result" in status
    
    @pytest.mark.asyncio
    async def test_known_good_state_storage(self, escalation_manager):
        """Test storing and retrieving known good states"""
        
        good_state = {
            "configuration": {"param1": "value1", "param2": "value2"},
            "performance_metrics": {"accuracy": 0.95, "speed": 1.2},
            "validation_status": "passed"
        }
        
        escalation_manager.store_known_good_state("analysis_task", good_state)
        
        # Verify storage
        stored_states = escalation_manager.known_good_states
        assert "analysis_task" in stored_states
        assert stored_states["analysis_task"]["state"] == good_state
        assert stored_states["analysis_task"]["verified"] == True
    
    @pytest.mark.asyncio
    async def test_escalation_metrics(self, escalation_manager, escalation_context):
        """Test escalation metrics tracking"""
        
        initial_metrics = escalation_manager.get_escalation_metrics()
        initial_count = initial_metrics["escalations_triggered"]
        
        # Trigger escalation
        await escalation_manager.trigger_escalation(
            EscalationTrigger.QUALITY_FAILURE,
            escalation_context,
            EscalationPriority.MEDIUM
        )
        
        # Check metrics update
        updated_metrics = escalation_manager.get_escalation_metrics()
        assert updated_metrics["escalations_triggered"] == initial_count + 1
        
        # Check metric structure
        expected_metrics = [
            "escalations_triggered", "escalations_resolved", "fallbacks_executed",
            "emergency_procedures_activated", "average_escalation_time",
            "escalation_success_rate", "fallback_success_rate", "system_availability"
        ]
        
        for metric in expected_metrics:
            assert metric in updated_metrics
    
    @pytest.mark.asyncio
    async def test_escalation_resolution(self, escalation_manager, escalation_context):
        """Test escalation resolution process"""
        
        # Trigger escalation
        escalation_result = await escalation_manager.trigger_escalation(
            EscalationTrigger.CONFLICT_UNRESOLVED,
            escalation_context,
            EscalationPriority.MEDIUM
        )
        
        escalation_id = escalation_result["escalation_id"]
        
        # Resolve escalation
        resolution = {
            "method": "manual_intervention",
            "result": "Conflict resolved through expert review",
            "confidence": 0.9
        }
        
        resolution_result = await escalation_manager.resolve_escalation(
            escalation_id, resolution, "human_expert"
        )
        
        assert resolution_result["status"] == "resolved"
        assert resolution_result["escalation_id"] == escalation_id
        
        # Verify escalation is no longer active
        status = escalation_manager.get_escalation_status(escalation_id)
        assert status is None  # Should be moved to history
        
        # Check history
        history = escalation_manager.get_escalation_history(limit=1)
        assert len(history) >= 1
        assert history[-1]["id"] == escalation_id
        assert history[-1]["status"] == "resolved"


class TestIntegratedQualitySystem:
    """Integration tests for the complete quality assurance system"""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated quality system for testing"""
        qa_system = QualityAssuranceSystem(quality_threshold=0.9, superhuman_standards=True)
        conflict_system = ConflictResolutionSystem(enable_auto_resolution=True)
        escalation_manager = EscalationManager(enable_auto_fallback=True)
        
        return {
            "qa": qa_system,
            "conflict": conflict_system,
            "escalation": escalation_manager
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_quality_workflow(self, integrated_system):
        """Test complete end-to-end quality assurance workflow"""
        
        # Simulate agent results with quality issues
        agent_results = [
            {
                "content": "High quality comprehensive analysis with detailed methodology and strong evidence base.",
                "confidence": 0.95,
                "agent": "expert_agent"
            },
            {
                "content": "Low quality analysis. Not good.",
                "confidence": 0.4,
                "agent": "weak_agent"
            }
        ]
        
        context = {
            "task_type": "analysis",
            "requirements": ["comprehensive analysis", "strong evidence"],
            "business_impact": 0.8
        }
        
        # Phase 1: Quality reviews
        qa_results = []
        for i, result in enumerate(agent_results):
            agent_name = result["agent"]
            qa_result = await integrated_system["qa"].conduct_quality_review(
                result, context, [agent_name], QualityLevel.SUPERHUMAN
            )
            qa_results.append(qa_result)
        
        # Phase 2: Check for conflicts if multiple approved results
        approved_results = [
            (i, result) for i, result in enumerate(agent_results)
            if qa_results[i]["quality_decision"]["approved"]
        ]
        
        if len(approved_results) > 1:
            # Test conflict resolution
            agents = [result[1]["agent"] for result in approved_results]
            results = [result[1] for result in approved_results]
            
            conflict_result = await integrated_system["conflict"].detect_and_resolve_conflict(
                agents, results, context
            )
            
            assert "final_result" in conflict_result
        
        # Phase 3: Handle quality failures with escalation
        failed_reviews = [
            qa_result for qa_result in qa_results
            if not qa_result["quality_decision"]["approved"]
        ]
        
        if failed_reviews:
            # Trigger escalation for quality failure
            escalation_result = await integrated_system["escalation"].trigger_escalation(
                EscalationTrigger.QUALITY_FAILURE,
                {**context, "failed_reviews": failed_reviews},
                EscalationPriority.MEDIUM
            )
            
            assert escalation_result["status"] == "initiated"
    
    @pytest.mark.asyncio
    async def test_superhuman_performance_validation(self, integrated_system):
        """Test superhuman performance validation across the system"""
        
        superhuman_result = {
            "content": "Exceptional analysis demonstrating superior methodology, comprehensive data analysis, innovative insights, and actionable recommendations that significantly exceed professional human capabilities in accuracy, depth, and strategic value.",
            "confidence": 0.98,
            "execution_time": 30,  # Much faster than human
            "validation_passed": True
        }
        
        context = {
            "task_type": "strategic_analysis",
            "human_baseline": {
                "accuracy": 0.85,
                "speed": 1.0,
                "quality": 0.8,
                "consistency": 0.75
            },
            "expected_time": 3600,  # Human would take 1 hour
            "speed_multiplier": 120  # 120x faster than human
        }
        
        qa_result = await integrated_system["qa"].conduct_quality_review(
            superhuman_result, context, ["superhuman_agent"], QualityLevel.SUPERHUMAN
        )
        
        # Verify superhuman performance validation
        superhuman_validation = qa_result["superhuman_validation"]
        assert superhuman_validation["achieves_superhuman"] == True
        assert superhuman_validation["overall_superhuman_ratio"] >= 1.2
        assert superhuman_validation["performance_tier"] in ["Superhuman", "Advanced Superhuman", "Extreme Superhuman"]
        
        # Quality decision should approve superhuman performance
        assert qa_result["quality_decision"]["approved"] == True
        assert qa_result["quality_decision"]["overall_score"] >= 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])