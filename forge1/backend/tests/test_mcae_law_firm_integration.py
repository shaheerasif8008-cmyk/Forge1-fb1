"""
MCAE Law Firm Integration Test

End-to-end integration test demonstrating the law firm scenario:
Intake → Lawyer → Research workflow running under MCAE orchestration
while maintaining Forge1's tenant isolation and enterprise features.
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any

from forge1.services.employee_manager import EmployeeManager
from forge1.models.employee_models import EmployeeRequirements, CommunicationStyle, ExpertiseLevel
from forge1.integrations.mcae_adapter import MCAEAdapter
from forge1.integrations.mcae_error_handler import MCAEErrorHandler
from forge1.core.tenancy import set_current_tenant, get_current_tenant
from forge1.core.database_config import get_database_manager
from forge1.core.model_router import ModelRouter
from forge1.core.memory_manager import MemoryManager


class TestMCAELawFirmIntegration:
    """Test suite for MCAE integration with law firm workflow scenario"""
    
    @pytest.fixture
    async def setup_test_environment(self):
        """Set up test environment with all necessary components"""
        
        # Create test tenant
        tenant_id = f"law_firm_test_{uuid.uuid4().hex[:8]}"
        set_current_tenant(tenant_id)
        
        # Initialize core components
        db_manager = await get_database_manager()
        model_router = ModelRouter()
        memory_manager = MemoryManager(db_manager)
        await memory_manager.initialize()
        
        # Initialize MCAE components
        mcae_error_handler = MCAEErrorHandler()
        mcae_adapter = MCAEAdapter(
            employee_manager=None,  # Will be set after employee manager creation
            model_router=model_router,
            memory_manager=memory_manager
        )
        
        # Initialize employee manager with MCAE
        employee_manager = EmployeeManager(
            db_manager=db_manager,
            model_router=model_router,
            memory_manager=memory_manager,
            mcae_adapter=mcae_adapter
        )
        await employee_manager.initialize()
        
        # Set employee manager reference in MCAE adapter
        mcae_adapter.employee_manager = employee_manager
        await mcae_adapter.initialize()
        
        return {
            "tenant_id": tenant_id,
            "employee_manager": employee_manager,
            "mcae_adapter": mcae_adapter,
            "mcae_error_handler": mcae_error_handler,
            "model_router": model_router,
            "memory_manager": memory_manager
        }
    
    @pytest.mark.asyncio
    async def test_law_firm_employee_creation_and_registration(self, setup_test_environment):
        """Test creating law firm employees and registering them with MCAE"""
        
        env = await setup_test_environment
        tenant_id = env["tenant_id"]
        employee_manager = env["employee_manager"]
        
        # Create Intake employee
        intake_requirements = EmployeeRequirements(
            role="Legal Intake Specialist",
            industry="Legal Services",
            expertise_areas=["Client Intake", "Case Assessment", "Legal Documentation"],
            communication_style=CommunicationStyle.PROFESSIONAL,
            expertise_level=ExpertiseLevel.INTERMEDIATE,
            tools_needed=["document_parser", "vector_search"],
            workflow_type="legal_intake",
            collaboration_mode="sequential"
        )
        
        intake_employee = await employee_manager.create_employee(tenant_id, intake_requirements)
        
        # Verify employee was created and registered with MCAE
        assert intake_employee is not None
        assert intake_employee.client_id == tenant_id
        assert intake_employee.role == "Legal Intake Specialist"
        assert intake_employee.workflow_id is not None
        
        # Create Lawyer employee
        lawyer_requirements = EmployeeRequirements(
            role="Senior Attorney",
            industry="Legal Services", 
            expertise_areas=["Contract Law", "Legal Analysis", "Case Strategy"],
            communication_style=CommunicationStyle.PROFESSIONAL,
            expertise_level=ExpertiseLevel.EXPERT,
            tools_needed=["document_parser", "vector_search", "legal_research"],
            workflow_type="legal_analysis",
            collaboration_mode="sequential"
        )
        
        lawyer_employee = await employee_manager.create_employee(tenant_id, lawyer_requirements)
        
        # Verify lawyer employee
        assert lawyer_employee is not None
        assert lawyer_employee.client_id == tenant_id
        assert lawyer_employee.role == "Senior Attorney"
        assert lawyer_employee.workflow_id is not None
        
        # Create Research employee
        research_requirements = EmployeeRequirements(
            role="Legal Researcher",
            industry="Legal Services",
            expertise_areas=["Legal Research", "Case Law", "Precedent Analysis"],
            communication_style=CommunicationStyle.TECHNICAL,
            expertise_level=ExpertiseLevel.EXPERT,
            tools_needed=["vector_search", "legal_database", "document_parser"],
            workflow_type="legal_research",
            collaboration_mode="sequential"
        )
        
        research_employee = await employee_manager.create_employee(tenant_id, research_requirements)
        
        # Verify research employee
        assert research_employee is not None
        assert research_employee.client_id == tenant_id
        assert research_employee.role == "Legal Researcher"
        assert research_employee.workflow_id is not None
        
        # Verify all employees are registered with different workflow IDs
        workflow_ids = {intake_employee.workflow_id, lawyer_employee.workflow_id, research_employee.workflow_id}
        assert len(workflow_ids) == 3, "All employees should have unique workflow IDs"
        
        return {
            "intake_employee": intake_employee,
            "lawyer_employee": lawyer_employee,
            "research_employee": research_employee
        }
    
    @pytest.mark.asyncio
    async def test_law_firm_workflow_execution(self, setup_test_environment):
        """Test end-to-end workflow execution: Intake → Lawyer → Research"""
        
        env = await setup_test_environment
        tenant_id = env["tenant_id"]
        employee_manager = env["employee_manager"]
        
        # Create employees first
        employees = await self.test_law_firm_employee_creation_and_registration(setup_test_environment)
        intake_employee = employees["intake_employee"]
        lawyer_employee = employees["lawyer_employee"]
        research_employee = employees["research_employee"]
        
        # Step 1: Intake employee processes new case
        intake_task = """
        New client case: Contract dispute with vendor ABC Corp.
        Client: XYZ Manufacturing
        Issue: Vendor failed to deliver goods per contract terms, seeking damages.
        Contract value: $500,000
        Delivery deadline missed by 30 days.
        """
        
        intake_result = await employee_manager.execute_employee_task(
            tenant_id, 
            intake_employee.id,
            intake_task,
            {"case_type": "contract_dispute", "priority": "high"}
        )
        
        # Verify intake result
        assert intake_result is not None
        assert intake_result["status"] == "completed"
        assert intake_result["tenant_id"] == tenant_id
        assert intake_result["employee_id"] == intake_employee.id
        assert "workflow_id" in intake_result
        
        # Step 2: Lawyer analyzes the case
        lawyer_task = f"""
        Analyze the contract dispute case from intake:
        {intake_result.get('result', {}).get('summary', 'Case details from intake')}
        
        Please provide:
        1. Legal assessment of the case
        2. Potential claims and defenses
        3. Recommended legal strategy
        4. Research areas needed
        """
        
        lawyer_result = await employee_manager.execute_employee_task(
            tenant_id,
            lawyer_employee.id, 
            lawyer_task,
            {
                "case_id": intake_result.get("session_id"),
                "intake_summary": intake_result.get("result", {})
            }
        )
        
        # Verify lawyer result
        assert lawyer_result is not None
        assert lawyer_result["status"] == "completed"
        assert lawyer_result["tenant_id"] == tenant_id
        assert lawyer_result["employee_id"] == lawyer_employee.id
        
        # Step 3: Research employee finds precedents
        research_task = f"""
        Research legal precedents for the contract dispute case:
        {lawyer_result.get('result', {}).get('legal_issues', 'Legal issues from lawyer analysis')}
        
        Find:
        1. Similar contract dispute cases
        2. Relevant case law and precedents
        3. Statutory requirements
        4. Damages calculation methods
        """
        
        research_result = await employee_manager.execute_employee_task(
            tenant_id,
            research_employee.id,
            research_task,
            {
                "case_id": intake_result.get("session_id"),
                "legal_analysis": lawyer_result.get("result", {})
            }
        )
        
        # Verify research result
        assert research_result is not None
        assert research_result["status"] == "completed"
        assert research_result["tenant_id"] == tenant_id
        assert research_result["employee_id"] == research_employee.id
        
        # Verify workflow isolation - all results should be for the same tenant
        all_results = [intake_result, lawyer_result, research_result]
        for result in all_results:
            assert result["tenant_id"] == tenant_id
            assert "workflow_id" in result
            assert "session_id" in result
        
        return {
            "intake_result": intake_result,
            "lawyer_result": lawyer_result,
            "research_result": research_result
        }
    
    @pytest.mark.asyncio
    async def test_tenant_isolation_enforcement(self, setup_test_environment):
        """Test that tenant isolation is enforced throughout the workflow"""
        
        env = await setup_test_environment
        tenant_id = env["tenant_id"]
        employee_manager = env["employee_manager"]
        
        # Create employee for first tenant
        employee_requirements = EmployeeRequirements(
            role="Legal Assistant",
            industry="Legal Services",
            expertise_areas=["Legal Research"],
            workflow_type="standard"
        )
        
        employee1 = await employee_manager.create_employee(tenant_id, employee_requirements)
        
        # Create second tenant
        other_tenant_id = f"other_law_firm_{uuid.uuid4().hex[:8]}"
        set_current_tenant(other_tenant_id)
        
        employee2 = await employee_manager.create_employee(other_tenant_id, employee_requirements)
        
        # Try to access employee1 from other tenant context - should fail
        set_current_tenant(other_tenant_id)
        
        with pytest.raises(Exception):  # Should raise TenantIsolationError or similar
            await employee_manager.execute_employee_task(
                other_tenant_id,  # Wrong tenant
                employee1.id,     # Employee from different tenant
                "Unauthorized access attempt"
            )
        
        # Verify employees are isolated
        assert employee1.client_id == tenant_id
        assert employee2.client_id == other_tenant_id
        assert employee1.workflow_id != employee2.workflow_id
    
    @pytest.mark.asyncio
    async def test_mcae_error_handling_and_fallback(self, setup_test_environment):
        """Test error handling and fallback to Forge1 native orchestration"""
        
        env = await setup_test_environment
        tenant_id = env["tenant_id"]
        employee_manager = env["employee_manager"]
        mcae_error_handler = env["mcae_error_handler"]
        
        # Create employee
        employee_requirements = EmployeeRequirements(
            role="Test Employee",
            industry="Legal Services",
            expertise_areas=["Testing"],
            workflow_type="standard"
        )
        
        employee = await employee_manager.create_employee(tenant_id, employee_requirements)
        
        # Simulate MCAE failure by temporarily disabling adapter
        original_adapter = employee_manager.mcae_adapter
        employee_manager.mcae_adapter = None
        
        try:
            # This should trigger fallback to Forge1 native orchestration
            result = await employee_manager.execute_employee_task(
                tenant_id,
                employee.id,
                "Test task that should fallback to native orchestration"
            )
            
            # Should still work but without MCAE
            # In a real implementation, this would use Forge1's native orchestration
            
        except Exception as e:
            # Handle the error through error handler
            error_result = await mcae_error_handler.handle_error(
                e,
                {"tenant_id": tenant_id, "employee_id": employee.id},
                "execute_employee_task"
            )
            
            assert error_result["error_handled"] == True
            assert "recovery_result" in error_result
            
        finally:
            # Restore adapter
            employee_manager.mcae_adapter = original_adapter
    
    @pytest.mark.asyncio
    async def test_workflow_status_and_metrics(self, setup_test_environment):
        """Test workflow status tracking and metrics collection"""
        
        env = await setup_test_environment
        tenant_id = env["tenant_id"]
        employee_manager = env["employee_manager"]
        mcae_adapter = env["mcae_adapter"]
        
        # Create employee
        employee_requirements = EmployeeRequirements(
            role="Metrics Test Employee",
            industry="Legal Services",
            expertise_areas=["Testing"],
            workflow_type="standard"
        )
        
        employee = await employee_manager.create_employee(tenant_id, employee_requirements)
        
        # Check workflow status
        status = await employee_manager.get_employee_workflow_status(tenant_id, employee.id)
        
        assert status["employee_id"] == employee.id
        assert status["workflow_registered"] == True
        assert "workflow_id" in status
        
        # Execute a task to generate metrics
        await employee_manager.execute_employee_task(
            tenant_id,
            employee.id,
            "Test task for metrics collection"
        )
        
        # Check MCAE metrics
        metrics = employee_manager.get_mcae_metrics()
        
        assert "mcae_workflows_created" in metrics
        assert "mcae_tasks_executed" in metrics
        assert metrics["mcae_adapter_available"] == True
        
        # Check adapter metrics
        adapter_metrics = mcae_adapter.get_metrics()
        
        assert "workflows_created" in adapter_metrics
        assert "workflows_executed" in adapter_metrics
        assert adapter_metrics["active_workflows"] >= 0
    
    @pytest.mark.asyncio
    async def test_workflow_cleanup(self, setup_test_environment):
        """Test workflow cleanup and resource management"""
        
        env = await setup_test_environment
        tenant_id = env["tenant_id"]
        employee_manager = env["employee_manager"]
        
        # Create employee
        employee_requirements = EmployeeRequirements(
            role="Cleanup Test Employee",
            industry="Legal Services",
            expertise_areas=["Testing"],
            workflow_type="standard"
        )
        
        employee = await employee_manager.create_employee(tenant_id, employee_requirements)
        
        # Verify workflow is registered
        status_before = await employee_manager.get_employee_workflow_status(tenant_id, employee.id)
        assert status_before["workflow_registered"] == True
        
        # Cleanup workflow
        cleanup_success = await employee_manager.cleanup_employee_workflow(tenant_id, employee.id)
        assert cleanup_success == True
        
        # Verify workflow is cleaned up
        status_after = await employee_manager.get_employee_workflow_status(tenant_id, employee.id)
        assert status_after["workflow_registered"] == False


# Utility functions for testing
async def create_test_law_firm_scenario():
    """Create a complete law firm scenario for manual testing"""
    
    # Set up test environment
    tenant_id = f"manual_test_law_firm_{uuid.uuid4().hex[:8]}"
    set_current_tenant(tenant_id)
    
    print(f"Creating law firm scenario for tenant: {tenant_id}")
    
    # Initialize components (simplified for manual testing)
    employee_manager = EmployeeManager()
    await employee_manager.initialize()
    
    # Create employees
    employees = {}
    
    # Intake Specialist
    intake_req = EmployeeRequirements(
        role="Legal Intake Specialist",
        industry="Legal Services",
        expertise_areas=["Client Intake", "Case Assessment"],
        workflow_type="legal_intake"
    )
    employees["intake"] = await employee_manager.create_employee(tenant_id, intake_req)
    
    # Senior Attorney
    lawyer_req = EmployeeRequirements(
        role="Senior Attorney", 
        industry="Legal Services",
        expertise_areas=["Contract Law", "Legal Analysis"],
        workflow_type="legal_analysis"
    )
    employees["lawyer"] = await employee_manager.create_employee(tenant_id, lawyer_req)
    
    # Legal Researcher
    research_req = EmployeeRequirements(
        role="Legal Researcher",
        industry="Legal Services", 
        expertise_areas=["Legal Research", "Case Law"],
        workflow_type="legal_research"
    )
    employees["researcher"] = await employee_manager.create_employee(tenant_id, research_req)
    
    print("Created employees:")
    for role, employee in employees.items():
        print(f"  {role}: {employee.name} (ID: {employee.id}, Workflow: {employee.workflow_id})")
    
    return tenant_id, employee_manager, employees


if __name__ == "__main__":
    """Run manual test scenario"""
    async def main():
        try:
            tenant_id, employee_manager, employees = await create_test_law_firm_scenario()
            
            # Execute sample workflow
            print("\nExecuting sample workflow...")
            
            # Intake task
            intake_result = await employee_manager.execute_employee_task(
                tenant_id,
                employees["intake"].id,
                "New contract dispute case: Client ABC vs Vendor XYZ, $100K contract breach"
            )
            print(f"Intake result: {intake_result['status']}")
            
            # Lawyer analysis
            lawyer_result = await employee_manager.execute_employee_task(
                tenant_id,
                employees["lawyer"].id,
                "Analyze the contract dispute case and provide legal strategy"
            )
            print(f"Lawyer result: {lawyer_result['status']}")
            
            # Research
            research_result = await employee_manager.execute_employee_task(
                tenant_id,
                employees["researcher"].id,
                "Research precedents for contract breach damages"
            )
            print(f"Research result: {research_result['status']}")
            
            print("\nWorkflow completed successfully!")
            
            # Show metrics
            metrics = employee_manager.get_mcae_metrics()
            print(f"\nMCAE Metrics:")
            print(f"  Workflows created: {metrics.get('mcae_workflows_created', 0)}")
            print(f"  Tasks executed: {metrics.get('mcae_tasks_executed', 0)}")
            
        except Exception as e:
            print(f"Error in manual test: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the manual test
    asyncio.run(main())