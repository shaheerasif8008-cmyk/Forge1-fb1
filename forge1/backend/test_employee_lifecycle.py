#!/usr/bin/env python3
"""
AI Employee Lifecycle End-to-End Test

This script demonstrates the complete AI employee lifecycle:
1. Client onboarding
2. Employee creation from requirements
3. Employee interaction with memory
4. Memory retrieval and search
5. Tenant isolation verification
"""

import asyncio
import os
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal

# Add the forge1 backend to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from forge1.services.client_manager import ClientManager
from forge1.services.employee_memory_manager import EmployeeMemoryManager
from forge1.services.employee_manager import EmployeeManager
from forge1.models.employee_models import (
    ClientInfo, ClientTier, SecurityLevel, EmployeeRequirements,
    CommunicationStyle, FormalityLevel, ExpertiseLevel, ResponseLength,
    EmployeeInteraction, EmployeeResponse, MemoryType
)
from forge1.core.database_config import DatabaseManager


async def test_complete_lifecycle():
    """Test the complete AI employee lifecycle"""
    print("🚀 Testing Complete AI Employee Lifecycle...\n")
    
    # Initialize managers
    print("🔧 Initializing services...")
    db_manager = DatabaseManager()
    await db_manager.initialize()
    
    client_manager = ClientManager(db_manager)
    await client_manager.initialize()
    
    memory_manager = EmployeeMemoryManager(db_manager)
    await memory_manager.initialize()
    
    employee_manager = EmployeeManager(db_manager, client_manager, memory_manager)
    await employee_manager.initialize()
    
    print("✅ Services initialized\n")
    
    # Step 1: Client Onboarding
    print("👥 Step 1: Client Onboarding")
    
    client_info = ClientInfo(
        name="Acme Legal Services",
        industry="legal",
        tier=ClientTier.PROFESSIONAL,
        max_employees=25,
        allowed_models=["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"],
        security_level=SecurityLevel.HIGH,
        compliance_requirements=["GDPR", "SOX", "Attorney-Client Privilege"]
    )
    
    try:
        client = await client_manager.onboard_client(client_info)
        print(f"✅ Client onboarded: {client.id} - {client.name}")
        print(f"   Tier: {client.tier.value}, Max employees: {client.configuration.max_employees}")
    except Exception as e:
        print(f"❌ Client onboarding failed: {e}")
        return False
    
    # Step 2: Employee Creation
    print(f"\n🤖 Step 2: Employee Creation")
    
    employee_requirements = EmployeeRequirements(
        role="Corporate Lawyer",
        industry="legal",
        expertise_areas=["Contract Law", "Mergers & Acquisitions", "Corporate Governance"],
        communication_style=CommunicationStyle.PROFESSIONAL,
        formality_level=FormalityLevel.FORMAL,
        expertise_level=ExpertiseLevel.EXPERT,
        response_length=ResponseLength.DETAILED,
        creativity_level=0.3,  # Low creativity for legal work
        empathy_level=0.6,     # Moderate empathy
        tools_needed=["legal_research", "document_review", "contract_analysis"],
        knowledge_domains=["Corporate Law", "Securities Regulations", "Contract Templates"],
        personality_traits={"attention_to_detail": "high", "risk_assessment": "conservative"}
    )
    
    try:
        employee = await employee_manager.create_employee(
            client.id, employee_requirements
        )
        print(f"✅ Employee created: {employee.id} - {employee.name}")
        print(f"   Role: {employee.role}")
        print(f"   Communication: {employee.personality.communication_style.value}")
        print(f"   Tools: {', '.join(employee.tool_access)}")
    except Exception as e:
        print(f"❌ Employee creation failed: {e}")
        return False
    
    # Step 3: Employee Interactions (Memory is initialized automatically)
    print(f"\n💬 Step 3: Employee Interactions")
    
    interactions = [
        {
            "message": "Hello, I need help reviewing a merger agreement for potential legal issues.",
            "response": "I'd be happy to help you review the merger agreement. As a corporate lawyer specializing in M&A, I'll examine the document for key legal considerations including due diligence requirements, liability provisions, regulatory compliance, and potential risk factors. Please provide the agreement and I'll conduct a thorough analysis."
        },
        {
            "message": "What are the key due diligence items I should focus on for this acquisition?",
            "response": "For M&A due diligence, focus on these critical areas: 1) Financial statements and audit reports, 2) Material contracts and customer agreements, 3) Intellectual property portfolio and licensing, 4) Regulatory compliance and pending litigation, 5) Employment agreements and benefit plans, 6) Environmental liabilities, 7) Tax compliance and structure. I recommend creating a comprehensive checklist and timeline for each workstream."
        },
        {
            "message": "Can you draft a confidentiality agreement for our discussions with the target company?",
            "response": "I'll draft a comprehensive mutual non-disclosure agreement (NDA) for your M&A discussions. The agreement will include: mutual confidentiality obligations, definition of confidential information, permitted uses and disclosures, return/destruction of materials, term duration (typically 2-3 years), and standard provisions for remedies and governing law. I'll ensure it's balanced and protects both parties' interests while facilitating productive negotiations."
        }
    ]
    
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    
    for i, interaction_data in enumerate(interactions, 1):
        try:
            # Process interaction using employee manager
            response = await employee_manager.interact_with_employee(
                client_id=client.id,
                employee_id=employee.id,
                message=interaction_data["message"],
                session_id=session_id,
                context={"interaction_number": i, "topic": "M&A Legal Review"}
            )
            
            print(f"  ✅ Interaction {i} processed successfully")
            print(f"     Model: {response.model_used}, Tokens: {response.tokens_used}")
            print(f"     Response: {response.message[:100]}...")
            
        except Exception as e:
            print(f"  ❌ Interaction {i} failed: {e}")
            return False
    
    # Step 4: Memory Retrieval and Search
    print(f"\n🔍 Step 4: Memory Retrieval and Search")
    
    try:
        # Get recent context
        recent_memories = await memory_manager.get_employee_context(
            client.id, employee.id, limit=5
        )
        print(f"✅ Retrieved {len(recent_memories)} recent memories")
        
        # Semantic search
        search_results, search_time = await memory_manager.search_employee_memory(
            client.id, employee.id, 
            query="due diligence checklist for mergers",
            limit=3
        )
        print(f"✅ Semantic search completed in {search_time:.2f}ms")
        print(f"   Found {len(search_results)} relevant memories")
        
        for i, memory in enumerate(search_results, 1):
            print(f"   {i}. Relevance: {memory.relevance_score:.3f} - {memory.content[:60]}...")
        
    except Exception as e:
        print(f"❌ Memory retrieval failed: {e}")
        return False
    
    # Step 5: Knowledge Base
    print(f"\n📚 Step 5: Knowledge Base Management")
    
    try:
        # Add knowledge source
        knowledge_id = await memory_manager.add_knowledge_source(
            client.id, employee.id,
            title="Delaware General Corporation Law - Section 251 (Merger Procedures)",
            content="Section 251 of the Delaware General Corporation Law governs merger procedures, requiring board approval, stockholder approval (unless certain exceptions apply), and proper notice requirements. Key provisions include the requirement for a merger agreement, appraisal rights for dissenting stockholders, and specific voting thresholds.",
            source_type="manual",
            keywords=["Delaware law", "merger procedures", "Section 251", "stockholder approval"],
            tags=["corporate law", "M&A", "Delaware", "statutory requirements"]
        )
        print(f"✅ Knowledge source added: {knowledge_id}")
        
        # Search knowledge base
        knowledge_results = await memory_manager.search_knowledge_base(
            client.id, employee.id,
            query="merger approval requirements",
            limit=3
        )
        print(f"✅ Knowledge search returned {len(knowledge_results)} results")
        
        for result in knowledge_results:
            print(f"   - {result['title']} (relevance: {result['relevance_score']:.3f})")
        
    except Exception as e:
        print(f"❌ Knowledge base operations failed: {e}")
        return False
    
    # Step 6: Memory Statistics
    print(f"\n📊 Step 6: Memory Statistics")
    
    try:
        stats = await memory_manager.get_memory_stats(client.id, employee.id)
        print(f"✅ Memory statistics retrieved:")
        print(f"   Total interactions: {stats['total_interactions']}")
        print(f"   Unique sessions: {stats['unique_sessions']}")
        print(f"   Knowledge items: {stats['total_knowledge_items']}")
        print(f"   Average importance: {stats['average_importance']:.3f}")
        
    except Exception as e:
        print(f"❌ Statistics retrieval failed: {e}")
        return False
    
    # Step 7: Client Statistics
    print(f"\n📈 Step 7: Client Statistics")
    
    try:
        client_stats = await client_manager.get_client_stats(client.id)
        print(f"✅ Client statistics retrieved:")
        print(f"   Total employees: {client_stats['total_employees']}")
        print(f"   Active employees: {client_stats['active_employees']}")
        print(f"   Total interactions: {client_stats['total_interactions']}")
        print(f"   Total cost: ${client_stats['total_cost']:.4f}")
        
    except Exception as e:
        print(f"❌ Client statistics failed: {e}")
        return False
    
    # Step 8: Tenant Isolation Test
    print(f"\n🔒 Step 8: Tenant Isolation Test")
    
    try:
        # Try to access employee from different client (should fail)
        fake_client_id = "client_fake_123"
        
        try:
            await memory_manager.get_employee_context(fake_client_id, employee.id)
            print(f"❌ Tenant isolation FAILED - unauthorized access allowed")
            return False
        except Exception:
            print(f"✅ Tenant isolation working - unauthorized access blocked")
        
    except Exception as e:
        print(f"❌ Tenant isolation test failed: {e}")
        return False
    
    # Cleanup
    await db_manager.cleanup()
    
    print(f"\n🎉 Complete AI Employee Lifecycle Test PASSED!")
    print(f"\n📋 Summary:")
    print(f"   ✅ Client onboarded: {client.name}")
    print(f"   ✅ Employee created: {employee.name} ({employee.role})")
    print(f"   ✅ Memory system working with {len(interactions)} interactions")
    print(f"   ✅ Semantic search functional")
    print(f"   ✅ Knowledge base operational")
    print(f"   ✅ Tenant isolation enforced")
    print(f"   ✅ Statistics and analytics available")
    
    return True


async def main():
    """Main test function"""
    print("🧪 Forge1 AI Employee Lifecycle Test Suite\n")
    
    # Check environment
    required_vars = ['POSTGRES_HOST', 'POSTGRES_DB', 'POSTGRES_USER', 'POSTGRES_PASSWORD']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print(f"✅ OpenAI API key configured")
    else:
        print(f"⚠️  OpenAI API key not found - using mock embeddings")
    
    # Run the complete test
    try:
        success = await test_complete_lifecycle()
        
        if success:
            print(f"\n🏆 ALL TESTS PASSED! The AI Employee Lifecycle system is working correctly.")
        else:
            print(f"\n💥 TESTS FAILED! Please check the error messages above.")
            
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())