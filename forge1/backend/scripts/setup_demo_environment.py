#!/usr/bin/env python3
"""
Demo Environment Setup Script

Creates "Hartwell & Associates" tenant with four employees and sample data
for demonstrating the LlamaIndex and workflows-py integration.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any

# Forge1 imports
from forge1.core.tenancy import set_current_tenant
from forge1.core.memory_manager import MemoryManager, MemoryContext, MemoryType, SecurityLevel
from forge1.services.employee_manager import EmployeeManager
from forge1.models.employee_models import Employee

logger = logging.getLogger(__name__)

class DemoEnvironmentSetup:
    """Setup demo environment for integration testing"""
    
    def __init__(self, memory_manager: MemoryManager, employee_manager: EmployeeManager):
        self.memory_manager = memory_manager
        self.employee_manager = employee_manager
        self.tenant_id = "hartwell_associates"
        
    async def setup_complete_demo(self) -> Dict[str, Any]:
        """Setup complete demo environment"""
        
        logger.info("Setting up Hartwell & Associates demo environment...")
        
        # Set tenant context
        set_current_tenant(self.tenant_id)
        
        # Setup employees
        employees = await self._setup_employees()
        
        # Setup sample documents
        documents = await self._setup_sample_documents()
        
        # Setup knowledge base
        kb_entries = await self._setup_knowledge_base()
        
        # Setup Slack channels (mock)
        slack_config = await self._setup_slack_channels()
        
        # Setup Google Drive folders (mock)
        drive_config = await self._setup_drive_folders()
        
        demo_config = {
            "tenant_id": self.tenant_id,
            "tenant_name": "Hartwell & Associates",
            "employees": employees,
            "sample_documents": documents,
            "knowledge_base_entries": len(kb_entries),
            "slack_channels": slack_config,
            "drive_folders": drive_config,
            "setup_completed_at": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Demo environment setup completed: {json.dumps(demo_config, indent=2)}")
        return demo_config
    
    async def _setup_employees(self) -> List[Dict[str, Any]]:
        """Setup four employees with different roles"""
        
        employees_data = [
            {
                "employee_id": "intake_001",
                "name": "Sarah Chen",
                "role": "intake",
                "email": "sarah.chen@hartwell.law",
                "permissions": ["knowledge_base:search", "drive:read"],
                "model_preference": "gpt-4o",
                "department": "Client Services"
            },
            {
                "employee_id": "lawyer_001", 
                "name": "Michael Rodriguez",
                "role": "lawyer",
                "email": "michael.rodriguez@hartwell.law",
                "permissions": ["document:read", "knowledge_base:search", "drive:read", "slack:post"],
                "model_preference": "claude-3-opus",
                "department": "Legal"
            },
            {
                "employee_id": "research_001",
                "name": "Dr. Emily Watson",
                "role": "researcher",
                "email": "emily.watson@hartwell.law", 
                "permissions": ["knowledge_base:search", "drive:read", "document:read"],
                "model_preference": "gpt-4o",
                "department": "Legal Research"
            },
            {
                "employee_id": "hr_001",
                "name": "James Thompson",
                "role": "hr",
                "email": "james.thompson@hartwell.law",
                "permissions": ["knowledge_base:search", "slack:post"],
                "model_preference": "gemini-pro",
                "department": "Human Resources"
            }
        ]
        
        created_employees = []
        
        for emp_data in employees_data:
            try:
                # Create employee in system
                employee = Employee(
                    id=emp_data["employee_id"],
                    client_id=self.tenant_id,
                    name=emp_data["name"],
                    role=emp_data["role"],
                    email=emp_data["email"],
                    department=emp_data["department"],
                    preferences={
                        "model_preference": emp_data["model_preference"],
                        "permissions": emp_data["permissions"]
                    }
                )
                
                # Register with employee manager
                await self.employee_manager.create_employee(employee)
                
                created_employees.append({
                    "employee_id": emp_data["employee_id"],
                    "name": emp_data["name"],
                    "role": emp_data["role"],
                    "email": emp_data["email"],
                    "department": emp_data["department"],
                    "permissions": emp_data["permissions"],
                    "model_preference": emp_data["model_preference"]
                })
                
                logger.info(f"Created employee: {emp_data['name']} ({emp_data['role']})")
                
            except Exception as e:
                logger.error(f"Failed to create employee {emp_data['name']}: {e}")
        
        return created_employees
    
    async def _setup_sample_documents(self) -> List[Dict[str, Any]]:
        """Setup sample NDA documents for testing"""
        
        sample_docs = [
            {
                "file_id": "nda_tech_startup_001",
                "file_name": "TechCorp_NDA_2024.pdf",
                "file_path": "/legal/ndas/pending/TechCorp_NDA_2024.pdf",
                "description": "Standard NDA from TechCorp for software development partnership",
                "case_id": "case_nda_001",
                "priority": "high",
                "assigned_to": "lawyer_001"
            },
            {
                "file_id": "nda_consulting_002", 
                "file_name": "ConsultingFirm_NDA_Draft.docx",
                "file_path": "/legal/ndas/pending/ConsultingFirm_NDA_Draft.docx",
                "description": "Mutual NDA for consulting engagement with financial services firm",
                "case_id": "case_nda_002",
                "priority": "medium",
                "assigned_to": "lawyer_001"
            },
            {
                "file_id": "nda_research_003",
                "file_name": "University_Research_NDA.pdf", 
                "file_path": "/legal/ndas/pending/University_Research_NDA.pdf",
                "description": "Research collaboration NDA with State University",
                "case_id": "case_nda_003",
                "priority": "low",
                "assigned_to": "research_001"
            }
        ]
        
        # Store document metadata in memory system
        for doc in sample_docs:
            try:
                await self.memory_manager.store_memory(
                    content={
                        "document_type": "nda",
                        "file_metadata": doc,
                        "status": "pending_review",
                        "created_at": datetime.now(timezone.utc).isoformat()
                    },
                    memory_type=MemoryType.DOCUMENT,
                    metadata={
                        "document_id": doc["file_id"],
                        "case_id": doc["case_id"],
                        "assigned_to": doc["assigned_to"],
                        "priority": doc["priority"],
                        "tenant_id": self.tenant_id
                    }
                )
                
                logger.info(f"Registered sample document: {doc['file_name']}")
                
            except Exception as e:
                logger.error(f"Failed to register document {doc['file_name']}: {e}")
        
        return sample_docs
    
    async def _setup_knowledge_base(self) -> List[str]:
        """Setup legal knowledge base entries"""
        
        kb_entries = [
            {
                "title": "Standard NDA Clauses and Best Practices",
                "content": """
                Non-Disclosure Agreements (NDAs) should include the following key elements:
                
                1. Definition of Confidential Information
                - Clearly define what constitutes confidential information
                - Include both written and oral disclosures
                - Specify exclusions (publicly available information, independently developed, etc.)
                
                2. Obligations and Restrictions
                - Duty to maintain confidentiality
                - Limitations on use of confidential information
                - Return or destruction of materials upon termination
                
                3. Term and Termination
                - Specify duration of confidentiality obligations
                - Conditions for early termination
                - Survival clauses for ongoing obligations
                
                4. Remedies and Enforcement
                - Monetary damages may be inadequate
                - Right to seek injunctive relief
                - Attorney fees and costs provisions
                """,
                "category": "legal_guidelines",
                "tags": ["nda", "confidentiality", "best_practices"]
            },
            {
                "title": "Common NDA Pitfalls and Red Flags",
                "content": """
                When reviewing NDAs, watch for these common issues:
                
                1. Overly Broad Definitions
                - Confidential information defined too broadly
                - Lack of reasonable exclusions
                - Indefinite time periods
                
                2. Unbalanced Obligations
                - One-sided confidentiality requirements
                - Disproportionate remedies
                - Unfair termination clauses
                
                3. Jurisdictional Issues
                - Inappropriate governing law
                - Inconvenient forum selection
                - Conflicting state law requirements
                
                4. Practical Concerns
                - Unrealistic compliance requirements
                - Insufficient carve-outs for business operations
                - Lack of clarity on permitted disclosures
                """,
                "category": "risk_assessment",
                "tags": ["nda", "risk", "red_flags", "review"]
            },
            {
                "title": "Technology Sector NDA Considerations",
                "content": """
                Special considerations for technology sector NDAs:
                
                1. Intellectual Property Protection
                - Software code and algorithms
                - Technical specifications and designs
                - Development methodologies and processes
                
                2. Data Security Requirements
                - Cybersecurity obligations
                - Data breach notification procedures
                - Compliance with privacy regulations (GDPR, CCPA)
                
                3. Open Source Considerations
                - Treatment of open source components
                - Contribution back requirements
                - License compatibility issues
                
                4. Rapid Development Cycles
                - Shorter confidentiality periods may be appropriate
                - Agile development disclosure needs
                - Continuous integration/deployment considerations
                """,
                "category": "industry_specific",
                "tags": ["nda", "technology", "software", "ip"]
            },
            {
                "title": "Financial Services NDA Requirements",
                "content": """
                Financial services NDAs must address regulatory compliance:
                
                1. Regulatory Disclosure Requirements
                - SEC reporting obligations
                - Banking regulatory requirements
                - FINRA compliance considerations
                
                2. Customer Information Protection
                - GLBA privacy requirements
                - PCI DSS compliance for payment data
                - Know Your Customer (KYC) information
                
                3. Market Sensitive Information
                - Material non-public information handling
                - Trading restrictions and blackout periods
                - Insider trading prevention measures
                
                4. Cross-Border Considerations
                - International banking regulations
                - Currency and trade restrictions
                - Data localization requirements
                """,
                "category": "industry_specific", 
                "tags": ["nda", "financial", "regulatory", "compliance"]
            },
            {
                "title": "NDA Negotiation Strategies",
                "content": """
                Effective strategies for NDA negotiations:
                
                1. Preparation Phase
                - Understand client's business objectives
                - Identify key confidential information
                - Assess counterparty's legitimate needs
                
                2. Key Negotiation Points
                - Scope of confidential information
                - Duration of obligations
                - Permitted uses and disclosures
                - Remedies and enforcement mechanisms
                
                3. Common Compromises
                - Mutual vs. one-way confidentiality
                - Carve-outs for regulatory compliance
                - Limited disclosure to advisors and employees
                
                4. Deal Breakers
                - Unreasonable time periods (>5 years)
                - Overly broad injunctive relief
                - Excessive damages or penalties
                - Inappropriate governing law
                """,
                "category": "negotiation",
                "tags": ["nda", "negotiation", "strategy", "compromise"]
            }
        ]
        
        stored_entries = []
        
        for entry in kb_entries:
            try:
                memory_id = await self.memory_manager.store_memory(
                    content={
                        "title": entry["title"],
                        "content": entry["content"],
                        "category": entry["category"],
                        "tags": entry["tags"],
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "source": "hartwell_legal_kb"
                    },
                    memory_type=MemoryType.KNOWLEDGE,
                    metadata={
                        "kb_category": entry["category"],
                        "kb_tags": entry["tags"],
                        "tenant_id": self.tenant_id,
                        "access_level": "internal"
                    }
                )
                
                stored_entries.append(memory_id)
                logger.info(f"Added KB entry: {entry['title']}")
                
            except Exception as e:
                logger.error(f"Failed to add KB entry {entry['title']}: {e}")
        
        return stored_entries
    
    async def _setup_slack_channels(self) -> Dict[str, Any]:
        """Setup Slack channel configuration"""
        
        slack_config = {
            "workspace": "hartwell-associates",
            "channels": {
                "#legal-reviews": {
                    "purpose": "Legal document reviews and approvals",
                    "members": ["lawyer_001", "research_001"],
                    "private": False
                },
                "#nda-alerts": {
                    "purpose": "NDA workflow notifications",
                    "members": ["lawyer_001", "intake_001"],
                    "private": False
                },
                "#case-updates": {
                    "purpose": "Case status updates and notifications", 
                    "members": ["lawyer_001", "research_001", "intake_001"],
                    "private": False
                },
                "#hr-notifications": {
                    "purpose": "HR announcements and updates",
                    "members": ["hr_001", "lawyer_001"],
                    "private": True
                }
            },
            "bot_token": "xoxb-mock-token-for-demo",
            "webhook_url": "https://hooks.slack.com/services/mock/webhook/url"
        }
        
        # Store Slack configuration
        try:
            await self.memory_manager.store_memory(
                content=slack_config,
                memory_type=MemoryType.CONFIGURATION,
                metadata={
                    "config_type": "slack_integration",
                    "tenant_id": self.tenant_id,
                    "service": "slack"
                }
            )
            
            logger.info("Configured Slack integration")
            
        except Exception as e:
            logger.error(f"Failed to configure Slack: {e}")
        
        return slack_config
    
    async def _setup_drive_folders(self) -> Dict[str, Any]:
        """Setup Google Drive folder structure"""
        
        drive_config = {
            "root_folder": "Hartwell & Associates Legal Files",
            "folder_structure": {
                "/legal": {
                    "folder_id": "1BxYz_mock_legal_folder_id",
                    "subfolders": {
                        "/ndas": {
                            "folder_id": "1CxYz_mock_ndas_folder_id",
                            "subfolders": {
                                "/pending": {"folder_id": "1DxYz_mock_pending_folder_id"},
                                "/approved": {"folder_id": "1ExYz_mock_approved_folder_id"},
                                "/templates": {"folder_id": "1FxYz_mock_templates_folder_id"}
                            }
                        },
                        "/contracts": {
                            "folder_id": "1GxYz_mock_contracts_folder_id"
                        },
                        "/research": {
                            "folder_id": "1HxYz_mock_research_folder_id"
                        }
                    }
                },
                "/hr": {
                    "folder_id": "1IxYz_mock_hr_folder_id",
                    "subfolders": {
                        "/policies": {"folder_id": "1JxYz_mock_policies_folder_id"},
                        "/employee_docs": {"folder_id": "1KxYz_mock_employee_folder_id"}
                    }
                }
            },
            "service_account_key": "mock_service_account_key.json",
            "permissions": {
                "lawyer_001": ["read", "write"],
                "research_001": ["read"],
                "intake_001": ["read"],
                "hr_001": ["read", "write"]
            }
        }
        
        # Store Drive configuration
        try:
            await self.memory_manager.store_memory(
                content=drive_config,
                memory_type=MemoryType.CONFIGURATION,
                metadata={
                    "config_type": "google_drive_integration",
                    "tenant_id": self.tenant_id,
                    "service": "google_drive"
                }
            )
            
            logger.info("Configured Google Drive integration")
            
        except Exception as e:
            logger.error(f"Failed to configure Google Drive: {e}")
        
        return drive_config

async def main():
    """Main setup function"""
    
    # This would normally initialize real Forge1 components
    # For demo purposes, we'll use mocks
    
    from unittest.mock import AsyncMock
    
    memory_manager = AsyncMock()
    employee_manager = AsyncMock()
    
    # Mock successful operations
    memory_manager.store_memory.return_value = f"memory_{datetime.now().timestamp()}"
    employee_manager.create_employee.return_value = True
    
    setup = DemoEnvironmentSetup(memory_manager, employee_manager)
    demo_config = await setup.setup_complete_demo()
    
    print("Demo environment setup completed!")
    print(json.dumps(demo_config, indent=2))

if __name__ == "__main__":
    asyncio.run(main())