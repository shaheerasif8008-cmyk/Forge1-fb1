#!/usr/bin/env python3
"""
NDA Workflow Demonstration Script

Executes the complete NDA workflow: 
Intake → MCAE handoff → Lawyer stage (workflows-py subgraph) → Research check → final Slack message

Captures logs/spans and generates observability artifacts for demonstration.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import AsyncMock, Mock

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

class NDAWorkflowDemo:
    """Complete NDA workflow demonstration"""
    
    def __init__(self):
        self.demo_start_time = datetime.now(timezone.utc)
        self.execution_log = []
        self.spans = []
        self.usage_metrics = []
        
    async def run_complete_demo(self) -> Dict[str, Any]:
        """Run the complete NDA workflow demonstration"""
        
        logger.info("🚀 Starting NDA Workflow Demonstration")
        logger.info("=" * 60)
        
        try:
            # Step 1: Setup demo environment
            demo_config = await self._setup_demo_environment()
            
            # Step 2: Intake stage
            intake_result = await self._simulate_intake_stage()
            
            # Step 3: MCAE handoff to Lawyer stage
            mcae_result = await self._simulate_mcae_handoff(intake_result)
            
            # Step 4: Lawyer stage with workflows-py subgraph
            lawyer_result = await self._execute_lawyer_workflow(mcae_result)
            
            # Step 5: Research check stage
            research_result = await self._simulate_research_check(lawyer_result)
            
            # Step 6: Final Slack notification
            final_result = await self._send_final_notification(research_result)
            
            # Generate demonstration artifacts
            artifacts = await self._generate_demo_artifacts()
            
            demo_summary = {
                "demo_id": f"nda_demo_{int(time.time())}",
                "tenant": "hartwell_associates",
                "workflow_type": "nda_review",
                "execution_time_seconds": (datetime.now(timezone.utc) - self.demo_start_time).total_seconds(),
                "stages_completed": [
                    "intake", "mcae_handoff", "lawyer_workflow", 
                    "research_check", "final_notification"
                ],
                "artifacts_generated": artifacts,
                "success": True
            }
            
            logger.info("✅ NDA Workflow Demonstration completed successfully!")
            logger.info(f"📊 Demo Summary: {json.dumps(demo_summary, indent=2)}")
            
            return demo_summary
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _setup_demo_environment(self) -> Dict[str, Any]:
        """Setup the demo environment"""
        
        self._log_step("setup", "Initializing Hartwell & Associates demo environment")
        
        # Mock Forge1 components
        demo_config = {
            "tenant_id": "hartwell_associates",
            "tenant_name": "Hartwell & Associates",
            "employees": {
                "intake_001": {
                    "name": "Sarah Chen",
                    "role": "intake",
                    "permissions": ["knowledge_base:search", "drive:read"]
                },
                "lawyer_001": {
                    "name": "Michael Rodriguez", 
                    "role": "lawyer",
                    "permissions": ["document:read", "knowledge_base:search", "drive:read", "slack:post"]
                },
                "research_001": {
                    "name": "Dr. Emily Watson",
                    "role": "researcher",
                    "permissions": ["knowledge_base:search", "drive:read", "document:read"]
                }
            },
            "case_id": "case_nda_001",
            "document_reference": {
                "file_id": "nda_tech_startup_001",
                "file_name": "TechCorp_NDA_2024.pdf",
                "file_path": "/legal/ndas/pending/TechCorp_NDA_2024.pdf"
            }
        }
        
        self._log_step("setup", f"✅ Environment configured for {demo_config['tenant_name']}")
        return demo_config
    
    async def _simulate_intake_stage(self) -> Dict[str, Any]:
        """Simulate the intake stage"""
        
        self._log_step("intake", "Sarah Chen (Intake) receives new NDA for review")
        
        # Simulate intake processing
        await asyncio.sleep(0.5)  # Simulate processing time
        
        intake_result = {
            "stage": "intake",
            "employee_id": "intake_001",
            "employee_name": "Sarah Chen",
            "case_id": "case_nda_001",
            "priority": "high",
            "client": "TechCorp Inc.",
            "document_type": "mutual_nda",
            "initial_assessment": "Standard technology partnership NDA requiring legal review",
            "assigned_to": "lawyer_001",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "next_stage": "legal_review"
        }
        
        self._create_span("intake_processing", "intake_001", 500, {"priority": "high"})
        self._log_step("intake", f"✅ Case {intake_result['case_id']} created and assigned to legal team")
        
        return intake_result
    
    async def _simulate_mcae_handoff(self, intake_result: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate MCAE orchestration handoff"""
        
        self._log_step("mcae", "MCAE orchestrator routing case to Lawyer stage")
        
        # Simulate MCAE processing
        await asyncio.sleep(0.3)
        
        mcae_result = {
            "stage": "mcae_handoff",
            "orchestrator": "mcae",
            "workflow_id": f"workflow_{intake_result['case_id']}",
            "source_stage": "intake",
            "target_stage": "lawyer_workflow",
            "context": {
                "tenant_id": "hartwell_associates",
                "employee_id": "lawyer_001",
                "case_id": intake_result["case_id"],
                "document_reference": {
                    "file_id": "nda_tech_startup_001",
                    "file_name": "TechCorp_NDA_2024.pdf"
                }
            },
            "handoff_time": datetime.now(timezone.utc).isoformat()
        }
        
        self._create_span("mcae_handoff", "system", 300, {"target_stage": "lawyer_workflow"})
        self._log_step("mcae", f"✅ Workflow {mcae_result['workflow_id']} routed to lawyer stage")
        
        return mcae_result
    
    async def _execute_lawyer_workflow(self, mcae_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the lawyer stage with workflows-py subgraph"""
        
        self._log_step("lawyer", "Michael Rodriguez (Lawyer) executing NDA review workflow")
        self._log_step("lawyer", "🔄 Starting workflows-py subgraph: drive_fetch → document_parser → kb_search → compose_draft → slack_post")
        
        context = mcae_result["context"]
        
        # Step 1: Drive Fetch
        drive_result = await self._simulate_drive_fetch(context)
        
        # Step 2: Document Parser
        parser_result = await self._simulate_document_parser(drive_result, context)
        
        # Step 3: KB Search
        kb_result = await self._simulate_kb_search(parser_result, context)
        
        # Step 4: Compose Draft
        draft_result = await self._simulate_compose_draft(parser_result, kb_result, context)
        
        # Step 5: Slack Post
        slack_result = await self._simulate_slack_post(draft_result, context)
        
        lawyer_result = {
            "stage": "lawyer_workflow",
            "employee_id": "lawyer_001",
            "employee_name": "Michael Rodriguez",
            "workflow_steps": {
                "drive_fetch": drive_result,
                "document_parser": parser_result,
                "kb_search": kb_result,
                "compose_draft": draft_result,
                "slack_post": slack_result
            },
            "final_output": {
                "review_completed": True,
                "recommendations": [
                    "Modify confidentiality definition in Section 2.1",
                    "Add carve-out for regulatory disclosures",
                    "Reduce term from 5 years to 3 years"
                ],
                "risk_level": "medium",
                "approval_status": "conditional"
            },
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        
        self._log_step("lawyer", "✅ NDA review workflow completed with conditional approval")
        return lawyer_result
    
    async def _simulate_drive_fetch(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate Google Drive document fetch"""
        
        self._log_step("lawyer.drive_fetch", "📁 Fetching NDA document from Google Drive")
        await asyncio.sleep(0.8)  # Simulate network call
        
        result = {
            "step": "drive_fetch",
            "success": True,
            "file_data": {
                "file_id": context["document_reference"]["file_id"],
                "name": context["document_reference"]["file_name"],
                "mime_type": "application/pdf",
                "size": "245KB",
                "content": "base64_encoded_pdf_content_mock"
            },
            "execution_time_ms": 800
        }
        
        self._create_span("drive_fetch", context["employee_id"], 800, {
            "file_name": result["file_data"]["name"],
            "file_size": result["file_data"]["size"]
        })
        
        self._track_usage("tool:drive_fetch", context["employee_id"], 0, 0, 800)
        self._log_step("lawyer.drive_fetch", f"✅ Retrieved {result['file_data']['name']} ({result['file_data']['size']})")
        
        return result
    
    async def _simulate_document_parser(self, drive_result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate document parsing with OCR"""
        
        self._log_step("lawyer.document_parser", "📄 Parsing PDF document and extracting text")
        await asyncio.sleep(1.2)  # Simulate parsing time
        
        result = {
            "step": "document_parser",
            "success": True,
            "parsed_content": {
                "text": "MUTUAL NON-DISCLOSURE AGREEMENT\n\nThis Mutual Non-Disclosure Agreement...",
                "text_length": 4250,
                "node_count": 12,
                "extraction_method": "standard_pdf",
                "key_sections": [
                    "Definition of Confidential Information",
                    "Obligations and Restrictions", 
                    "Term and Termination",
                    "Remedies and Enforcement"
                ]
            },
            "execution_time_ms": 1200
        }
        
        self._create_span("document_parser", context["employee_id"], 1200, {
            "text_length": result["parsed_content"]["text_length"],
            "node_count": result["parsed_content"]["node_count"],
            "extraction_method": result["parsed_content"]["extraction_method"]
        })
        
        self._track_usage("tool:document_parser", context["employee_id"], 0, 0, 1200)
        self._log_step("lawyer.document_parser", f"✅ Extracted {result['parsed_content']['text_length']} characters in {result['parsed_content']['node_count']} sections")
        
        return result
    
    async def _simulate_kb_search(self, parser_result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate knowledge base search"""
        
        self._log_step("lawyer.kb_search", "🔍 Searching legal knowledge base for relevant precedents")
        await asyncio.sleep(0.9)  # Simulate search time
        
        result = {
            "step": "kb_search",
            "success": True,
            "query": "NDA confidentiality mutual technology partnership",
            "results": [
                {
                    "memory_id": "kb_001",
                    "title": "Standard NDA Clauses and Best Practices",
                    "similarity_score": 0.89,
                    "summary": "Comprehensive guide to NDA structure and key provisions"
                },
                {
                    "memory_id": "kb_002", 
                    "title": "Technology Sector NDA Considerations",
                    "similarity_score": 0.85,
                    "summary": "Special considerations for tech industry NDAs including IP protection"
                },
                {
                    "memory_id": "kb_003",
                    "title": "Common NDA Pitfalls and Red Flags",
                    "similarity_score": 0.78,
                    "summary": "Common issues to watch for in NDA reviews"
                }
            ],
            "total_results": 3,
            "execution_time_ms": 900
        }
        
        self._create_span("kb_search", context["employee_id"], 900, {
            "query": result["query"],
            "results_count": len(result["results"])
        })
        
        self._track_usage("tool:kb_search", context["employee_id"], 0, 0, 900)
        self._log_step("lawyer.kb_search", f"✅ Found {len(result['results'])} relevant knowledge base entries")
        
        return result
    
    async def _simulate_compose_draft(self, parser_result: Dict[str, Any], kb_result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate LLM-powered draft composition"""
        
        self._log_step("lawyer.compose_draft", "✍️ Generating legal review draft using Claude-3-Opus")
        await asyncio.sleep(2.1)  # Simulate LLM processing
        
        draft_text = """
        LEGAL REVIEW: TechCorp NDA Analysis
        
        EXECUTIVE SUMMARY:
        This mutual NDA establishes confidentiality obligations for a technology partnership between TechCorp Inc. and Hartwell & Associates. The agreement is generally well-structured but requires several modifications to better protect our client's interests.
        
        KEY FINDINGS:
        1. Confidentiality Definition (Section 2.1): Overly broad - recommend narrowing scope
        2. Term Duration: 5-year term is excessive for this type of partnership
        3. Remedies: Adequate injunctive relief provisions
        4. Governing Law: Delaware law is acceptable
        
        RECOMMENDATIONS:
        - Modify confidentiality definition to exclude publicly available information
        - Reduce term from 5 years to 3 years
        - Add regulatory disclosure carve-out
        - Include return/destruction of materials clause
        
        RISK ASSESSMENT: MEDIUM
        With recommended modifications, this NDA is acceptable for execution.
        """
        
        result = {
            "step": "compose_draft",
            "success": True,
            "draft_text": draft_text.strip(),
            "word_count": len(draft_text.split()),
            "model_used": "claude-3-opus",
            "execution_time_ms": 2100
        }
        
        self._create_span("compose_draft", context["employee_id"], 2100, {
            "model": "claude-3-opus",
            "word_count": result["word_count"]
        })
        
        # Track LLM usage
        self._track_usage("claude-3-opus", context["employee_id"], 850, 320, 2100, 0.025)
        self._log_step("lawyer.compose_draft", f"✅ Generated {result['word_count']}-word legal review using {result['model_used']}")
        
        return result
    
    async def _simulate_slack_post(self, draft_result: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate Slack notification"""
        
        self._log_step("lawyer.slack_post", "💬 Posting review to #legal-reviews Slack channel")
        await asyncio.sleep(0.4)  # Simulate API call
        
        slack_message = f"""
        🔍 **NDA Review Complete** - Case ID: {context['case_id']}
        
        Document: TechCorp_NDA_2024.pdf
        Reviewed by: Michael Rodriguez (lawyer_001)
        
        **Legal Analysis:**
        {draft_result['draft_text'][:500]}...
        
        📎 Citations: 3 relevant precedents found
        ⏱️ Completed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
        
        *This is an automated legal review. Please have a qualified attorney review before making final decisions.*
        """
        
        result = {
            "step": "slack_post",
            "success": True,
            "channel": "#legal-reviews",
            "message_ts": f"{int(time.time())}.123456",
            "message_length": len(slack_message),
            "execution_time_ms": 400
        }
        
        self._create_span("slack_post", context["employee_id"], 400, {
            "channel": result["channel"],
            "message_length": result["message_length"]
        })
        
        self._track_usage("tool:slack_post", context["employee_id"], 0, 0, 400)
        self._log_step("lawyer.slack_post", f"✅ Posted review to {result['channel']} (message length: {result['message_length']} chars)")
        
        return result
    
    async def _simulate_research_check(self, lawyer_result: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate research team validation"""
        
        self._log_step("research", "Dr. Emily Watson (Research) validating legal analysis")
        await asyncio.sleep(1.0)  # Simulate research time
        
        research_result = {
            "stage": "research_check",
            "employee_id": "research_001",
            "employee_name": "Dr. Emily Watson",
            "validation_results": {
                "precedent_accuracy": "verified",
                "legal_citations": "accurate",
                "risk_assessment": "confirmed",
                "recommendations": "sound"
            },
            "additional_notes": "Analysis aligns with current industry standards. Recommend proceeding with suggested modifications.",
            "approval": "approved",
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        
        self._create_span("research_validation", "research_001", 1000, {
            "validation_status": "approved"
        })
        
        self._log_step("research", "✅ Research validation completed - analysis approved")
        return research_result
    
    async def _send_final_notification(self, research_result: Dict[str, Any]) -> Dict[str, Any]:
        """Send final workflow completion notification"""
        
        self._log_step("final", "Sending final workflow completion notification")
        await asyncio.sleep(0.3)
        
        final_result = {
            "stage": "final_notification",
            "workflow_status": "completed",
            "final_decision": "conditional_approval",
            "next_steps": [
                "Incorporate recommended modifications",
                "Schedule client review meeting",
                "Prepare final execution version"
            ],
            "notification_sent": True,
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        
        self._create_span("final_notification", "system", 300, {
            "workflow_status": "completed",
            "decision": "conditional_approval"
        })
        
        self._log_step("final", "✅ NDA workflow completed successfully with conditional approval")
        return final_result
    
    def _log_step(self, stage: str, message: str):
        """Log a workflow step"""
        timestamp = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "timestamp": timestamp,
            "stage": stage,
            "message": message,
            "tenant_id": "hartwell_associates"
        }
        self.execution_log.append(log_entry)
        logger.info(f"[{stage.upper()}] {message}")
    
    def _create_span(self, operation: str, employee_id: str, duration_ms: float, attributes: Dict[str, Any]):
        """Create a mock OpenTelemetry span"""
        span = {
            "trace_id": f"trace_{int(time.time() * 1000000)}",
            "span_id": f"span_{int(time.time() * 1000000)}",
            "operation_name": operation,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "duration_ms": duration_ms,
            "attributes": {
                "tenant.id": "hartwell_associates",
                "employee.id": employee_id,
                "service.name": "forge1-workflows",
                **attributes
            },
            "status": "OK"
        }
        self.spans.append(span)
    
    def _track_usage(self, model: str, employee_id: str, tokens_in: int, tokens_out: int, latency_ms: float, cost: float = 0.0):
        """Track usage metrics"""
        usage = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tenant_id": "hartwell_associates",
            "employee_id": employee_id,
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
            "cost_estimate": cost,
            "tool_calls": "[]"
        }
        self.usage_metrics.append(usage)
    
    async def _generate_demo_artifacts(self) -> Dict[str, Any]:
        """Generate demonstration artifacts"""
        
        # Generate usage CSV
        csv_content = "timestamp,tenant_id,employee_id,model,tokens_in,tokens_out,latency_ms,cost_estimate,tool_calls\n"
        for usage in self.usage_metrics:
            csv_content += f"{usage['timestamp']},{usage['tenant_id']},{usage['employee_id']},{usage['model']},{usage['tokens_in']},{usage['tokens_out']},{usage['latency_ms']},{usage['cost_estimate']},{usage['tool_calls']}\n"
        
        # Generate execution log
        execution_log_json = json.dumps(self.execution_log, indent=2)
        
        # Generate spans JSON
        spans_json = json.dumps(self.spans, indent=2)
        
        artifacts = {
            "usage_csv": {
                "filename": f"nda_demo_usage_{int(time.time())}.csv",
                "content": csv_content,
                "records": len(self.usage_metrics)
            },
            "execution_log": {
                "filename": f"nda_demo_execution_{int(time.time())}.json",
                "content": execution_log_json,
                "entries": len(self.execution_log)
            },
            "otel_spans": {
                "filename": f"nda_demo_spans_{int(time.time())}.json", 
                "content": spans_json,
                "spans": len(self.spans)
            }
        }
        
        # Write artifacts to files
        for artifact_type, artifact_data in artifacts.items():
            try:
                with open(f"/tmp/{artifact_data['filename']}", 'w') as f:
                    f.write(artifact_data['content'])
                logger.info(f"📄 Generated {artifact_type}: {artifact_data['filename']}")
            except Exception as e:
                logger.warning(f"Failed to write {artifact_type}: {e}")
        
        return artifacts

async def main():
    """Run the NDA workflow demonstration"""
    
    print("🏛️  Hartwell & Associates - NDA Workflow Demonstration")
    print("=" * 60)
    print("This demo shows the complete integration of LlamaIndex and workflows-py")
    print("within the Forge1 platform for automated legal document review.")
    print()
    
    demo = NDAWorkflowDemo()
    result = await demo.run_complete_demo()
    
    if result.get("success"):
        print("\n🎉 Demonstration completed successfully!")
        print(f"⏱️  Total execution time: {result['execution_time_seconds']:.2f} seconds")
        print(f"📊 Artifacts generated: {len(result['artifacts_generated'])} files")
        print("\nArtifacts available in /tmp/:")
        for artifact_type, artifact_data in result['artifacts_generated'].items():
            print(f"  - {artifact_data['filename']} ({artifact_data.get('records', artifact_data.get('entries', artifact_data.get('spans')))} items)")
    else:
        print(f"\n❌ Demonstration failed: {result.get('error')}")

if __name__ == "__main__":
    asyncio.run(main())