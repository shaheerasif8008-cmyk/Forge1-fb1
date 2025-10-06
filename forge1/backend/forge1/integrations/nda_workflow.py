"""
NDA Workflow Implementation using workflows-py

Implements the complete NDA processing workflow as a sub-orchestration within
MCAE stages. The workflow follows the pattern:
drive_fetch → document_parser → kb_search → compose_draft → slack_post

Each step is idempotent, supports retries, and maintains proper handoff packets.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import json

# workflows-py imports (with fallback)
try:
    from workflows import workflow, step
    WORKFLOWS_AVAILABLE = True
except ImportError:
    WORKFLOWS_AVAILABLE = False
    def workflow(cls):
        return cls
    def step(func):
        return func

# Forge1 imports
from forge1.integrations.workflows_adapter import (
    Forge1Workflow, Forge1WorkflowContext, HandoffPacket, StepResult, StepStatus
)
from forge1.integrations.llamaindex_adapter import ExecutionContext, ToolType
from forge1.integrations.llamaindex_tools import (
    DocumentParserTool, KBSearchTool, DriveFetchTool, SlackPostTool
)
from forge1.integrations.llamaindex_model_shim import ModelShimFactory

logger = logging.getLogger(__name__)

@workflow
class NDAWorkflow(Forge1Workflow):
    """
    NDA processing workflow that handles legal document analysis and response generation.
    
    Workflow Steps:
    1. drive_fetch: Retrieve NDA document from Google Drive
    2. document_parser: Parse and extract text from the document
    3. kb_search: Search knowledge base for relevant legal precedents
    4. compose_draft: Generate draft response using LLM
    5. slack_post: Post the draft to appropriate Slack channel
    """
    
    def __init__(self, workflow_id: str, context: Forge1WorkflowContext, **kwargs):
        super().__init__(workflow_id, context, **kwargs)
        
        # Initialize tools
        self.drive_tool = None
        self.parser_tool = None
        self.kb_tool = None
        self.slack_tool = None
        
        # Workflow configuration
        self.config = kwargs.get('config', {})
        self.target_channel = self.config.get('slack_channel', '#legal-reviews')
        self.kb_search_limit = self.config.get('kb_search_limit', 5)
        
    async def _execute_steps(self) -> bool:
        """Execute the complete NDA workflow"""
        try:
            # Initialize tools
            await self._initialize_tools()
            
            # Step 1: Fetch document from Google Drive
            drive_result = await self._execute_step_with_retry(
                "drive_fetch",
                self._drive_fetch_step
            )
            
            if drive_result.status != StepStatus.COMPLETED:
                logger.error(f"Drive fetch failed: {drive_result.error}")
                return False
            
            self.context.add_step_result(drive_result)
            
            # Step 2: Parse the document
            parser_result = await self._execute_step_with_retry(
                "document_parser",
                self._document_parser_step,
                drive_result.output
            )
            
            if parser_result.status != StepStatus.COMPLETED:
                logger.error(f"Document parsing failed: {parser_result.error}")
                return False
            
            self.context.add_step_result(parser_result)
            
            # Step 3: Search knowledge base
            kb_result = await self._execute_step_with_retry(
                "kb_search",
                self._kb_search_step,
                parser_result.output
            )
            
            if kb_result.status != StepStatus.COMPLETED:
                logger.error(f"KB search failed: {kb_result.error}")
                return False
            
            self.context.add_step_result(kb_result)
            
            # Step 4: Compose draft response
            compose_result = await self._execute_step_with_retry(
                "compose_draft",
                self._compose_draft_step,
                parser_result.output,
                kb_result.output
            )
            
            if compose_result.status != StepStatus.COMPLETED:
                logger.error(f"Draft composition failed: {compose_result.error}")
                return False
            
            self.context.add_step_result(compose_result)
            
            # Step 5: Post to Slack
            slack_result = await self._execute_step_with_retry(
                "slack_post",
                self._slack_post_step,
                compose_result.output
            )
            
            if slack_result.status != StepStatus.COMPLETED:
                logger.error(f"Slack post failed: {slack_result.error}")
                return False
            
            self.context.add_step_result(slack_result)
            
            # Update final handoff packet
            final_packet = self.context.get_handoff_packet()
            final_packet.summary = "NDA workflow completed successfully"
            final_packet.step_outputs = {
                "drive_fetch": drive_result.output,
                "document_parser": parser_result.output,
                "kb_search": kb_result.output,
                "compose_draft": compose_result.output,
                "slack_post": slack_result.output
            }
            
            return True
            
        except Exception as e:
            logger.error(f"NDA workflow execution failed: {e}")
            return False
    
    async def _initialize_tools(self) -> None:
        """Initialize all required tools"""
        try:
            # Get LlamaIndex adapter from context
            llamaindex_adapter = self.context.llamaindex_adapter
            
            # Create tool instances
            self.drive_tool = DriveFetchTool(
                tool_type=ToolType.DRIVE_FETCH,
                security_manager=llamaindex_adapter.security_manager,
                secret_manager=llamaindex_adapter.secret_manager,
                audit_logger=llamaindex_adapter.audit_logger,
                model_router=self.context.model_router,
                memory_manager=self.context.memory_manager
            )
            
            self.parser_tool = DocumentParserTool(
                tool_type=ToolType.DOCUMENT_PARSER,
                security_manager=llamaindex_adapter.security_manager,
                secret_manager=llamaindex_adapter.secret_manager,
                audit_logger=llamaindex_adapter.audit_logger,
                model_router=self.context.model_router,
                memory_manager=self.context.memory_manager
            )
            
            self.kb_tool = KBSearchTool(
                tool_type=ToolType.KB_SEARCH,
                security_manager=llamaindex_adapter.security_manager,
                secret_manager=llamaindex_adapter.secret_manager,
                audit_logger=llamaindex_adapter.audit_logger,
                model_router=self.context.model_router,
                memory_manager=self.context.memory_manager
            )
            
            self.slack_tool = SlackPostTool(
                tool_type=ToolType.SLACK_POST,
                security_manager=llamaindex_adapter.security_manager,
                secret_manager=llamaindex_adapter.secret_manager,
                audit_logger=llamaindex_adapter.audit_logger,
                model_router=self.context.model_router,
                memory_manager=self.context.memory_manager
            )
            
            logger.info("NDA workflow tools initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize workflow tools: {e}")
            raise
    
    @step
    async def _drive_fetch_step(self) -> Dict[str, Any]:
        """Step 1: Fetch NDA document from Google Drive"""
        try:
            # Get document reference from handoff packet
            packet = self.context.get_handoff_packet()
            document_ref = packet.metadata.get('document_reference')
            
            if not document_ref:
                raise ValueError("No document reference provided in workflow input")
            
            # Fetch document
            result = await self.drive_tool.acall(
                context=self.context.execution_context,
                file_id=document_ref.get('file_id'),
                file_path=document_ref.get('file_path')
            )
            
            if not result.get('success'):
                raise ValueError(f"Drive fetch failed: {result.get('error')}")
            
            # Update handoff packet
            packet.attachment_refs.append({
                "type": "google_drive",
                "file_id": result['file_data']['file_id'],
                "file_name": result['file_data']['name'],
                "mime_type": result['file_data']['mime_type'],
                "size": result['file_data']['size']
            })
            
            logger.info(f"Successfully fetched document: {result['file_data']['name']}")
            return result
            
        except Exception as e:
            logger.error(f"Drive fetch step failed: {e}")
            raise
    
    @step
    async def _document_parser_step(self, drive_output: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Parse the fetched document"""
        try:
            file_data = drive_output['file_data']
            
            # Parse document
            result = await self.parser_tool.acall(
                context=self.context.execution_context,
                document_content=file_data['content'],
                document_format=self._get_format_from_mime_type(file_data['mime_type']),
                use_ocr=True
            )
            
            if not result.get('success'):
                raise ValueError(f"Document parsing failed: {result.get('error')}")
            
            # Update handoff packet with parsed content summary
            packet = self.context.get_handoff_packet()
            packet.summary = f"Parsed NDA document: {file_data['name']}"
            packet.metadata['parsed_content'] = {
                "text_length": result['parsed_content']['text_length'],
                "node_count": result['parsed_content']['node_count'],
                "extraction_method": result['parsed_content']['extraction_method']
            }
            
            logger.info(f"Successfully parsed document with {result['parsed_content']['node_count']} nodes")
            return result
            
        except Exception as e:
            logger.error(f"Document parser step failed: {e}")
            raise
    
    @step
    async def _kb_search_step(self, parser_output: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Search knowledge base for relevant information"""
        try:
            parsed_content = parser_output['parsed_content']
            
            # Extract key terms for search (simplified approach)
            search_query = self._extract_search_terms(parsed_content['text'])
            
            # Search knowledge base
            result = await self.kb_tool.acall(
                context=self.context.execution_context,
                query=search_query,
                max_results=self.kb_search_limit,
                similarity_threshold=0.7
            )
            
            if not result.get('success'):
                raise ValueError(f"KB search failed: {result.get('error')}")
            
            # Update handoff packet with search results
            packet = self.context.get_handoff_packet()
            packet.citations = [
                {
                    "memory_id": res['memory_id'],
                    "summary": res['summary'],
                    "similarity_score": res['similarity_score'],
                    "memory_type": res['memory_type']
                }
                for res in result['results']
            ]
            
            logger.info(f"Found {len(result['results'])} relevant knowledge base entries")
            return result
            
        except Exception as e:
            logger.error(f"KB search step failed: {e}")
            raise
    
    @step
    async def _compose_draft_step(
        self, 
        parser_output: Dict[str, Any], 
        kb_output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Step 4: Compose draft response using LLM"""
        try:
            # Create model shim for LLM access
            model_shim_factory = ModelShimFactory(
                model_router=self.context.model_router,
                audit_logger=self.context.llamaindex_adapter.audit_logger
            )
            
            llm = model_shim_factory.create_llm_shim(
                tenant_id=self.context.execution_context.tenant_id,
                employee_id=self.context.execution_context.employee_id
            )
            
            # Prepare prompt
            prompt = self._create_draft_prompt(
                parser_output['parsed_content'],
                kb_output['results']
            )
            
            # Generate draft
            response = await llm.acomplete(prompt)
            draft_text = response.text
            
            # Update handoff packet
            packet = self.context.get_handoff_packet()
            packet.summary = "Generated NDA review draft"
            packet.metadata['draft_response'] = {
                "text": draft_text,
                "word_count": len(draft_text.split()),
                "generated_at": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"Successfully generated draft response ({len(draft_text.split())} words)")
            
            return {
                "success": True,
                "draft_text": draft_text,
                "word_count": len(draft_text.split())
            }
            
        except Exception as e:
            logger.error(f"Draft composition step failed: {e}")
            raise
    
    @step
    async def _slack_post_step(self, compose_output: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: Post draft to Slack channel"""
        try:
            draft_text = compose_output['draft_text']
            
            # Format message for Slack
            packet = self.context.get_handoff_packet()
            slack_message = self._format_slack_message(draft_text, packet)
            
            # Post to Slack
            result = await self.slack_tool.acall(
                context=self.context.execution_context,
                channel=self.target_channel,
                message=slack_message
            )
            
            if not result.get('success'):
                raise ValueError(f"Slack post failed: {result.get('error')}")
            
            # Update handoff packet
            packet.metadata['slack_post'] = {
                "channel": self.target_channel,
                "message_ts": result['message_ts'],
                "posted_at": result['post_time']
            }
            
            logger.info(f"Successfully posted to Slack channel: {self.target_channel}")
            return result
            
        except Exception as e:
            logger.error(f"Slack post step failed: {e}")
            raise
    
    def _get_format_from_mime_type(self, mime_type: str) -> str:
        """Convert MIME type to document format"""
        mime_map = {
            "application/pdf": "pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            "application/msword": "docx",
            "text/plain": "txt",
            "image/png": "image",
            "image/jpeg": "image"
        }
        return mime_map.get(mime_type, "auto")
    
    def _extract_search_terms(self, text: str) -> str:
        """Extract key search terms from document text"""
        # Simplified approach - look for legal keywords
        legal_keywords = [
            "confidentiality", "non-disclosure", "proprietary", "trade secret",
            "intellectual property", "liability", "indemnification", "term",
            "termination", "governing law", "jurisdiction", "damages"
        ]
        
        text_lower = text.lower()
        found_terms = [term for term in legal_keywords if term in text_lower]
        
        # Create search query
        if found_terms:
            return f"NDA legal review {' '.join(found_terms[:5])}"
        else:
            return "NDA non-disclosure agreement legal review"
    
    def _create_draft_prompt(
        self, 
        parsed_content: Dict[str, Any], 
        kb_results: List[Dict[str, Any]]
    ) -> str:
        """Create prompt for draft generation"""
        
        # Extract relevant text snippets
        document_text = parsed_content['text'][:2000]  # Limit for prompt size
        
        # Format knowledge base context
        kb_context = ""
        if kb_results:
            kb_context = "\n\nRelevant precedents and guidelines:\n"
            for i, result in enumerate(kb_results[:3], 1):
                kb_context += f"{i}. {result.get('summary', 'No summary available')}\n"
        
        prompt = f"""
You are a legal expert reviewing an NDA (Non-Disclosure Agreement). Please analyze the following document and provide a comprehensive review.

DOCUMENT CONTENT:
{document_text}

{kb_context}

Please provide a structured legal review that includes:

1. EXECUTIVE SUMMARY
   - Brief overview of the NDA's purpose and scope
   - Key strengths and concerns

2. DETAILED ANALYSIS
   - Definition of confidential information
   - Obligations and restrictions
   - Term and termination provisions
   - Liability and remedies
   - Governing law and jurisdiction

3. RECOMMENDATIONS
   - Suggested modifications or clarifications
   - Risk assessment
   - Negotiation points

4. CONCLUSION
   - Overall assessment
   - Recommendation to proceed, modify, or reject

Please ensure your review is thorough, professional, and actionable for business decision-making.
"""
        
        return prompt
    
    def _format_slack_message(self, draft_text: str, packet: HandoffPacket) -> str:
        """Format draft for Slack posting"""
        
        # Get document info
        doc_info = ""
        if packet.attachment_refs:
            doc_ref = packet.attachment_refs[0]
            doc_info = f"Document: {doc_ref.get('file_name', 'Unknown')}"
        
        # Format message
        message = f"""
🔍 **NDA Review Complete** - Case ID: {packet.case_id}

{doc_info}
Reviewed by: {packet.employee_id}

**Legal Analysis:**
{draft_text[:1500]}{'...' if len(draft_text) > 1500 else ''}

📎 Citations: {len(packet.citations)} relevant precedents found
⏱️ Completed: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

*This is an automated legal review. Please have a qualified attorney review before making final decisions.*
"""
        
        return message.strip()

# Workflow registration function
async def register_nda_workflow(workflows_adapter) -> None:
    """Register the NDA workflow with the workflows adapter"""
    try:
        await workflows_adapter.register_workflow_definition(
            workflow_name="nda_workflow",
            workflow_class=NDAWorkflow
        )
        logger.info("NDA workflow registered successfully")
    except Exception as e:
        logger.error(f"Failed to register NDA workflow: {e}")
        raise