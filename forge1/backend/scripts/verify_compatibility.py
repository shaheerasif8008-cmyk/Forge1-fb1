#!/usr/bin/env python3
"""
Compatibility verification script for LlamaIndex and workflows-py integration.
Ensures all dependencies work together and core functionality is available.
"""

import sys
import importlib
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_import(module_name: str, description: str) -> Tuple[bool, str]:
    """Check if a module can be imported successfully."""
    try:
        importlib.import_module(module_name)
        return True, f"✓ {description}"
    except ImportError as e:
        return False, f"✗ {description}: {e}"
    except Exception as e:
        return False, f"✗ {description}: Unexpected error - {e}"

def verify_llamaindex_compatibility() -> List[Tuple[bool, str]]:
    """Verify LlamaIndex components are compatible and functional."""
    results = []
    
    # Core LlamaIndex imports
    core_modules = [
        ("llama_index.core", "LlamaIndex Core"),
        ("llama_index.core.llms", "LlamaIndex LLM Interface"),
        ("llama_index.core.embeddings", "LlamaIndex Embeddings"),
        ("llama_index.core.node_parser", "LlamaIndex Node Parser"),
        ("llama_index.core.vector_stores", "LlamaIndex Vector Stores"),
        ("llama_index.core.tools", "LlamaIndex Tools"),
    ]
    
    for module, desc in core_modules:
        results.append(check_import(module, desc))
    
    # Specific integrations
    integration_modules = [
        ("llama_index.embeddings.openai", "OpenAI Embeddings"),
        ("llama_index.llms.openai", "OpenAI LLM"),
        ("llama_index.readers.file", "File Readers"),
        ("llama_index.vector_stores.postgres", "PostgreSQL Vector Store"),
    ]
    
    for module, desc in integration_modules:
        results.append(check_import(module, desc))
    
    return results

def verify_workflows_compatibility() -> List[Tuple[bool, str]]:
    """Verify workflows-py components are compatible and functional."""
    results = []
    
    # workflows-py core imports
    workflow_modules = [
        ("workflows", "workflows-py Core"),
        ("workflows.workflow", "Workflow Definition"),
        ("workflows.step", "Step Definition"),
        ("workflows.context", "Workflow Context"),
    ]
    
    for module, desc in workflow_modules:
        results.append(check_import(module, desc))
    
    return results

def verify_document_processing() -> List[Tuple[bool, str]]:
    """Verify document processing dependencies."""
    results = []
    
    doc_modules = [
        ("pypdf", "PDF Processing"),
        ("docx", "DOCX Processing"),
        ("pytesseract", "OCR Support"),
        ("PIL", "Image Processing"),
    ]
    
    for module, desc in doc_modules:
        results.append(check_import(module, desc))
    
    return results

def verify_external_integrations() -> List[Tuple[bool, str]]:
    """Verify external service integrations."""
    results = []
    
    external_modules = [
        ("googleapiclient", "Google API Client"),
        ("google.auth", "Google Authentication"),
        ("slack_sdk", "Slack SDK"),
        ("pgvector", "PostgreSQL Vector Extension"),
        ("sentence_transformers", "Sentence Transformers"),
    ]
    
    for module, desc in external_modules:
        results.append(check_import(module, desc))
    
    return results

def verify_observability() -> List[Tuple[bool, str]]:
    """Verify OpenTelemetry observability components."""
    results = []
    
    otel_modules = [
        ("opentelemetry.api", "OpenTelemetry API"),
        ("opentelemetry.sdk", "OpenTelemetry SDK"),
        ("opentelemetry.instrumentation", "OpenTelemetry Instrumentation"),
        ("opentelemetry.exporter.jaeger", "Jaeger Exporter"),
    ]
    
    for module, desc in otel_modules:
        results.append(check_import(module, desc))
    
    return results

def test_basic_functionality() -> List[Tuple[bool, str]]:
    """Test basic functionality of key components."""
    results = []
    
    try:
        # Test LlamaIndex basic functionality
        from llama_index.core.llms.mock import MockLLM
        from llama_index.core.embeddings.mock import MockEmbedding
        
        # Create mock instances
        llm = MockLLM()
        embedding = MockEmbedding(embed_dim=1536)
        
        results.append((True, "✓ LlamaIndex LLM instantiation"))
        results.append((True, "✓ LlamaIndex Embedding instantiation"))
        
    except Exception as e:
        results.append((False, f"✗ LlamaIndex basic functionality: {e}"))
    
    try:
        # Test workflows-py basic functionality
        import workflows
        from workflows import workflow, step
        
        @workflow
        class TestWorkflow:
            @step
            def test_step(self):
                return "test"
        
        results.append((True, "✓ workflows-py workflow definition"))
        
    except Exception as e:
        results.append((False, f"✗ workflows-py basic functionality: {e}"))
    
    return results

def main():
    """Run all compatibility checks."""
    logger.info("Starting compatibility verification for LlamaIndex and workflows-py integration")
    
    all_results = []
    
    # Run all verification checks
    checks = [
        ("LlamaIndex Core", verify_llamaindex_compatibility),
        ("workflows-py", verify_workflows_compatibility),
        ("Document Processing", verify_document_processing),
        ("External Integrations", verify_external_integrations),
        ("Observability", verify_observability),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    for check_name, check_func in checks:
        logger.info(f"\n--- {check_name} ---")
        results = check_func()
        all_results.extend(results)
        
        for success, message in results:
            if success:
                logger.info(message)
            else:
                logger.error(message)
    
    # Summary
    total_checks = len(all_results)
    successful_checks = sum(1 for success, _ in all_results if success)
    failed_checks = total_checks - successful_checks
    
    logger.info(f"\n--- Summary ---")
    logger.info(f"Total checks: {total_checks}")
    logger.info(f"Successful: {successful_checks}")
    logger.info(f"Failed: {failed_checks}")
    
    if failed_checks > 0:
        logger.error(f"Compatibility verification failed with {failed_checks} errors")
        sys.exit(1)
    else:
        logger.info("✓ All compatibility checks passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()