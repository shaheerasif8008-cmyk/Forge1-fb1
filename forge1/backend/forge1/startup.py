"""
Forge1 Backend Startup Script

Ensures the backend can start even with missing optional dependencies.
This script handles graceful degradation for all OSS integrations.
"""

import os
import sys
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_and_mock_dependencies():
    """Check for missing dependencies and create mocks if needed"""
    
    # Mock Azure Monitor if not available
    try:
        import azure.monitor.opentelemetry
    except ImportError:
        logger.warning("Azure Monitor not available - creating mock")
        import sys
        from unittest.mock import MagicMock
        
        # Create mock modules
        sys.modules['azure'] = MagicMock()
        sys.modules['azure.monitor'] = MagicMock()
        sys.modules['azure.monitor.opentelemetry'] = MagicMock()
    
    # Mock OpenTelemetry if not available
    try:
        import opentelemetry
    except ImportError:
        logger.warning("OpenTelemetry not available - creating mock")
        import sys
        from unittest.mock import MagicMock
        
        # Create mock modules for OpenTelemetry
        otel_modules = [
            'opentelemetry',
            'opentelemetry.trace',
            'opentelemetry.metrics',
            'opentelemetry.exporter.jaeger.thrift',
            'opentelemetry.exporter.otlp.proto.grpc.trace_exporter',
            'opentelemetry.exporter.otlp.proto.grpc.metric_exporter',
            'opentelemetry.instrumentation.fastapi',
            'opentelemetry.instrumentation.celery',
            'opentelemetry.instrumentation.redis',
            'opentelemetry.instrumentation.psycopg2',
            'opentelemetry.instrumentation.requests',
            'opentelemetry.instrumentation.httpx',
            'opentelemetry.sdk.trace',
            'opentelemetry.sdk.trace.export',
            'opentelemetry.sdk.metrics',
            'opentelemetry.sdk.metrics.export',
            'opentelemetry.sdk.resources',
            'opentelemetry.semconv.resource',
            'opentelemetry.propagate',
            'opentelemetry.propagators.b3',
            'opentelemetry.propagators.jaeger',
            'opentelemetry.propagators.composite'
        ]
        
        for module in otel_modules:
            sys.modules[module] = MagicMock()
    
    # Mock Celery if not available
    try:
        import celery
    except ImportError:
        logger.warning("Celery not available - creating mock")
        import sys
        from unittest.mock import MagicMock
        
        sys.modules['celery'] = MagicMock()
        sys.modules['kombu'] = MagicMock()
    
    # Mock Weaviate if not available
    try:
        import weaviate
    except ImportError:
        logger.warning("Weaviate not available - creating mock")
        import sys
        from unittest.mock import MagicMock
        
        sys.modules['weaviate'] = MagicMock()
    
    # Mock Kafka if not available
    try:
        import kafka
    except ImportError:
        logger.warning("Kafka not available - creating mock")
        import sys
        from unittest.mock import MagicMock
        
        sys.modules['kafka'] = MagicMock()
        sys.modules['kafka.producer'] = MagicMock()

def start_application():
    """Start the Forge1 application with dependency checks"""

    logger.info("Starting Forge1 Backend...")

    # Check and mock missing dependencies
    check_and_mock_dependencies()

    # Set environment variables for graceful degradation
    os.environ.setdefault('OTEL_CONSOLE_EXPORTER', 'false')
    os.environ.setdefault('DISABLE_AZURE_MONITOR', 'true')
    os.environ.setdefault('DISABLE_MCAE_INTEGRATION', 'true')

    if os.getenv("FORGE1_FORCE_MINIMAL", "false").lower() in {"1", "true", "yes"}:
        logger.info("FORGE1_FORCE_MINIMAL is set; returning minimal application without importing main app.")
        return _create_fallback_app(reason="forced_minimal")

    try:
        # Import and start the main application
        from forge1.main import app
        logger.info("✓ Forge1 main application imported successfully")
        return app
        
    except Exception as e:
        logger.error(f"Failed to import main application: {e}")
        return _create_fallback_app(reason=str(e))


def _create_fallback_app(reason: str) -> "FastAPI":
    from fastapi import FastAPI

    logger.info("Creating minimal fallback application...")
    fallback_app = FastAPI(
        title="Forge1 Backend (Minimal Mode)",
        description="Forge1 running in minimal mode due to missing dependencies",
        version="1.0.0"
    )

    @fallback_app.get("/")
    async def root():
        return {
            "message": "Forge1 Backend running in minimal mode",
            "status": "ok",
            "details": reason,
        }

    @fallback_app.get("/health")
    async def health():
        return {
            "status": "ok",
            "mode": "minimal",
            "details": reason,
        }

    @fallback_app.get("/metrics")
    async def metrics():
        return {"status": "ok", "mode": "minimal"}

    return fallback_app

# Create the app instance
app = start_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
