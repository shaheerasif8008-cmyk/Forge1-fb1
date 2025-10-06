"""
MCAE Tool Adapters

Tenant-aware tool implementations that provide MCAE agents with access
to enterprise tools while maintaining strict tenant isolation and
employee-specific permissions.
"""

from .document_parser import TenantAwareDocumentParser
from .vector_search import TenantAwareVectorSearch
from .slack_poster import TenantAwareSlackPoster
from .drive_fetcher import TenantAwareDriveFetcher

__all__ = [
    'TenantAwareDocumentParser',
    'TenantAwareVectorSearch', 
    'TenantAwareSlackPoster',
    'TenantAwareDriveFetcher'
]