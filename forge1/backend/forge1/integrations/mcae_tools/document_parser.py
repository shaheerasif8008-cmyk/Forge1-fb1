"""
Tenant-Aware Document Parser

Document parsing tool with tenant isolation that ensures employees
can only access and parse documents within their tenant's scope.
"""

import asyncio
import logging
import mimetypes
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import uuid
from datetime import datetime, timezone

from forge1.core.tenancy import get_current_tenant, set_current_tenant
from forge1.core.dlp import redact_payload

logger = logging.getLogger(__name__)


class DocumentParsingError(Exception):
    """Exception raised when document parsing fails"""
    pass


class TenantAccessError(Exception):
    """Exception raised when tenant access is denied"""
    pass


class TenantAwareDocumentParser:
    """
    Document parsing tool with tenant isolation.
    
    Provides document parsing capabilities while ensuring strict tenant
    boundaries and employee-specific access controls.
    """
    
    def __init__(self, tenant_id: str, employee_id: str, base_storage_path: Optional[str] = None):
        """
        Initialize the tenant-aware document parser.
        
        Args:
            tenant_id: Tenant ID for isolation
            employee_id: Employee ID for access control
            base_storage_path: Base path for tenant document storage
        """
        self.tenant_id = tenant_id
        self.employee_id = employee_id
        
        # Set up tenant-scoped storage path
        self.base_storage_path = base_storage_path or "/var/forge1/documents"
        self.tenant_storage_path = os.path.join(self.base_storage_path, f"tenant_{tenant_id}")
        
        # Supported file types
        self.supported_types = {
            '.txt': self._parse_text,
            '.md': self._parse_text,
            '.pdf': self._parse_pdf,
            '.docx': self._parse_docx,
            '.doc': self._parse_doc,
            '.html': self._parse_html,
            '.json': self._parse_json,
            '.csv': self._parse_csv,
            '.xml': self._parse_xml
        }
        
        # Parsing statistics
        self.stats = {
            "documents_parsed": 0,
            "total_pages": 0,
            "total_characters": 0,
            "errors": 0,
            "access_denied": 0
        }
        
    async def parse_document(self, document_path: str, options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Parse document with tenant-scoped access.
        
        Args:
            document_path: Path to document (relative to tenant storage)
            options: Parsing options (extract_metadata, apply_dlp, etc.)
            
        Returns:
            Dictionary containing parsed content and metadata
            
        Raises:
            TenantAccessError: If document is outside tenant scope
            DocumentParsingError: If parsing fails
        """
        try:
            # Set tenant context
            set_current_tenant(self.tenant_id)
            
            # Validate and resolve document path
            full_path = await self._validate_document_path(document_path)
            
            # Check if file exists and is accessible
            if not os.path.exists(full_path):
                raise DocumentParsingError(f"Document not found: {document_path}")
            
            if not os.access(full_path, os.R_OK):
                raise TenantAccessError(f"Access denied to document: {document_path}")
            
            # Get file information
            file_info = await self._get_file_info(full_path)
            
            # Determine parser based on file extension
            file_ext = Path(full_path).suffix.lower()
            if file_ext not in self.supported_types:
                raise DocumentParsingError(f"Unsupported file type: {file_ext}")
            
            # Parse the document
            parser_func = self.supported_types[file_ext]
            content = await parser_func(full_path, options or {})
            
            # Apply DLP if requested
            if options and options.get('apply_dlp', True):
                content['text'], dlp_violations = redact_payload(content['text'])
                content['dlp_violations'] = dlp_violations
            
            # Extract metadata if requested
            metadata = {}
            if options and options.get('extract_metadata', True):
                metadata = await self._extract_metadata(full_path, content)
            
            # Prepare result
            result = {
                "document_id": str(uuid.uuid4()),
                "tenant_id": self.tenant_id,
                "employee_id": self.employee_id,
                "document_path": document_path,
                "file_info": file_info,
                "content": content,
                "metadata": metadata,
                "parsed_at": datetime.now(timezone.utc).isoformat(),
                "parsing_options": options or {}
            }
            
            # Update statistics
            self.stats["documents_parsed"] += 1
            self.stats["total_characters"] += len(content.get('text', ''))
            if 'page_count' in content:
                self.stats["total_pages"] += content['page_count']
            
            logger.info(f"Successfully parsed document {document_path} for tenant {self.tenant_id}")
            return result
            
        except TenantAccessError:
            self.stats["access_denied"] += 1
            logger.warning(f"Access denied to document {document_path} for tenant {self.tenant_id}")
            raise
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Failed to parse document {document_path}: {e}")
            raise DocumentParsingError(f"Document parsing failed: {e}")
    
    async def list_documents(self, directory: str = "", file_pattern: str = "*") -> List[Dict[str, Any]]:
        """
        List documents in tenant storage.
        
        Args:
            directory: Subdirectory within tenant storage
            file_pattern: File pattern to match
            
        Returns:
            List of document information dictionaries
        """
        try:
            # Set tenant context
            set_current_tenant(self.tenant_id)
            
            # Validate directory path
            if directory:
                search_path = await self._validate_document_path(directory)
            else:
                search_path = self.tenant_storage_path
            
            if not os.path.exists(search_path):
                return []
            
            # List files
            documents = []
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.tenant_storage_path)
                    
                    # Check if file matches pattern and is supported
                    file_ext = Path(file).suffix.lower()
                    if file_pattern != "*" and file_pattern not in file:
                        continue
                    
                    if file_ext in self.supported_types:
                        file_info = await self._get_file_info(file_path)
                        documents.append({
                            "path": relative_path,
                            "name": file,
                            "type": file_ext,
                            "size": file_info["size"],
                            "modified": file_info["modified"],
                            "supported": True
                        })
            
            logger.debug(f"Listed {len(documents)} documents for tenant {self.tenant_id}")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    async def _validate_document_path(self, document_path: str) -> str:
        """
        Validate and resolve document path within tenant scope.
        
        Args:
            document_path: Relative or absolute document path
            
        Returns:
            Validated absolute path within tenant storage
            
        Raises:
            TenantAccessError: If path is outside tenant scope
        """
        # Convert to absolute path within tenant storage
        if os.path.isabs(document_path):
            # Absolute path - check if it's within tenant storage
            full_path = os.path.normpath(document_path)
        else:
            # Relative path - resolve within tenant storage
            full_path = os.path.normpath(os.path.join(self.tenant_storage_path, document_path))
        
        # Ensure path is within tenant storage (prevent directory traversal)
        if not full_path.startswith(self.tenant_storage_path):
            raise TenantAccessError(f"Document path outside tenant scope: {document_path}")
        
        return full_path
    
    async def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information"""
        stat = os.stat(file_path)
        mime_type, _ = mimetypes.guess_type(file_path)
        
        return {
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "mime_type": mime_type,
            "extension": Path(file_path).suffix.lower()
        }
    
    async def _parse_text(self, file_path: str, options: Dict) -> Dict[str, Any]:
        """Parse plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            return {
                "text": text,
                "format": "text",
                "character_count": len(text),
                "line_count": text.count('\n') + 1
            }
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
            
            return {
                "text": text,
                "format": "text",
                "character_count": len(text),
                "line_count": text.count('\n') + 1,
                "encoding": "latin-1"
            }
    
    async def _parse_pdf(self, file_path: str, options: Dict) -> Dict[str, Any]:
        """Parse PDF files (mock implementation)"""
        # In a real implementation, you would use a library like PyPDF2 or pdfplumber
        return {
            "text": f"[PDF Content from {os.path.basename(file_path)}]",
            "format": "pdf",
            "page_count": 1,
            "character_count": 50,
            "note": "PDF parsing requires additional libraries"
        }
    
    async def _parse_docx(self, file_path: str, options: Dict) -> Dict[str, Any]:
        """Parse DOCX files (mock implementation)"""
        # In a real implementation, you would use python-docx
        return {
            "text": f"[DOCX Content from {os.path.basename(file_path)}]",
            "format": "docx",
            "character_count": 50,
            "note": "DOCX parsing requires python-docx library"
        }
    
    async def _parse_doc(self, file_path: str, options: Dict) -> Dict[str, Any]:
        """Parse DOC files (mock implementation)"""
        # In a real implementation, you would use python-docx or antiword
        return {
            "text": f"[DOC Content from {os.path.basename(file_path)}]",
            "format": "doc",
            "character_count": 50,
            "note": "DOC parsing requires additional libraries"
        }
    
    async def _parse_html(self, file_path: str, options: Dict) -> Dict[str, Any]:
        """Parse HTML files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Simple HTML text extraction (in real implementation, use BeautifulSoup)
            import re
            text = re.sub(r'<[^>]+>', '', html_content)
            text = re.sub(r'\s+', ' ', text).strip()
            
            return {
                "text": text,
                "html": html_content,
                "format": "html",
                "character_count": len(text)
            }
        except Exception as e:
            raise DocumentParsingError(f"HTML parsing failed: {e}")
    
    async def _parse_json(self, file_path: str, options: Dict) -> Dict[str, Any]:
        """Parse JSON files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import json
                data = json.load(f)
            
            # Convert JSON to readable text
            text = json.dumps(data, indent=2)
            
            return {
                "text": text,
                "json_data": data,
                "format": "json",
                "character_count": len(text)
            }
        except Exception as e:
            raise DocumentParsingError(f"JSON parsing failed: {e}")
    
    async def _parse_csv(self, file_path: str, options: Dict) -> Dict[str, Any]:
        """Parse CSV files"""
        try:
            import csv
            rows = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    rows.append(row)
            
            # Convert CSV to text
            text = '\n'.join([','.join(row) for row in rows])
            
            return {
                "text": text,
                "rows": rows,
                "format": "csv",
                "row_count": len(rows),
                "character_count": len(text)
            }
        except Exception as e:
            raise DocumentParsingError(f"CSV parsing failed: {e}")
    
    async def _parse_xml(self, file_path: str, options: Dict) -> Dict[str, Any]:
        """Parse XML files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            
            # Simple XML text extraction
            import re
            text = re.sub(r'<[^>]+>', '', xml_content)
            text = re.sub(r'\s+', ' ', text).strip()
            
            return {
                "text": text,
                "xml": xml_content,
                "format": "xml",
                "character_count": len(text)
            }
        except Exception as e:
            raise DocumentParsingError(f"XML parsing failed: {e}")
    
    async def _extract_metadata(self, file_path: str, content: Dict) -> Dict[str, Any]:
        """Extract metadata from document"""
        metadata = {
            "file_name": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
            "format": content.get("format", "unknown"),
            "character_count": content.get("character_count", 0),
            "tenant_id": self.tenant_id,
            "employee_id": self.employee_id
        }
        
        # Add format-specific metadata
        if "page_count" in content:
            metadata["page_count"] = content["page_count"]
        if "line_count" in content:
            metadata["line_count"] = content["line_count"]
        if "row_count" in content:
            metadata["row_count"] = content["row_count"]
        
        return metadata
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parsing statistics"""
        return {
            **self.stats,
            "tenant_id": self.tenant_id,
            "employee_id": self.employee_id,
            "supported_types": list(self.supported_types.keys())
        }
    
    def reset_stats(self):
        """Reset parsing statistics"""
        self.stats = {
            "documents_parsed": 0,
            "total_pages": 0,
            "total_characters": 0,
            "errors": 0,
            "access_denied": 0
        }
        
        logger.info(f"Reset document parser stats for tenant {self.tenant_id}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Check if tenant storage directory exists
            storage_exists = os.path.exists(self.tenant_storage_path)
            storage_writable = os.access(self.tenant_storage_path, os.W_OK) if storage_exists else False
            
            return {
                "status": "healthy" if storage_exists else "degraded",
                "tenant_id": self.tenant_id,
                "employee_id": self.employee_id,
                "storage_path": self.tenant_storage_path,
                "storage_exists": storage_exists,
                "storage_writable": storage_writable,
                "supported_types": len(self.supported_types),
                "stats": self.get_stats()
            }
        except Exception as e:
            logger.error(f"Document parser health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "tenant_id": self.tenant_id,
                "employee_id": self.employee_id
            }