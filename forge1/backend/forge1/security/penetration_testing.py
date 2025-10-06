"""
Penetration Testing & Bug Bounty Management
Automated DAST/SAST, external pen-tests, and bug bounty program management
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import logging
import json
import subprocess
from pathlib import Path
import tempfile

from forge1.backend.forge1.core.monitoring import MetricsCollector
from forge1.backend.forge1.core.memory import MemoryManager
from forge1.backend.forge1.core.security import SecretManager


class TestType(Enum):
    """Types of security tests"""
    SAST = "sast"  # Static Application Security Testing
    DAST = "dast"  # Dynamic Application Security Testing
    IAST = "iast"  # Interactive Application Security Testing
    PENETRATION_TEST = "penetration_test"
    BUG_BOUNTY = "bug_bounty"


class FindingSeverity(Enum):
    """Security finding severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingStatus(Enum):
    """Status of security findings"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ACCEPTED_RISK = "accepted_risk"
    FALSE_POSITIVE = "false_positive"


@dataclass
class SecurityFinding:
    """Security finding from testing"""
    finding_id: str
    title: str
    description: str
    severity: FindingSeverity
    status: FindingStatus
    test_type: TestType
    
    # Technical details
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None
    cvss_score: Optional[float] = None
    
    # Location information
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    url: Optional[str] = None
    parameter: Optional[str] = None
    
    # Remediation
    remediation: str = ""
    references: List[str] = field(default_factory=list)
    
    # Tracking
    discovered_date: datetime = field(default_factory=datetime.utcnow)
    due_date: Optional[datetime] = None
    resolved_date: Optional[datetime] = None
    
    # Metadata
    scanner: Optional[str] = None
    confidence: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)