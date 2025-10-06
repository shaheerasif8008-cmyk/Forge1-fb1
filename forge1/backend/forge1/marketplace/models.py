"""
Marketplace Data Models
Comprehensive models for agents, tools, templates, and marketplace operations
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from decimal import Decimal
import json


class MarketplaceItemType(Enum):
    """Types of marketplace items"""
    AGENT_TEMPLATE = "agent_template"
    TOOL_CONNECTOR = "tool_connector"
    WORKFLOW_TEMPLATE = "workflow_template"
    INTEGRATION_PACKAGE = "integration_package"
    CUSTOM_MODEL = "custom_model"


class PublishingStatus(Enum):
    """Publishing status for marketplace items"""
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"


class InstallationStatus(Enum):
    """Installation status for marketplace items"""
    NOT_INSTALLED = "not_installed"
    INSTALLING = "installing"
    INSTALLED = "installed"
    UPDATING = "updating"
    FAILED = "failed"
    UNINSTALLING = "uninstalling"


class CompatibilityLevel(Enum):
    """Compatibility levels for marketplace items"""
    COMPATIBLE = "compatible"
    PARTIALLY_COMPATIBLE = "partially_compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


@dataclass
class Version:
    """Semantic version with metadata"""
    major: int
    minor: int
    patch: int
    pre_release: Optional[str] = None
    build_metadata: Optional[str] = None
    
    def __str__(self) -> str:
        version_str = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre_release:
            version_str += f"-{self.pre_release}"
        if self.build_metadata:
            version_str += f"+{self.build_metadata}"
        return version_str
    
    def __lt__(self, other: 'Version') -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def __eq__(self, other: 'Version') -> bool:
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
    
    @classmethod
    def from_string(cls, version_str: str) -> 'Version':
        """Parse version from string"""
        # Simplified parsing - would use proper semver library in production
        parts = version_str.split('.')
        return cls(
            major=int(parts[0]),
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2]) if len(parts) > 2 else 0
        )


@dataclass
class Dependency:
    """Dependency specification for marketplace items"""
    name: str
    version_constraint: str  # e.g., ">=1.0.0,<2.0.0"
    item_type: MarketplaceItemType
    required: bool = True
    description: Optional[str] = None


@dataclass
class SecurityScan:
    """Security scan results for marketplace items"""
    scan_id: str
    scan_date: datetime
    scanner_version: str
    vulnerabilities_found: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    scan_passed: bool
    detailed_report: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SBOM:
    """Software Bill of Materials"""
    sbom_id: str
    format_version: str  # CycloneDX version
    components: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    licenses: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_cyclonedx(self) -> Dict[str, Any]:
        """Export as CycloneDX format"""
        return {
            "bomFormat": "CycloneDX",
            "specVersion": self.format_version,
            "serialNumber": f"urn:uuid:{self.sbom_id}",
            "version": 1,
            "metadata": {
                "timestamp": self.generated_at.isoformat(),
                "tools": [
                    {
                        "vendor": "Cognisia",
                        "name": "Forge1 Marketplace",
                        "version": "1.0.0"
                    }
                ]
            },
            "components": self.components,
            "dependencies": self.dependencies
        }


@dataclass
class MarketplaceItem:
    """Base marketplace item with common properties"""
    item_id: str
    name: str
    display_name: str
    description: str
    item_type: MarketplaceItemType
    version: Version
    publisher_id: str
    publisher_name: str
    
    # Publishing metadata
    publishing_status: PublishingStatus = PublishingStatus.DRAFT
    published_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Categorization
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    
    # Dependencies and compatibility
    dependencies: List[Dependency] = field(default_factory=list)
    platform_requirements: Dict[str, str] = field(default_factory=dict)
    
    # Security and compliance
    security_scan: Optional[SecurityScan] = None
    sbom: Optional[SBOM] = None
    license: str = "proprietary"
    
    # Marketplace metadata
    download_count: int = 0
    rating: float = 0.0
    review_count: int = 0
    price: Decimal = Decimal('0.00')
    
    # Configuration
    configuration_schema: Dict[str, Any] = field(default_factory=dict)
    default_configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Documentation
    readme: str = ""
    changelog: str = ""
    documentation_url: Optional[str] = None
    support_url: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTemplate(MarketplaceItem):
    """Agent template marketplace item"""
    
    # Agent-specific properties
    vertical_type: str = "general"  # cx, revops, finance, etc.
    capabilities: List[str] = field(default_factory=list)
    supported_frameworks: List[str] = field(default_factory=list)  # langchain, crewai, autogen
    
    # Performance characteristics
    expected_performance: Dict[str, float] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Template definition
    template_definition: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.item_type != MarketplaceItemType.AGENT_TEMPLATE:
            self.item_type = MarketplaceItemType.AGENT_TEMPLATE


@dataclass
class ToolConnector(MarketplaceItem):
    """Tool connector marketplace item"""
    
    # Tool-specific properties
    connector_type: str = "api"  # api, webhook, database, file_system
    supported_protocols: List[str] = field(default_factory=list)
    authentication_methods: List[str] = field(default_factory=list)
    
    # Integration details
    api_specification: Optional[Dict[str, Any]] = None
    connector_implementation: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.item_type != MarketplaceItemType.TOOL_CONNECTOR:
            self.item_type = MarketplaceItemType.TOOL_CONNECTOR


@dataclass
class WorkflowTemplate(MarketplaceItem):
    """Workflow template marketplace item"""
    
    # Workflow-specific properties
    workflow_type: str = "sequential"  # sequential, parallel, conditional
    estimated_duration: Optional[int] = None  # seconds
    
    # Workflow definition
    workflow_definition: Dict[str, Any] = field(default_factory=dict)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.item_type != MarketplaceItemType.WORKFLOW_TEMPLATE:
            self.item_type = MarketplaceItemType.WORKFLOW_TEMPLATE


@dataclass
class Installation:
    """Installation record for a marketplace item"""
    installation_id: str
    item_id: str
    item_version: str
    tenant_id: str
    installed_by: str
    
    # Installation metadata
    installation_status: InstallationStatus = InstallationStatus.NOT_INSTALLED
    installed_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Configuration
    installation_config: Dict[str, Any] = field(default_factory=dict)
    custom_configuration: Dict[str, Any] = field(default_factory=dict)
    
    # Health and monitoring
    health_status: str = "unknown"
    last_health_check: Optional[datetime] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Installation artifacts
    installation_path: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    
    # Rollback information
    previous_version: Optional[str] = None
    rollback_available: bool = False


@dataclass
class MarketplaceReview:
    """User review for marketplace item"""
    review_id: str
    item_id: str
    reviewer_id: str
    reviewer_name: str
    
    # Review content
    rating: int  # 1-5 stars
    title: str
    content: str
    
    # Review metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    verified_purchase: bool = False
    helpful_votes: int = 0
    
    # Moderation
    moderation_status: str = "pending"  # pending, approved, rejected
    moderated_at: Optional[datetime] = None
    moderated_by: Optional[str] = None


@dataclass
class PublishingRequest:
    """Request to publish an item to the marketplace"""
    request_id: str
    item_id: str
    publisher_id: str
    
    # Request details
    requested_status: PublishingStatus
    submission_notes: str = ""
    
    # Review process
    review_status: str = "pending"  # pending, in_review, approved, rejected
    reviewer_id: Optional[str] = None
    review_notes: str = ""
    
    # Timestamps
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None
    
    # Compliance checks
    security_check_passed: bool = False
    compliance_check_passed: bool = False
    quality_check_passed: bool = False


@dataclass
class MarketplaceMetrics:
    """Marketplace performance metrics"""
    
    # Overall metrics
    total_items: int = 0
    total_downloads: int = 0
    total_publishers: int = 0
    total_tenants: int = 0
    
    # Item type breakdown
    items_by_type: Dict[str, int] = field(default_factory=dict)
    downloads_by_type: Dict[str, int] = field(default_factory=dict)
    
    # Popular items
    most_downloaded: List[Dict[str, Any]] = field(default_factory=list)
    highest_rated: List[Dict[str, Any]] = field(default_factory=list)
    
    # Publisher metrics
    top_publishers: List[Dict[str, Any]] = field(default_factory=list)
    
    # Time-based metrics
    daily_downloads: Dict[str, int] = field(default_factory=dict)
    monthly_revenue: Dict[str, Decimal] = field(default_factory=dict)
    
    # Quality metrics
    avg_rating: float = 0.0
    security_scan_pass_rate: float = 0.0
    
    # Generated metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TenantMarketplaceConfig:
    """Tenant-specific marketplace configuration"""
    tenant_id: str
    
    # Access controls
    allow_public_marketplace: bool = True
    allow_private_publishing: bool = True
    require_approval_for_installs: bool = False
    
    # Restrictions
    allowed_item_types: List[MarketplaceItemType] = field(default_factory=lambda: list(MarketplaceItemType))
    blocked_publishers: List[str] = field(default_factory=list)
    allowed_categories: List[str] = field(default_factory=list)
    
    # Budget and billing
    monthly_budget: Optional[Decimal] = None
    current_spend: Decimal = Decimal('0.00')
    
    # Custom marketplace
    custom_marketplace_url: Optional[str] = None
    private_registry_enabled: bool = False
    
    # Notifications
    notify_on_new_items: bool = True
    notify_on_updates: bool = True
    notification_channels: List[str] = field(default_factory=list)
    
    # Configuration metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)