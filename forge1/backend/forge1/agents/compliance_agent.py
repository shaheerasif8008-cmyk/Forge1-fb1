# forge1/backend/forge1/agents/compliance_agent.py
"""
Compliance Agent for Forge 1

Specialized agent for compliance monitoring and validation.
"""

from forge1.agents.enhanced_base_agent import EnhancedBaseAgent

class ComplianceAgent(EnhancedBaseAgent):
    """Compliance monitoring agent"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.compliance_standards = ["GDPR", "CCPA", "HIPAA", "SOX"]