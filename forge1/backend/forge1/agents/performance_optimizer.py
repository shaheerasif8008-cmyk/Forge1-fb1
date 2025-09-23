# forge1/backend/forge1/agents/performance_optimizer.py
"""
Performance Optimizer Agent for Forge 1

Specialized agent for performance monitoring and optimization.
"""

from forge1.agents.enhanced_base_agent import EnhancedBaseAgent

class PerformanceOptimizerAgent(EnhancedBaseAgent):
    """Performance optimization agent"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.optimization_targets = ["speed", "accuracy", "cost"]