# forge1/backend/forge1/core/performance_optimizer.py
"""
Performance Optimization Engine

Analyzes collected metrics to:
- Detect hotspots and regressions
- Recommend optimization protocols (caching, routing, batching)
- Trigger dynamic scaling hooks (mocked)
- Provide simple predictive capacity planning
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

from forge1.core.model_router import ModelRouter
from forge1.core.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class PerformanceOptimizationEngine:
    """Optimization engine for runtime performance improvements"""

    def __init__(self, monitor: PerformanceMonitor, router: ModelRouter):
        self.monitor = monitor
        self.router = router

    async def analyze(self) -> Dict[str, Any]:
        """Analyze metrics and identify hotspots"""
        hotspots: List[Dict[str, Any]] = []
        for endpoint, m in self.monitor.metrics.items():
            if m["avg_duration"] > 1.0 or m["max_duration"] > 2.0:
                hotspots.append({
                    "endpoint": endpoint,
                    "avg": round(m["avg_duration"], 3),
                    "max": round(m["max_duration"], 3),
                    "count": m["total_requests"],
                })
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "hotspots": hotspots,
        }

    async def optimize(self) -> Dict[str, Any]:
        """Apply simple optimization protocols based on analysis"""
        analysis = await self.analyze()
        actions: List[str] = []

        # Example protocol: For hotspots, encourage cheaper/faster models by nudging scores
        if analysis["hotspots"]:
            for name, cap in self.router.models.items():
                try:
                    if cap.cost_per_1k_tokens <= 0.002 and cap.performance_score >= 0.8:
                        cap.performance_score = min(1.0, cap.performance_score + 0.02)
                        actions.append(f"nudged_performance_score:{name}")
                except Exception:
                    continue

        # Mock hooks for dynamic scaling
        scaled = self._maybe_scale_resources(analysis)
        if scaled:
            actions.append("scaled_resources")

        return {"analysis": analysis, "actions": actions}

    def _maybe_scale_resources(self, analysis: Dict[str, Any]) -> bool:
        """Mock: decide to scale based on hotspots"""
        return len(analysis.get("hotspots", [])) >= 3

    async def predict_capacity(self) -> Dict[str, Any]:
        """Very light predictive capacity planning based on moving averages"""
        avg_latency = [m["avg_duration"] for m in self.monitor.metrics.values() if m["total_requests"] > 0]
        if not avg_latency:
            return {"prediction": "insufficient_data"}
        mean = sum(avg_latency) / len(avg_latency)
        return {
            "prediction": "stable" if mean < 0.5 else ("watch" if mean < 1.0 else "risk"),
            "mean_latency": round(mean, 3)
        }

