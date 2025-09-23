from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class KPI:
    name: str
    description: str
    target: float
    unit: str = ""


@dataclass
class VerticalKPIs:
    vertical: str
    kpis: List[KPI] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vertical": self.vertical,
            "kpis": [vars(k) for k in self.kpis],
        }


def default_kpis(vertical: str) -> VerticalKPIs:
    presets = {
        "cx": [
            KPI("first_response_time", "Time to first reply", 5, "seconds"),
            KPI("deflection_rate", "Automated resolution of scoped intents", 0.95, "ratio"),
            KPI("csat", "Customer satisfaction score", 0.90, "ratio"),
        ],
        "revops": [
            KPI("forecast_accuracy", "Delta between forecast and actual", 0.90, "ratio"),
            KPI("quote_turnaround", "Time to quote approval", 7200, "seconds"),
        ],
        "finance": [
            KPI("close_cycle_time", "Monthly close duration", 3, "days"),
            KPI("variance_accuracy", "Accuracy of variance analysis", 0.95, "ratio"),
        ],
        "legal": [
            KPI("contract_cycle_time", "Time to execute NDA/MSA/SOW", 7, "days"),
            KPI("risk_detection_precision", "Clause risk precision", 0.90, "ratio"),
        ],
        "itops": [
            KPI("mttr", "Mean time to resolve incidents", 3600, "seconds"),
            KPI("false_positive_reduction", "Reduction in FP alerts", 0.50, "ratio"),
        ],
        "software": [
            KPI("pr_throughput", "Pull requests completed per week", 50, "count"),
            KPI("defect_escape_rate", "Post-release defect rate", 0.02, "ratio"),
        ],
    }
    return VerticalKPIs(vertical, presets.get(vertical, []))

