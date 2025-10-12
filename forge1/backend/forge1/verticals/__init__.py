"""Forge1 vertical solutions package."""

from .phase8_accuracy import (
    DomainEvaluation,
    MetricResult,
    run_finance_accuracy_evaluation,
    run_legal_accuracy_evaluation,
    run_phase8_evaluations,
)

__all__ = [
    "DomainEvaluation",
    "MetricResult",
    "run_finance_accuracy_evaluation",
    "run_legal_accuracy_evaluation",
    "run_phase8_evaluations",
]
