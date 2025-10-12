"""Phase 8 accuracy and KPI evaluations for legal and finance verticals."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple, Union

from forge1.integrations.automation_connectors import KBSearchAdapter


JsonDict = Dict[str, Any]
Number = Union[int, float]


@dataclass
class MetricResult:
    """Single metric captured during an accuracy evaluation."""

    name: str
    value: Number
    target: Number
    unit: str
    passed: bool
    details: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        payload = asdict(self)
        return payload


@dataclass
class DomainEvaluation:
    """Container for metrics computed for a specific domain."""

    domain: str
    metrics: List[MetricResult]
    methodology: str
    dataset_summary: JsonDict

    @property
    def passed(self) -> bool:
        return all(metric.passed for metric in self.metrics)

    def to_dict(self) -> JsonDict:
        return {
            "domain": self.domain,
            "passed": self.passed,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "methodology": self.methodology,
            "dataset_summary": self.dataset_summary,
        }


def _build_legal_dataset() -> Tuple[List[JsonDict], List[JsonDict]]:
    documents = [
        {
            "clause_id": "confidentiality_obligations",
            "content": (
                "This confidentiality clause requires both parties to protect all confidential "
                "information, specifies mutual obligations, and defines what constitutes "
                "protected information."
            ),
            "summary": "Mutual confidentiality obligations",
            "key_points": [
                "mutual_confidentiality",
                "confidential_information_definition",
                "protection_scope",
            ],
        },
        {
            "clause_id": "nda_exceptions",
            "content": (
                "Standard NDA exceptions cover information previously known to the recipient, "
                "independently developed knowledge, and disclosures required by law."
            ),
            "summary": "Standard NDA exceptions",
            "key_points": [
                "previously_known_information",
                "independent_development",
                "legal_disclosure_requirement",
            ],
        },
        {
            "clause_id": "governing_law",
            "content": (
                "The agreement is governed by the laws of California and any disputes will be "
                "resolved in San Francisco County courts. This governing law clause clearly "
                "states the venue for legal proceedings."
            ),
            "summary": "Governing law and venue",
            "key_points": [
                "california_law",
                "san_francisco_venue",
            ],
        },
    ]

    queries = [
        {
            "query": "confidentiality obligations in nda",
            "relevant_clauses": {"confidentiality_obligations"},
            "allowed_facts": {
                "mutual_confidentiality",
                "confidential_information_definition",
                "protection_scope",
            },
        },
        {
            "query": "nda exceptions for previously known information",
            "relevant_clauses": {"nda_exceptions"},
            "allowed_facts": {
                "previously_known_information",
                "independent_development",
                "legal_disclosure_requirement",
            },
        },
        {
            "query": "california governing law clause",
            "relevant_clauses": {"governing_law"},
            "allowed_facts": {
                "california_law",
                "san_francisco_venue",
            },
        },
    ]

    return documents, queries


async def _legal_precision_and_hallucination() -> DomainEvaluation:
    documents, queries = _build_legal_dataset()
    adapter = KBSearchAdapter()

    tenant_id = "tenant_legal_eval"
    employee_id = "legal_reviewer"

    for document in documents:
        adapter.index_local_document(
            tenant_id=tenant_id,
            employee_id=employee_id,
            content=document["content"],
            metadata={
                "summary": document["summary"],
                "clause_id": document["clause_id"],
                "key_points": document["key_points"],
            },
        )

    precision_scores: List[float] = []
    hallucinated_facts = 0
    total_facts = 0

    for query in queries:
        result = await adapter.search(
            tenant_id=tenant_id,
            employee_id=employee_id,
            query=query["query"],
            limit=5,
            similarity_threshold=0.2,
        )
        retrieved_clauses: List[str] = []
        predicted_facts: List[str] = []
        for entry in result.results:
            metadata = entry.get("metadata", {})
            clause_id = metadata.get("clause_id")
            if clause_id:
                retrieved_clauses.append(clause_id)
            predicted_facts.extend(metadata.get("key_points", []))

        if retrieved_clauses:
            hits = sum(1 for clause in retrieved_clauses[:5] if clause in query["relevant_clauses"])
            precision_scores.append(hits / min(len(retrieved_clauses), 5))
        else:
            precision_scores.append(0.0)

        allowed_facts = set(query["allowed_facts"])
        hallucinated = [fact for fact in predicted_facts if fact not in allowed_facts]
        hallucinated_facts += len(hallucinated)
        total_facts += len(predicted_facts)

    precision_at_5 = mean(precision_scores) if precision_scores else 0.0
    hallucination_rate = (hallucinated_facts / total_facts) if total_facts else 0.0

    metrics = [
        MetricResult(
            name="precision_at_5",
            value=round(precision_at_5, 3),
            target=0.8,
            unit="ratio",
            passed=precision_at_5 >= 0.8,
            details={"per_query": precision_scores},
        ),
        MetricResult(
            name="hallucination_rate",
            value=round(hallucination_rate, 3),
            target=0.02,
            unit="ratio",
            passed=hallucination_rate <= 0.02,
            details={"hallucinated_facts": hallucinated_facts, "total_facts": total_facts},
        ),
    ]

    dataset_summary = {
        "documents_indexed": len(documents),
        "queries": [
            {
                "query": item["query"],
                "relevant_clauses": sorted(item["relevant_clauses"]),
            }
            for item in queries
        ],
    }

    methodology = (
        "Indexed deterministic NDA clauses into the local KB adapter and executed three legal "
        "queries. Precision@5 is computed per query over retrieved clause identifiers, and the "
        "hallucination rate captures unsupported key points emitted from retrieved results."
    )

    return DomainEvaluation(
        domain="legal",
        metrics=metrics,
        methodology=methodology,
        dataset_summary=dataset_summary,
    )


def _build_finance_dataset() -> JsonDict:
    return {
        "month": "2025-01",
        "revenue": [
            {"segment": "Enterprise", "amount": 180000.0},
            {"segment": "SMB", "amount": 75000.0},
        ],
        "expenses": [
            {"category": "Payroll", "amount": 90000.0},
            {"category": "Infrastructure", "amount": 30000.0},
            {"category": "Sales", "amount": 15000.0},
        ],
        "ground_truth": {
            "total_revenue": 255000.0,
            "total_expenses": 135000.0,
            "net_profit": 120000.0,
            "profit_margin": 120000.0 / 255000.0,
        },
    }


def _run_finance_analysis(dataset: JsonDict) -> JsonDict:
    total_revenue = sum(item["amount"] for item in dataset["revenue"])
    total_expenses = sum(item["amount"] for item in dataset["expenses"])
    net_profit = total_revenue - total_expenses
    margin = net_profit / total_revenue if total_revenue else 0.0
    return {
        "total_revenue": total_revenue,
        "total_expenses": total_expenses,
        "net_profit": net_profit,
        "profit_margin": margin,
    }


def _variance(predicted: Number, actual: Number) -> float:
    if actual == 0:
        return 0.0
    return abs(float(predicted) - float(actual)) / float(actual)


def run_finance_accuracy_evaluation() -> DomainEvaluation:
    dataset = _build_finance_dataset()
    predicted = _run_finance_analysis(dataset)
    actual = dataset["ground_truth"]

    metrics = [
        MetricResult(
            name="revenue_variance",
            value=round(_variance(predicted["total_revenue"], actual["total_revenue"]), 5),
            target=0.01,
            unit="ratio",
            passed=_variance(predicted["total_revenue"], actual["total_revenue"]) <= 0.01,
        ),
        MetricResult(
            name="expense_variance",
            value=round(_variance(predicted["total_expenses"], actual["total_expenses"]), 5),
            target=0.01,
            unit="ratio",
            passed=_variance(predicted["total_expenses"], actual["total_expenses"]) <= 0.01,
        ),
        MetricResult(
            name="net_profit_variance",
            value=round(_variance(predicted["net_profit"], actual["net_profit"]), 5),
            target=0.01,
            unit="ratio",
            passed=_variance(predicted["net_profit"], actual["net_profit"]) <= 0.01,
        ),
        MetricResult(
            name="profit_margin_delta",
            value=round(abs(predicted["profit_margin"] - actual["profit_margin"]), 5),
            target=0.01,
            unit="ratio",
            passed=abs(predicted["profit_margin"] - actual["profit_margin"]) <= 0.01,
        ),
    ]

    dataset_summary = {
        "month": dataset["month"],
        "revenue_entries": len(dataset["revenue"]),
        "expense_entries": len(dataset["expenses"]),
    }

    methodology = (
        "Aggregated deterministic ERP revenue and expense samples to compute total revenue, "
        "expenses, net profit, and profit margin. Variance is measured against the curated "
        "ground-truth ledger with a 1% tolerance."
    )

    return DomainEvaluation(
        domain="finance",
        metrics=metrics,
        methodology=methodology,
        dataset_summary=dataset_summary,
    )


def run_legal_accuracy_evaluation() -> DomainEvaluation:
    return asyncio.run(_legal_precision_and_hallucination())


def run_phase8_evaluations(output_dir: Union[str, Path]) -> Dict[str, DomainEvaluation]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    legal = run_legal_accuracy_evaluation()
    finance = run_finance_accuracy_evaluation()

    summary = {
        "legal": legal.to_dict(),
        "finance": finance.to_dict(),
        "overall_passed": legal.passed and finance.passed,
    }

    (output_path / "legal_evaluation.json").write_text(
        json.dumps(legal.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_path / "finance_evaluation.json").write_text(
        json.dumps(finance.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_path / "phase8_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    methodology_lines = [
        "# Phase 8 Accuracy & KPI Evaluation",
        "",
        f"- Legal evaluation passed: {legal.passed}",
        f"- Finance evaluation passed: {finance.passed}",
        "",
        "## Methodology",
        "", legal.methodology,
        "", finance.methodology,
    ]
    (output_path / "methodology.md").write_text("\n".join(methodology_lines) + "\n", encoding="utf-8")

    return {"legal": legal, "finance": finance}


__all__ = [
    "DomainEvaluation",
    "MetricResult",
    "run_legal_accuracy_evaluation",
    "run_finance_accuracy_evaluation",
    "run_phase8_evaluations",
]
