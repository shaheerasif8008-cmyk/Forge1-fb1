import pytest

from forge1.core.memory import MemGPTSummarizer
from forge1.models.employee_models import SummaryType
from forge1.services.employee_memory_manager import EmployeeMemoryManager


def test_memgpt_summarizer_truncates_when_needed():
    summarizer = MemGPTSummarizer(max_length=10)
    long_text = "abcdefghijk"
    result = summarizer.summarize(long_text)
    assert result.startswith("abcdefghij")
    assert result.endswith(" …")


@pytest.mark.asyncio
async def test_employee_memory_manager_heuristic_summary(monkeypatch):
    manager = EmployeeMemoryManager(db_manager=None)
    manager._summarizer = None

    summary = await manager._generate_summary(
        ["User: hello", "Assistant: hi there"], SummaryType.CONVERSATION
    )
    assert "Conversation summary" in summary
