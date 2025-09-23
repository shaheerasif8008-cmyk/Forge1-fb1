import pytest
import asyncio

from forge1.verticals.playbook_engine import PlaybookEngine


@pytest.mark.asyncio
async def test_list_verticals_and_playbooks_and_execute_dry_run():
    eng = PlaybookEngine()
    verticals = eng.list_verticals()
    assert set(["cx", "revops", "finance", "legal", "itops", "software"]).issubset(set(verticals))

    for v in ["cx", "revops", "finance", "legal", "itops", "software"]:
        pbs = eng.list_playbooks(v)
        assert len(pbs) >= 1
        name = pbs[0]["name"]
        res = await eng.execute(v, name, context={}, dry_run=True)
        assert res["dry_run"] is True
        assert res["vertical"] == v
        assert len(res["steps"]) >= 1
        assert "kpis" in res

