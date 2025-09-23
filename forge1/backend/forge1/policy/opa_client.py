# forge1/backend/forge1/policy/opa_client.py
"""
OPA (Open Policy Agent) client wrapper.

Evaluates input against a Rego policy path via OPA's REST API.
"""

from typing import Any, Dict, Optional
import httpx


class OPAClient:
    def __init__(self, base_url: str, timeout_ms: int = 2000):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_ms / 1000.0

    async def evaluate(self, policy_path: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """POST /v1/data/<policy_path> with input and return result.

        Expected Rego to return a document containing an `allow` boolean and optional `reason`.
        """
        url = f"{self.base_url}/v1/data/{policy_path.lstrip('/')}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json={"input": input_data})
            resp.raise_for_status()
            data = resp.json()
            # OPA returns {"result": {...}}
            return data.get("result", {})


__all__ = ["OPAClient"]

