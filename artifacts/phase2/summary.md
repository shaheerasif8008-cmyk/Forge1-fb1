# Phase 2 — Working Tool Adapters

## Overview
- Implemented functional automation connectors with optional dependency guards in `forge1/backend/forge1/integrations/automation_connectors.py`.
- Updated LlamaIndex tool wiring to use the new connectors via dependency injection in `forge1/backend/forge1/integrations/llamaindex_tools.py`.
- Added targeted tool tests under `forge1/backend/tests/tools/test_automation_connectors.py` and relaxed fixtures in `forge1/backend/tests/conftest.py` to skip missing optional dependencies.

## Testing
- `pytest forge1/backend/tests/tools -q`
- `ruff check forge1/backend/forge1/integrations/automation_connectors.py forge1/backend/forge1/integrations/llamaindex_tools.py forge1/backend/tests/tools/test_automation_connectors.py`

Test outputs are captured in sibling artifact files.
