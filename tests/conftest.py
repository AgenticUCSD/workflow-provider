"""Pytest configuration for workflow-provider tests.

Never emit or flush real traces during tests. deepeval auto-loads the repo `.env`
(which may carry a CONFIDENT_API_KEY) on import, which would otherwise make the
retrieval-span `traced(...)` helper active and flush to Confident AI. Force
tracing off for the whole test session. setdefault keeps an explicit
CONFIDENT_TRACING from the caller respected.
"""

import os

os.environ.setdefault("CONFIDENT_TRACING", "false")
