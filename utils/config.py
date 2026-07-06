import logging
import os
from dotenv import load_dotenv

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# Phase 1 shared-store flag. "chroma" (default) keeps the per-service Chroma store;
# "pg" routes the template store to the shared planner schema (planner.workflow_templates)
# via PLANNER_DATABASE_URL. Off by default — nothing changes until the flag is flipped.
STORE_BACKEND = os.getenv("STORE_BACKEND", "chroma").strip().lower()
PLANNER_DATABASE_URL = os.getenv("PLANNER_DATABASE_URL", "")


def make_template_store():
    """Return the configured template store.

    ``STORE_BACKEND=pg`` → the Postgres-backed ``PGTemplateStore`` (planner schema);
    anything else → the default Chroma-backed ``TemplateStore``. Imports are local so
    the selected backend's dependencies are only touched when it's chosen (the Chroma
    default never imports the pg driver)."""
    if STORE_BACKEND == "pg":
        from utils.pg_template_store import PGTemplateStore

        return PGTemplateStore()
    from utils.template_store import TemplateStore

    return TemplateStore()


def make_workflow_store():
    """Return the configured flat workflow store.

    ``STORE_BACKEND=pg`` → the Postgres-backed ``PGWorkflowStore`` (planner.workflows);
    anything else → the default ``ChromaVectorStore``. Imports are local so the selected
    backend's dependencies are only touched when it's chosen."""
    if STORE_BACKEND == "pg":
        from utils.pg_workflow_store import PGWorkflowStore

        return PGWorkflowStore()
    from utils.chroma import ChromaVectorStore

    return ChromaVectorStore()

# Placeholder used only when OPENAI_API_KEY is unset, so eager client/embedding
# construction at import time can't crash the container on startup (that took down
# the Cloud Run service — the container never listened on PORT). Real LLM/embedding
# calls still fail until a real key is configured; the service just starts and can
# serve /health.
_OPENAI_KEY_PLACEHOLDER = "sk-openai-key-not-configured"


def openai_api_key_or_placeholder() -> str:
    """Return ``OPENAI_API_KEY``, or a non-empty placeholder (with a warning) when
    it is unset, so importing the app never crashes on a missing key."""
    if OPENAI_API_KEY:
        return OPENAI_API_KEY
    logging.getLogger(__name__).warning(
        "OPENAI_API_KEY is not set; starting so /health works, but LLM/embedding "
        "calls will fail until it is configured."
    )
    return _OPENAI_KEY_PLACEHOLDER
