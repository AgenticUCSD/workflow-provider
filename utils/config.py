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


def template_near_dup_distance():
    """Optional cosine-distance threshold for *semantic* near-dup template dedup.

    Read at call time so it's easy to toggle/test. Unset or unparseable → ``None``
    = OFF (exact content-hash dedup only, i.e. today's behavior). Set
    ``TEMPLATE_NEAR_DUP_DISTANCE`` to a small value (e.g. ``0.02``) to also collapse
    near-identical templates on create. Default OFF because the right threshold
    needs calibration on real data — a too-large value would wrongly merge distinct
    templates."""
    raw = os.getenv("TEMPLATE_NEAR_DUP_DISTANCE", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


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


def make_instance_store():
    """Return the configured enriched-instance store.

    ``STORE_BACKEND=pg`` → the Postgres-backed ``PGInstanceStore`` (persists to
    planner.enriched_instances); anything else → a no-op store (no persistence, which
    is today's behavior — there is no pre-existing instance store to fall back to)."""
    if STORE_BACKEND == "pg":
        from utils.pg_instance_store import PGInstanceStore

        return PGInstanceStore()
    from utils.pg_instance_store import NullInstanceStore

    return NullInstanceStore()

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
