import logging
import os
from dotenv import load_dotenv

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

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
