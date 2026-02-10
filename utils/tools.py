from typing import Optional
import requests
from langchain.tools import tool

@tool
def curl(url: str, timeout_seconds: Optional[float] = 10.0) -> str:
    """HTTP GET a URL (e.g., wttr.in) and return the response text."""
    # Prepend https:// if caller omits scheme (matches skill examples)
    normalized = url if url.startswith(("http://", "https://")) else f"https://{url}"

    # Try HTTPS first, then fall back to HTTP if it times out (wttr.in allows both)
    schemes = [normalized]
    if normalized.startswith("https://"):
        schemes.append("http://" + normalized[len("https://"):])

    last_error = None
    for target in schemes:
        try:
            resp = requests.get(
                target,
                timeout=timeout_seconds or 20,
                headers={"User-Agent": "langgraph-skill-agent/1.0"},
            )
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.RequestException as exc:  # include timeouts
            last_error = exc
            continue

    # If all attempts fail, surface the last error
    raise last_error if last_error else RuntimeError("curl failed with no error captured")