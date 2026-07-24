"""Conservative, stdlib-only email address value normalizer.

A conservative repair pass for loose email-typed slot values extracted by the
LLM (e.g. "Bob Smith <bob@x.com>", "BOB@X.COM") into bare lowercased
address(es) (e.g. "bob@x.com"). Only touches values that contain a
recognizable email address via a small, fixed regex — it never guesses, never
blanks a value, and leaves anything with no address (display names only,
whole-team references, malformed addresses) exactly as extracted. No
email-validator/dnspython dependency; this is a pure string-extraction layer,
not a real email validator.
"""

import re
from typing import Optional

EMAIL_FINDER = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def normalize_email(value: Optional[str]) -> Optional[str]:
    """Extract and normalize email address(es) from a loose string.

    Idempotent and conservative: values with no recognizable address
    (including None, blank strings, and name-only text) are returned
    unchanged. When one or more addresses are found, they're lowercased,
    de-duplicated (first-occurrence order), and comma-joined; display names
    and surrounding text are dropped. Never blanks a value.
    """
    if value is None:
        return None
    if not value.strip():
        return value
    matches = EMAIL_FINDER.findall(value)
    if not matches:
        return value            # name-only / no address -> unchanged
    seen = set()
    out = []
    for m in matches:
        low = m.lower()
        if low not in seen:
            seen.add(low)
            out.append(low)
    return ", ".join(out)
