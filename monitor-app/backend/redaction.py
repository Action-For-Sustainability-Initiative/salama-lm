"""Privacy/redaction utility.

Everything that leaves the collectors passes through redact() before it is
broadcast or recorded. We redact by PATTERN, not by allowlist, so new secret
shapes still get caught:
  - key=value pairs whose key names a credential
  - bearer/authorization headers
  - long high-entropy tokens (hex/base64-ish >= 24 chars)
We never collect environment variable values, file contents, or request
bodies in the first place — redaction is the second line of defence.
"""

from __future__ import annotations

import re

_PATTERNS = [
    re.compile(r"(?i)((?:api[-_]?key|token|secret|passw(?:or)?d|auth|credential|bearer)[=:\s]+)(\S+)"),
    re.compile(r"(?i)(--?(?:api[-_]?key|token|secret|password|auth)[= ])(\S+)"),
    re.compile(r"\b(hf_|sk-|ghp_|gho_|xox[bp]-)[A-Za-z0-9_\-]{10,}"),
    re.compile(r"\b[A-Fa-f0-9]{32,}\b"),
    re.compile(r"\b[A-Za-z0-9+/_\-]{40,}={0,2}\b"),
]


def redact(text: str) -> str:
    if not text:
        return text
    for pat in _PATTERNS[:2]:
        text = pat.sub(lambda m: m.group(1) + "[REDACTED]", text)
    for pat in _PATTERNS[2:]:
        text = pat.sub("[REDACTED]", text)
    return text


def redact_obj(obj):
    """Recursively redact every string in a JSON-like structure."""
    if isinstance(obj, str):
        return redact(obj)
    if isinstance(obj, list):
        return [redact_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    return obj
