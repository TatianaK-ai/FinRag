"""
Auth and spend guards for the HTTP surface.

Every /api/ask call costs real money against the operator's OpenAI key - several
model calls plus embeddings - so an open endpoint is a billing liability before it
is a data one. These bound who can spend and how fast; the filings themselves are
public SEC documents.
"""
from __future__ import annotations

import hmac
import os
import time

API_KEY = os.getenv("APP_API_KEY") or None
PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))
PER_DAY = int(os.getenv("RATE_LIMIT_PER_DAY", "500"))


def auth_enabled() -> bool:
    return bool(API_KEY)


def bind_host() -> str:
    """
    With no key configured the server must not be reachable off-box. Refusing to
    bind beyond loopback is safer than serving an open, billable endpoint.
    """
    return os.getenv("BIND_HOST", "0.0.0.0") if API_KEY else "127.0.0.1"


def key_matches(presented: str | None) -> bool:
    # compare_digest is constant-time, so a wrong key cannot be found byte-by-byte.
    if not API_KEY or not presented:
        return False
    return hmac.compare_digest(presented, API_KEY)


_buckets: dict[str, dict] = {}
_daily = {"count": 0, "reset_at": time.time() + 86_400}


def check_rate(client: str) -> tuple[bool, str, int]:
    """
    Two independent limits: per-client-per-minute bounds one caller hammering the
    endpoint; per-day bounds total spend across every caller against one API key.
    In-memory and therefore per-process - a multi-instance deployment needs a
    shared store.
    """
    now = time.time()

    if now > _daily["reset_at"]:
        _daily.update(count=0, reset_at=now + 86_400)
    if _daily["count"] >= PER_DAY:
        return False, f"daily request cap of {PER_DAY} reached; this bounds API spend", int(_daily["reset_at"] - now)

    b = _buckets.get(client)
    if not b or now > b["reset_at"]:
        _buckets[client] = {"count": 1, "reset_at": now + 60}
    elif b["count"] >= PER_MINUTE:
        return False, f"rate limit of {PER_MINUTE} requests/minute exceeded", int(b["reset_at"] - now)
    else:
        b["count"] += 1

    _daily["count"] += 1
    # Drop expired buckets so the map cannot grow without bound.
    for k in [k for k, v in _buckets.items() if now > v["reset_at"]]:
        _buckets.pop(k, None)
    return True, "", 0


def guard_status(port: int) -> list[str]:
    if API_KEY:
        return [f"auth: enabled (APP_API_KEY set) - limits {PER_MINUTE}/min, {PER_DAY}/day"]
    return [
        "! APP_API_KEY is not set - binding to 127.0.0.1 only.",
        "! Anyone who can reach this port can spend your OpenAI credit.",
        f"! Set APP_API_KEY=<secret> to enable auth and listen on {os.getenv('BIND_HOST', '0.0.0.0')}:{port}.",
    ]
