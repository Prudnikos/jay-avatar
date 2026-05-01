"""
HMAC-signed token validation.

The Cloudflare Worker signs short-lived tokens (5 min TTL) using a shared
secret (JAY_AVATAR_TOKEN_SECRET) and the GPU pod verifies them.

Token format (compact, no dependency on JWT library on Worker side):
    payload.signature
    where payload = base64url(JSON({sid, slug, exp}))
          signature = base64url(HMAC-SHA256(payload, secret))

Tokens are single-use in spirit — we just validate `exp`, not enforce
single use, since a session is itself short-lived.
"""
import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Optional

from config import CFG


@dataclass
class TokenPayload:
    sid: str            # session id (uuid)
    slug: str           # business_slug
    exp: int            # unix epoch seconds
    avatar: str = "default"


class TokenError(Exception):
    pass


def _b64url_decode(s: str) -> bytes:
    pad = 4 - (len(s) % 4)
    if pad != 4:
        s += "=" * pad
    return base64.urlsafe_b64decode(s)


def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def verify_token(token: str) -> TokenPayload:
    """Verify a Worker-issued token. Returns payload or raises TokenError."""
    if not CFG.token_secret:
        raise TokenError("Server misconfigured: token secret not set")
    if not token or "." not in token:
        raise TokenError("Malformed token")

    try:
        payload_b64, sig_b64 = token.split(".", 1)
    except ValueError:
        raise TokenError("Malformed token")

    expected_sig = hmac.new(
        CFG.token_secret.encode(),
        payload_b64.encode(),
        hashlib.sha256,
    ).digest()

    try:
        provided_sig = _b64url_decode(sig_b64)
    except Exception:
        raise TokenError("Malformed signature")

    if not hmac.compare_digest(expected_sig, provided_sig):
        raise TokenError("Invalid signature")

    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception:
        raise TokenError("Malformed payload")

    if not isinstance(payload, dict):
        raise TokenError("Malformed payload")
    for key in ("sid", "slug", "exp"):
        if key not in payload:
            raise TokenError(f"Missing field: {key}")

    if int(payload["exp"]) < int(time.time()):
        raise TokenError("Token expired")

    return TokenPayload(
        sid=str(payload["sid"]),
        slug=str(payload["slug"]),
        exp=int(payload["exp"]),
        avatar=str(payload.get("avatar", "default")),
    )


# ── For local dev / testing only ─────────────────────────────────────────
def issue_dev_token(sid: str, slug: str, ttl_s: int = 300, avatar: str = "default") -> str:
    """Issue a token using the local secret. Use only in dev."""
    payload = {
        "sid": sid,
        "slug": slug,
        "avatar": avatar,
        "exp": int(time.time()) + ttl_s,
    }
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(CFG.token_secret.encode(), payload_b64.encode(), hashlib.sha256).digest()
    return f"{payload_b64}.{_b64url_encode(sig)}"
