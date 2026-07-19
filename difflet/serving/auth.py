"""Optional API-key authentication for the HTTP serving surface."""

from __future__ import annotations

import hashlib
import secrets
from typing import Any, Awaitable, Callable

_UNAUTHORIZED_BODY = b'{"error":"Unauthorized"}'


class ApiKeyAuthenticationMiddleware:
    """Require one Bearer token for versioned serving API routes.

    The middleware is installed only when ``difflet serve`` has an API key.
    Health and readiness routes intentionally remain unauthenticated so load
    balancers and process supervisors can probe the service.
    """

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        *,
        api_key: str,
        protected_prefix: str = "/v1",
    ) -> None:
        self.app = app
        self._api_key_hash = hashlib.sha256(api_key.encode("utf-8")).digest()
        self._protected_prefix = protected_prefix.rstrip("/")

    async def __call__(self, scope: dict[str, Any], receive, send) -> None:
        if not self._requires_authentication(scope):
            await self.app(scope, receive, send)
            return

        candidate = self._bearer_token(scope)
        if candidate is not None:
            candidate_hash = hashlib.sha256(candidate.encode("utf-8")).digest()
            if secrets.compare_digest(candidate_hash, self._api_key_hash):
                await self.app(scope, receive, send)
                return

        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(_UNAUTHORIZED_BODY)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": _UNAUTHORIZED_BODY})

    def _requires_authentication(self, scope: dict[str, Any]) -> bool:
        if scope.get("type") != "http" or scope.get("method", "").upper() == "OPTIONS":
            return False
        path = scope.get("path", "")
        root_path = scope.get("root_path", "")
        if root_path and path.startswith(root_path):
            path = path[len(root_path) :]
        return path == self._protected_prefix or path.startswith(f"{self._protected_prefix}/")

    @staticmethod
    def _bearer_token(scope: dict[str, Any]) -> str | None:
        authorization_values = [
            value for name, value in scope.get("headers", ()) if name.lower() == b"authorization"
        ]
        if len(authorization_values) != 1:
            return None
        try:
            authorization = authorization_values[0].decode("latin-1")
        except UnicodeDecodeError:
            return None
        scheme, separator, token = authorization.partition(" ")
        if not separator or scheme.lower() != "bearer" or not token:
            return None
        return token
