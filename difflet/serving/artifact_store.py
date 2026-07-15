"""Artifact storage backends for generated media."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid
from dataclasses import dataclass
from typing import Protocol

from difflet.serving.errors import DiffletServingError
from difflet.serving.errors import internal_error

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactRef:
    file_id: str
    uri: str
    mime_type: str


class ArtifactStore(Protocol):
    async def put_bytes(
        self,
        *,
        data: bytes,
        mime_type: str,
        suffix: str,
        ttl_seconds: int,
    ) -> ArtifactRef: ...

    async def get_url(self, ref: ArtifactRef, *, ttl_seconds: int) -> str: ...


class MemoryArtifactStore:
    """In-memory store for unit tests and local development."""

    def __init__(self, *, base_url: str = "memory://difflet") -> None:
        self.base_url = base_url.rstrip("/")
        self._objects: dict[str, bytes] = {}

    async def put_bytes(
        self,
        *,
        data: bytes,
        mime_type: str,
        suffix: str,
        ttl_seconds: int,
    ) -> ArtifactRef:
        file_id = f"{uuid.uuid4().hex}{suffix}"
        self._objects[file_id] = bytes(data)
        return ArtifactRef(file_id=file_id, uri=f"{self.base_url}/{file_id}", mime_type=mime_type)

    async def get_url(self, ref: ArtifactRef, *, ttl_seconds: int) -> str:
        return f"{self.base_url}/{ref.file_id}"

    def get_bytes(self, file_id: str) -> bytes:
        return self._objects[file_id]


class S3ArtifactStore:
    """Private S3-compatible store using presigned URLs for object access."""

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
        prefix: str = "difflet",
        region_name: str | None = None,
        addressing_style: str | None = None,
        client_timeout: float = 60.0,
    ) -> None:
        if bool(access_key_id) != bool(secret_access_key):
            raise ValueError("S3 access key ID and secret access key must be configured together")
        if session_token and not access_key_id:
            raise ValueError("S3 session token requires explicit access key credentials")
        if addressing_style not in (None, "auto", "virtual", "path"):
            raise ValueError("S3 addressing style must be auto, virtual, or path")
        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.session_token = session_token
        self.prefix = prefix.strip("/")
        self.region_name = region_name
        self.addressing_style = addressing_style or ("virtual" if endpoint_url is None else "auto")
        self.client_timeout = float(client_timeout)
        self._client_instance = None
        self._client_lock = threading.Lock()

    @classmethod
    def from_env(cls, *, client_timeout: float = 60.0) -> "S3ArtifactStore":
        missing = []
        if not (os.environ.get("DIFFLET_S3_BUCKET") or "").strip():
            missing.append("DIFFLET_S3_BUCKET")

        access_key_id = (os.environ.get("DIFFLET_S3_ACCESS_KEY_ID") or "").strip()
        secret_access_key = (os.environ.get("DIFFLET_S3_SECRET_ACCESS_KEY") or "").strip()
        session_token = (os.environ.get("DIFFLET_S3_SESSION_TOKEN") or "").strip()
        if access_key_id or secret_access_key or session_token:
            if not access_key_id:
                missing.append("DIFFLET_S3_ACCESS_KEY_ID")
            if not secret_access_key:
                missing.append("DIFFLET_S3_SECRET_ACCESS_KEY")
        if missing:
            raise DiffletServingError(
                503,
                "artifact_store_unavailable",
                "S3 artifact store is not configured; missing " + ", ".join(missing),
                error_type="server_error",
            )
        return cls(
            bucket=os.environ["DIFFLET_S3_BUCKET"].strip(),
            endpoint_url=(os.environ.get("DIFFLET_S3_ENDPOINT_URL") or "").strip() or None,
            access_key_id=access_key_id or None,
            secret_access_key=secret_access_key or None,
            session_token=session_token or None,
            prefix=os.environ.get("DIFFLET_S3_PREFIX", "difflet"),
            region_name=(
                (os.environ.get("DIFFLET_S3_REGION") or "").strip()
                or (os.environ.get("AWS_REGION") or "").strip()
                or (os.environ.get("AWS_DEFAULT_REGION") or "").strip()
                or None
            ),
            addressing_style=(os.environ.get("DIFFLET_S3_ADDRESSING_STYLE") or "").strip() or None,
            client_timeout=float(os.environ.get("DIFFLET_S3_CLIENT_TIMEOUT", client_timeout)),
        )

    @classmethod
    def from_env_if_configured(cls, *, client_timeout: float = 60.0) -> "S3ArtifactStore | None":
        selectors = (
            "DIFFLET_S3_BUCKET",
            "DIFFLET_S3_ENDPOINT_URL",
            "DIFFLET_S3_REGION",
            "DIFFLET_S3_ACCESS_KEY_ID",
            "DIFFLET_S3_SECRET_ACCESS_KEY",
            "DIFFLET_S3_SESSION_TOKEN",
        )
        configured = [name for name in selectors if name in os.environ]
        if not configured:
            return None
        return cls.from_env(client_timeout=client_timeout)

    def _client(self):
        if self._client_instance is not None:
            return self._client_instance

        import boto3
        from botocore.config import Config

        with self._client_lock:
            if self._client_instance is None:
                client_options = {
                    "region_name": self.region_name,
                    "config": Config(
                        signature_version="s3v4",
                        connect_timeout=min(self.client_timeout, 10.0),
                        read_timeout=self.client_timeout,
                        retries={"max_attempts": 2},
                        s3={"addressing_style": self.addressing_style},
                    ),
                }
                if self.endpoint_url is not None:
                    client_options["endpoint_url"] = self.endpoint_url
                if self.access_key_id is not None:
                    client_options["aws_access_key_id"] = self.access_key_id
                    client_options["aws_secret_access_key"] = self.secret_access_key
                    if self.session_token is not None:
                        client_options["aws_session_token"] = self.session_token
                self._client_instance = boto3.client("s3", **client_options)
        return self._client_instance

    async def put_bytes(
        self,
        *,
        data: bytes,
        mime_type: str,
        suffix: str,
        ttl_seconds: int,
    ) -> ArtifactRef:
        file_id = f"{uuid.uuid4().hex}{suffix}"
        key = f"{self.prefix}/{file_id}" if self.prefix else file_id

        def _put() -> None:
            client = self._client()
            client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=mime_type,
            )

        try:
            await asyncio.to_thread(_put)
        except Exception as exc:
            logger.exception("S3 artifact upload failed")
            raise internal_error("Internal artifact storage error") from exc
        return ArtifactRef(file_id=file_id, uri=f"s3://{self.bucket}/{key}", mime_type=mime_type)

    async def get_url(self, ref: ArtifactRef, *, ttl_seconds: int) -> str:
        key = f"{self.prefix}/{ref.file_id}" if self.prefix else ref.file_id

        def _sign() -> str:
            client = self._client()
            return client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=int(ttl_seconds),
            )

        try:
            return await asyncio.to_thread(_sign)
        except Exception as exc:
            logger.exception("S3 artifact presign failed")
            raise internal_error("Internal artifact storage error") from exc


async def put_with_timeout(
    store: ArtifactStore,
    *,
    data: bytes,
    mime_type: str,
    suffix: str,
    ttl_seconds: int,
    timeout_s: float,
) -> ArtifactRef:
    try:
        return await asyncio.wait_for(
            store.put_bytes(
                data=data,
                mime_type=mime_type,
                suffix=suffix,
                ttl_seconds=ttl_seconds,
            ),
            timeout=float(timeout_s),
        )
    except DiffletServingError:
        raise
    except asyncio.TimeoutError as exc:
        raise DiffletServingError(
            502,
            "artifact_upload_failed",
            "artifact upload timed out",
            error_type="server_error",
        ) from exc


async def get_url_with_timeout(
    store: ArtifactStore,
    ref: ArtifactRef,
    *,
    ttl_seconds: int,
    timeout_s: float,
) -> str:
    try:
        return await asyncio.wait_for(
            store.get_url(ref, ttl_seconds=ttl_seconds),
            timeout=float(timeout_s),
        )
    except DiffletServingError:
        raise
    except asyncio.TimeoutError as exc:
        raise DiffletServingError(
            502,
            "artifact_upload_failed",
            "artifact presign timed out",
            error_type="server_error",
        ) from exc
