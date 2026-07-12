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


class R2ArtifactStore:
    """Cloudflare R2 store using the S3-compatible boto3 client."""

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
        prefix: str = "difflet",
        public_base_url: str | None = None,
        client_timeout: float = 60.0,
    ) -> None:
        self.bucket = bucket
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.prefix = prefix.strip("/")
        self.public_base_url = public_base_url.rstrip("/") if public_base_url else None
        self.client_timeout = float(client_timeout)
        self._client_instance = None
        self._client_lock = threading.Lock()

    @classmethod
    def from_env(cls, *, client_timeout: float = 60.0) -> "R2ArtifactStore":
        missing = [
            name
            for name in (
                "DIFFLET_R2_BUCKET",
                "DIFFLET_R2_ENDPOINT_URL",
                "DIFFLET_R2_ACCESS_KEY_ID",
                "DIFFLET_R2_SECRET_ACCESS_KEY",
            )
            if not os.environ.get(name)
        ]
        if missing:
            raise DiffletServingError(
                503,
                "artifact_store_unavailable",
                "R2 artifact store is not configured; missing " + ", ".join(missing),
                error_type="server_error",
            )
        return cls(
            bucket=os.environ["DIFFLET_R2_BUCKET"],
            endpoint_url=os.environ["DIFFLET_R2_ENDPOINT_URL"],
            access_key_id=os.environ["DIFFLET_R2_ACCESS_KEY_ID"],
            secret_access_key=os.environ["DIFFLET_R2_SECRET_ACCESS_KEY"],
            prefix=os.environ.get("DIFFLET_R2_PREFIX", "difflet"),
            public_base_url=os.environ.get("DIFFLET_R2_PUBLIC_BASE_URL"),
            client_timeout=float(os.environ.get("DIFFLET_R2_CLIENT_TIMEOUT", client_timeout)),
        )

    def _client(self):
        if self._client_instance is not None:
            return self._client_instance

        import boto3
        from botocore.config import Config

        with self._client_lock:
            if self._client_instance is None:
                self._client_instance = boto3.client(
                    "s3",
                    endpoint_url=self.endpoint_url,
                    aws_access_key_id=self.access_key_id,
                    aws_secret_access_key=self.secret_access_key,
                    region_name="auto",
                    config=Config(
                        connect_timeout=min(self.client_timeout, 10.0),
                        read_timeout=self.client_timeout,
                        retries={"max_attempts": 2},
                    ),
                )
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
            logger.exception("R2 artifact upload failed")
            raise internal_error("Internal artifact storage error") from exc
        return ArtifactRef(file_id=file_id, uri=f"s3://{self.bucket}/{key}", mime_type=mime_type)

    async def get_url(self, ref: ArtifactRef, *, ttl_seconds: int) -> str:
        key = f"{self.prefix}/{ref.file_id}" if self.prefix else ref.file_id
        if self.public_base_url:
            return f"{self.public_base_url}/{key}"

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
            logger.exception("R2 artifact presign failed")
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
