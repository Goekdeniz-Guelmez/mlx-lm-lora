"""MCP server for agent-driven, tenant-scoped MLX-LM-LoRA training.

The MCP SDK is an optional dependency. Install it with::

    pip install -U "mlx-lm-lora[mcp]"

The default transport is ``stdio`` so an MCP host can launch this module as a
subprocess. Streamable HTTP is also available for a shared service deployment.
Training jobs are intentionally serialized because MLX training jobs compete
for the same Apple Silicon memory and GPU resources.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import threading
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised in an installation check
    FastMCP = None  # type: ignore[assignment,misc]
    _MCP_IMPORT_ERROR = exc
else:
    _MCP_IMPORT_ERROR = None


LOGGER = logging.getLogger(__name__)
TENANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
JOB_STATES = frozenset({"queued", "running", "succeeded", "failed", "cancelled"})
TRAINING_MODES = (
    "sft",
    "dpo",
    "ftpo",
    "cpo",
    "orpo",
    "grpo",
    "online_dpo",
    "xpo",
    "rlhf_reinforce",
    "ppo",
)
TRAINING_TYPES = ("lora", "dora", "full")


class TenantError(ValueError):
    """Raised when a tenant or tenant-owned path is invalid."""


class JobNotFoundError(FileNotFoundError):
    """Raised when a job does not exist in the requested tenant."""


def _utc_now() -> str:
    """Return an RFC 3339 timestamp in UTC."""

    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically and restrict the file to the current user."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(temporary_path, 0o600)
    os.replace(temporary_path, path)


def _read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON object from ``path``."""

    with path.open(encoding="utf-8") as input_file:
        value = json.load(input_file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def validate_tenant_id(tenant_id: str) -> str:
    """Validate and normalize a tenant identifier.

    Tenant IDs are deliberately narrower than a general filesystem path. This
    makes the tenant ID safe to use as one directory component and keeps the
    tenant boundary easy to audit.
    """

    if not isinstance(tenant_id, str):
        raise TenantError("tenant_id must be a string")
    normalized = tenant_id.strip()
    if not TENANT_ID_PATTERN.fullmatch(normalized):
        raise TenantError(
            "tenant_id must contain 1-64 letters, numbers, '.', '_' or '-' "
            "and must not start with punctuation"
        )
    return normalized


def _resolve_inside(root: Path, value: Path) -> Path:
    """Resolve ``value`` and require it to remain inside ``root``."""

    resolved_root = root.expanduser().resolve(strict=False)
    resolved_value = value.expanduser().resolve(strict=False)
    if resolved_value != resolved_root and resolved_root not in resolved_value.parents:
        raise TenantError(f"Path must stay inside the tenant workspace: {value}")
    return resolved_value


def _resolve_in_roots(value: str, roots: Sequence[Path]) -> Path:
    """Resolve a local path under one of the approved roots."""

    candidate = Path(value).expanduser()
    for root in roots:
        resolved_root = root.expanduser().resolve(strict=False)
        resolved_candidate = candidate.resolve(strict=False)
        if (
            resolved_candidate == resolved_root
            or resolved_root in resolved_candidate.parents
        ):
            return resolved_candidate
    raise TenantError(f"Local path is outside the approved workspace: {value}")


@dataclass(frozen=True)
class ServerSettings:
    """Runtime settings for the MCP server."""

    tenant_root: Path = field(
        default_factory=lambda: Path("~/.mlx-lm-lora/tenants").expanduser()
    )
    tenant_id: Optional[str] = None
    allowed_tenants: Optional[frozenset] = None
    shared_root: Optional[Path] = None
    transport: str = "stdio"
    host: str = "127.0.0.1"
    port: int = 8000
    auth_tokens: Mapping[str, str] = field(default_factory=dict)
    auth_issuer_url: Optional[str] = None
    auth_resource_url: Optional[str] = None
    stateless_http: bool = True
    json_response: bool = True

    def __post_init__(self) -> None:
        """Validate settings at the process boundary."""

        if self.tenant_id is not None:
            validate_tenant_id(self.tenant_id)
        if self.allowed_tenants is not None:
            for allowed_tenant in self.allowed_tenants:
                validate_tenant_id(allowed_tenant)
        if self.transport not in {"stdio", "streamable-http"}:
            raise ValueError("transport must be 'stdio' or 'streamable-http'")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.auth_tokens and self.transport != "streamable-http":
            raise ValueError("bearer-token authentication is only available over HTTP")
        if bool(self.auth_issuer_url) != bool(self.auth_resource_url):
            raise ValueError(
                "auth_issuer_url and auth_resource_url must be configured together"
            )
        for token_tenant in self.auth_tokens.values():
            validate_tenant_id(token_tenant)

    @classmethod
    def from_environment(cls) -> "ServerSettings":
        """Build settings from environment variables.

        Environment variables are used instead of putting tenant roots or
        bearer tokens in an agent's tool arguments. See the README MCP section
        for the full list.
        """

        allowed_value = os.environ.get("MLX_LM_LORA_ALLOWED_TENANTS")
        allowed_tenants = None
        if allowed_value:
            allowed_tenants = frozenset(
                validate_tenant_id(value.strip())
                for value in allowed_value.split(",")
                if value.strip()
            )

        tokens_value = os.environ.get("MLX_LM_LORA_AUTH_TOKENS_JSON", "")
        auth_tokens: Dict[str, str] = {}
        if tokens_value:
            try:
                decoded_tokens = json.loads(tokens_value)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "MLX_LM_LORA_AUTH_TOKENS_JSON must be a JSON object mapping "
                    "bearer tokens to tenant IDs"
                ) from exc
            if not isinstance(decoded_tokens, dict):
                raise ValueError("MLX_LM_LORA_AUTH_TOKENS_JSON must be a JSON object")
            for token, token_tenant in decoded_tokens.items():
                if not isinstance(token, str) or not token:
                    raise ValueError("Authentication token keys must be non-empty strings")
                if not isinstance(token_tenant, str):
                    raise ValueError("Authentication token tenant IDs must be strings")
                auth_tokens[token] = validate_tenant_id(token_tenant)

        tenant_id = os.environ.get("MLX_LM_LORA_TENANT_ID")
        shared_root = os.environ.get("MLX_LM_LORA_SHARED_ROOT")
        return cls(
            tenant_root=Path(
                os.environ.get(
                    "MLX_LM_LORA_TENANT_ROOT", "~/.mlx-lm-lora/tenants"
                )
            ).expanduser(),
            tenant_id=validate_tenant_id(tenant_id) if tenant_id else None,
            allowed_tenants=allowed_tenants,
            shared_root=Path(shared_root).expanduser() if shared_root else None,
            transport=os.environ.get("MLX_LM_LORA_MCP_TRANSPORT", "stdio"),
            host=os.environ.get("MLX_LM_LORA_MCP_HOST", "127.0.0.1"),
            port=int(os.environ.get("MLX_LM_LORA_MCP_PORT", "8000")),
            auth_tokens=auth_tokens,
            auth_issuer_url=os.environ.get("MLX_LM_LORA_AUTH_ISSUER_URL"),
            auth_resource_url=os.environ.get("MLX_LM_LORA_AUTH_RESOURCE_URL"),
            stateless_http=os.environ.get(
                "MLX_LM_LORA_MCP_STATELESS_HTTP", "true"
            ).lower()
            in {"1", "true", "yes"},
            json_response=os.environ.get(
                "MLX_LM_LORA_MCP_JSON_RESPONSE", "true"
            ).lower()
            in {"1", "true", "yes"},
        )


class TenantManager:
    """Own tenant selection and tenant-safe filesystem operations."""

    def __init__(self, settings: ServerSettings):
        self.settings = settings
        self.tenant_root = settings.tenant_root.expanduser().resolve(strict=False)

    def resolve_tenant(
        self,
        requested_tenant_id: Optional[str],
        authenticated_tenant_id: Optional[str] = None,
    ) -> str:
        """Resolve a requested tenant and enforce process/authentication pinning."""

        requested = (
            validate_tenant_id(requested_tenant_id)
            if requested_tenant_id is not None
            else None
        )
        authenticated = (
            validate_tenant_id(authenticated_tenant_id)
            if authenticated_tenant_id is not None
            else None
        )
        configured = self.settings.tenant_id

        if configured and requested and configured != requested:
            raise PermissionError("The MCP process is pinned to another tenant")
        if authenticated and requested and authenticated != requested:
            raise PermissionError("The authenticated principal cannot access this tenant")

        resolved = configured or authenticated or requested
        if resolved is None:
            raise TenantError(
                "tenant_id is required; configure MLX_LM_LORA_TENANT_ID for a "
                "single-tenant agent or pass tenant_id for a shared server"
            )
        if self.settings.allowed_tenants and resolved not in self.settings.allowed_tenants:
            raise PermissionError("Tenant is not allowed by the server policy")
        return resolved

    def workspace(self, tenant_id: str, create: bool = False) -> Path:
        """Return the tenant workspace, optionally creating its directories."""

        tenant_id = validate_tenant_id(tenant_id)
        workspace = _resolve_inside(self.tenant_root, self.tenant_root / tenant_id)
        if create:
            for directory in (
                workspace,
                workspace / "inputs",
                workspace / "runs",
                workspace / "artifacts",
            ):
                directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        return workspace

    def tenant_path(self, tenant_id: str, value: str) -> Path:
        """Resolve a ``tenant://`` path or relative path in a tenant workspace."""

        workspace = self.workspace(tenant_id, create=False)
        if value.startswith("tenant://"):
            relative_value = value[len("tenant://") :]
        else:
            relative_value = value
        if not relative_value or Path(relative_value).is_absolute():
            raise TenantError("Tenant paths must contain a relative path")
        return _resolve_inside(workspace, workspace / relative_value)

    def approved_input_roots(self, tenant_id: str) -> Tuple[Path, ...]:
        """Return roots allowed for local model/dataset inputs."""

        roots = [self.workspace(tenant_id, create=False)]
        if self.settings.shared_root is not None:
            roots.append(self.settings.shared_root.expanduser().resolve(strict=False))
        return tuple(roots)


def _is_local_reference(value: str) -> bool:
    """Return whether a user value is explicitly a local filesystem reference."""

    path = Path(value).expanduser()
    return (
        value.startswith("tenant://")
        or path.is_absolute()
        or value.startswith("./")
        or value.startswith("../")
        or path.exists()
    )


def _normalize_local_reference(
    value: Any, label: str, tenant_id: str, tenants: TenantManager
) -> Any:
    """Normalize an explicit local input without changing Hub identifiers."""

    if not isinstance(value, str) or not _is_local_reference(value):
        return value
    if value.startswith("tenant://") or value.startswith("./") or value.startswith("../"):
        return str(tenants.tenant_path(tenant_id, value))
    return str(_resolve_in_roots(value, tenants.approved_input_roots(tenant_id)))


def _training_config_keys() -> frozenset:
    """Return config keys supported by the existing training entry point."""

    from . import train

    parser = train.build_parser()
    parser_keys = {
        action.dest for action in parser._actions if action.dest != "help"  # pylint: disable=protected-access
    }
    return frozenset(parser_keys | set(train.CONFIG_DEFAULTS))


def normalize_training_config(
    config: Mapping[str, Any],
    tenant_id: str,
    tenants: TenantManager,
    job_id: str,
) -> Dict[str, Any]:
    """Validate and normalize a training request for one tenant.

    The returned mapping can be passed directly to ``mlx_lm_lora.train.main``.
    Output paths are always absolute paths under the selected tenant workspace.
    Model and dataset Hub identifiers remain unchanged; explicit local inputs
    must be under the tenant workspace or the optional shared read-only root.
    """

    if not isinstance(config, Mapping):
        raise ValueError("config must be a JSON object")
    config_keys = set(config)
    unsupported_keys = sorted(config_keys - set(_training_config_keys()))
    if unsupported_keys:
        raise ValueError(
            "Unsupported training config keys: " + ", ".join(unsupported_keys)
        )
    if "config" in config:
        raise ValueError(
            "The config-file option is not accepted by MCP; pass training "
            "options directly in the config object"
        )

    normalized = dict(config)
    model = normalized.get("model")
    data = normalized.get("data")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("config.model is required")
    if not isinstance(data, str) or not data.strip():
        raise ValueError("config.data is required")

    tenants.workspace(tenant_id, create=True)
    normalized["model"] = _normalize_local_reference(
        model, "model", tenant_id, tenants
    )
    normalized["data"] = _normalize_local_reference(data, "data", tenant_id, tenants)

    for key in ("reference_model_path", "judge"):
        if key in normalized and normalized[key] is not None:
            normalized[key] = _normalize_local_reference(
                normalized[key], key, tenant_id, tenants
            )
    for key in ("resume_adapter_file", "reward_functions_file"):
        if key in normalized and normalized[key] is not None:
            value = normalized[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty local path")
            normalized[key] = str(
                _resolve_in_roots(
                    value,
                    tenants.approved_input_roots(tenant_id),
                )
                if not value.startswith(("tenant://", "./", "../"))
                else tenants.tenant_path(tenant_id, value)
            )

    workspace = tenants.workspace(tenant_id, create=True)
    adapter_path = normalized.get("adapter_path")
    if adapter_path is None or adapter_path == "":
        normalized["adapter_path"] = str(workspace / "artifacts" / job_id)
    else:
        if not isinstance(adapter_path, str):
            raise ValueError("adapter_path must be a path string")
        normalized["adapter_path"] = str(
            _resolve_inside(workspace, workspace / adapter_path)
            if not Path(adapter_path).expanduser().is_absolute()
            else _resolve_inside(workspace, Path(adapter_path))
        )

    if normalized.get("lm_studio_name"):
        raise ValueError(
            "lm_studio_name is not supported by MCP because it writes outside "
            "the tenant workspace; use adapter_path instead"
        )
    normalized["train"] = True
    return normalized


@dataclass
class JobRecord:
    """Persisted metadata for a training job."""

    job_id: str
    tenant_id: str
    status: str
    submitted_at: str
    run_dir: str
    adapter_path: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable job record."""

        return asdict(self)


class TrainingJobManager:
    """Queue and persist tenant-scoped training jobs."""

    def __init__(self, tenants: TenantManager) -> None:
        self.tenants = tenants
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-train")
        self._futures: Dict[Tuple[str, str], Future] = {}
        self._lock = threading.RLock()

    def start(self, tenant_id: str, config: Mapping[str, Any]) -> JobRecord:
        """Validate, persist, and enqueue a training job."""

        job_id = uuid.uuid4().hex
        normalized = normalize_training_config(config, tenant_id, self.tenants, job_id)
        workspace = self.tenants.workspace(tenant_id, create=True)
        run_dir = workspace / "runs" / job_id
        run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        record = JobRecord(
            job_id=job_id,
            tenant_id=tenant_id,
            status="queued",
            submitted_at=_utc_now(),
            run_dir=str(run_dir),
            adapter_path=str(normalized["adapter_path"]),
        )
        _write_json(run_dir / "request.json", normalized)
        _write_json(run_dir / "status.json", record.as_dict())

        with self._lock:
            future = self._executor.submit(self._run, record, normalized)
            self._futures[(tenant_id, job_id)] = future
        return record

    def _persist(self, record: JobRecord) -> None:
        """Persist the current job status."""

        _write_json(Path(record.run_dir) / "status.json", record.as_dict())

    def _run(self, record: JobRecord, config: Mapping[str, Any]) -> None:
        """Run one training request and persist its terminal state."""

        record.status = "running"
        record.started_at = _utc_now()
        self._persist(record)
        log_path = Path(record.run_dir) / "training.log"
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                    from . import train

                    train.main(dict(config))
        except Exception as exc:  # The worker boundary must persist failures.
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
            with log_path.open("a", encoding="utf-8") as log_file:
                traceback.print_exc(file=log_file)
            LOGGER.exception("Training job %s failed", record.job_id)
        else:
            record.status = "succeeded"
        finally:
            record.completed_at = _utc_now()
            self._persist(record)

    def get(self, tenant_id: str, job_id: str) -> JobRecord:
        """Load a job only from the requested tenant's workspace."""

        if not JOB_ID_PATTERN.fullmatch(job_id):
            raise JobNotFoundError("Invalid job_id")
        run_dir = self.tenants.workspace(tenant_id, create=False) / "runs" / job_id
        run_dir = _resolve_inside(self.tenants.workspace(tenant_id), run_dir)
        status_path = run_dir / "status.json"
        if not status_path.is_file():
            raise JobNotFoundError(f"No training job {job_id!r} for this tenant")
        value = _read_json(status_path)
        if value.get("tenant_id") != tenant_id:
            raise PermissionError("Job does not belong to the requested tenant")
        return JobRecord(**value)

    def list(self, tenant_id: str, limit: int = 20) -> Sequence[JobRecord]:
        """List the most recent jobs belonging to one tenant."""

        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        runs_dir = self.tenants.workspace(tenant_id, create=False) / "runs"
        if not runs_dir.is_dir():
            return []
        records = []
        for status_path in runs_dir.glob("*/status.json"):
            try:
                safe_status_path = _resolve_inside(
                    self.tenants.workspace(tenant_id), status_path
                )
                value = _read_json(safe_status_path)
                if value.get("tenant_id") == tenant_id:
                    records.append(JobRecord(**value))
            except (OSError, TypeError, ValueError, TenantError):
                LOGGER.warning("Skipping malformed job status %s", status_path)
        records.sort(key=lambda item: item.submitted_at, reverse=True)
        return records[:limit]

    def log_tail(self, tenant_id: str, job_id: str, max_chars: int = 12000) -> str:
        """Return a bounded tail of a tenant's training log."""

        if not 1 <= max_chars <= 50000:
            raise ValueError("max_chars must be between 1 and 50000")
        record = self.get(tenant_id, job_id)
        log_path = _resolve_inside(Path(record.run_dir), Path(record.run_dir) / "training.log")
        if not log_path.is_file():
            return ""
        return log_path.read_text(encoding="utf-8", errors="replace")[-max_chars:]

    def cancel(self, tenant_id: str, job_id: str) -> JobRecord:
        """Cancel a queued job, if it has not started running yet."""

        record = self.get(tenant_id, job_id)
        with self._lock:
            future = self._futures.get((tenant_id, job_id))
            if future is None or not future.cancel():
                if record.status == "running":
                    raise RuntimeError(
                        "The job is already running; cancellation is only supported "
                        "before MLX training starts"
                    )
                return record
            record.status = "cancelled"
            record.completed_at = _utc_now()
            self._persist(record)
            return record


def _authenticated_tenant_id() -> Optional[str]:
    """Return the tenant claim attached by the MCP HTTP auth middleware."""

    try:
        from mcp.server.auth.middleware.auth_context import get_access_token
    except ImportError:
        return None
    access_token = get_access_token()
    if access_token is None:
        return None
    claims = getattr(access_token, "claims", None) or {}
    tenant_id = claims.get("tenant_id") or getattr(access_token, "client_id", None)
    return validate_tenant_id(tenant_id) if tenant_id else None


class StaticTokenVerifier:
    """Minimal bearer-token verifier for local/shared deployments.

    Production deployments should usually replace this with an OAuth/JWT
    verifier at the identity-provider or reverse-proxy layer. The static
    mapping is useful for a small private service and keeps tokens out of MCP
    tool arguments.
    """

    def __init__(self, token_to_tenant: Mapping[str, str]) -> None:
        self._token_to_tenant = dict(token_to_tenant)

    async def verify_token(self, token: str) -> Any:
        """Return an MCP access token for a configured bearer token."""

        tenant_id = self._token_to_tenant.get(token)
        if tenant_id is None:
            return None
        from mcp.server.auth.provider import AccessToken

        return AccessToken(
            token=token,
            client_id=tenant_id,
            subject=tenant_id,
            scopes=["mlx-lm-lora:read", "mlx-lm-lora:write"],
            claims={"tenant_id": tenant_id},
        )


def _build_auth_options(settings: ServerSettings) -> Dict[str, Any]:
    """Build official MCP SDK HTTP auth options for static bearer tokens."""

    if not settings.auth_tokens:
        return {}
    if not settings.auth_issuer_url or not settings.auth_resource_url:
        raise ValueError(
            "Static bearer auth requires MLX_LM_LORA_AUTH_ISSUER_URL and "
            "MLX_LM_LORA_AUTH_RESOURCE_URL"
        )
    try:
        from mcp.server.auth.settings import AuthSettings
    except ImportError as exc:  # pragma: no cover - depends on installed SDK version
        raise RuntimeError(
            "The installed MCP SDK does not provide HTTP auth support; upgrade "
            "the optional dependency with pip install -U 'mlx-lm-lora[mcp]'"
        ) from exc
    return {
        "auth": AuthSettings(
            issuer_url=settings.auth_issuer_url,
            resource_server_url=settings.auth_resource_url,
            required_scopes=["mlx-lm-lora:read"],
        ),
        "token_verifier": StaticTokenVerifier(settings.auth_tokens),
    }


def create_server(settings: Optional[ServerSettings] = None) -> Any:
    """Create and register the MLX-LM-LoRA MCP server.

    Keeping construction in a function avoids importing the MCP SDK or
    creating filesystem state when the normal training CLI is imported.
    """

    if FastMCP is None:
        raise RuntimeError(
            "MCP support is not installed. Install it with "
            "pip install -U 'mlx-lm-lora[mcp]'"
        ) from _MCP_IMPORT_ERROR
    settings = settings or ServerSettings.from_environment()
    tenants = TenantManager(settings)
    jobs = TrainingJobManager(tenants)
    server = FastMCP(
        "mlx-lm-lora",
        instructions=(
            "Use mlx_lm_lora_get_capabilities first. Validate a complete training "
            "config, then start a job and poll its status. Every operation is "
            "tenant-scoped."
        ),
        **_build_auth_options(settings),
    )

    def resolve_tenant(tenant_id: Optional[str]) -> str:
        """Resolve the tenant for an MCP tool call."""

        return tenants.resolve_tenant(tenant_id, _authenticated_tenant_id())

    @server.tool()
    def mlx_lm_lora_get_capabilities() -> Dict[str, Any]:
        """Describe supported training modes and MCP tenant behavior."""

        return {
            "server": "mlx-lm-lora",
            "training_modes": list(TRAINING_MODES),
            "training_types": list(TRAINING_TYPES),
            "transports": ["stdio", "streamable-http"],
            "job_behavior": "Training jobs are queued and run one at a time.",
            "tenant_behavior": (
                "Each tenant has isolated inputs, run metadata, logs, and artifacts. "
                "Use tenant:// paths for local tenant files."
            ),
            "required_config": ["model", "data"],
        }

    @server.tool()
    def mlx_lm_lora_validate_training_config(
        config: Dict[str, Any], tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate a training config without starting a job."""

        selected_tenant = resolve_tenant(tenant_id)
        try:
            normalized = normalize_training_config(
                config, selected_tenant, tenants, "validation-preview"
            )
        except (TenantError, ValueError, PermissionError) as exc:
            return {"valid": False, "tenant_id": selected_tenant, "errors": [str(exc)]}
        return {
            "valid": True,
            "tenant_id": selected_tenant,
            "normalized_config": normalized,
            "notes": [
                "The train flag is forced to true when a job is started.",
                "The default adapter_path is isolated under the tenant artifact directory.",
            ],
        }

    @server.tool()
    def mlx_lm_lora_start_training(
        config: Dict[str, Any], tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Queue a tenant-scoped MLX-LM-LoRA training job."""

        selected_tenant = resolve_tenant(tenant_id)
        record = jobs.start(selected_tenant, config)
        return record.as_dict()

    @server.tool()
    def mlx_lm_lora_get_training_status(
        job_id: str, tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get status and artifact locations for one tenant's training job."""

        selected_tenant = resolve_tenant(tenant_id)
        return jobs.get(selected_tenant, job_id).as_dict()

    @server.tool()
    def mlx_lm_lora_list_training_runs(
        tenant_id: Optional[str] = None, limit: int = 20
    ) -> Sequence[Dict[str, Any]]:
        """List recent training jobs for one tenant."""

        selected_tenant = resolve_tenant(tenant_id)
        return [record.as_dict() for record in jobs.list(selected_tenant, limit)]

    @server.tool()
    def mlx_lm_lora_get_training_log(
        job_id: str, tenant_id: Optional[str] = None, max_chars: int = 12000
    ) -> Dict[str, Any]:
        """Read a bounded tail of one tenant's training log."""

        selected_tenant = resolve_tenant(tenant_id)
        return {
            "job_id": job_id,
            "tenant_id": selected_tenant,
            "log_tail": jobs.log_tail(selected_tenant, job_id, max_chars),
        }

    @server.tool()
    def mlx_lm_lora_cancel_training(
        job_id: str, tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Cancel a queued job before MLX training begins."""

        selected_tenant = resolve_tenant(tenant_id)
        return jobs.cancel(selected_tenant, job_id).as_dict()

    return server


def build_parser() -> argparse.ArgumentParser:
    """Build the MCP server command-line parser."""

    parser = argparse.ArgumentParser(
        description="Expose MLX-LM-LoRA training through the Model Context Protocol."
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default=None,
        help="Transport used by the MCP host (default: MLX_LM_LORA_MCP_TRANSPORT or stdio).",
    )
    parser.add_argument("--host", default=None, help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=None, help="HTTP bind port.")
    parser.add_argument("--tenant-root", type=Path, default=None)
    parser.add_argument("--tenant-id", default=None)
    parser.add_argument(
        "--allowed-tenants",
        default=None,
        help="Comma-separated tenant allow-list; otherwise use the environment value.",
    )
    parser.add_argument("--shared-root", type=Path, default=None)
    parser.add_argument(
        "--insecure-http",
        action="store_true",
        help="Allow HTTP on a non-loopback host without bearer-token auth (development only).",
    )
    parser.add_argument("--stateful-http", action="store_true")
    parser.add_argument("--sse-json-response", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _settings_from_args(args: argparse.Namespace) -> ServerSettings:
    """Merge CLI overrides into environment-backed settings."""

    environment_settings = ServerSettings.from_environment()
    allowed_tenants = environment_settings.allowed_tenants
    if args.allowed_tenants is not None:
        allowed_tenants = frozenset(
            validate_tenant_id(value)
            for value in args.allowed_tenants.split(",")
            if value.strip()
        )
    settings = ServerSettings(
        tenant_root=args.tenant_root or environment_settings.tenant_root,
        tenant_id=(
            validate_tenant_id(args.tenant_id)
            if args.tenant_id is not None
            else environment_settings.tenant_id
        ),
        allowed_tenants=allowed_tenants,
        shared_root=args.shared_root or environment_settings.shared_root,
        transport=args.transport or environment_settings.transport,
        host=args.host or environment_settings.host,
        port=args.port or environment_settings.port,
        auth_tokens=environment_settings.auth_tokens,
        auth_issuer_url=environment_settings.auth_issuer_url,
        auth_resource_url=environment_settings.auth_resource_url,
        stateless_http=not args.stateful_http,
        json_response=args.sse_json_response or environment_settings.json_response,
    )
    if (
        settings.transport == "streamable-http"
        and settings.host not in {"127.0.0.1", "localhost", "::1"}
        and not settings.auth_tokens
        and not args.insecure_http
    ):
        raise ValueError(
            "Refusing unauthenticated HTTP on a non-loopback host. Configure "
            "MLX_LM_LORA_AUTH_TOKENS_JSON or pass --insecure-http for development."
        )
    return settings


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the MCP server from the terminal."""

    args = build_parser().parse_args(argv)
    settings = _settings_from_args(args)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    server = create_server(settings)
    if settings.transport == "stdio":
        server.run(transport="stdio")
        return
    server.run(
        transport="streamable-http",
        host=settings.host,
        port=settings.port,
        stateless_http=settings.stateless_http,
        json_response=settings.json_response,
    )


if __name__ == "__main__":
    main()
