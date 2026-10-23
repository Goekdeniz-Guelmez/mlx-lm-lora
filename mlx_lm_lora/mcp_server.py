"""MCP server for mlx-lm-lora: start, watch and stop training runs from an agent.

Each run is ``python -m mlx_lm_lora.train --config <run>/config.yaml`` started as
a detached background process. Everything about a run lives in its own folder
under the runs root (default ``~/.mlx-lm-lora/runs``)::

    <run_id>/config.yaml   generated training config
    <run_id>/run.json      run state (survives server restarts)
    <run_id>/train.log     stdout + stderr of the training process
    <run_id>/adapters/     adapters, checkpoints and (if fused) the merged model

Run it with ``mlx-lm-lora-mcp`` (stdio) or
``mlx-lm-lora-mcp --transport http --port 8765``. Requires ``pip install "mlx-lm-lora[mcp]"``.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hmac
import json
import math
import os
import re
import secrets
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

TOKEN_ENV = "MLX_LM_LORA_MCP_TOKEN"
RUNS_DIR_ENV = "MLX_LM_LORA_RUNS_DIR"
PYTHON_ENV = "MLX_LM_LORA_PYTHON"
DEFAULT_RUNS_DIR = "~/.mlx-lm-lora/runs"

TRAIN_MODES = {
    "sft": "Supervised fine-tuning. Data rows: messages, prompt/completion or text.",
    "dpo": "Direct Preference Optimization. Data rows: prompt, chosen, rejected.",
    "cpo": "Contrastive Preference Optimization. Data rows: prompt, chosen, rejected.",
    "orpo": "Odds Ratio Preference Optimization. Data rows: prompt, chosen, rejected.",
    "dsla": "Latent alignment on preference pairs. Data rows: prompt, chosen, rejected.",
    "ftpo": "Final-token preference optimization (Antidoom data).",
    "grpo": "Group Relative Policy Optimization with reward functions. Data rows: prompt, answer.",
    "klpo": "KL-regularized policy optimization with rewards. Data rows: prompt, answer.",
    "online_dpo": "Online DPO, a judge model scores samples. Data rows: prompt. Needs judge.",
    "xpo": "Exploratory preference optimization. Data rows: prompt. Needs judge.",
    "rlhf_reinforce": "RLHF with REINFORCE. Data rows: prompt. Needs judge.",
    "ppo": "Proximal Policy Optimization. Data rows: prompt. Needs judge.",
}
JUDGE_MODES = {"online_dpo", "xpo", "rlhf_reinforce", "ppo"}
TRAIN_TYPES = ("lora", "dora", "full")
OPTIMIZERS = ("adam", "adamw", "muon")
QUANT_KEYS = {4: "load_in_4bits", 6: "load_in_6bits", 8: "load_in_8bits"}
ACTIVE = {"starting", "running"}

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_ITER = re.compile(r"\bIter (\d+):")
_VAL_LOSS = re.compile(r"\bVal loss[=: ]\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", re.I)
_LOSS = re.compile(r"\bloss[=: ]\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", re.I)
_BAR = re.compile(r"(\d+)/(\d+) \[")
_HF_ID = re.compile(r"^[\w.-]+/[\w.-]+$")

# Runs started by this process, so finished children get reaped (no zombies).
_PROCS: dict[str, subprocess.Popen] = {}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def known_config_keys() -> set[str]:
    """Keys of ``train.CONFIG_DEFAULTS``, read from source so MLX is not imported."""
    try:
        tree = ast.parse((Path(__file__).parent / "train.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "CONFIG_DEFAULTS" for t in node.targets
            ):
                return {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    except (OSError, SyntaxError):
        pass
    return set()


def reward_functions() -> list[str]:
    try:
        from .trainer.grpo_reward_functions import list_available_reward_functions

        return sorted(list_available_reward_functions())
    except Exception:  # pragma: no cover - optional information only
        return []


def _resolve_source(value: str, what: str, errors: list[str], notes: list[str]) -> str:
    """Local paths become absolute; anything shaped like owner/name is a Hub id."""
    if not isinstance(value, str):
        errors.append(f"{what} must be a string.")
        return ""
    value = value.strip()
    if not value:
        errors.append(f"{what} is required.")
        return value
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())
    if _HF_ID.match(value) and not value.startswith((".", "~", "/")):
        notes.append(
            f"{what} '{value}' is not a local path; using it as a Hugging Face id."
        )
        return value
    errors.append(
        f"{what} '{value}' was not found locally and is not a Hugging Face id."
    )
    return value


def build_config(
    model: str,
    data: str,
    train_mode: str = "sft",
    train_type: str = "lora",
    iters: int = 100,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    num_layers: int = 16,
    max_seq_length: int = 2048,
    quantize_bits: int = 0,
    fuse: bool = False,
    extra_config: str = "",
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Return ``(config, errors, notes)`` for a training run. Paths are absolute."""
    errors: list[str] = []
    notes: list[str] = []
    cfg: dict[str, Any] = {
        "model": model,
        "data": data,
        "train": True,
        "train_mode": train_mode,
        "train_type": train_type,
        "iters": iters,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_layers": num_layers,
        "max_seq_length": max_seq_length,
        "fuse": fuse,
    }
    if quantize_bits:
        if quantize_bits not in QUANT_KEYS:
            errors.append("quantize_bits must be 0, 4, 6 or 8.")
        else:
            cfg[QUANT_KEYS[quantize_bits]] = True

    if extra_config.strip():
        try:
            extra = yaml.safe_load(extra_config)
        except yaml.YAMLError as exc:
            extra = None
            errors.append(f"extra_config is not valid JSON/YAML: {exc}")
        if not isinstance(extra, dict):
            errors.append(
                'extra_config must be a JSON/YAML object, e.g. {"beta": 0.1}.'
            )
        elif any(not isinstance(key, str) for key in extra):
            errors.append("extra_config keys must be strings.")
        elif extra:
            known = known_config_keys() | {"wandb"}
            unknown = sorted(k for k in extra if known and k not in known)
            if unknown:
                notes.append(
                    f"Unknown config keys passed through as-is: {', '.join(unknown)}."
                )
            for key in ("adapter_path", "config"):
                if key in extra:
                    notes.append(
                        f"Ignored '{key}' in extra_config; the server manages it."
                    )
                    extra.pop(key)
            cfg.update(extra)

    # Validate the final config, including overrides supplied by extra_config.
    for key in ("model", "data"):
        cfg[key] = _resolve_source(cfg[key], key, errors, notes)
    mode = cfg["train_mode"]
    if not isinstance(mode, str) or mode not in TRAIN_MODES:
        errors.append(
            f"Unknown train_mode '{mode}'. Use one of: {', '.join(TRAIN_MODES)}."
        )
    if cfg["train_type"] not in TRAIN_TYPES:
        errors.append(f"train_type must be one of: {', '.join(TRAIN_TYPES)}.")
    for key in ("iters", "batch_size", "max_seq_length"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            errors.append(f"{key} must be a positive integer.")
    rate = cfg["learning_rate"]
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        errors.append("learning_rate must be a finite number greater than 0.")

    # The trainer runs in its own directory, so resolve input paths here.
    for key in ("judge", "reference_model_path"):
        if cfg.get(key) is not None:
            cfg[key] = _resolve_source(cfg[key], key, errors, notes)
    for key in ("resume_adapter_file", "reward_functions_file"):
        value = cfg.get(key)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{key} must be a nonempty file path.")
            continue
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            errors.append(f"{key} '{value}' was not found or is not a file.")
        cfg[key] = str(path)

    if isinstance(mode, str) and mode in JUDGE_MODES and not cfg.get("judge"):
        errors.append(
            f"train_mode '{cfg['train_mode']}' needs a judge model: "
            'pass extra_config {"judge": "<model id or path>"}.'
        )
    data_path = Path(cfg["data"])
    if data_path.is_dir() and not (data_path / "train.jsonl").exists():
        errors.append(f"Data folder {data_path} has no train.jsonl.")
    return cfg, errors, notes


def parse_log(text: str) -> dict[str, Any]:
    """Pull iteration, losses and errors out of a training log."""
    text = _ANSI.sub("", text)
    info: dict[str, Any] = {}
    for line in re.split(r"[\r\n]+", text):
        if m := _ITER.search(line):
            info["iteration"] = int(m.group(1))
        if m := _BAR.search(line):
            info["bar_step"], info["bar_total"] = int(m.group(1)), int(m.group(2))
        if m := _VAL_LOSS.search(line):
            info["val_loss"] = float(m.group(1))
        elif "Saved adapter weights" in line:
            info["last_save"] = line.strip()
        elif m := _LOSS.search(line):
            info["train_loss"] = float(m.group(1))
        if line.startswith("Traceback") or re.match(
            r"^\w*(Error|Exception)\b.*:", line
        ):
            info["error"] = line.strip()
    return info


def _read_tail(path: Path, max_bytes: int = 256_000) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(max(0, path.stat().st_size - max_bytes))
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunManager:
    """Training runs on disk. No MCP or MLX imports, so it is easy to test."""

    def __init__(self, runs_dir: str | Path | None = None, python: str | None = None):
        root = runs_dir or os.environ.get(RUNS_DIR_ENV) or DEFAULT_RUNS_DIR
        self.root = Path(root).expanduser().resolve()
        self.python = python or os.environ.get(PYTHON_ENV) or sys.executable

    # -- state -------------------------------------------------------------
    def _save(self, run: dict[str, Any]) -> None:
        path = self.root / run["id"] / "run.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(run, indent=2))
        tmp.replace(path)

    def _load(self, run_id: str) -> dict[str, Any] | None:
        try:
            return json.loads((self.root / run_id / "run.json").read_text())
        except (OSError, ValueError):
            return None

    def _refresh(self, run: dict[str, Any]) -> dict[str, Any]:
        """Settle the status of a run whose process may have ended."""
        if run["status"] not in ACTIVE:
            return run
        proc = _PROCS.get(run["id"])
        alive = proc.poll() is None if proc else _pid_alive(run.get("pid"))
        if alive:
            return run
        _PROCS.pop(run["id"], None)
        exit_file = self.root / run["id"] / "exit_code"
        code = None
        if exit_file.exists():
            try:
                code = int(exit_file.read_text().strip())
            except ValueError:
                pass
        run["exit_code"] = code
        run["ended_at"] = run.get("ended_at") or _now()
        if run.get("stop_requested"):
            run["status"] = "stopped"
        elif code == 0:
            run["status"] = "completed"
        else:
            run["status"] = "failed"
        self._save(run)
        return run

    def runs(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        found = [self._load(p.name) for p in self.root.iterdir() if p.is_dir()]
        runs = [self._refresh(r) for r in found if r]
        return sorted(runs, key=lambda r: r["created_at"], reverse=True)

    def get(self, run_id: str = "") -> dict[str, Any]:
        """Find a run by id or name; empty or 'latest' means the newest run."""
        runs = self.runs()
        if not runs:
            raise LookupError("No training runs yet.")
        if run_id in ("", "latest"):
            return runs[0]
        for run in runs:
            if run_id in (run["id"], run.get("name")):
                return run
        matches = [r for r in runs if r["id"].startswith(run_id)]
        if len(matches) == 1:
            return matches[0]
        raise LookupError(f"No run '{run_id}'. Use list_training_runs to see run ids.")

    # -- actions -----------------------------------------------------------
    def start(self, name: str = "", **options: Any) -> dict[str, Any]:
        cfg, errors, notes = build_config(**options)
        if errors:
            raise ValueError(" ".join(errors))
        self.root.mkdir(parents=True, exist_ok=True)
        # Separate MCP clients can run separate server processes. Keep the check,
        # launch and initial state write under one filesystem lock shared by all.
        with open(self.root / ".start.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._start(name, cfg, notes)

    def _start(
        self, name: str, cfg: dict[str, Any], notes: list[str]
    ) -> dict[str, Any]:
        active = [r for r in self.runs() if r["status"] in ACTIVE]
        if active:
            raise RuntimeError(
                f"Run {active[0]['id']} is still running. Stop it first with stop_training."
            )
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_id = f"{stamp}-{cfg['train_mode']}-{secrets.token_hex(2)}"
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True)
        cfg["adapter_path"] = str(run_dir / "adapters")
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

        command = [
            self.python,
            "-m",
            "mlx_lm_lora.train",
            "--config",
            str(run_dir / "config.yaml"),
        ]
        # sh records the exit code, so the outcome is known even after a server restart.
        wrapper = [
            "/bin/sh",
            "-c",
            '"$@"; rc=$?; echo $rc > exit_code; exit $rc',
            "sh",
            *command,
        ]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        with open(run_dir / "train.log", "ab") as log:
            proc = subprocess.Popen(
                wrapper,
                cwd=run_dir,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,  # own process group; never touches MCP stdio
            )
        _PROCS[run_id] = proc
        run = {
            "id": run_id,
            "name": name.strip() or None,
            "status": "running",
            "pid": proc.pid,
            "created_at": _now(),
            "ended_at": None,
            "exit_code": None,
            "stop_requested": False,
            "train_mode": cfg["train_mode"],
            "train_type": cfg["train_type"],
            "model": cfg["model"],
            "data": cfg["data"],
            "iters": cfg["iters"],
            "run_dir": str(run_dir),
            "adapter_path": cfg["adapter_path"],
            "log_path": str(run_dir / "train.log"),
            "command": command,
            "notes": notes,
        }
        self._save(run)
        return run

    def stop(self, run_id: str, wait_s: float = 5.0) -> dict[str, Any]:
        run = self.get(run_id)
        if run["status"] not in ACTIVE:
            return run
        run["stop_requested"] = True
        self._save(run)
        try:
            os.killpg(run["pid"], signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and self._refresh(run)["status"] in ACTIVE:
            time.sleep(0.2)
        if run["status"] in ACTIVE:
            try:
                os.killpg(run["pid"], signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            time.sleep(0.2)
            self._refresh(run)
        return run

    def progress(self, run: dict[str, Any]) -> dict[str, Any]:
        return parse_log(_read_tail(Path(run["log_path"])))

    def log_tail(self, run: dict[str, Any], lines: int = 30) -> str:
        text = _ANSI.sub("", _read_tail(Path(run["log_path"]), 64_000))
        # tqdm redraws with \r; keep only the last redraw of each line.
        rows = [line.split("\r")[-1].rstrip() for line in text.split("\n")]
        rows = [r for r in rows if r]
        return "\n".join(rows[-max(1, min(lines, 200)) :])[-6000:]


# -- text for small models --------------------------------------------------
def describe_run(run: dict[str, Any], info: dict[str, Any]) -> str:
    label = f"{run['id']}" + (f" ({run['name']})" if run.get("name") else "")
    parts = [f"Run {label}: {run['status']}"]
    step = info.get("iteration") or info.get("bar_step")
    total = run.get("iters") or info.get("bar_total")
    if step:
        parts.append(
            f"iter {step}/{total} ({100 * step // total}%)" if total else f"iter {step}"
        )
    if "train_loss" in info:
        parts.append(f"train loss {info['train_loss']:.4g}")
    if "val_loss" in info:
        parts.append(f"val loss {info['val_loss']:.4g}")
    text = ", ".join(parts) + "."
    text += f"\nMode {run['train_mode']}/{run['train_type']}, model {run['model']}."
    if run["status"] == "failed":
        reason = info.get("error") or f"exit code {run.get('exit_code')}"
        text += f"\nFailed: {reason}. Use training_log to see the end of the log."
    if run["status"] in ("completed", "stopped") or info.get("last_save"):
        text += f"\nAdapters: {run['adapter_path']}"
    return text


def summary(run: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "id",
        "name",
        "status",
        "train_mode",
        "train_type",
        "model",
        "iters",
        "created_at",
        "ended_at",
        "exit_code",
        "adapter_path",
        "log_path",
    )
    out = {k: run.get(k) for k in keep}
    out.update(
        {
            k: info[k]
            for k in ("iteration", "train_loss", "val_loss", "error")
            if k in info
        }
    )
    return out


# -- MCP server --------------------------------------------------------------
def create_server(manager: RunManager | None = None):
    """Build the MCP server. Imports the ``mcp`` SDK only here."""
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
    from mcp.types import CallToolResult, TextContent, ToolAnnotations

    runs = manager or RunManager()
    server = MCPServer(
        "mlx-lm-lora",
        instructions=(
            "Fine-tune LLMs on Apple silicon. Call check_training_config before "
            "start_training, then training_status to follow the run."
        ),
    )
    read_only = ToolAnnotations(read_only_hint=True)

    def result(text: str, data: Any):
        return CallToolResult(
            content=[TextContent(type="text", text=text)],
            structured_content=data if isinstance(data, dict) else {"items": data},
        )

    def find(run_id: str) -> dict[str, Any] | None:
        """The run, None when there are no runs at all, ToolError for a bad id."""
        if not runs.runs():
            return None
        try:
            return runs.get(run_id)
        except LookupError as exc:
            raise ToolError(str(exc)) from exc

    @server.tool(annotations=read_only)
    def list_training_modes() -> str:
        """List the training modes, train types and GRPO reward functions you can use."""
        lines = [f"- {mode}: {desc}" for mode, desc in TRAIN_MODES.items()]
        rewards = reward_functions()
        return (
            "Training modes:\n"
            + "\n".join(lines)
            + f"\nTrain types: {', '.join(TRAIN_TYPES)} (lora is the usual choice)."
            + f"\nOptimizers (extra_config 'optimizer'): {', '.join(OPTIMIZERS)}."
            + (f"\nGRPO reward functions: {', '.join(rewards)}." if rewards else "")
            + "\nData is a folder with train.jsonl (and valid.jsonl) or a Hugging Face dataset id."
        )

    @server.tool(annotations=read_only)
    def inspect_dataset(data: str, model: str, train_mode: str = "sft",
                        extra_config: str = ""):
        """Inspect dataset access and validate rows with the training dataset loader.

        Loads the tokenizer but no model weights. Errors block training.
        Uses the same feature mappings and dataset settings as training.
        """
        try:
            from types import SimpleNamespace
            from transformers import AutoTokenizer
            from .trainer.datasets import load_dataset

            cfg, errors, _ = build_config(
                model, data, train_mode=train_mode, extra_config=extra_config
            )
            if errors:
                raise ValueError("; ".join(errors))
            cfg.setdefault("test", False)
            tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=False)
            splits = load_dataset(SimpleNamespace(**cfg), tokenizer)
            sizes = {}
            for name, split in zip(("train", "valid", "test"), splits):
                sizes[name] = len(split)
                for index in range(len(split)):
                    split[index]
            if not sizes["train"]:
                raise ValueError("Training dataset is empty.")
            return result("Dataset inspection passed.", {"valid": True, "rows": sizes})
        except Exception as exc:
            raise ToolError(f"Dataset inspection failed: {exc}") from exc

    @server.tool(annotations=read_only)
    def validate_model_support(model: str):
        """Validate model access and architecture against installed MLX-LM.

        Reads config only, without downloading weights or executing remote code.
        This checks architecture support, not GPU fit or successful training.
        """
        try:
            from huggingface_hub import hf_hub_download
            from mlx_lm.utils import _get_classes

            path = Path(model).expanduser()
            config_path = path / "config.json" if path.is_dir() else Path(
                hf_hub_download(model, "config.json")
            )
            config = json.loads(config_path.read_text())
            _get_classes(config)
            return result("Model support validation passed.", {
                "valid": True, "model": model, "model_type": config.get("model_type")
            })
        except Exception as exc:
            raise ToolError(f"Model support validation failed: {exc}") from exc

    @server.tool(annotations=read_only)
    def check_training_config(
        model: str,
        data: str,
        train_mode: str = "sft",
        train_type: str = "lora",
        iters: int = 100,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        num_layers: int = 16,
        max_seq_length: int = 2048,
        quantize_bits: int = 0,
        fuse: bool = False,
        extra_config: str = "",
    ) -> str:
        """Check training settings without starting anything. Same arguments as start_training."""
        cfg, errors, notes = build_config(
            model,
            data,
            train_mode,
            train_type,
            iters,
            batch_size,
            learning_rate,
            num_layers,
            max_seq_length,
            quantize_bits,
            fuse,
            extra_config,
        )
        head = (
            "Config has problems:\n- " + "\n- ".join(errors)
            if errors
            else "Config looks good."
        )
        if notes:
            head += "\nNotes:\n- " + "\n- ".join(notes)
        return head + "\n\nconfig.yaml:\n" + yaml.safe_dump(cfg, sort_keys=False)

    @server.tool()
    def start_training(
        model: str,
        data: str,
        train_mode: str = "sft",
        train_type: str = "lora",
        iters: int = 100,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        num_layers: int = 16,
        max_seq_length: int = 2048,
        quantize_bits: int = 0,
        fuse: bool = False,
        extra_config: str = "",
        name: str = "",
    ):
        """Start a fine-tuning run in the background and return its run id.

        model: Hugging Face id or local path. data: folder with train.jsonl or a
        Hugging Face dataset id. quantize_bits: 0, 4, 6 or 8 (QLoRA). fuse: merge
        adapters into a full model at the end. extra_config: JSON object with any
        other mlx_lm_lora.train key, e.g. {"beta": 0.1}. name: optional label.
        """
        try:
            run = runs.start(
                name=name,
                model=model,
                data=data,
                train_mode=train_mode,
                train_type=train_type,
                iters=iters,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_layers=num_layers,
                max_seq_length=max_seq_length,
                quantize_bits=quantize_bits,
                fuse=fuse,
                extra_config=extra_config,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            raise ToolError(str(exc)) from exc
        text = (
            f"Started run {run['id']} ({run['train_mode']}/{run['train_type']}, "
            f"{run['iters']} iters) on {run['model']}.\n"
            f"Adapters will be saved to {run['adapter_path']}.\n"
            "Use training_status to follow it."
        )
        if run["notes"]:
            text += "\nNotes: " + " ".join(run["notes"])
        return result(text, summary(run, {}))

    @server.tool(annotations=read_only)
    def training_status(run_id: str = ""):
        """Show progress and loss of a run. Leave run_id empty for the latest run."""
        run = find(run_id)
        if run is None:
            return "No training runs yet."
        info = runs.progress(run)
        return result(describe_run(run, info), summary(run, info))

    @server.tool(annotations=read_only)
    def training_log(run_id: str = "", lines: int = 30) -> str:
        """Show the last lines of a run's training log. Leave run_id empty for the latest run."""
        run = find(run_id)
        if run is None:
            return "No training runs yet."
        return runs.log_tail(run, lines) or f"Run {run['id']} has no log output yet."

    @server.tool(annotations=read_only)
    def list_training_runs(limit: int = 10):
        """List recent training runs, newest first, with their status."""
        items = runs.runs()[: max(1, min(limit, 50))]
        if not items:
            return result("No training runs yet.", [])
        rows = []
        for run in items:
            info = runs.progress(run)
            loss = f", loss {info['train_loss']:.4g}" if "train_loss" in info else ""
            rows.append(
                f"- {run['id']}{' (' + run['name'] + ')' if run.get('name') else ''}: "
                f"{run['status']}, {run['train_mode']} on {run['model']}{loss}"
            )
        return result("\n".join(rows), [summary(r, {}) for r in items])

    @server.tool(annotations=ToolAnnotations(destructive_hint=True))
    def stop_training(run_id: str = "") -> str:
        """Stop a running training run. Saved checkpoints are kept. Empty run_id means the latest run."""
        run = find(run_id)
        if run is None:
            return "No training runs yet."
        if run["status"] not in ACTIVE:
            return f"Run {run['id']} is not running (status: {run['status']})."
        run = runs.stop(run["id"])
        return f"Run {run['id']} is now {run['status']}. Checkpoints so far are in {run['adapter_path']}."

    return server


class BearerAuth:
    """Tiny ASGI middleware: require ``Authorization: Bearer <token>``."""

    def __init__(self, app: Any, token: str):
        self.app, self.token = app, token.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            given = headers.get(b"authorization", b"")
            if not hmac.compare_digest(given, b"Bearer " + self.token):
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            (b"content-type", b"text/plain"),
                            (b"www-authenticate", b"Bearer"),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": b"Unauthorized"})
                return
        await self.app(scope, receive, send)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MCP server for mlx-lm-lora training.")
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--runs-dir",
        help=f"Where runs are stored (env {RUNS_DIR_ENV}, default {DEFAULT_RUNS_DIR}).",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="Extra Host header allowed over HTTP, e.g. 192.168.1.20:*",
    )
    args = parser.parse_args(argv)

    try:
        server = create_server(RunManager(args.runs_dir))
    except ImportError:
        sys.exit('The MCP server needs the mcp package: pip install "mlx-lm-lora[mcp]"')

    if args.transport == "stdio":
        server.run(transport="stdio")
        return

    import uvicorn
    from mcp.server.transport_security import TransportSecuritySettings

    hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*", *args.allowed_host]
    if args.host not in ("127.0.0.1", "localhost", "::1", "0.0.0.0", "::"):
        hosts.append(f"{args.host}:*")
    app = server.streamable_http_app(
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(allowed_hosts=hosts),
    )
    token = os.environ.get(TOKEN_ENV, "")
    if token:
        app = BearerAuth(app, token)
    elif args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"Warning: serving on {args.host} without {TOKEN_ENV}; anyone on the network can start training.",
            file=sys.stderr,
        )
    print(
        f"mlx-lm-lora MCP server on http://{args.host}:{args.port}/mcp", file=sys.stderr
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
