"""Tests for the MCP server. No MLX needed: training is replaced by a fake script."""

import asyncio
import contextlib
import json
import os
import re
import socket
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from mlx_lm_lora import mcp_server as ms

try:
    import mcp  # noqa: F401

    HAS_MCP = True
except ImportError:  # pragma: no cover
    HAS_MCP = False

REPO = Path(__file__).resolve().parents[1]

FAKE_TRAINER = """#!/bin/sh
# Stands in for "python -m mlx_lm_lora.train --config <file>".
printf 'Training:  10%%|#   | 1/10 [00:01<00:09]\\r'
printf 'Training:  50%%|##  | 5/10 [00:05<00:05]\\r'
printf '\\nIter 5: loss 1.234, lr 1.000e-05, it/s 1.0\\n'
printf 'Iter 5: Val loss 1.500, Val took 0.1s\\n'
sleep "${FAKE_SLEEP:-0}"
if [ -n "$FAKE_FAIL" ]; then echo 'ValueError: bad dataset'; exit 1; fi
printf 'Iter 10: loss 0.900, lr 1.000e-05\\n'
printf 'Iter 10: Saved adapter weights to adapters/adapters.safetensors.\\n'
"""


def wait_for(manager, run_id, statuses, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = manager.get(run_id)
        if run["status"] in statuses:
            return run
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} never reached {statuses}: {run['status']}")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.data = root / "data"
        self.data.mkdir()
        (self.data / "train.jsonl").write_text('{"text": "hi"}\n')
        self.fake = root / "fake_python"
        self.fake.write_text(FAKE_TRAINER)
        self.fake.chmod(self.fake.stat().st_mode | stat.S_IEXEC)
        self.manager = ms.RunManager(root / "runs", python=str(self.fake))
        ms._PROCS.clear()

    def tearDown(self):
        for proc in list(ms._PROCS.values()):
            if proc.poll() is None:
                os.killpg(proc.pid, 9)
            proc.wait()
        ms._PROCS.clear()
        os.environ.pop("FAKE_SLEEP", None)
        os.environ.pop("FAKE_FAIL", None)
        self.tmp.cleanup()

    def start(self, **kw):
        kw.setdefault("model", "mlx-community/Qwen3-0.6B-4bit")
        kw.setdefault("data", str(self.data))
        return self.manager.start(**kw)


class ConfigTests(Base):
    def test_extra_config_overrides_are_validated(self):
        cases = [
            ("iters", -1),
            ("batch_size", 0),
            ("max_seq_length", True),
            ("learning_rate", "fast"),
            ("learning_rate", float("nan")),
            ("learning_rate", float("inf")),
            ("train_mode", "unknown"),
            ("train_mode", []),
            ("train_type", "unknown"),
            ("model", ""),
            ("data", None),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                _, errors, _ = ms.build_config(
                    "a/b", str(self.data), extra_config=json.dumps({key: value})
                )
                self.assertTrue(errors)
                with self.assertRaises(ValueError):
                    self.start(extra_config=json.dumps({key: value}))
        self.assertFalse(self.manager.root.exists())

    def test_valid_overrides_replace_invalid_flat_arguments(self):
        cfg, errors, _ = ms.build_config(
            "",
            "",
            train_mode="unknown",
            iters=0,
            extra_config=json.dumps(
                {
                    "model": "a/b",
                    "data": str(self.data),
                    "train_mode": "sft",
                    "iters": 5,
                }
            ),
        )
        self.assertEqual(errors, [])
        self.assertEqual(cfg["iters"], 5)

    def test_extra_config_requires_string_keys(self):
        _, errors, _ = ms.build_config(
            "a/b", str(self.data), extra_config="{1: value, other: value}"
        )
        self.assertIn("extra_config keys must be strings.", errors)

    def test_missing_input_files_are_reported(self):
        for key in ("resume_adapter_file", "reward_functions_file"):
            with self.subTest(key=key):
                _, errors, _ = ms.build_config(
                    "a/b",
                    str(self.data),
                    extra_config=json.dumps({key: str(self.data / "missing")}),
                )
                self.assertTrue(any(key in error for error in errors))

    def test_valid_config_uses_absolute_paths(self):
        cfg, errors, notes = ms.build_config(
            "mlx-community/Qwen3-0.6B-4bit", str(self.data)
        )
        self.assertEqual(errors, [])
        self.assertEqual(cfg["data"], str(self.data.resolve()))
        self.assertTrue(cfg["train"])
        self.assertIn("Hugging Face id", notes[0])

    def test_errors_are_reported(self):
        empty = Path(self.tmp.name) / "empty"
        empty.mkdir()
        _, errors, _ = ms.build_config("", str(empty), train_mode="nope", iters=0)
        text = " ".join(errors)
        for needle in (
            "model is required",
            "Unknown train_mode",
            "iters",
            "no train.jsonl",
        ):
            self.assertIn(needle, text)

    def test_judge_modes_need_a_judge(self):
        _, errors, _ = ms.build_config("a/b", str(self.data), train_mode="online_dpo")
        self.assertTrue(any("judge" in e for e in errors))
        _, errors, _ = ms.build_config(
            "a/b",
            str(self.data),
            train_mode="online_dpo",
            extra_config='{"judge": "c/d"}',
        )
        self.assertEqual(errors, [])

    def test_extra_config_and_quantization(self):
        cfg, errors, notes = ms.build_config(
            "a/b",
            str(self.data),
            quantize_bits=4,
            extra_config='{"beta": 0.2, "adapter_path": "/x", "not_a_key": 1}',
        )
        self.assertEqual(errors, [])
        self.assertEqual(cfg["beta"], 0.2)
        self.assertTrue(cfg["load_in_4bits"])
        self.assertNotIn("adapter_path", cfg)
        self.assertTrue(any("not_a_key" in n for n in notes))
        _, errors, _ = ms.build_config("a/b", str(self.data), extra_config="[1, 2]")
        self.assertTrue(errors)

    def test_modes_and_keys_match_train_py(self):
        source = (REPO / "mlx_lm_lora" / "train.py").read_text()
        block = re.search(r'"--train-mode",.*?choices=\[(.*?)\]', source, re.S).group(1)
        self.assertEqual(set(re.findall(r'"(\w+)"', block)), set(ms.TRAIN_MODES))
        self.assertIn("learning_rate", ms.known_config_keys())

    def test_parse_log(self):
        log = (
            "\x1b[36mLoading\x1b[0m\nTraining: 3/10 [00:01]\rTraining: 4/10 [00:02]\r"
            "\nIter 4: loss 2.500, lr 1e-5\nIter 4: Val loss 2.700, Val took 1s\n"
            "Iter 6:\n----\nLoss: 1.250\nTotal Rewards: 0.5\n"
        )
        info = ms.parse_log(log)
        self.assertEqual(info["iteration"], 6)
        self.assertEqual(info["train_loss"], 1.25)
        self.assertEqual(info["val_loss"], 2.7)
        self.assertEqual((info["bar_step"], info["bar_total"]), (4, 10))


class RunTests(Base):
    def test_extra_config_paths_work_in_trainer_directory(self):
        root = Path(self.tmp.name)
        for name in ("model", "judge", "reference"):
            (root / name).mkdir()
        for name in ("checkpoint.safetensors", "rewards.py"):
            (root / name).touch()
        extra = {
            "model": "model",
            "data": "data",
            "judge": "judge",
            "reference_model_path": "reference",
            "resume_adapter_file": "checkpoint.safetensors",
            "reward_functions_file": "rewards.py",
        }
        # This trainer opens the real configured inputs from the run directory.
        self.fake.write_text(
            f"#!{sys.executable}\n"
            "import sys, yaml\n"
            "from pathlib import Path\n"
            "cfg = yaml.safe_load(Path(sys.argv[-1]).read_text())\n"
            f"for key in {tuple(extra)!r}:\n"
            "    path = Path(cfg[key])\n"
            "    assert path.is_absolute() and path.exists(), (key, path)\n"
            "for key in ('resume_adapter_file', 'reward_functions_file'):\n"
            "    Path(cfg[key]).read_bytes()\n"
        )
        with contextlib.chdir(root):
            run = self.start(extra_config=json.dumps(extra))
        run = wait_for(self.manager, run["id"], {"completed", "failed"})
        self.assertEqual(run["status"], "completed", self.manager.log_tail(run))

    def test_concurrent_server_processes_allow_only_one_run(self):
        root = Path(self.tmp.name)
        gate = root / "start"
        script = """
import sys
import time
from pathlib import Path
from unittest import mock
from mlx_lm_lora import mcp_server as ms

runs, python, data, ready, gate = sys.argv[1:]
original_popen = ms.subprocess.Popen
def delayed_popen(*args, **kwargs):
    # Widen the gap between the active-run check and writing run.json.
    time.sleep(0.3)
    return original_popen(*args, **kwargs)
Path(ready).touch()
while not Path(gate).exists():
    time.sleep(0.01)
with mock.patch.object(ms.subprocess, "Popen", delayed_popen):
    try:
        ms.RunManager(runs, python=python).start(model="a/b", data=data)
    except RuntimeError:
        print("blocked", flush=True)
    else:
        print("started", flush=True)
"""
        env = {**os.environ, "PYTHONPATH": str(REPO), "FAKE_SLEEP": "30"}
        clients = []
        try:
            for index in range(2):
                clients.append(
                    subprocess.Popen(
                        [
                            sys.executable, "-c", script, str(self.manager.root),
                            str(self.fake), str(self.data),
                            str(root / f"ready{index}"), str(gate),
                        ],
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                )
            deadline = time.monotonic() + 10
            while not all((root / f"ready{i}").exists() for i in range(2)):
                if time.monotonic() >= deadline:
                    self.fail("server processes did not become ready")
                time.sleep(0.01)
            gate.touch()
            outcomes = []
            for client in clients:
                output, error = client.communicate(timeout=10)
                self.assertEqual(client.returncode, 0, error)
                outcomes.append(output.strip())
            self.assertCountEqual(outcomes, ["started", "blocked"])
            self.assertEqual(len(self.manager.runs()), 1)
        finally:
            for client in clients:
                if client.poll() is None:
                    client.kill()
                client.communicate()
            # Trainers outlive the short-lived client/server processes.
            for run in self.manager.runs():
                try:
                    os.killpg(run["pid"], 9)
                except ProcessLookupError:
                    pass

    def test_popen_is_detached_from_stdio(self):
        with mock.patch.object(ms.subprocess, "Popen") as popen:
            popen.return_value.pid = 4242
            popen.return_value.poll.return_value = None
            run = self.start()
        ms._PROCS.clear()
        args, kwargs = popen.call_args
        command = args[0]
        self.assertEqual(
            command[-5:-1], [str(self.fake), "-m", "mlx_lm_lora.train", "--config"]
        )
        self.assertTrue(command[-1].endswith("config.yaml"))
        self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
        self.assertTrue(kwargs["start_new_session"])
        self.assertNotIn(kwargs["stdout"], (None, sys.stdout))
        saved = json.loads((Path(run["run_dir"]) / "run.json").read_text())
        self.assertEqual(saved["status"], "running")
        self.assertEqual(saved["pid"], 4242)

    def test_completed_run_and_restart(self):
        run = self.start(name="demo", iters=10)
        run = wait_for(self.manager, run["id"], {"completed", "failed"})
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["exit_code"], 0)
        info = self.manager.progress(run)
        self.assertEqual((info["iteration"], info["train_loss"]), (10, 0.9))
        text = ms.describe_run(run, info)
        self.assertIn("completed", text)
        self.assertIn("iter 10/10 (100%)", text)
        # A fresh manager (server restart) sees the same run by name.
        ms._PROCS.clear()
        again = ms.RunManager(self.manager.root).get("demo")
        self.assertEqual(again["status"], "completed")
        tail = self.manager.log_tail(again, 3)
        self.assertIn("Saved adapter weights", tail)
        self.assertNotIn("\r", tail)

    def test_failed_run(self):
        os.environ["FAKE_FAIL"] = "1"
        run = wait_for(self.manager, self.start()["id"], {"completed", "failed"})
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["exit_code"], 1)
        text = ms.describe_run(run, self.manager.progress(run))
        self.assertIn("ValueError: bad dataset", text)

    def test_stop_and_single_active_run(self):
        os.environ["FAKE_SLEEP"] = "30"
        run = self.start()
        with self.assertRaises(RuntimeError):
            self.start()
        # Status survives a restart while the process is still alive.
        procs = dict(ms._PROCS)
        ms._PROCS.clear()
        self.assertEqual(ms.RunManager(self.manager.root).get()["status"], "running")
        ms._PROCS.update(procs)
        stopped = self.manager.stop(run["id"])
        self.assertEqual(stopped["status"], "stopped")
        self.assertEqual(self.manager.get(run["id"])["status"], "stopped")

    def test_get_errors(self):
        with self.assertRaises(LookupError):
            self.manager.get()


class AuthTests(unittest.TestCase):
    def test_bearer_auth(self):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})

        async def call(headers):
            sent = []

            async def send(msg):
                sent.append(msg)

            await ms.BearerAuth(app, "s3cret")(
                {"type": "http", "headers": headers}, None, send
            )
            return sent[0]["status"]

        self.assertEqual(asyncio.run(call([])), 401)
        self.assertEqual(asyncio.run(call([(b"authorization", b"Bearer nope")])), 401)
        self.assertEqual(asyncio.run(call([(b"authorization", b"Bearer s3cret")])), 200)


def text_of(result):
    return "\n".join(c.text for c in result.content if c.type == "text")


@unittest.skipUnless(HAS_MCP, "mcp is not installed")
class MCPRoundTripTests(Base):
    TOOLS = {
        "calculate_iters",
        "inspect_dataset",
        "validate_model_support",
        "list_training_modes",
        "check_training_config",
        "start_training",
        "training_status",
        "training_log",
        "list_training_runs",
        "stop_training",
    }

    def test_in_process_client(self):
        from mcp import Client

        async def scenario():
            async with Client(ms.create_server(self.manager)) as client:
                empty = await client.call_tool("training_status", {})
                self.assertFalse(empty.is_error)
                self.assertEqual(text_of(empty), "No training runs yet.")
                tools = (await client.list_tools()).tools
                self.assertEqual({t.name for t in tools}, self.TOOLS)
                for tool in tools:
                    self.assertLess(len(tool.description), 600, tool.name)

                modes = await client.call_tool("list_training_modes", {})
                self.assertIn("grpo", text_of(modes))

                for rows, batch_size, epochs, expected in (
                    (101, 8, 3, 39), (100, 10, 2, 20), (1, 8, 1, 1)
                ):
                    budget = await client.call_tool("calculate_iters", {
                        "num_samples": rows, "batch_size": batch_size, "epochs": epochs
                    })
                    self.assertFalse(budget.is_error, text_of(budget))
                    self.assertEqual(budget.structured_content["iters"], expected)
                for key in ("num_samples", "batch_size", "epochs"):
                    for value in (0, -1):
                        args = {"num_samples": 100, "batch_size": 8, "epochs": 3}
                        args[key] = value
                        invalid = await client.call_tool("calculate_iters", args)
                        self.assertTrue(invalid.is_error)

                check = await client.call_tool(
                    "check_training_config", {"model": "a/b", "data": str(self.data)}
                )
                self.assertIn("Config looks good", text_of(check))

                started = await client.call_tool(
                    "start_training",
                    {"model": "a/b", "data": str(self.data), "iters": 10},
                )
                self.assertFalse(started.is_error, text_of(started))
                run_id = started.structured_content["id"]
                wait_for(self.manager, run_id, {"completed"})

                status = await client.call_tool("training_status", {})
                self.assertIn("completed", text_of(status))
                self.assertEqual(status.structured_content["train_loss"], 0.9)

                listed = await client.call_tool("list_training_runs", {})
                self.assertIn(run_id, text_of(listed))
                log = await client.call_tool(
                    "training_log", {"run_id": run_id, "lines": 2}
                )
                self.assertIn("Saved adapter", text_of(log))

                missing = await client.call_tool("training_status", {"run_id": "nope"})
                self.assertTrue(missing.is_error)
                bad = await client.call_tool(
                    "start_training", {"model": "", "data": "x"}
                )
                self.assertTrue(bad.is_error)

        asyncio.run(scenario())

    def _env(self):
        from mcp.client.stdio import get_default_environment

        return {**get_default_environment(), "PYTHONPATH": str(REPO)}

    def test_stdio_subprocess(self):
        from mcp import Client, StdioServerParameters

        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mlx_lm_lora.mcp_server", "--runs-dir", str(self.manager.root)],
            env=self._env(),
        )

        async def scenario():
            async with Client(params) as client:
                tools = (await client.list_tools()).tools
                self.assertEqual({t.name for t in tools}, self.TOOLS)
                runs = await client.call_tool("list_training_runs", {})
                self.assertEqual(text_of(runs), "No training runs yet.")

        asyncio.run(scenario())

    def test_http_with_token(self):
        from mcp import Client
        from mcp.client.streamable_http import (
            create_mcp_http_client,
            streamable_http_client,
        )

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        env = {**self._env(), ms.TOKEN_ENV: "s3cret"}
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mlx_lm_lora.mcp_server",
                "--transport",
                "http",
                "--port",
                str(port),
                "--runs-dir",
                str(self.manager.root),
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        url = f"http://127.0.0.1:{port}/mcp"
        try:
            for _ in range(100):
                try:
                    socket.create_connection(("127.0.0.1", port), 0.1).close()
                    break
                except OSError:
                    time.sleep(0.1)

            async def scenario(headers):
                transport = streamable_http_client(
                    url, http_client=create_mcp_http_client(headers=headers)
                )
                async with Client(transport) as client:
                    return {t.name for t in (await client.list_tools()).tools}

            # Keep a system HTTP proxy away from the local server.
            local = {
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            }
            with mock.patch.dict(os.environ, local):
                good = asyncio.run(scenario({"Authorization": "Bearer s3cret"}))
                self.assertEqual(good, self.TOOLS)
                with self.assertRaises(BaseException):
                    asyncio.run(scenario({"Authorization": "Bearer wrong"}))
        finally:
            proc.terminate()
            proc.wait(10)


if __name__ == "__main__":
    unittest.main()
