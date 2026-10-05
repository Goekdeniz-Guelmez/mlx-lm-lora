import ast
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import mlx_lm_lora
from mlx_lm_lora import mcp


class McpTenantTest(unittest.TestCase):
    def make_settings(self, root):
        return mcp.ServerSettings(
            tenant_root=Path(root),
            allowed_tenants=frozenset({"acme", "beta"}),
        )

    def test_tenant_ids_reject_path_traversal(self):
        with self.assertRaises(mcp.TenantError):
            mcp.validate_tenant_id("../other-tenant")

    def test_pinned_tenant_cannot_be_overridden(self):
        settings = mcp.ServerSettings(
            tenant_root=Path("/tmp/tenants"), tenant_id="acme"
        )
        tenants = mcp.TenantManager(settings)
        with self.assertRaises(PermissionError):
            tenants.resolve_tenant("beta")

    def test_pinned_tenant_rejects_other_authenticated_tenant(self):
        settings = mcp.ServerSettings(
            tenant_root=Path("/tmp/tenants"), tenant_id="acme"
        )
        tenants = mcp.TenantManager(settings)
        with self.assertRaises(PermissionError):
            tenants.resolve_tenant(None, "beta")

    def test_tenant_path_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as root:
            tenants = mcp.TenantManager(self.make_settings(root))
            with self.assertRaises(mcp.TenantError):
                tenants.tenant_path("acme", "tenant://../beta/secrets.json")

    def test_training_config_is_tenant_scoped(self):
        with tempfile.TemporaryDirectory() as root:
            tenants = mcp.TenantManager(self.make_settings(root))
            normalized = mcp.normalize_training_config(
                {
                    "model": "org/model",
                    "data": "org/dataset",
                    "train_mode": "sft",
                },
                "acme",
                tenants,
                "a" * 32,
            )
            adapter_path = Path(normalized["adapter_path"])
            self.assertEqual(adapter_path.parent.name, "artifacts")
            self.assertEqual(adapter_path.parent.parent.name, "acme")
            self.assertTrue(normalized["train"])

    def test_training_config_rejects_external_adapter_path(self):
        with tempfile.TemporaryDirectory() as root:
            tenants = mcp.TenantManager(self.make_settings(root))
            with self.assertRaises(mcp.TenantError):
                mcp.normalize_training_config(
                    {
                        "model": "org/model",
                        "data": "org/dataset",
                        "adapter_path": "../escape",
                    },
                    "acme",
                    tenants,
                    "a" * 32,
                )

    def test_training_config_requires_hugging_face_dataset_repo(self):
        with tempfile.TemporaryDirectory() as root:
            tenants = mcp.TenantManager(self.make_settings(root))
            for dataset in (
                "tenant://inputs/train.jsonl",
                "./train.jsonl",
                "https://huggingface.co/datasets/org/dataset",
            ):
                with self.subTest(dataset=dataset), self.assertRaises(ValueError):
                    mcp.normalize_training_config(
                        {
                            "model": "org/model",
                            "data": dataset,
                            "train_mode": "sft",
                        },
                        "acme",
                        tenants,
                        "a" * 32,
                    )

    def test_training_config_rejects_invalid_cli_values(self):
        with tempfile.TemporaryDirectory() as root:
            tenants = mcp.TenantManager(self.make_settings(root))
            with self.assertRaises(ValueError):
                mcp.normalize_training_config(
                    {
                        "model": "org/model",
                        "data": "org/dataset",
                        "train_mode": "not-a-mode",
                    },
                    "acme",
                    tenants,
                    "a" * 32,
                )

    def test_http_auth_settings_are_tenant_validated(self):
        with tempfile.TemporaryDirectory() as root:
            settings = mcp.ServerSettings(
                tenant_root=Path(root),
                transport="streamable-http",
                auth_tokens={"secret-token": "acme"},
                auth_issuer_url="https://issuer.example",
                auth_resource_url="https://mcp.example/mcp",
            )
            self.assertEqual(settings.auth_tokens["secret-token"], "acme")

    def test_install_skill_copies_complete_skill_for_each_target(self):
        with tempfile.TemporaryDirectory() as root:
            home = Path(root)
            for target in ("codex", "claude", "hermes"):
                with self.subTest(target=target):
                    destination = mcp.install_skill(target, home_dir=home)
                    self.assertEqual(
                        destination,
                        home / f".{target}" / "skills" / "mlx_lm_lora",
                    )
                    self.assertTrue((destination / "SKILL.md").is_file())
                    self.assertTrue(
                        (destination / "references" / "config.md").is_file()
                    )

    def test_install_skill_updates_existing_files_without_removing_other_skills(self):
        with tempfile.TemporaryDirectory() as root:
            home = Path(root)
            destination = home / ".codex" / "skills" / "mlx_lm_lora"
            destination.mkdir(parents=True)
            (destination / "old-file.md").write_text("keep", encoding="utf-8")
            (home / ".codex" / "skills" / "other-skill").mkdir(parents=True)

            installed = mcp.install_skill("codex", home_dir=home)

            self.assertEqual(installed, destination)
            self.assertEqual(
                (destination / "old-file.md").read_text(encoding="utf-8"), "keep"
            )
            self.assertTrue((home / ".codex" / "skills" / "other-skill").is_dir())

    def test_install_skill_rejects_unknown_target(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                mcp.install_skill("unknown", home_dir=Path(root))

    def test_parser_accepts_skill_install_target(self):
        args = mcp.build_parser().parse_args(["--install-skill", "codex"])
        self.assertEqual(args.install_skill, "codex")

    def test_server_defaults_to_streaming_http_on_port_8008(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            settings = mcp.ServerSettings.from_environment()

        self.assertEqual(settings.transport, "streamable-http")
        self.assertEqual(settings.host, "127.0.0.1")
        self.assertEqual(settings.port, 8008)
        self.assertFalse(settings.json_response)

    @mock.patch.object(mcp, "create_server")
    def test_http_startup_logs_complete_mcp_endpoint(self, create_server):
        with self.assertLogs(mcp.LOGGER, level="INFO") as logs:
            mcp.main(
                [
                    "--transport",
                    "streamable-http",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8765",
                    "--tenant-id",
                    "test",
                ]
            )

        create_server.return_value.run.assert_called_once_with(
            transport="streamable-http",
            host="127.0.0.1",
            port=8765,
            stateless_http=True,
            json_response=mock.ANY,
        )
        self.assertIn(
            "MCP Streamable HTTP endpoint: http://127.0.0.1:8765/mcp",
            "\n".join(logs.output),
        )


class McpBackendFeatureTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings = mcp.ServerSettings(
            tenant_root=Path(self.directory.name), tenant_id="acme"
        )
        self.tenants = mcp.TenantManager(self.settings)

    def normalize(self, **options):
        return mcp.normalize_training_config(
            {"model": "org/model", "data": "org/dataset", **options},
            "acme",
            self.tenants,
            "a" * 32,
        )

    def server_tools(self):
        class ToolServer:
            def __init__(self, *args, **kwargs):
                self.tools = {}

            def tool(self):
                def register(function):
                    self.tools[function.__name__] = function
                    return function

                return register

        with mock.patch.object(mcp, "FastMCP", ToolServer):
            return mcp.create_server(self.settings).tools

    def test_backend_fields_and_choices_are_exposed(self):
        # Inspect source without importing the GPU-dependent training module.
        source = Path(mcp.__file__).with_name("train.py").read_text()
        tree = ast.parse(source)
        defaults = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "CONFIG_DEFAULTS"
                for target in node.targets
            )
        )
        self.assertFalse(set(defaults) - mcp.TRAINING_CONFIG_KEYS)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
            ):
                continue
            choice_node = next(
                (item.value for item in node.keywords if item.arg == "choices"), None
            )
            if choice_node is None:
                continue
            key = ast.literal_eval(node.args[0]).removeprefix("--").replace("-", "_")
            with self.subTest(field=key):
                self.assertEqual(
                    tuple(ast.literal_eval(choice_node)),
                    mcp.TRAINING_CONFIG_CHOICES[key],
                )

    def test_capabilities_explain_new_features(self):
        capabilities = self.server_tools()["mlx_lm_lora_get_capabilities"]()
        self.assertIn("dsla", capabilities["training_modes"])
        self.assertIn("klpo", capabilities["training_modes"])
        self.assertIn("micro_batch_size", capabilities["training_config_keys"])
        self.assertIn("recurrence_chunk_size", capabilities["training_config_keys"])
        self.assertIn("load_in_mxfp4", capabilities["features"]["quantized_loading"])
        self.assertIn("dsla", capabilities["features"]["qat_modes"])
        self.assertNotIn(
            "dsla", capabilities["features"]["efficient_long_context_modes"]
        )
        self.assertEqual("acme", capabilities["configured_tenant_id"])
        json.dumps(capabilities)

    def test_dsla_and_klpo_options_reach_worker_and_persist(self):
        requests = [
            {
                "train_mode": "dsla",
                "dsla_loss": "orpo",
                "latent_weight": 0.2,
                "latent_margin": 0.1,
                "latent_gamma": 8.0,
                "latent_variant": "direction",
                "latent_pooling": "last_token",
                "latent_layer": "middle",
                "qat_enable": True,
                "qat_group_size": 0,
                "recurrence_chunk_size": 32,
            },
            {
                "train_mode": "klpo",
                "klpo_route": "sequence",
                "klpo_kl_estimator": "topk",
                "klpo_mc_samples": 16,
                "klpo_top_k": 32,
                "klpo_tail_floor": 0.0001,
                "load_in_mxfp4": True,
                "recurrence_chunk_size": 16,
            },
        ]
        jobs = mcp.TrainingJobManager(self.tenants)
        self.addCleanup(jobs._executor.shutdown)
        fake_train = types.ModuleType("mlx_lm_lora.train")
        fake_train.main = mock.Mock()
        with mock.patch.dict(
            sys.modules, {"mlx_lm_lora.train": fake_train}
        ), mock.patch.object(mlx_lm_lora, "train", fake_train, create=True):
            for request in requests:
                with self.subTest(mode=request["train_mode"]):
                    record = jobs.start(
                        "acme", {"model": "org/model", "data": "org/dataset", **request}
                    )
                    jobs._futures[("acme", record.job_id)].result(timeout=5)
                    self.assertEqual(
                        "succeeded", jobs.get("acme", record.job_id).status
                    )
                    saved = json.loads(
                        (Path(record.run_dir) / "request.json").read_text()
                    )
                    self.assertEqual(saved, fake_train.main.call_args.args[0])
                    for key, value in request.items():
                        self.assertEqual(value, saved[key])

    def test_online_scoring_microbatches(self):
        for mode in mcp.ONLINE_JUDGE_MODES:
            with self.subTest(mode=mode):
                normalized = self.normalize(
                    train_mode=mode, judge="org/judge", micro_batch_size=1
                )
                self.assertEqual(1, normalized["micro_batch_size"])
                with self.assertRaisesRegex(ValueError, "requires a judge"):
                    self.normalize(train_mode=mode)

    def test_invalid_feature_settings_fail_before_queueing(self):
        cases = [
            {"train_mode": "dsla", "dsla_loss": "sft"},
            {"latent_variant": "unknown"},
            {"latent_pooling": "mean"},
            {"latent_layer": -1},
            {"latent_layer": True},
            {"latent_layer": "unknown"},
            {"latent_weight": -0.1},
            {"latent_margin": float("nan")},
            {"latent_gamma": 0},
            {"latent_gamma": float("inf")},
            {"train_mode": "dsla", "beta": 0},
            {"train_mode": "dsla", "delta": -1},
            {"train_mode": "dsla", "max_seq_length": 1},
            {"train_mode": "dsla", "efficient_long_context": True},
            {"klpo_route": "batch"},
            {"klpo_kl_estimator": "unknown"},
            {"klpo_mc_samples": 0},
            {"klpo_top_k": False},
            {"klpo_tail_floor": 0},
            {"klpo_tail_floor": 1},
            {"klpo_tail_floor": float("nan")},
            {"train_mode": "klpo", "klpo_route": "sequence", "klpo_mc_samples": 1},
            {"train_mode": "klpo", "temperature": 0},
            {"train_mode": "klpo", "qat_enable": True},
            {"micro_batch_size": 0},
            {"micro_batch_size": 1.5},
            {"recurrence_chunk_size": 0},
            {"recurrence_chunk_size": True},
            {"qat_group_size": -1},
            {"qat_bits": 1},
            {"qat_bits": 17},
            {"load_in_mxfp4": True, "load_in_4bits": True},
            {"qat_enable": "false"},
            {"learning_rate": float("nan")},
            {"list_reward_functions": True},
        ]
        jobs = mcp.TrainingJobManager(self.tenants)
        self.addCleanup(jobs._executor.shutdown)
        with mock.patch.object(jobs._executor, "submit") as submit:
            for options in cases:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    jobs.start(
                        "acme", {"model": "org/model", "data": "org/dataset", **options}
                    )
            submit.assert_not_called()
        self.assertEqual([], jobs.list("acme"))

    def test_layer_indices_and_zero_alignment_weight_are_supported(self):
        normalized = self.normalize(
            train_mode="dsla",
            latent_layer=3,
            latent_weight=0,
            latent_margin=0,
            qat_enable=True,
            qat_bits=2,
            qat_group_size=0,
        )
        self.assertEqual("3", normalized["latent_layer"])
        self.assertEqual(0, normalized["latent_weight"])

    def test_validation_tool_returns_structured_errors(self):
        validate = self.server_tools()["mlx_lm_lora_validate_training_config"]
        for config in (
            [],
            {"model": "org/model", "data": "org/dataset", "klpo_route": "bad"},
        ):
            result = validate(config)
            self.assertFalse(result["valid"])
            self.assertEqual("acme", result["tenant_id"])
            self.assertTrue(result["errors"])

    def test_reward_discovery_does_not_load_custom_files(self):
        rewards = self.server_tools()["mlx_lm_lora_list_reward_functions"]()
        self.assertIn("r1_accuracy_reward_func", rewards["reward_functions"])
        self.assertTrue(
            set(rewards["default_reward_functions"]) <= set(rewards["reward_functions"])
        )
        self.assertEqual(["grpo", "klpo"], rewards["training_modes"])


if __name__ == "__main__":
    unittest.main()
