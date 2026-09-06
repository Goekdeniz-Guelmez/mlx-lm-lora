import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
