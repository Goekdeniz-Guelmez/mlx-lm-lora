import json
import tempfile
import unittest
from pathlib import Path

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
        settings = mcp.ServerSettings(tenant_root=Path("/tmp/tenants"), tenant_id="acme")
        tenants = mcp.TenantManager(settings)
        with self.assertRaises(PermissionError):
            tenants.resolve_tenant("beta")

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
                    {"model": "org/model", "data": "org/dataset", "adapter_path": "../escape"},
                    "acme",
                    tenants,
                    "a" * 32,
                )

    def test_settings_parse_auth_tokens_without_exposing_token_values(self):
        with tempfile.TemporaryDirectory() as root:
            settings = mcp.ServerSettings(
                tenant_root=Path(root),
                transport="streamable-http",
                auth_tokens={"secret-token": "acme"},
                auth_issuer_url="https://issuer.example",
                auth_resource_url="https://mcp.example/mcp",
            )
            self.assertEqual(settings.auth_tokens["secret-token"], "acme")
            self.assertEqual(json.dumps(settings.auth_tokens), '{"secret-token": "acme"}')


if __name__ == "__main__":
    unittest.main()
