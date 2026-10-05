import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mlx_lm_lora import skills


class SkillInstallerTest(unittest.TestCase):
    def test_cli_installs_only_selected_targets_with_all_references(self):
        with tempfile.TemporaryDirectory() as root:
            home = Path(root)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                skills.main(["--codex", "--claude", "--home-dir", root])
            for target in ("codex", "claude"):
                destination = home / skills.SKILL_TARGETS[target] / skills.SKILL_NAME
                self.assertTrue((destination / "SKILL.md").is_file())
                for reference in ("dsla.md", "klpo.md", "memory.md", "config.md"):
                    self.assertTrue((destination / "references" / reference).is_file())
                self.assertIn(str(destination), output.getvalue())
            self.assertFalse((home / ".hermes").exists())

    def test_all_and_explicit_target_install_once_per_harness(self):
        with tempfile.TemporaryDirectory() as root:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                skills.main(["--all", "--codex", "--home-dir", root])
            self.assertEqual(
                len(skills.SKILL_TARGETS), len(output.getvalue().splitlines())
            )
            for relative in skills.SKILL_TARGETS.values():
                self.assertTrue(
                    (Path(root) / relative / skills.SKILL_NAME / "SKILL.md").is_file()
                )

    def test_cli_requires_target_and_does_not_create_directories(self):
        with tempfile.TemporaryDirectory() as root:
            home = Path(root) / "unused"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                SystemExit
            ) as result:
                skills.main(["--home-dir", str(home)])
            self.assertEqual(2, result.exception.code)
            self.assertFalse(home.exists())

    def test_module_command_updates_skill_and_preserves_unrelated_files(self):
        with tempfile.TemporaryDirectory() as root:
            home = Path(root)
            destination = home / ".hermes" / "skills" / skills.SKILL_NAME
            destination.mkdir(parents=True)
            (destination / "SKILL.md").write_text("outdated")
            extra = destination.parent / "other-skill.txt"
            extra.write_text("keep")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlx_lm_lora",
                    "skills",
                    "--hermes",
                    "--home-dir",
                    root,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn(str(destination), result.stdout)
            self.assertEqual(
                (skills._skill_source_dir() / "SKILL.md").read_text(),
                (destination / "SKILL.md").read_text(),
            )
            self.assertEqual("keep", extra.read_text())

    def test_installer_does_not_import_training_or_mcp_dependencies(self):
        script = (
            "import sys\n"
            "from mlx_lm_lora import skills\n"
            "assert not any(name == 'mlx' or name.startswith('mlx.') or "
            "name == 'mcp' or name.startswith('mcp.') for name in sys.modules)\n"
        )
        subprocess.run([sys.executable, "-c", script], check=True)


if __name__ == "__main__":
    unittest.main()
