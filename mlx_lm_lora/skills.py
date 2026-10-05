"""Install the bundled training skill without importing the MCP or ML backend."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

SKILL_NAME = "mlx_lm_lora"
SKILL_TARGETS = {
    "codex": Path(".codex") / "skills",
    "claude": Path(".claude") / "skills",
    "hermes": Path(".hermes") / "skills",
}


def _skill_source_dir() -> Path:
    """Return the packaged harness skill directory."""

    source_dir = Path(__file__).resolve().parent.parent / "skills" / SKILL_NAME
    if not source_dir.is_dir() or not (source_dir / "SKILL.md").is_file():
        raise FileNotFoundError(
            "The packaged mlx_lm_lora skill is missing from the installation: "
            f"{source_dir}"
        )
    return source_dir


def install_skill(
    target: str,
    *,
    home_dir: Path | None = None,
    source_dir: Path | None = None,
) -> Path:
    """Install the bundled harness skill for one supported agent.

    Args:
        target: Harness name: ``codex``, ``claude``, or ``hermes``.
        home_dir: Optional home directory override, primarily for testing.
        source_dir: Optional skill source override, primarily for testing.

    Returns:
        The installed skill directory.

    Raises:
        ValueError: If ``target`` is not supported.
        FileNotFoundError: If the bundled skill is unavailable.
        FileExistsError: If the destination is not a directory.
    """

    if target not in SKILL_TARGETS:
        supported_targets = ", ".join(sorted(SKILL_TARGETS))
        raise ValueError(f"target must be one of: {supported_targets}")

    source = (source_dir or _skill_source_dir()).expanduser().resolve()
    if not source.is_dir() or not (source / "SKILL.md").is_file():
        raise FileNotFoundError(f"Skill source directory is invalid: {source}")

    home = (home_dir or Path.home()).expanduser()
    destination = home / SKILL_TARGETS[target] / SKILL_NAME
    if source == destination.resolve(strict=False):
        return destination
    if destination.is_symlink():
        raise FileExistsError(
            f"Refusing to install through a symbolic-link destination: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone skill installer CLI."""

    parser = argparse.ArgumentParser(
        description="Install the MLX-LM-LoRA training skill for your agent."
    )
    for target in SKILL_TARGETS:
        parser.add_argument(
            f"--{target}",
            action="store_true",
            help=f"Install into ~/{SKILL_TARGETS[target]}/{SKILL_NAME}.",
        )
    parser.add_argument(
        "--all", action="store_true", help="Install for Codex, Claude Code, and Hermes."
    )
    parser.add_argument(
        "--home-dir", type=Path, help="Use this home directory instead of ~."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Install the bundled skill for one or more selected agents."""

    parser = build_parser()
    args = parser.parse_args(argv)
    targets = [target for target in SKILL_TARGETS if args.all or getattr(args, target)]
    if not targets:
        parser.error("select at least one of --codex, --claude, --hermes, or --all")
    for target in targets:
        destination = install_skill(target, home_dir=args.home_dir)
        print(f"Installed {SKILL_NAME} skill for {target} to {destination}")


if __name__ == "__main__":
    main()
