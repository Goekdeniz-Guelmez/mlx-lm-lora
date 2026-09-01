import sys
from pathlib import Path

from setuptools import setup

package_dir = Path(__file__).parent / "mlx_lm_lora"
with open("requirements.txt") as fid:
    requirements = [l.strip() for l in fid.readlines()]

sys.path.append(str(package_dir))
from _version import __version__

setup(
    name="mlx-lm-lora",
    version=__version__,
    description="Train LLMs on Apple silicon with MLX and the Hugging Face Hub",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    author_email="goekdenizguelmez@gmail.com",
    author="Gökdeniz Gülmez",
    url="https://github.com/Goekdeniz-Guelmez/mlx-lm-lora",
    license="MIT",
    install_requires=requirements,
    packages=[
        "mlx_lm_lora",
        "mlx_lm_lora.trainer",
        "skills",
        "skills.mlx_lm_lora",
    ],
    python_requires=">=3.10",
    extras_require={
        # Keep the historical extra as a compatibility alias. MCP is now a
        # core dependency through requirements.txt, so new installs do not
        # need to request this extra.
        "mcp": ["mcp[cli]>=1.13.0"],
    },
    package_data={
        "skills.mlx_lm_lora": ["SKILL.md", "references/*.md"],
    },
    entry_points={
        "console_scripts": [
            "mlx_lm_lora.train = mlx_lm_lora.train:main",
            "mlx_lm_lora.mcp = mlx_lm_lora.mcp:main",
        ]
    },
)
