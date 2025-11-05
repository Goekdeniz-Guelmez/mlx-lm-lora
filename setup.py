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
    packages=["mlx_lm_lora", "mlx_lm_lora.trainer"],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "mlx_lm_lora.train = mlx_lm_lora.train:main",
            "mlx_lm_lora.synthetic_sft = mlx_lm_lora.synthetic_sft:main",
            "mlx_lm_lora.synthetic_dpo = mlx_lm_lora.synthetic_dpo:main",
        ]
    },
)
