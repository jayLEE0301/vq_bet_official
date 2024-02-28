import io
import os
import re
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


VERSION = find_version("vq_behavior_transformer", "__init__.py")
requirements = ["torch>=1.6,<3", "einops>=0.4", "tqdm"]

setup(
    name="vq-bet",
    version=VERSION,
    description="PyTorch implementation of VQ-BeT: Behavior Generation with Latent Actions",
    url="https://github.com/jayLEE0301/vq_bet_official",
    author="Seungjae Lee",
    author_email="ysz0301@gmail.com",
    license="MIT License",
    install_requires=requirements,
    long_description_content_type="text/markdown",
    long_description=read("README.md"),
    packages=find_packages(exclude=["examples", "tests"]),
    keywords=[
        "artificial intelligence",
        "deep learning",
        "transformers",
        "vq-bet",
        "multimodal behaviors",
        "behavior cloning",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
