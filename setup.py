import setuptools
from setuptools import find_packages

with open("requirements.txt", "r") as fh:
    install_requires = fh.read().split('\n')

setuptools.setup(
    name="LoFTR", 
    version="0.1.0",
    author="Jiaming Sun, Zehong Shen, Yu'ang Wang, Hujun Bao, Xiaowei Zhou",
    description="LoFTR: Detector-Free Local Feature Matching with Transformers",
    url="https://github.com/parallel-domain/LoFTR",
    install_requires=install_requires,
    packages=find_packages(),
    extras_require={
        "dev": ["flake8", "black==22.3.0", "isort", "twine", "pytest", "wheel"],
    },
    python_requires=">=3.8",
)