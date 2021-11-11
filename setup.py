"""
package builder for apulu
"""
import sys
import subprocess
from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install

with open("./requirements.txt") as f:
    REQUIRED = f.read().splitlines()

# build package
setup(
    install_requires=REQUIRED,
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    zip_safe=False,
    python_requires=">=3.8",
    dependency_links=[
        "https://download.pytorch.org/whl/torch_stable.html",
    ],
    # cmdclass={"install": PostInstallCommand},
)
