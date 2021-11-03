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
# torch version detection
CUDA = "cpu"
TORCH_VERSION = ""
TORCH_EXISTS = False
for item in REQUIRED:
    if item[:5] == "torch":
        TORCH_VERSION = item[5:].strip("==")
        TORCH_EXISTS = True
if TORCH_VERSION:
    pass
elif not TORCH_VERSION and TORCH_EXISTS:
    REQUIRED.remove("torch")
    TORCH_VERSION = "1.9.0"  # default latest
    REQUIRED.append(f"torch=={TORCH_VERSION}")
else:
    TORCH_VERSION = "1.9.0"  # default latest
    REQUIRED.append(f"torch=={TORCH_VERSION}")

# TODO torch geometric dependency
# class PostInstallCommand(install):
#     """Post-installation for installation mode."""

#     def run(self):
#         install.run(self)
#         # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION

#     torch_geometrics = [
#         "torch-scatter",
#         "torch-sparse",
#         "torch-geometric",
#         f"-f https://data.pyg.org/whl/torch-{TORCH_VERSION}+{CUDA}.html",
#     ]
#     subprocess.check_call([sys.executable, "-m", "pip", "install", *torch_geometrics])


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
