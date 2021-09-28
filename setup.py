import sys
import os
import setuptools
from setuptools.command.install import install

with open("README.md", "r") as fh:
    long_description = fh.read()

VERSION = "0.2.2"


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = 'verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)


setuptools.setup(
    name="deep_utils",
    version=VERSION,
    author="Pooya Mohammadi Kazaj",
    author_email="pooyamohammadikazaj@gmial.com",
    download_url=f"https://github.com/Practical-AI/deep_utils/archive/refs/tags/{VERSION}.tar.gz",
    description="Deep Utils",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy', "requests", "tqdm"
    ],
    cmdclass={
        'verify': VerifyVersionCommand,
    }
)
