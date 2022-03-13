import sys
import os
import setuptools
from setuptools.command.install import install

VERSION = "0.8.22"

try:
    import pypandoc

    long_description = pypandoc.convert('Readme.md', 'rst')
except (IOError, ImportError):
    long_description = open('Readme.md', mode='r').read()


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""
    description = 'verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('GIT_TAG')

        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)


# Module dependencies
requirements, dependency_links = [], []
with open('requirements.txt') as f:
    for line in f.read().splitlines():
        if line.startswith('-e git+'):
            dependency_links.append(line.replace('-e ', ''))
        else:
            requirements.append(line)

cv_requirements = [
    "opencv-python>=4.5.5.58"
]

tf_requirements = [
                      "tensorflow>=2.6.0",
                  ] + cv_requirements

torch_requirements = [
                         "torch>=1.8.0",
                         "torchvision>=0.10.0"
                     ] + cv_requirements

setuptools.setup(
    name="deep_utils",
    version=VERSION,
    author="Pooya Mohammadi Kazaj",
    author_email="pooyamohammadikazaj@gmial.com",
    download_url=f"https://github.com/Practical-AI/deep_utils/archive/refs/tags/{VERSION}.tar.gz",
    description="Deep Utils",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pooya-mohammadi/deep_utils",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    extras_require={"cv": cv_requirements, "tf": tf_requirements, "torch": torch_requirements},
    install_requires=requirements,
    dependency_links=dependency_links,
    python_requires='>=3.6',
)
