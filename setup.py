import setuptools

VERSION = "1.4.4"

long_description = open("Readme.md", mode="r", encoding="utf-8").read()

# Module dependencies
dependency_links = []

requirements = [
    "numpy>=1.21.0",
    "requests>=2.27.1",
    "tqdm>=4.62.3",
]

cv_requirements = ["opencv-python>=4.5.5.58"]

tf_requirements = ["tensorflow>=2.6.0"] + cv_requirements

torch_requirements = [
    "torch>=1.8.0,<1.12.0",
    "torchvision>=0.10.0",
    "torchaudio>=1.10.0",
]

torchvision_requirements = [
                               "torch>=1.8.0,<1.12.0",
                               "torchvision>=0.10.0",
                           ] + cv_requirements
torch_transformers_requirements = torch_requirements + ["transformers>=4.18.0"]
setuptools.setup(
    name="deep_utils",
    version=VERSION,
    author="Pooya Mohammadi Kazaj",
    author_email="pooyamohammadikazaj@gmial.com",
    download_url=f"https://github.com/pooya-mohammadi/deep_utils/archive/refs/tags/{VERSION}.tar.gz",
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
    extras_require={
        "cv": cv_requirements,
        "tf": tf_requirements,
        "torch": torch_requirements,
        "torchvision": torchvision_requirements,
        "torch_transformers": torch_transformers_requirements,
    },
    install_requires=requirements,
    dependency_links=dependency_links,
    python_requires=">=3.6",
    license_files=('LICENSE',),
)
