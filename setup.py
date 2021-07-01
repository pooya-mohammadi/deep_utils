import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deep_utils",
    version="0.1.0",
    author="Pooya Mohammadi Kazaj",
    author_email="pooyamohammadikazaj@gmial.com",
    download_url="https://github.com/Practical-AI/deep_utils/archive/refs/tags/1.0.0.tar.gz",
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
        'numpy'
    ]
)
