from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="euraculus",
    version="0.1",
    description="FEVD networks for asset volatility spillovers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="felixbrunner",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "requests",
        "scipy",
        "networkx",
        "glmnet-py",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
