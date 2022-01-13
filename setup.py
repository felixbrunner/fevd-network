from setuptools import find_packages, setup

setup(
    name="euraculus",
    version="0.1",
    description="FEVD networks for asset volatility spillovers",
    author="felixbrunner",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "sklearn",
        "requests",
        "scipy",
        "networkx",
    ],
)
