from setuptools import find_packages, setup

with open("quactuary/requirements.txt") as f:
    reqs = f.read().splitlines()

setup(
    name="quactuary",
    version="0.0.1",
    description="Quantum-powered actuarial tools",
    packages=find_packages("quactuary"),
    package_dir={"": "quactuary"},
    install_requires=reqs,
)
