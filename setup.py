from setuptools import find_packages, setup

with open("quactuary/requirements.txt") as f:
    reqs = f.read().splitlines()

setup(
    use_scm_version={
        "write_to": "quactuary/_version.py"
    },
    setup_requires=["setuptools_scm"],
    packages=find_packages("quactuary"),
    package_dir={"": "quactuary"},
    description="Quantum-powered actuarial tools",
    install_requires=reqs,
)
