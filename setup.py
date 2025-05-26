from setuptools import find_packages, setup

try:
    with open("quactuary/requirements.txt") as f:
        reqs = f.read().splitlines()
except FileNotFoundError:
    with open("quactuary/quactuary/requirements.txt") as f:
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
    extras_require={
        "docs": [
            "sphinx>=8.0",
            "sphinx-rtd-theme",
            "sphinx-sitemap",
            "sphinxcontrib-napoleon"
        ],
        "mcp": [
            "mcp>=0.9.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "quactuary-mcp=quactuary.mcp.server:main",
        ],
        "mcp_servers": [
            "quactuary=quactuary.mcp.server:mcp",
        ],
    },
)
