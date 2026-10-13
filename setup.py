# Modified for the RepForcing source release; see NOTICE.md.
"""RepForcing source-package metadata."""
from setuptools import find_namespace_packages, setup

setup(
    name="repforcing",
    version="0.1.0",
    description="Representation-supervised initialization and distillation for causal video generation",
    python_requires=">=3.10",
    packages=find_namespace_packages(include=[
        "configs", "configs.*", "model", "model.*", "trainer", "trainer.*",
        "pipeline", "pipeline.*", "utils", "utils.*", "wan", "wan.*",
        "demo_utils", "demo_utils.*",
    ]),
    package_data={"configs": ["*.yaml"]},
)
