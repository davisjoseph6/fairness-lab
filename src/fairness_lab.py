"""Shared configuration and utility functions for the fairness lab.

This module contains project-wide constants and reproducibility helpers.
Dataset loading, model training, fairness auditing and mitigation functions
will be added during the later missions.
"""

from __future__ import annotations

import json
import platform
import random
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.30

POSITIVE_LABEL: int = 1
NEGATIVE_LABEL: int = 0

YOUNG_GROUP: str = "young"
OLD_GROUP: str = "old"
AGE_THRESHOLD: int = 25


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
TABLES_DIR: Path = OUTPUT_DIR / "tables"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
MODELS_DIR: Path = OUTPUT_DIR / "models"

REPORT_DIR: Path = PROJECT_ROOT / "report"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"


def ensure_project_directories() -> None:
    """Create all generated-output directories if they do not exist."""

    directories = (
        OUTPUT_DIR,
        TABLES_DIR,
        FIGURES_DIR,
        MODELS_DIR,
        REPORT_DIR,
        NOTEBOOKS_DIR,
    )

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    """Set random seeds used by Python and NumPy."""

    random.seed(seed)
    np.random.seed(seed)


def define_age_group(age: float | int) -> str:
    """Convert an applicant's age into the lab's sensitive group."""

    numeric_age = float(age)

    if not np.isfinite(numeric_age):
        raise ValueError("Age must be a finite numeric value.")

    if numeric_age < 0:
        raise ValueError("Age cannot be negative.")

    if numeric_age < AGE_THRESHOLD:
        return YOUNG_GROUP

    return OLD_GROUP


def installed_version(distribution_name: str) -> str:
    """Return the installed version of a Python distribution."""

    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return "not installed"


def get_environment_info() -> dict[str, Any]:
    """Collect software and reproducibility information."""

    package_names = (
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "fairlearn",
        "openml",
        "jupyterlab",
        "ipykernel",
    )

    package_versions = {
        package_name: installed_version(package_name)
        for package_name in package_names
    }

    return {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "positive_label": POSITIVE_LABEL,
        "negative_label": NEGATIVE_LABEL,
        "age_threshold": AGE_THRESHOLD,
        "young_group_definition": f"age < {AGE_THRESHOLD}",
        "old_group_definition": f"age >= {AGE_THRESHOLD}",
        "package_versions": package_versions,
    }


def save_environment_info(destination: Path | None = None) -> Path:
    """Save environment details as a JSON file."""

    ensure_project_directories()

    output_path = destination or (OUTPUT_DIR / "environment.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    environment_info = get_environment_info()

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(environment_info, file, indent=2, ensure_ascii=False)

    return output_path


def initialise_project() -> dict[str, Any]:
    """Initialise directories, random seeds and environment metadata."""

    ensure_project_directories()
    set_global_seed()

    environment_path = save_environment_info()
    environment_info = get_environment_info()

    print("Fairness lab project initialised.")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Random seed: {RANDOM_STATE}")
    print(f"Environment information: {environment_path}")

    return environment_info


def main() -> None:
    """Run the project-initialisation checks."""

    environment_info = initialise_project()

    print("\nInstalled package versions:")

    for package_name, version in environment_info["package_versions"].items():
        print(f"- {package_name}: {version}")


if __name__ == "__main__":
    main()
