"""Shared YAML import utility.

Centralizes the optional PyYAML import so it's not duplicated
across parsers, exporters, and CLI modules.
"""

import importlib
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _load_yaml() -> Any | None:
    try:
        return importlib.import_module("yaml")
    except ModuleNotFoundError:  # pragma: no cover
        return None


def get_yaml() -> Any | None:
    return _load_yaml()


def yaml_available() -> bool:
    return get_yaml() is not None
