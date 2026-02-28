from __future__ import annotations
import importlib


def require_package(import_name, pip_name= None):
    _ = pip_name
    return importlib.import_module(import_name)
