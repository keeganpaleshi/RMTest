"""Compatibility wrapper for :mod:`radon_inference`."""

from importlib import import_module


_core = import_module("radon_inference")

run_radon_inference = _core.run_radon_inference


__all__ = ["run_radon_inference"]

