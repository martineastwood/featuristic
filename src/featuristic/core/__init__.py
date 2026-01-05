"""
Core utilities for Featuristic.

This module contains utility functions that complement the Rust backend:
- Registry for custom symbolic functions
- Preprocessing utilities
"""

from . import registry
from . import preprocess

__all__ = ["registry", "preprocess"]
