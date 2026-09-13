"""XIOS - adaptive-depth chat architecture.

Separates the two reasons models are big: knowledge (bits, pushed to disk)
and computation (depth, allocated at runtime from a small shared core).
"""
__version__ = "0.1.0"

from .config import XiosConfig, get_config, PRESETS
from .model import XiosChat, XiosOutput
from .baseline import BaselineTransformer, matched_baseline
from .tokenizer import ByteBPE

__all__ = ["XiosConfig", "get_config", "PRESETS", "XiosChat", "XiosOutput",
           "BaselineTransformer", "matched_baseline", "ByteBPE", "__version__"]
