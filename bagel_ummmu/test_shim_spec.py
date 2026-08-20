# -*- coding: utf-8 -*-
"""Verify the installed shim survives importlib.util.find_spec probing
(the exact check transformers performs)."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flash_attn_sdpa_shim import install_shim

install_shim()

spec = importlib.util.find_spec("flash_attn")
assert spec is not None, "find_spec returned None"
print("find_spec ok:", spec)

from flash_attn import flash_attn_varlen_func  # noqa: F401
print("shim import ok")

try:
    import importlib.metadata
    importlib.metadata.version("flash_attn")
    print("WARNING: unexpected dist metadata found")
except importlib.metadata.PackageNotFoundError:
    print("no dist metadata (transformers will treat flash-attn as unavailable) - ok")
