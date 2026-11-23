"""SFT prototype: instruction-formatted corpus + versioned chat template.

Builds on the M2 pretrained checkpoint; does not touch ``evals/`` or the
frozen M1.5 eval spec. See SFT.md.
"""

from .corpus import SFT_VERSION, build, verify

__all__ = ["SFT_VERSION", "build", "verify"]
