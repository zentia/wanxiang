"""M2 minimal data pipeline prototype (see M2.md).

Produces a versioned, checksum-sealed character-level corpus and tokenizer.
Deliberately does NOT touch ``evals/`` or the M1.5 frozen eval spec: this is a
data-pipeline prototype, not a new measurement instrument.
"""

from .pipeline import CORPUS_VERSION, build, clean_text, extract_text, verify
from .source import SOURCES, fetch

__all__ = ["SOURCES", "fetch", "build", "verify", "clean_text",
           "extract_text", "CORPUS_VERSION"]
