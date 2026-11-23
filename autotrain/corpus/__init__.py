"""General-purpose text corpus pipeline (domain-agnostic).

This package is the language-model data entry point. Unlike the historical
idiom-specific ``m2`` pipeline, it accepts any public natural-language text
(Chinese Wikipedia by default), deterministically splits a frozen holdout,
builds a tokenizer, and writes the flat uint16 token streams consumed by the
unchanged trainer. No game / idiom / retrieval assumptions live here.
"""

from .pipeline import CorpusError, build, clean_text, verify

__all__ = ["CorpusError", "build", "clean_text", "verify"]
