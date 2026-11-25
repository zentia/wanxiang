"""Local source-code corpus pipeline for a small, locally-runnable code LM.

Design constraints (deliberate, do not relax):

* Source code is a LOCAL, proprietary asset. Only derived artifacts under
  ``runs/`` (gitignored) are produced. No path, filename, author, repository
  name or source text ever enters a tracked file or a manifest/report.
* Newlines and indentation are PRESERVED. The general-purpose corpus cleaner
  in ``autotrain.corpus.pipeline`` flattens them to spaces, which destroys code
  syntax, so this module has its own cleaner.
* Holdout is split by DOCUMENT (file), never by token offset, so a training
  cursor that wraps around can never see evaluation code.
* The tokenizer is a ``BPETokenizer``-compatible vocab so the existing bundle
  contract, ``load_tokenizer()`` and the CPU inference server all work as-is.
"""
