"""Wanxiang eval-4: retrieval usability instrument.

eval-3 measures ZERO-SHOT generalisation on word-level holdout idioms (the
definition never entered training). eval-4 measures a different question:
"when Wanxiang is allowed its retrieval tool, does it answer correctly?"

The two are INCOMPARABLE by design. A high eval-4 is NOT evidence that the
language model learned anything -- the index deliberately contains the whole
dictionary including holdout words. eval-4 exists to quantify the honest
usability of the retrieval path and to contrast it with the generator.
"""

from .runner import Eval4Runner, compare_eval4, eval4_hash, run_retrieval
from .spec import (EVAL4_VERSION, REPORT_VERSION, COMPARABLE_WITH_EVAL3,
                   INDEX_KEY, QUALITY_THRESHOLD)

__all__ = [
    "EVAL4_VERSION", "REPORT_VERSION", "COMPARABLE_WITH_EVAL3", "INDEX_KEY",
    "QUALITY_THRESHOLD", "Eval4Runner", "run_retrieval", "compare_eval4",
    "eval4_hash",
]
