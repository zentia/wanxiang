"""wanxiang-eval-2: a *separate*, frozen evaluation instrument for text/SFT models.

M1.5's ``evals/`` stays byte-identical and its ``evalspec_hash`` is untouched.
This package is a parallel instrument with its own spec hash, so reports from
the two eras are automatically INCOMPARABLE (by design, see VERIFY.md).

All metrics are objective and deterministic: no LLM-as-a-judge.
"""

from .spec import EVAL2_VERSION, eval2_hash
from .corpus import freeze, read_qa, verify_corpus
from .runner import COMPAT_KEYS, Eval2Runner, compare_reports

__all__ = ["EVAL2_VERSION", "eval2_hash", "freeze", "verify_corpus",
           "read_qa", "Eval2Runner", "compare_reports", "COMPAT_KEYS"]
