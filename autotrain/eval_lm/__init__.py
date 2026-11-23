"""General-purpose language-model evaluation (perplexity + continuation).

Independent instrument: measures next-token modelling on a frozen text
holdout. Explicitly NOT comparable with eval2/3/4 (idiom answer quality) or
eval4 (retrieval) — different question, different unit, different family.
"""

from .runner import compare_reports, generate_continuation, run, score_bundle

__all__ = ["score_bundle", "compare_reports", "run", "generate_continuation"]
