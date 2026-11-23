"""wanxiang-eval-3: word-level holdout answer-quality instrument.

Unlike eval-2 (token-offset holdout, leaked under the long run), eval-3 picks
holdout IDIOM WORDS at the source and excludes every sample of those words
from both the M2 pretrain and SFT streams. A report is therefore a genuine
generalisation measurement, not a memorisation score.

Reuses eval-2's objective character-level metrics (pure functions) but seals
its own corpus + spec, so eval-2 and eval-3 reports are INCOMPARABLE by design.
"""

from .runner import Eval3Runner, compare_reports, compare_tokenizers

__all__ = ["Eval3Runner", "compare_reports", "compare_tokenizers"]
