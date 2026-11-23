"""wanxiang-eval-2 runner: bundle -> sealed answer-quality report.

Read-only with respect to the frozen corpus and the bundle: it never writes
into ``evals2/`` and never imports the training loop. ``COMPAT_KEYS`` includes
``evalspec_hash``, so a report from the M1.5 era is automatically
INCOMPARABLE with an eval-2 report (the intended "no moving the goalposts"
behaviour).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..inference import (CharTokenizer, ChatTemplate, GenerationConfig,
                          InferenceEngine)
from .corpus import EVAL2_DIR, read_qa, verify_corpus
from .metrics import aggregate, empty_reply_rate, score_pair
from .spec import (BASELINE_THRESHOLD, CONTROL_TOLERANCE, GEN_TOKENS,
                   MIN_IMPROVEMENT, QUALITY_THRESHOLD, REPORT_VERSION,
                   eval2_hash)

COMPAT_KEYS = (
    "format_version",
    "evalspec_hash",
    "corpus_manifest_hash",
    "bundle_config_hash",
    "device",
    "tokenizer_hash",
    "chat_template_hash",
)


class Eval2Error(ValueError):
    """Evaluation inputs are inconsistent."""


class Eval2Runner:
    def __init__(
        self,
        bundle_dir: str | Path,
        tokenizer_path: str | Path,
        chat_template_path: str | Path,
        *,
        eval_dir: str | Path | None = None,
        device: str | None = None,
        threads: int | None = None,
    ):
        self.eval_dir = Path(eval_dir) if eval_dir is not None else EVAL2_DIR
        self.manifest = verify_corpus(self.eval_dir)
        self.corpus_hash = self.manifest["corpus_manifest_hash"]
        self.spec_hash = eval2_hash(self.corpus_hash)
        self.qa, self.control = read_qa(self.eval_dir)

        self.tok = CharTokenizer.from_file(tokenizer_path)
        self.tmpl = ChatTemplate.from_file(chat_template_path)
        self.tmpl.validate_chars(self.tok)
        self.engine = InferenceEngine.from_bundle(
            bundle_dir, device=device, threads=threads
        )
        if self.tok.vocab_size != self.engine.vocab_size:
            raise Eval2Error(
                f"tokenizer vocab {self.tok.vocab_size} != model vocab "
                f"{self.engine.vocab_size}"
            )
        self.tokenizer_hash = self.tok.as_dict()["sha256"]
        self.template_hash = self.tmpl.template_hash
        self.bundle_config_hash = self.engine.info.config_hash
        self.bundle_step = self.engine.info.step
        self.device = self.engine.info.device

    def _reply(self, question: str) -> str:
        ids = self.tmpl.encode_messages(
            [{"role": "user", "content": question}], self.tok
        )
        cfg = GenerationConfig(
            max_new_tokens=GEN_TOKENS,
            do_sample=False,
            eos_token_id=self.tok.eos_token_id,
        )
        result = self.engine.generate(ids, cfg)
        return self.tmpl.extract_reply(self.tok.decode(result.token_ids))

    def run(self) -> dict[str, Any]:
        replies = [self._reply(row["question"]) for row in self.qa]
        scores = [score_pair(reply, row["reference"])
                  for reply, row in zip(replies, self.qa)]
        control_scores = [score_pair(reply, row["reference"])
                          for reply, row in zip(replies, self.control)]
        metrics = aggregate(scores)
        control = aggregate(control_scores)

        checks = {
            "control_consistent": (
                control["composite"] - metrics["composite"] <= CONTROL_TOLERANCE
            ),
            "quality_passed": metrics["composite"] >= QUALITY_THRESHOLD,
        }
        if not checks["control_consistent"]:
            status = "INVALID"
        elif checks["quality_passed"]:
            status = "QUALITY_PASSED"
        elif metrics["composite"] <= BASELINE_THRESHOLD:
            status = "BASELINE"
        else:
            status = "PARTIAL"

        return {
            "format_version": REPORT_VERSION,
            "evalspec_hash": self.spec_hash,
            "corpus_manifest_hash": self.corpus_hash,
            "bundle_config_hash": self.bundle_config_hash,
            "bundle_step": self.bundle_step,
            "device": self.device,
            "tokenizer_hash": self.tokenizer_hash,
            "chat_template_hash": self.template_hash,
            "qa_count": len(self.qa),
            "gen_tokens": GEN_TOKENS,
            "metrics": metrics,
            "negative_control": control,
            "empty_reply_rate": empty_reply_rate(replies),
            "checks": checks,
            "status": status,
            "examples": [
                {
                    "question": self.qa[i]["question"],
                    "reference": self.qa[i]["reference"][:60],
                    "reply": replies[i][:60],
                    "composite": round(scores[i]["composite"], 4),
                }
                for i in range(min(3, len(self.qa)))
            ],
            "note": "Objective character-level metrics only, no LLM-as-a-judge. "
                    "M1.5 evals/ and evalspec_hash are untouched; reports from "
                    "the two eras are INCOMPARABLE by design.",
        }


def compare_reports(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """A/B verdict. Reports with different frozen specs are INCOMPARABLE."""
    differences = [k for k in COMPAT_KEYS if before.get(k) != after.get(k)]
    result: dict[str, Any] = {
        "format_version": "wanxiang-eval2-compare-1",
        "compatible": not differences,
        "differences": differences,
        "before_step": before.get("bundle_step"),
        "after_step": after.get("bundle_step"),
    }
    if differences:
        result["verdict"] = "INCOMPARABLE"
        result["note"] = "Never mix M1.5 (Markov) reports with eval-2 (text QA) reports."
        return result

    delta = (after["metrics"]["composite"] - before["metrics"]["composite"])
    result.update({
        "delta_composite": delta,
        "min_improvement": MIN_IMPROVEMENT,
        "before_status": before.get("status"),
        "after_status": after.get("status"),
        "verdict": (
            "IMPROVED" if delta >= MIN_IMPROVEMENT
            else "REGRESSED" if delta <= -MIN_IMPROVEMENT
            else "NO_CHANGE"
        ),
    })
    return result
