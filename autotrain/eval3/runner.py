"""wanxiang-eval-3 runner: bundle -> sealed word-holdout answer-quality report.

Reuses eval-2's pure metrics but seals with eval-3's spec + corpus hash, so an
eval-2 report and an eval-3 report are automatically INCOMPARABLE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..eval2.metrics import aggregate, empty_reply_rate, score_pair
from ..inference import (ChatTemplate, GenerationConfig, InferenceEngine,
                         load_tokenizer)
from .corpus import EVAL3_DIR, read_qa, verify_corpus
from .spec import (BASELINE_THRESHOLD, CONTROL_TOLERANCE, GEN_TOKENS,
                   MIN_IMPROVEMENT, QUALITY_THRESHOLD, REPORT_VERSION,
                   eval3_hash)

COMPAT_KEYS = (
    "format_version",
    "evalspec_hash",
    "corpus_manifest_hash",
    "device",
    "tokenizer_hash",
    "chat_template_hash",
)


class Eval3Error(ValueError):
    """Evaluation inputs are inconsistent."""


class Eval3Runner:
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
        self.eval_dir = Path(eval_dir) if eval_dir is not None else EVAL3_DIR
        self.manifest = verify_corpus(self.eval_dir)
        self.corpus_hash = self.manifest["corpus_manifest_hash"]
        self.spec_hash = eval3_hash(self.corpus_hash)
        self.qa, self.control = read_qa(self.eval_dir)

        self.tok = load_tokenizer(tokenizer_path)
        self.tmpl = ChatTemplate.from_file(chat_template_path)
        self.tmpl.validate_chars(self.tok)
        self.engine = InferenceEngine.from_bundle(
            bundle_dir, device=device, threads=threads
        )
        if self.tok.vocab_size != self.engine.vocab_size:
            raise Eval3Error(
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
            "tokenizer_type": self.tok.type,
            "tokenizer_vocab_size": self.tok.vocab_size,
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
                for i in range(min(5, len(self.qa)))
            ],
            "note": "Word-level holdout: questions are unseen idioms excluded "
                    "from pretrain+SFT. Objective character metrics only.",
        }


def compare_reports(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """A/B verdict. Reports with different frozen specs are INCOMPARABLE."""
    differences = [k for k in COMPAT_KEYS if before.get(k) != after.get(k)]
    result: dict[str, Any] = {
        "format_version": "wanxiang-eval3-compare-1",
        "compatible": not differences,
        "differences": differences,
        "before_step": before.get("bundle_step"),
        "after_step": after.get("bundle_step"),
    }
    if differences:
        result["verdict"] = "INCOMPARABLE"
        result["note"] = "Never mix eval-2 and eval-3 or re-split reports."
        return result

    delta = after["metrics"]["composite"] - before["metrics"]["composite"]
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


# Keys that must stay identical for a tokenizer-only A/B. ``tokenizer_hash``
# and ``tokenizer_type`` are intentionally excluded: they are the variable
# being tested. Everything that defines the QUESTIONS and the SCORING must
# still match, so this is a genuine "same ruler, different encoder" comparison.
_CROSS_PROTOCOL_KEYS = (
    "format_version",
    "evalspec_hash",
    "corpus_manifest_hash",
    "device",
    "chat_template_hash",
    "qa_count",
    "gen_tokens",
)


def compare_tokenizers(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """A/B across a tokenizer change on the SAME frozen eval-3 questions.

    Unlike ``compare_reports`` (which refuses any tokenizer change), this
    function is for the M3 experiment: it verifies that the frozen question
    set, answers, scoring spec, template, device and generation budget are
    byte-identical, and that only the tokenizer differs. The verdict is
    therefore a valid measurement of *encoding granularity* -- not a mix of
    two different rulers like eval-2 vs eval-3.

    Hard checks (any failure -> INCOMPARABLE):
      * all ``_CROSS_PROTOCOL_KEYS`` equal;
      * tokenizer vocabulary size unchanged (model compatibility);
      * ``tokenizer_hash`` actually differs (else use ``compare_reports``).
    """
    protocol_diff = [k for k in _CROSS_PROTOCOL_KEYS if before.get(k) != after.get(k)]
    same_vocab = (before.get("tokenizer_vocab_size")
                  == after.get("tokenizer_vocab_size")
                  and before.get("tokenizer_vocab_size") is not None)
    tokenizer_changed = before.get("tokenizer_hash") != after.get("tokenizer_hash")

    result: dict[str, Any] = {
        "format_version": "wanxiang-eval3-tokenizer-compare-1",
        "cross_tokenizer": True,
        "before_tokenizer": {
            "type": before.get("tokenizer_type"),
            "hash": (before.get("tokenizer_hash") or "")[:16],
            "vocab_size": before.get("tokenizer_vocab_size"),
        },
        "after_tokenizer": {
            "type": after.get("tokenizer_type"),
            "hash": (after.get("tokenizer_hash") or "")[:16],
            "vocab_size": after.get("tokenizer_vocab_size"),
        },
        "protocol_differences": protocol_diff,
        "vocab_size_matched": same_vocab,
        "tokenizer_changed": tokenizer_changed,
        "before_step": before.get("bundle_step"),
        "after_step": after.get("bundle_step"),
    }
    if protocol_diff or not same_vocab or not tokenizer_changed:
        result["compatible"] = False
        result["verdict"] = "INCOMPARABLE"
        result["note"] = (
            "Tokenizer A/B requires identical eval spec/corpus/device/template/"
            "budget and vocab size, with a genuinely different tokenizer."
        )
        return result

    delta = after["metrics"]["composite"] - before["metrics"]["composite"]
    result.update({
        "compatible": True,
        "delta_composite": delta,
        "min_improvement": MIN_IMPROVEMENT,
        "before_composite": before["metrics"]["composite"],
        "after_composite": after["metrics"]["composite"],
        "before_status": before.get("status"),
        "after_status": after.get("status"),
        "verdict": (
            "IMPROVED" if delta >= MIN_IMPROVEMENT
            else "REGRESSED" if delta <= -MIN_IMPROVEMENT
            else "NO_CHANGE"
        ),
        "note": "Same frozen eval-3 questions/answers/scoring/gen budget; "
                "only tokenizer encoding changed (char -> BPE or reverse).",
    })
    return result
