"""wanxiang-eval-3: word-level holdout, sealed corpus, A/B isolation.

Self-contained: a tiny source + bundle are built in tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotrain.eval3 import compare_reports
from autotrain.eval3.corpus import (Eval3CorpusError, freeze, read_qa,
                                     verify_corpus)
from autotrain.eval3.runner import COMPAT_KEYS, Eval3Runner
from autotrain.eval3.spec import EVAL3_VERSION, eval3_hash, spec_payload
from autotrain.export.exporter import export_initial
from autotrain.inference import CharTokenizer, ChatTemplate

TINY_TEXT = (
    "用户问 助手答 系统说 \n请解释成语《甲乙丙丁戊》的意思是什么含义"
    "请说出比喻指形容犹原不一人之。，《》？ "
) * 30

TINY_YAML = """\
experiment: eval3-tiny
seed: 1
data:
  mix: tiny
  seq_len: 32
  val_fraction: 0.02
model:
  layers: 1
  dim: 32
  heads: 2
  vocab_size: 256
train:
  global_batch_tokens: 1024
  lr_peak: 0.001
  warmup_steps: 1
  max_steps: 1
  precision: fp32
  ckpt_every_steps: 1
  val_every_steps: 1
  grad_clip: 1.0
recover:
  max_retries: 0
  loss_spike_sigma: 3.0
  spike_window: 10
  early_stop_patience: 1
eval:
  harness_tasks: []
  milestones: [1.0]
"""


def _rows(n: int = 30):
    return [{"word": f"词{i}", "explanation": f"比喻第{i}种意思。"} for i in range(n)]


def _fixture(tmp_path: Path, qa_count: int = 5, holdout_n: int = 10):
    tok = CharTokenizer.build_from_text(TINY_TEXT, vocab_size=256)
    tok.save(tmp_path / "tokenizer.json")
    tok = CharTokenizer.from_file(tmp_path / "tokenizer.json")
    tmpl = ChatTemplate.default()
    tmpl.save(tmp_path / "chat_template.json")
    tmpl.validate_chars(tok)

    src = tmp_path / "idiom.json"
    src.write_text(json.dumps(_rows(), ensure_ascii=False), encoding="utf-8")

    eval_dir = tmp_path / "evals3"
    freeze(src, tok, tmpl, eval_dir, count=qa_count, seed=42, holdout_n=holdout_n)

    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(TINY_YAML, encoding="utf-8")
    bundle = tmp_path / "bundle"
    export_initial(cfg, bundle, force=True)
    return bundle, tmp_path / "tokenizer.json", tmp_path / "chat_template.json", eval_dir


class TestEval3Spec:
    def test_seal_changes_with_corpus_hash(self):
        h1 = eval3_hash("a" * 64)
        h2 = eval3_hash("b" * 64)
        assert h1 != h2 and len(h1) == 64

    def test_distinct_from_eval2(self):
        from autotrain.eval2 import eval2_hash
        assert eval3_hash("x" * 64) != eval2_hash("x" * 64)

    def test_payload_has_holdout_constants(self):
        p = spec_payload("h" * 64)
        assert "holdout_seed" in p and "holdout_count" in p


class TestEval3Corpus:
    def test_frozen_and_holdout_sealed(self, tmp_path):
        _, tok_p, tmpl_p, ed = _fixture(tmp_path)
        doc = verify_corpus(ed)
        assert doc["frozen"] is True and doc["format_version"] == EVAL3_VERSION
        assert doc["holdout_seed"] == 42 and doc["holdout_count"] == 10
        qa, control = read_qa(ed)
        assert len(qa) == len(control) == 5
        assert control[0]["reference"] != qa[0]["reference"]

        words = json.loads((ed / "holdout_words.json").read_text(encoding="utf-8"))
        assert words["seed"] == 42 and len(words["words"]) == 10
        # every scored question's word comes from the holdout set
        assert all(q["word"] in set(words["words"]) for q in qa)

    def test_reject_extra_file(self, tmp_path):
        _, _, _, ed = _fixture(tmp_path)
        (ed / "extra.txt").write_text("x", encoding="utf-8")
        with pytest.raises(Eval3CorpusError, match="file set mismatch"):
            verify_corpus(ed)

    def test_reject_tampered_qa(self, tmp_path):
        _, _, _, ed = _fixture(tmp_path)
        qa = ed / "qa.jsonl"
        qa.write_text(qa.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with pytest.raises(Eval3CorpusError, match="checksum mismatch"):
            verify_corpus(ed)


class TestEval3RunnerCompare:
    def test_report_sealed(self, tmp_path):
        bundle, tok_p, tmpl_p, ed = _fixture(tmp_path)
        r = Eval3Runner(bundle, tok_p, tmpl_p, eval_dir=ed, device="cpu")
        rep = r.run()
        assert rep["format_version"] == "wanxiang-eval3-report-1"
        assert rep["status"] in {"QUALITY_PASSED", "PARTIAL", "BASELINE", "INVALID"}
        assert rep["qa_count"] == 5 and len(rep["examples"]) <= 5
        assert 0.0 <= rep["metrics"]["composite"] <= 1.0

    def test_compare_verdicts(self):
        base = {"format_version": "wanxiang-eval3-report-1",
                "evalspec_hash": "e" * 64, "corpus_manifest_hash": "c" * 64,
                "device": "cpu", "tokenizer_hash": "t" * 64,
                "chat_template_hash": "p" * 64, "bundle_step": 100,
                "metrics": {"composite": 0.10}, "status": "PARTIAL"}
        improved = dict(base, bundle_step=200, metrics={"composite": 0.15})
        assert compare_reports(base, improved)["verdict"] == "IMPROVED"
        regressed = dict(base, bundle_step=200, metrics={"composite": 0.05})
        assert compare_reports(base, regressed)["verdict"] == "REGRESSED"
        same = dict(base, bundle_step=200, metrics={"composite": 0.105})
        assert compare_reports(base, same)["verdict"] == "NO_CHANGE"

    def test_incomparable_across_specs(self):
        base = {"format_version": "wanxiang-eval3-report-1",
                "evalspec_hash": "e" * 64, "corpus_manifest_hash": "c" * 64,
                "device": "cpu", "tokenizer_hash": "t" * 64,
                "chat_template_hash": "p" * 64}
        other = dict(base, evalspec_hash="f" * 64)
        out = compare_reports(base, other)
        assert out["compatible"] is False and out["verdict"] == "INCOMPARABLE"
        assert "evalspec_hash" in out["differences"]
