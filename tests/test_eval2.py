"""wanxiang-eval-2: frozen corpus, objective metrics, sealed reports.

Fully self-contained: a tiny bundle + tiny frozen QA set are built in
tmp_path, so no ``runs/`` artifacts or network are needed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from autotrain.eval.evalspec import corpus_manifest_hash, evalspec_hash, verify_corpus as verify_m15
from autotrain.eval2 import EVAL2_VERSION, compare_reports, eval2_hash
from autotrain.eval2.corpus import (Eval2CorpusError, freeze, read_qa,
                                     verify_corpus)
from autotrain.eval2.metrics import (aggregate, char_f1, edit_similarity,
                                      empty_reply_rate, exact_match, normalize,
                                      rouge_l, score_pair)
from autotrain.eval2.runner import COMPAT_KEYS, Eval2Runner
from autotrain.export.exporter import export_initial
from autotrain.inference import BOS_ID, EOS_ID, CharTokenizer, ChatTemplate

TINY_TEXT = (
    "用户问 助手答 系统说 \n请解释成语《甲乙丙》的意思是什么含义"
    "请说出答案比喻指形容犹原不一人之。，《》？ "
) * 30

TINY_YAML = """\
experiment: eval2-tiny
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


def _fixture(tmp_path: Path):
    tok = CharTokenizer.build_from_text(TINY_TEXT, vocab_size=256)
    tok.save(tmp_path / "tokenizer.json")
    tok = CharTokenizer.from_file(tmp_path / "tokenizer.json")
    tmpl = ChatTemplate.default()
    tmpl.save(tmp_path / "chat_template.json")
    tmpl.validate_chars(tok)

    stream: list[int] = []
    for w, a in [("甲", "答案甲"), ("乙", "答案乙"), ("丙", "答案丙")] * 3:
        text = tmpl.training_text(f"请解释成语《{w}》的意思", a)
        stream.extend([BOS_ID] + tok.encode(text) + [EOS_ID])
    np.asarray(stream, dtype="<u2").tofile(tmp_path / "sft_tokens.bin")

    eval_dir = tmp_path / "evals2"
    freeze(tmp_path / "sft_tokens.bin", tok, tmpl, eval_dir,
           count=3, min_offset=0)

    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(TINY_YAML, encoding="utf-8")
    bundle = tmp_path / "bundle"
    export_initial(cfg, bundle, force=True)
    return bundle, tmp_path / "tokenizer.json", tmp_path / "chat_template.json", eval_dir


# ---------------------------------------------------------------- metrics

def test_objective_metrics():
    assert normalize("比喻，做了多余的事！") == "比喻做了多余的事"
    assert exact_match("比喻多余", "比喻多余！") == 1.0
    assert exact_match("指人。", "比喻多余") == 0.0
    assert char_f1("abc", "abc") == 1.0
    assert char_f1("abc", "xyz") == 0.0          # no-overlap edge (no ZeroDivision)
    assert rouge_l("abc", "abc") == 1.0
    assert rouge_l("abc", "xyz") == 0.0
    assert edit_similarity("abc", "abc") == 1.0
    assert edit_similarity("abc", "abd") == pytest.approx(2 / 3)
    assert empty_reply_rate(["", "指人。", "   "]) == pytest.approx(2 / 3)


def test_score_pair_and_aggregate():
    s = score_pair("abc", "abc")
    assert s["composite"] == pytest.approx(1.0)
    assert aggregate([s, score_pair("x", "y")])["composite"] == pytest.approx(0.5)


# ---------------------------------------------------------------- frozen corpus

def test_frozen_corpus_and_tamper_detection(tmp_path):
    _, _, _, eval_dir = _fixture(tmp_path)
    doc = verify_corpus(eval_dir)
    assert doc["frozen"] is True and doc["format_version"] == EVAL2_VERSION
    qa, control = read_qa(eval_dir)
    assert len(qa) == len(control) == 3
    assert control[0]["reference"] != qa[0]["reference"]  # shuffled

    (eval_dir / "extra.txt").write_text("x", encoding="utf-8")
    with pytest.raises(Eval2CorpusError, match="file set mismatch"):
        verify_corpus(eval_dir)
    (eval_dir / "extra.txt").unlink()

    qa_file = eval_dir / "qa.jsonl"
    qa_file.write_text(qa_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(Eval2CorpusError, match="checksum mismatch"):
        verify_corpus(eval_dir)

    # restore by rebuilding the frozen set
    freeze(tmp_path / "sft_tokens.bin",
           CharTokenizer.from_file(tmp_path / "tokenizer.json"),
           ChatTemplate.from_file(tmp_path / "chat_template.json"),
           eval_dir, count=3, min_offset=0)
    m = eval_dir / "manifest.json"
    mdoc = json.loads(m.read_text(encoding="utf-8"))
    mdoc["corpus_manifest_hash"] = "0" * 64
    m.write_text(json.dumps(mdoc, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(Eval2CorpusError, match="seal mismatch"):
        verify_corpus(eval_dir)


def test_eval2_spec_hash_differs_from_m15():
    old = evalspec_hash(corpus_manifest_hash(verify_m15()))
    new = eval2_hash("dummy-corpus-hash")
    assert old != new
    assert eval2_hash("a") != eval2_hash("b")   # corpus change changes the seal


# ---------------------------------------------------------------- runner / report

def test_runner_is_read_only_and_sealed(tmp_path):
    original_threads = torch.get_num_threads()
    bundle, tok_p, tmpl_p, eval_dir = _fixture(tmp_path)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in eval_dir.iterdir() if p.is_file()}
    runner = Eval2Runner(bundle, tok_p, tmpl_p, eval_dir=eval_dir, threads=1)
    report = runner.run()
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in eval_dir.iterdir() if p.is_file()}
    assert before == after, "evaluator wrote into the frozen corpus"
    torch.set_num_threads(original_threads)

    assert report["format_version"].startswith("wanxiang-eval2")
    assert report["evalspec_hash"] == eval2_hash(verify_corpus(eval_dir)["corpus_manifest_hash"])
    assert report["qa_count"] == 3
    assert report["status"] in ("INVALID", "BASELINE", "PARTIAL", "QUALITY_PASSED")
    assert 0.0 <= report["metrics"]["composite"] <= 1.0
    assert len(report["examples"]) == 3
    for key in COMPAT_KEYS:
        assert key in report


def test_compare_reports_four_states(tmp_path):
    original_threads = torch.get_num_threads()
    bundle, tok_p, tmpl_p, eval_dir = _fixture(tmp_path)
    runner = Eval2Runner(bundle, tok_p, tmpl_p, eval_dir=eval_dir, threads=1)
    base = runner.run()
    torch.set_num_threads(original_threads)

    hi = {**base, "metrics": {**base["metrics"], "composite": 0.50}}
    assert compare_reports(hi, hi)["verdict"] == "NO_CHANGE"
    better = {**hi, "metrics": {**hi["metrics"], "composite": 0.60}}
    assert compare_reports(hi, better)["verdict"] == "IMPROVED"
    worse = {**hi, "metrics": {**hi["metrics"], "composite": 0.40}}
    assert compare_reports(hi, worse)["verdict"] == "REGRESSED"

    # A report from the M1.5 era must be INCOMPARABLE (different spec).
    m15 = {
        "format_version": "wanxiang-evalreport-1",
        "evalspec_hash": evalspec_hash(corpus_manifest_hash(verify_m15())),
        "corpus_manifest_hash": "m15",
        "bundle_config_hash": "x",
        "device": "cpu",
        "bundle_step": 40,
        "metrics": {"structured": {}},
    }
    out = compare_reports(m15, base)
    assert out["verdict"] == "INCOMPARABLE"
    assert "evalspec_hash" in out["differences"]
    assert out["compatible"] is False
