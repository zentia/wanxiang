"""eval-lm tests: token-level metrics, sealed spec, strict comparison.

Pins that perplexity is a distinct, non-commensurable instrument: reports
carry an eval_lm_hash bound to corpus/holdout/tokenizer/config, and the
comparator refuses cross-hash or cross-family comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from autotrain.corpus.pipeline import build
from autotrain.eval_lm.metrics import (
    aggregate_losses,
    cross_entropy_from_logits,
    perplexity,
)
from autotrain.eval_lm.runner import (
    EvalLMError,
    compare_reports,
    generate_continuation,
    generate_tokens,
    score_bundle,
)
from autotrain.eval_lm.spec import build_spec

DOCS = [
    {"text": "自然语言处理是人工智能的重要分支，研究计算机如何理解和生成人类语言。" * 30},
    {"text": "机器学习通过数据训练模型，使模型能从经验中改进性能而无需显式编程。" * 30},
    {"text": "深度学习使用多层神经网络自动学习数据的层次化表示，在语音和视觉任务表现突出。" * 30},
    {"text": "语言模型估计词序列的概率分布，是机器翻译问答和文本生成的核心组件。" * 30},
    {"text": "注意力机制让模型在编码每个位置时都能关注输入序列的相关部分。" * 30},
]


class TestMetrics:
    def test_cross_entropy_of_certain_prediction_is_zero(self):
        logits = torch.zeros(2, 3, 5)
        targets = torch.tensor([[0, 1, 2], [3, 4, 0]])
        logits[0, 0, 0] = logits[0, 1, 1] = logits[0, 2, 2] = 10.0
        logits[1, 0, 3] = logits[1, 1, 4] = logits[1, 2, 0] = 10.0
        assert cross_entropy_from_logits(logits, targets) == pytest.approx(0.0, abs=1e-3)

    def test_cross_entropy_ignores_minus_one(self):
        logits = torch.zeros(1, 2, 3)
        logits[0, 0, 0] = logits[0, 1, 1] = 10.0
        targets = torch.tensor([[0, -1]])
        assert cross_entropy_from_logits(logits, targets) == pytest.approx(0.0, abs=1e-3)

    def test_perplexity_exp_and_guard(self):
        assert perplexity(0.0) == pytest.approx(1.0)
        assert perplexity(np.log(10)) == pytest.approx(10.0)
        with pytest.raises(ValueError):
            perplexity(-0.1)

    def test_aggregate(self):
        agg = aggregate_losses(np.log(10) * 100, 100)
        assert agg["perplexity"] == pytest.approx(10.0)
        assert agg["tokens"] == 100.0


def _spec(**overrides):
    base = dict(corpus_id="abc", corpus_version="wanxiang-corpus-lm-1",
                holdout_sha256="d" * 64, holdout_tokens=5000,
                tokenizer_sha256="e" * 64, seq_len=32, batch_size=8, seed=42)
    base.update(overrides)
    return build_spec(**base)


def _report(hash_: str, ppl: float, family: str = "wanxiang-eval-lm"):
    return {
        "family": family,
        "eval_lm_hash": hash_,
        "scores": {"perplexity": ppl, "cross_entropy": np.log(ppl)},
    }


class TestCompare:
    def test_same_hash_compares_lower_ppl_as_improved(self):
        spec = _spec()
        base = _report(spec["eval_lm_hash"], 20.0)
        cand = _report(spec["eval_lm_hash"], 15.0)
        result = compare_reports(base, cand, threshold=1.0)
        assert result["verdict"] == "IMPROVED"
        assert result["perplexity_delta"] == pytest.approx(5.0)

    def test_small_delta_is_no_change(self):
        spec = _spec()
        base = _report(spec["eval_lm_hash"], 20.0)
        cand = _report(spec["eval_lm_hash"], 19.5)
        assert compare_reports(base, cand, threshold=1.0)["verdict"] == "NO_CHANGE"

    def test_cross_hash_refused(self):
        a, b = _spec(), _spec(seq_len=64)
        with pytest.raises(EvalLMError):
            compare_reports(_report(a["eval_lm_hash"], 20.0),
                            _report(b["eval_lm_hash"], 15.0))

    def test_cross_family_refused(self):
        spec = _spec()
        with pytest.raises(EvalLMError):
            compare_reports(
                _report(spec["eval_lm_hash"], 20.0),
                _report(spec["eval_lm_hash"], 15.0, family="wanxiang-eval-3"),
            )

    def test_spec_binds_incomparable_instruments(self):
        spec = _spec()
        assert "wanxiang-eval-4" in spec["incomparable_with"]
        assert spec["comparable_with"] == ["wanxiang-eval-lm"]


class TestScoreSmoke:
    def test_score_smoke_bundle(self, tmp_path):
        """End-to-end perplexity on a tiny corpus with the smoke bundle."""
        src = tmp_path / "src.jsonl"
        src.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in DOCS),
                       encoding="utf-8")
        corpus = tmp_path / "corpus"
        build(src, corpus, vocab_size=512, holdout_fraction=0.2, seed=7)
        bundle = Path(__file__).resolve().parents[1] / "runs/v2-smoke/bundle_step0"
        if not (bundle / "model.safetensors").is_file():
            pytest.skip("smoke bundle not present")
        report = score_bundle(bundle, corpus, seq_len=32, batch_size=4, device="cpu")
        assert report["family"] == "wanxiang-eval-lm"
        assert report["scores"]["perplexity"] > 0
        assert report["scores"]["tokens"] > 0
        assert report["windows_scored"] >= 1

    def test_generate_tokens_stops_on_eos(self):
        """A model that always predicts EOS emits no new tokens."""
        class AlwaysEOS:
            max_seq_len = 32

            def eval(self):
                pass

            def forward_cached(self, idx, cache, start_pos):
                logits = torch.zeros(1, idx.size(1), 5)
                logits[:, :, 4] = 10.0  # argmax == eos id 4
                return logits, cache

        out = generate_tokens(AlwaysEOS(), [0, 1, 2], max_new_tokens=10,
                              device="cpu", eos_token_id=4)
        assert out == [0, 1, 2]

    def test_generate_uses_explicit_tokenizer_override(self, monkeypatch, tmp_path):
        """Generation may decode BPE ids, while scoring stays on the frozen ruler."""
        import autotrain.eval_lm.runner as runner

        class FakeTok:
            supports_text = True
            eos_token_id = None

            def encode(self, text, *, add_bos=False):
                return [7]

            def decode(self, ids):
                return "BPE:" + ",".join(str(i) for i in ids)

        class FakeModel:
            max_seq_len = 8

            def eval(self):
                pass

            def forward_cached(self, idx, cache, start_pos):
                logits = torch.zeros(1, idx.size(1), 10)
                logits[:, :, -1] = 5.0
                return logits, cache

        called = []

        def fake_load_tokenizer(path):
            called.append(Path(path))
            return FakeTok()

        info = SimpleNamespace(device="cpu", step=1, config={}, dtype="fp32")
        monkeypatch.setattr(runner, "load_tokenizer", fake_load_tokenizer)
        monkeypatch.setattr(runner, "load_bundle", lambda *a, **k: (FakeModel(), info))

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        override = tmp_path / "bpe.json"
        override.write_text("{}", encoding="utf-8")

        out = runner.generate_continuation(
            tmp_path / "bundle", corpus, "hi", max_new_tokens=2,
            device="cpu", tokenizer_path=override,
        )
        assert called == [override]
        assert out["continuation"].startswith("BPE:")
