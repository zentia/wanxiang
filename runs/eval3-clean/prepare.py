"""One-shot data prep for the eval-3 clean retrain.

Builds (all under runs/eval3-clean/):
  m2_corpus/   M2 pretrain stream, holdout words EXCLUDED, tokenizer still full
  sft_corpus/  SFT v2 stream, holdout EXCLUDED, answers cleaned, + label mask
And freezes evals3/ (word-level holdout QA) from the pinned source.

Idempotent: rebuilds deterministically and prints all seals for verification.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autotrain.inference import CharTokenizer, ChatTemplate  # noqa: E402
from autotrain.m2.pipeline import build as m2_build, verify as m2_verify  # noqa: E402
from autotrain.sft.corpus import build as sft_build, verify as sft_verify  # noqa: E402
from autotrain.split import HOLDOUT_SEED, holdout_words  # noqa: E402
from autotrain.eval3.corpus import freeze as eval3_freeze  # noqa: E402

SRC = ROOT / "runs/m2/sources/idiom.json"
BASE = ROOT / "runs/eval3-clean"
M2_OUT = BASE / "m2_corpus"
SFT_OUT = BASE / "sft_corpus"
EVAL3_OUT = ROOT / "evals3"
FULL_TOK = ROOT / "runs/m2/corpus/tokenizer.json"


def main() -> None:
    rows = json.loads(SRC.read_text(encoding="utf-8"))
    holdout = set(holdout_words(rows, seed=HOLDOUT_SEED))
    print(f"holdout_words={len(holdout)} seed={HOLDOUT_SEED}")

    # 1) clean M2 pretrain stream (tokenizer from FULL text -> frozen vocab).
    # source_meta keeps the tokenizer's provenance identical to the original
    # pipeline so its hash stays byte-for-byte comparable with the old bundle.
    m2 = m2_build(SRC, M2_OUT, exclude_words=holdout,
                  source_meta={"name": "idiom"})
    m2v = m2_verify(M2_OUT)
    assert m2["corpus_id"] == m2v["corpus_id"]
    assert m2["tokenizer"]["sha256"] == CharTokenizer.from_file(FULL_TOK).as_dict()["sha256"]
    assert not (set(m2["extraction"]["excluded_words"]) != holdout)
    print(f"M2  tokens={m2['files']['corpus_tokens.bin']['tokens']} "
          f"entries={m2['extraction']['entries']} id={m2['corpus_id']}")

    # 2) clean SFT v2 stream (clean answers + answer-side labels)
    sft = sft_build(SRC, str(M2_OUT / "tokenizer.json"), SFT_OUT,
                    exclude_words=holdout)
    sftv = sft_verify(SFT_OUT)
    assert sft["corpus_id"] == sftv["corpus_id"]
    assert sft["extraction"]["entries_used"] > 0
    if (EVAL3_OUT / "holdout_words.json").exists():
        frozen_words = set(
            json.loads((EVAL3_OUT / "holdout_words.json").read_text())["words"]
        )
        assert set(sft["extraction"]["excluded_words"]) == frozen_words
    print(f"SFT tokens={sft['files']['sft_tokens.bin']['tokens']} "
          f"entries={sft['extraction']['entries_used']} id={sft['corpus_id']} "
          f"labels={sft['files']['sft_labels.bin']['tokens']}")

    # 3) freeze eval-3 word-holdout corpus
    if EVAL3_OUT.exists():
        print(f"evals3 exists; leaving untouched (delete to re-freeze): {EVAL3_OUT}")
    else:
        tok = CharTokenizer.from_file(str(M2_OUT / "tokenizer.json"))
        tmpl = ChatTemplate.default()
        d = eval3_freeze(SRC, tok, tmpl, EVAL3_OUT)
        print(f"eval3 frozen: qa={d['qa_count']} "
              f"manifest_hash={d['corpus_manifest_hash'][:16]}")

    summary = {
        "holdout_seed": HOLDOUT_SEED,
        "holdout_count": len(holdout),
        "m2_corpus_id": m2["corpus_id"],
        "m2_tokens": m2["files"]["corpus_tokens.bin"]["tokens"],
        "sft_corpus_id": sft["corpus_id"],
        "sft_tokens": sft["files"]["sft_tokens.bin"]["tokens"],
        "tokenizer_hash": m2["tokenizer"]["sha256"],
    }
    (BASE / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PREPARE_OK=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
