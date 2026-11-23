"""One-shot data prep for the eval-3 BPE (subword) A/B experiment.

Builds under runs/eval3-bpe/:
  m2_corpus/   pretrain stream, holdout words EXCLUDED, BPE tokenizer
  sft_corpus/  SFT v2 stream, holdout EXCLUDED, cleaned answers + label mask

The frozen evals3/ corpus is NEVER regenerated: the same 150 questions and
references are used for both the char and BPE arms. The only changed variable
is tokenizer encoding granularity; vocab_size stays 8192.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from autotrain.eval3.corpus import verify_corpus  # noqa: E402
from autotrain.inference import BPETokenizer, CharTokenizer, load_tokenizer  # noqa: E402
from autotrain.m2.pipeline import build as m2_build, verify as m2_verify  # noqa: E402
from autotrain.sft.corpus import build as sft_build, verify as sft_verify  # noqa: E402
from autotrain.split import HOLDOUT_SEED, holdout_words  # noqa: E402

SRC = ROOT / "runs/m2/sources/idiom.json"
BASE = ROOT / "runs/eval3-bpe"
M2_OUT = BASE / "m2_corpus"
SFT_OUT = BASE / "sft_corpus"
CHAR_TOK = ROOT / "runs/eval3-clean/m2_corpus/tokenizer.json"
EVAL3_DIR = ROOT / "evals3"


def main() -> None:
    rows = json.loads(SRC.read_text(encoding="utf-8"))
    holdout = set(holdout_words(rows, seed=HOLDOUT_SEED))
    frozen = verify_corpus(EVAL3_DIR)
    frozen_words = set(
        json.loads((EVAL3_DIR / "holdout_words.json").read_text(encoding="utf-8"))["words"]
    )
    assert frozen_words == holdout, "frozen eval3 holdout must match split.py"
    print(f"holdout_words={len(holdout)} seed={HOLDOUT_SEED} "
          f"eval3_hash={frozen['corpus_manifest_hash'][:16]}")

    # 1) BPE pretrain stream. Tokenizer is learned from the FULL text so the
    # held-out words remain exactly encodable, while their rows are excluded
    # from the written token stream.
    m2 = m2_build(SRC, M2_OUT, exclude_words=holdout,
                  source_meta={"name": "idiom"}, tokenizer_kind="bpe")
    m2v = m2_verify(M2_OUT)
    assert m2["corpus_id"] == m2v["corpus_id"]
    assert set(m2["extraction"]["excluded_words"]) == holdout

    tok = load_tokenizer(M2_OUT / "tokenizer.json")
    assert isinstance(tok, BPETokenizer)
    char_tok = CharTokenizer.from_file(CHAR_TOK)
    assert tok.vocab_size == char_tok.vocab_size == 8192
    assert tok.type == "bpe" and char_tok.type == "char"
    assert tok.as_dict()["sha256"] != char_tok.as_dict()["sha256"]
    missing = [ch for q in frozen_words for ch in q if tok.encode(ch)[0] == tok.unk_token_id]
    assert not missing, f"holdout chars not encodable: {missing[:8]}"
    print(f"M2  tokens={m2['files']['corpus_tokens.bin']['tokens']} "
          f"entries={m2['extraction']['entries']} id={m2['corpus_id']}")
    print(f"TOK type={tok.type} vocab={tok.vocab_size} pieces={tok.n_merges} "
          f"used_chars={tok.used_chars} hash={tok.as_dict()['sha256'][:16]}")

    # 2) BPE SFT v2 stream (same clean answers as the frozen eval instrument).
    sft = sft_build(SRC, str(M2_OUT / "tokenizer.json"), SFT_OUT,
                    exclude_words=holdout)
    sftv = sft_verify(SFT_OUT)
    assert sft["corpus_id"] == sftv["corpus_id"]
    assert sft["extraction"]["entries_used"] > 0
    assert set(sft["extraction"]["excluded_words"]) == holdout
    print(f"SFT tokens={sft['files']['sft_tokens.bin']['tokens']} "
          f"entries={sft['extraction']['entries_used']} id={sft['corpus_id']} "
          f"labels={sft['files']['sft_labels.bin']['tokens']}")

    summary = {
        "holdout_seed": HOLDOUT_SEED,
        "holdout_count": len(holdout),
        "eval3_corpus_manifest_hash": frozen["corpus_manifest_hash"],
        "m2_corpus_id": m2["corpus_id"],
        "m2_tokens": m2["files"]["corpus_tokens.bin"]["tokens"],
        "sft_corpus_id": sft["corpus_id"],
        "sft_tokens": sft["files"]["sft_tokens.bin"]["tokens"],
        "sft_labels": sft["files"]["sft_labels.bin"]["tokens"],
        "tokenizer_type": tok.type,
        "tokenizer_hash": tok.as_dict()["sha256"],
        "tokenizer_vocab_size": tok.vocab_size,
        "tokenizer_n_merges": tok.n_merges,
        "char_tokenizer_hash": char_tok.as_dict()["sha256"],
    }
    (BASE / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PREPARE_OK=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
