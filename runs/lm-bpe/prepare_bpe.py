"""Build a char-ID-aligned BPE train stream WITHOUT touching the eval ruler.

Key design decision for "same hash BPE":
* The BPE tokenizer is learned from the FROZEN base documents only, so all
  special ids and every used character id are byte-identical to the frozen
  char tokenizer. The 380 subword pieces occupy exactly the char tokenizer's
  former <unused_N> slots (7812..8191).
* Training uses BPE encoding of the same ext1 documents as runs/lm-scaled.
* Evaluation still loads runs/corpus/lm/holdout_tokens.bin (char ids) directly.
  No re-tokenisation, no new manifest, no new spec. Therefore eval_lm_hash
  stays 2b70b6f6793432ab and strict A/B against the char model is legal.
* This is deliberately conservative: a BPE model may put probability mass on
  subword ids, which are simply wrong answers under the char target stream.

Outputs under --run (default runs/lm-bpe):
    tokenizer.json       learned BPE tokenizer (diagnostics/generation)
    train_tokens.bin     uint16 BPE train stream
    bpe.json             sealed provenance and alignment checks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autotrain.corpus.pipeline import CorpusError, read_documents  # noqa: E402
from autotrain.export.contract import sha256_file  # noqa: E402
from autotrain.inference.tokenizer import BPETokenizer, load_tokenizer  # noqa: E402

FROZEN_CORPUS = ROOT / "runs/corpus/lm"
BASE_SOURCE = ROOT / "runs/corpus/sources/wiki_zh.jsonl"
EXT_SOURCE = ROOT / "runs/corpus/sources/wiki_zh_ext.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs/lm-bpe"))
    args = ap.parse_args()
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)

    # 1) frozen char tokenizer and split must be intact
    manifest = json.loads((FROZEN_CORPUS / "manifest.json").read_text(encoding="utf-8"))
    char = load_tokenizer(FROZEN_CORPUS / "tokenizer.json")
    if char.type != "char" or char.as_dict()["sha256"] != manifest["tokenizer"]["sha256"]:
        raise CorpusError("frozen char tokenizer mismatch")
    if sha256_file(FROZEN_CORPUS / "tokenizer.json") != manifest["files"]["tokenizer.json"]["sha256"]:
        raise CorpusError("frozen tokenizer file changed on disk")
    holdout_idx = set(manifest["split"]["holdout_indices"])

    # 2) learn BPE from the SAME full base documents that built the char tokenizer
    base_docs = read_documents(BASE_SOURCE)
    bpe = BPETokenizer.build_from_text(
        "".join(base_docs), vocab_size=char.vocab_size,
        source={"corpus_version": manifest["version"],
                "tokenizer_kind": "bpe",
                "note": "char-id-aligned BPE from frozen base docs; eval unchanged"},
    )
    if bpe.vocab_size != char.vocab_size:
        raise CorpusError("BPE vocab size drifted from frozen char vocab")
    if any(char.id_to_token[i] != bpe.id_to_token[i] for i in range(4)):
        raise CorpusError("special token ids are not aligned with frozen char tokenizer")
    char_ids = list(range(4, 4 + char.used_chars))
    if any(char.id_to_token[i] != bpe.id_to_token[i] for i in char_ids):
        raise CorpusError("character ids are not aligned with frozen char tokenizer; "
                          "frozen holdout tokens would mean different characters")

    char_unused = {i for i, t in enumerate(char.id_to_token) if t.startswith("<unused_")}
    bpe_piece_ids = {i for i, t in enumerate(bpe.id_to_token) if t in bpe.pieces}
    if char_unused != bpe_piece_ids:
        raise CorpusError("BPE pieces must occupy exactly the char tokenizer's unused slots")
    print(f"[align] char_ids=4..{char_ids[-1]} pieces={bpe.n_merges} "
          f"piece_ids={min(bpe_piece_ids)}..{max(bpe_piece_ids)} aligned=True")

    # 3) reconstruct the exact base train text and verify against frozen char stream
    base_train_text = "".join(d for i, d in enumerate(base_docs) if i not in holdout_idx)
    frozen_train = np.fromfile(FROZEN_CORPUS / "train_tokens.bin", dtype="<u2")
    char_train = np.asarray(char.encode(base_train_text), dtype="<u2")
    if char_train.size != frozen_train.size or not np.array_equal(char_train, frozen_train):
        raise CorpusError("reconstructed base train text does not match frozen char train stream")
    print(f"[verify] base train char tokens={char_train.size} match frozen stream")

    # 4) append ext1 documents, encode the combined train text with BPE
    ext_text = "".join(read_documents(EXT_SOURCE))
    train_text = base_train_text + ext_text
    train_ids = bpe.encode(train_text)
    unk_id = bpe.unk_token_id
    n_unk = sum(1 for t in train_ids if t == unk_id)
    unk_rate = n_unk / max(1, len(train_ids))
    if unk_rate > 0.15:
        raise CorpusError(f"BPE train UNK rate {unk_rate:.2%} too high — aborting")

    tok_path = run / "tokenizer.json"
    tok_doc = bpe.save(tok_path)
    out_path = run / "train_tokens.bin"
    np.asarray(train_ids, dtype="<u2").tofile(out_path)

    prov = {
        "experiment": "wanxiang-lm-bpe",
        "method": "char-id-aligned-bpe-on-ext1-train-only",
        "eval_corpus": {
            "path": str(FROZEN_CORPUS.resolve()),
            "corpus_id": manifest["corpus_id"],
            "holdout_sha256": manifest["files"]["holdout_tokens.bin"]["sha256"],
            "tokenizer_sha256": manifest["tokenizer"]["sha256"],
            "note": "UNCHANGED — eval_lm_hash must stay 2b70b6f6793432ab; "
                    "BPE model is scored on the frozen CHAR holdout tokens directly",
        },
        "tokenizer": {
            "char_sha256": char.as_dict()["sha256"],
            "bpe_sha256": tok_doc["sha256"],
            "vocab_size": bpe.vocab_size,
            "used_chars": bpe.used_chars,
            "n_merges": bpe.n_merges,
            "subword_id_range": [min(bpe_piece_ids), max(bpe_piece_ids)],
            "char_id_alignment": True,
        },
        "train_stream": {
            "path": str(out_path.resolve()),
            "sha256": sha256_file(out_path),
            "bytes": out_path.stat().st_size,
            "base_chars": len(base_train_text),
            "ext1_chars": len(ext_text),
            "bpe_tokens": len(train_ids),
            "char_equivalent_tokens": int(frozen_train.size + len(char.encode(ext_text))),
            "unk_count": n_unk,
            "unk_rate": unk_rate,
        },
        "alignment": {
            "special_ids_identical": True,
            "used_char_ids_identical": True,
            "pieces_occupy_char_unused_slots": True,
        },
        "note": "eval_diagnostic: use run/tokenizer.json only for generation decoding; "
                "score_bundle must keep using the frozen corpus tokenizer",
    }
    (run / "bpe.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[write] {out_path} bpe_tokens={len(train_ids)} "
          f"char_equivalent={prov['train_stream']['char_equivalent_tokens']} "
          f"unk_rate={unk_rate:.4%}")
    print(f"[seal] {run / 'bpe.json'} bpe_sha={tok_doc['sha256'][:16]}")


if __name__ == "__main__":
    main()
