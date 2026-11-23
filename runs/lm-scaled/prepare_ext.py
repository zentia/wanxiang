"""Grow the TRAIN stream with disjoint Wikipedia rows, keeping eval frozen.

Design invariants (critical for a valid A/B):
* The eval instrument keeps pointing at the ORIGINAL frozen corpus
  (``runs/corpus/lm``): holdout_tokens.bin, tokenizer.json and manifest.json
  are byte-for-byte untouched. So eval_lm_hash stays identical.
* The new source (rows 400..1199) is disjoint from the base rows 0..399 that
  produced the 20 holdout documents. No eval text ever enters training.
* The extended train stream's PREFIX equals the original train stream bytes,
  so a restored checkpoint's absolute data_cursor stays valid.
* The frozen tokenizer is reused verbatim; OOV characters in new rows map to
  <unk> (standard fixed-vocab behaviour). The UNK rate is measured and sealed
  into provenance so the expansion cannot silently flood training with <unk>.

Outputs (under --run, default runs/lm-scaled):
    train_tokens.bin     extended uint16 stream for the trainer
    expansion.json       sealed provenance + integrity checks
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autotrain.corpus.pipeline import (  # noqa: E402
    CorpusError,
    clean_text,
    read_documents,
)
from autotrain.corpus.source import fetch  # noqa: E402
from autotrain.export.contract import sha256_file  # noqa: E402
from autotrain.inference.tokenizer import load_tokenizer  # noqa: E402

FROZEN_CORPUS = ROOT / "runs/corpus/lm"
BASE_SOURCE = ROOT / "runs/corpus/sources/wiki_zh.jsonl"
EXT_SOURCE = ROOT / "runs/corpus/sources/wiki_zh_ext.jsonl"


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs/lm-scaled"))
    ap.add_argument("--source", default="wiki_zh_ext")
    args = ap.parse_args()
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)

    # 1) fetch (cached on re-run, sha256 sealed)
    meta = fetch(args.source, str(ROOT / "runs/corpus/sources"))
    print(f"[fetch] rows={meta['rows']} sha256={meta['sha256'][:16]}")

    # 2) load the FROZEN tokenizer and verify it against the frozen manifest
    manifest = json.loads((FROZEN_CORPUS / "manifest.json").read_text(encoding="utf-8"))
    tok_path = FROZEN_CORPUS / "tokenizer.json"
    tok = load_tokenizer(tok_path)
    if tok.as_dict()["sha256"] != manifest["tokenizer"]["sha256"]:
        raise CorpusError("frozen tokenizer checksum mismatch — will not expand")
    if sha256_file(tok_path) != manifest["files"]["tokenizer.json"]["sha256"]:
        raise CorpusError("frozen tokenizer file changed on disk")

    # 3) verify base / extension row ranges are disjoint by document id/url
    # NOTE: split on "\n" only — str.splitlines() also breaks on U+2028/U+0085
    # which occur inside Wikipedia text and would truncate a JSON line.
    base_rows = [json.loads(l) for l in BASE_SOURCE.read_text(encoding="utf-8").split("\n") if l.strip()]
    ext_rows = [json.loads(l) for l in EXT_SOURCE.read_text(encoding="utf-8").split("\n") if l.strip()]
    base_ids = {r.get("id") for r in base_rows}
    ext_ids = {r.get("id") for r in ext_rows}
    overlap = base_ids & ext_ids
    if overlap:
        raise CorpusError(f"base/extension document id overlap: {len(overlap)} rows — refusing")
    print(f"[disjoint] base={len(base_rows)} ext={len(ext_rows)} overlap={len(overlap)}")

    # 4) read + clean extension docs, encode with the FROZEN tokenizer
    docs = read_documents(EXT_SOURCE)
    ext_text = "".join(docs)
    ext_ids_tokens = tok.encode(ext_text)
    unk_id = tok.unk_token_id
    n_unk = sum(1 for t in ext_ids_tokens if t == unk_id)
    unk_rate = n_unk / max(1, len(ext_ids_tokens))
    print(f"[encode] ext_chars={len(ext_text)} ext_tokens={len(ext_ids_tokens)} "
          f"unk={n_unk} unk_rate={unk_rate:.4%} unk_id={unk_id}")
    if unk_rate > 0.15:
        raise CorpusError(
            f"UNK rate {unk_rate:.2%} too high — extension would flood training "
            f"with <unk>; aborting (consider a larger frozen vocab)"
        )

    # 5) append to the ORIGINAL train stream, preserving byte-identical prefix
    base_train = np.fromfile(FROZEN_CORPUS / "train_tokens.bin", dtype="<u2")
    new_arr = np.concatenate([base_train, np.asarray(ext_ids_tokens, dtype="<u2")])
    out_path = run / "train_tokens.bin"
    new_arr.tofile(out_path)

    recheck = np.fromfile(out_path, dtype="<u2")
    prefix_ok = (
        recheck.size == base_train.size + len(ext_ids_tokens)
        and np.array_equal(recheck[: base_train.size], base_train)
    )
    if not prefix_ok:
        raise CorpusError("extended stream prefix does not match original — cursor would break")

    base_sha = sha256_file(FROZEN_CORPUS / "train_tokens.bin")
    prov = {
        "experiment": "wanxiang-lm-scaled",
        "method": "append-disjoint-wiki-rows-to-train-only",
        "eval_corpus": {
            "path": str(FROZEN_CORPUS.resolve()),
            "corpus_id": manifest["corpus_id"],
            "holdout_sha256": manifest["files"]["holdout_tokens.bin"]["sha256"],
            "tokenizer_sha256": manifest["tokenizer"]["sha256"],
            "note": "UNCHANGED — eval_lm_hash must stay 2b70b6f6793432ab",
        },
        "extension_source": {
            "name": meta["name"], "pin": meta["pin"], "sha256": meta["sha256"],
            "rows": meta["rows"], "bytes": meta["bytes"],
        },
        "disjointness": {
            "base_rows": len(base_rows), "ext_rows": len(ext_rows),
            "overlapping_ids": len(overlap),
        },
        "train_stream": {
            "path": str(out_path.resolve()),
            "sha256": sha256_file(out_path),
            "bytes": out_path.stat().st_size,
            "base_tokens": int(base_train.size),
            "base_sha256": base_sha,
            "appended_tokens": len(ext_ids_tokens),
            "total_tokens": int(new_arr.size),
            "prefix_identical_to_base": prefix_ok,
        },
        "tokenizer": {
            "kind": tok.type,
            "sha256": tok.as_dict()["sha256"],
            "unk_id": unk_id,
            "unk_count": n_unk,
            "unk_rate": unk_rate,
            "note": "frozen tokenizer reused verbatim; OOV -> <unk>",
        },
    }
    (run / "expansion.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[write] {out_path} tokens={new_arr.size} (base {base_train.size} + ext {len(ext_ids_tokens)})")
    print(f"[seal] prefix_identical={prefix_ok} unk_rate={unk_rate:.4%} "
          f"-> {run / 'expansion.json'}")


if __name__ == "__main__":
    main()
