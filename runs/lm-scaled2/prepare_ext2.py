"""Append a SECOND disjoint Wikipedia block to the TRAIN stream only.

Same discipline as runs/lm-scaled/prepare_ext.py:
* runs/corpus/lm stays byte-for-byte frozen (holdout, tokenizer, manifest);
* rows 1200..1999 are disjoint from base rows 0..399 and ext rows 400..1199;
* the prior extended stream is a byte prefix of the new stream, so cursor
  semantics and all prior checkpoints remain structurally valid;
* the frozen char tokenizer is reused verbatim; OOV -> <unk>, with the rate
  measured and sealed into expansion.json.

Outputs under --run (default runs/lm-scaled2):
    train_tokens.bin     uint16 train stream (base + ext1 + ext2)
    expansion.json       sealed provenance and integrity checks
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
from autotrain.corpus.source import fetch  # noqa: E402
from autotrain.export.contract import sha256_file  # noqa: E402
from autotrain.inference.tokenizer import load_tokenizer  # noqa: E402

FROZEN_CORPUS = ROOT / "runs/corpus/lm"
PRIOR_RUN = ROOT / "runs/lm-scaled"
SOURCES_DIR = ROOT / "runs/corpus/sources"
BASE_SOURCE = SOURCES_DIR / "wiki_zh.jsonl"
EXT_SOURCE = SOURCES_DIR / "wiki_zh_ext.jsonl"
EXT2_SOURCE = SOURCES_DIR / "wiki_zh_ext2.jsonl"


def _read_ids(path: Path) -> list[str]:
    # split("\\n") only: Wikipedia text contains U+2028/U+0085, which
    # str.splitlines() would mistake for JSONL record boundaries.
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    return [str(row.get("id")) for row in rows if row.get("id") is not None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs/lm-scaled2"))
    ap.add_argument("--source", default="wiki_zh_ext2")
    args = ap.parse_args()
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)

    # 1) fetch and seal the new source block
    meta = fetch(args.source, str(SOURCES_DIR))
    print(f"[fetch] rows={meta['rows']} sha256={meta['sha256'][:16]}")

    # 2) frozen eval corpus must be untouched
    manifest = json.loads((FROZEN_CORPUS / "manifest.json").read_text(encoding="utf-8"))
    tok_path = FROZEN_CORPUS / "tokenizer.json"
    tok = load_tokenizer(tok_path)
    if tok.as_dict()["sha256"] != manifest["tokenizer"]["sha256"]:
        raise CorpusError("frozen tokenizer checksum mismatch — will not expand")
    if sha256_file(tok_path) != manifest["files"]["tokenizer.json"]["sha256"]:
        raise CorpusError("frozen tokenizer file changed on disk")

    # 3) prior stream must be the sealed ext1 stream
    prior_prov = json.loads((PRIOR_RUN / "expansion.json").read_text(encoding="utf-8"))
    prior_path = PRIOR_RUN / "train_tokens.bin"
    if sha256_file(prior_path) != prior_prov["train_stream"]["sha256"]:
        raise CorpusError("prior lm-scaled train stream was modified — refusing to append")
    prior = np.fromfile(prior_path, dtype="<u2")
    base_train = np.fromfile(FROZEN_CORPUS / "train_tokens.bin", dtype="<u2")
    if prior.size < base_train.size or not np.array_equal(prior[:base_train.size], base_train):
        raise CorpusError("prior stream does not extend the frozen base train prefix")

    # 4) document-id disjointness across all three source blocks
    base_ids, ext_ids, ext2_ids = map(_read_ids, (BASE_SOURCE, EXT_SOURCE, EXT2_SOURCE))
    if not (base_ids and ext_ids and ext2_ids):
        raise CorpusError("one or more source blocks have no document ids")
    if set(base_ids) & set(ext_ids) or set(base_ids) & set(ext2_ids) or set(ext_ids) & set(ext2_ids):
        raise CorpusError("source blocks overlap by document id — refusing to train on eval text")
    print(f"[disjoint] base={len(base_ids)} ext={len(ext_ids)} ext2={len(ext2_ids)} overlap=0")

    # 5) encode ext2 with the frozen tokenizer
    docs = read_documents(EXT2_SOURCE)
    ext2_text = "".join(docs)
    ext2_tokens = tok.encode(ext2_text)
    unk_id = tok.unk_token_id
    n_unk = sum(1 for t in ext2_tokens if t == unk_id)
    unk_rate = n_unk / max(1, len(ext2_tokens))
    print(f"[encode] ext2_chars={len(ext2_text)} ext2_tokens={len(ext2_tokens)} "
          f"unk={n_unk} unk_rate={unk_rate:.4%}")
    if unk_rate > 0.15:
        raise CorpusError(f"ext2 UNK rate {unk_rate:.2%} too high — aborting")

    # 6) append; prefix must remain byte-identical
    new_arr = np.concatenate([prior, np.asarray(ext2_tokens, dtype="<u2")])
    out_path = run / "train_tokens.bin"
    new_arr.tofile(out_path)
    recheck = np.fromfile(out_path, dtype="<u2")
    prefix_ok = (
        recheck.size == prior.size + len(ext2_tokens)
        and np.array_equal(recheck[:prior.size], prior)
    )
    if not prefix_ok:
        raise CorpusError("new stream prefix does not match prior stream — cursor would break")

    prov = {
        "experiment": "wanxiang-lm-scaled2",
        "method": "append-second-disjoint-wiki-block-to-train-only",
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
            "base_rows": len(base_ids), "ext_rows": len(ext_ids),
            "ext2_rows": len(ext2_ids), "overlapping_ids": 0,
        },
        "train_stream": {
            "path": str(out_path.resolve()),
            "sha256": sha256_file(out_path),
            "bytes": out_path.stat().st_size,
            "base_tokens": int(base_train.size),
            "ext1_tokens": int(prior.size - base_train.size),
            "ext2_tokens": len(ext2_tokens),
            "prior_tokens": int(prior.size),
            "total_tokens": int(new_arr.size),
            "prefix_identical_to_prior": prefix_ok,
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
    print(f"[write] {out_path} tokens={new_arr.size} "
          f"(prior {prior.size} + ext2 {len(ext2_tokens)})")
    print(f"[seal] prefix_identical={prefix_ok} unk_rate={unk_rate:.4%} "
          f"-> {run / 'expansion.json'}")


if __name__ == "__main__":
    main()
