"""Frozen generation-quality ruler.

The teacher-forced rulers (``eval_lm`` / ``sft``) measure next-token
cross-entropy, which CANNOT see autoregressive loop/degeneration failures.
This module scores greedy continuations on a frozen set of holdout prompts
with objective repetition statistics, so "the model repeats itself" becomes a
number instead of an eyeballed impression.

``gen_hash`` binds only the ruler (prompt set, lengths, seed, corpus). Decode
controls are deliberately NOT part of the hash: they are the variable under
test, so reports with different decode configs stay comparable.
"""
