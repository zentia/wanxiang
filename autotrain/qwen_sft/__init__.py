"""QLoRA adaptation of an open code-LLM base on the anonymised local corpus.

This is a NEW model family: the base tokenizer has ~152k tokens, so every
artifact from the from-scratch experiments (bundle contract, corpus_id,
eval_lm_hash, gen_hash, sft_hash) is INCOMPARABLE and must be rebuilt.

What survives from the earlier work: the anonymised source jsonl, the
document-level frozen split, block-aligned sample extraction, answer-side
loss masking, seal/verify discipline, and the objective generation ruler.
"""
