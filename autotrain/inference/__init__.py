"""Wanxiang inference service (M1: token-id level, see INFERENCE.md).

Public surface::

    from autotrain.inference import InferenceEngine, GenerationConfig
    engine = InferenceEngine.from_bundle("runs/android-verify/bundle")
    result = engine.generate([1, 2, 3], GenerationConfig(max_new_tokens=16))
    print(result.token_ids)
"""

from .chat import ChatTemplate, ChatTemplateError
from .engine import GenerationConfig, GenerationResult, InferenceEngine, default_threads
from .tokenizer import (BOS_ID, EOS_ID, PAD_ID, UNK_ID, BPETokenizer,
                        BPETokenizerError, CharTokenizer, CharTokenizerError,
                        IdentityTokenizer, Tokenizer, TokenizerUnavailable,
                        load_tokenizer, validate_token_ids)

__all__ = [
    "ChatTemplate",
    "ChatTemplateError",
    "BOS_ID", "EOS_ID", "PAD_ID", "UNK_ID",
    "InferenceEngine",
    "GenerationConfig",
    "GenerationResult",
    "IdentityTokenizer",
    "CharTokenizer",
    "CharTokenizerError",
    "BPETokenizer",
    "BPETokenizerError",
    "load_tokenizer",
    "Tokenizer",
    "TokenizerUnavailable",
    "validate_token_ids",
    "default_threads",
]
