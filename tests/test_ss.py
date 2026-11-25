"""Pure-function tests for the exposure-bias (scheduled sampling) intervention.

These lock the semantics that matter: targets must stay real, position 0 must
never be perturbed, the ramp is deterministic, and the default is a no-op.
"""

from __future__ import annotations

import pytest
import torch

from autotrain.trainer.runner import prediction_aware_perturb, ss_probability


def test_ss_probability_ramp():
    kw = dict(max_prob=0.3, ramp_steps=200, start_step=4800)
    assert ss_probability(0, **kw) == 0.0
    assert ss_probability(4800, **kw) == 0.0
    assert ss_probability(4900, **kw) == pytest.approx(0.15)
    assert ss_probability(5000, **kw) == pytest.approx(0.3)
    assert ss_probability(6000, **kw) == pytest.approx(0.3)  # clamped


def test_ss_disabled_is_a_strict_noop():
    assert ss_probability(999, max_prob=0.0, ramp_steps=200,
                          start_step=0) == 0.0
    x = torch.tensor([[1, 2, 3, 4]])
    logits = torch.zeros(1, 4, 8)
    gen = torch.Generator().manual_seed(0)
    assert torch.equal(prediction_aware_perturb(x, logits, 0.0, gen), x)


def test_perturb_uses_shifted_argmax_and_never_touches_position_zero():
    x = torch.tensor([[1, 2, 3, 4]])
    logits = torch.full((1, 4, 10), -1.0)
    logits[:, :, 7] = 5.0  # argmax is token 7 at every position
    gen = torch.Generator().manual_seed(1)
    out = prediction_aware_perturb(x, logits, 1.0, gen)
    assert out[0, 0].item() == 1, "no history at position 0"
    assert bool((out[0, 1:] == 7).all()), "replaced by shifted prediction"
    assert out.dtype == x.dtype and out.shape == x.shape


def test_perturb_is_deterministic_given_seed():
    x = torch.arange(16).reshape(1, 16)
    logits = torch.randn(1, 16, 32, generator=torch.Generator().manual_seed(9))
    a = prediction_aware_perturb(x, logits, 0.5,
                                 torch.Generator().manual_seed(42))
    b = prediction_aware_perturb(x, logits, 0.5,
                                 torch.Generator().manual_seed(42))
    assert torch.equal(a, b)
    assert not torch.equal(a, x), "some positions should actually change"


def test_targets_are_not_perturbed():
    """The intervention changes inputs only; the caller keeps the real y."""
    x = torch.tensor([[1, 2, 3, 4]])
    y = torch.tensor([[2, 3, 4, -1]])  # shifted real targets, one ignored
    logits = torch.full((1, 4, 10), -1.0)
    logits[:, :, 7] = 5.0
    x_pert = prediction_aware_perturb(
        x, logits, 1.0, torch.Generator().manual_seed(3)
    )
    assert not torch.equal(x_pert, x)
    assert torch.equal(y, torch.tensor([[2, 3, 4, -1]]))
