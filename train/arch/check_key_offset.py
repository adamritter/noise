"""Forward-only regression checks for the experimental raw-key offset.

Run on a bf16-capable CUDA GPU: uv run train/arch/check_key_offset.py
No autograd or backward oracle is used.
"""
from dataclasses import replace
import json

import torch
from torch.nn import functional as F

import model as architecture
import run as training


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    torch.manual_seed(31)
    torch.set_float32_matmul_precision('highest')
    layer = architecture.ShiftedKey(16, 2).double()
    # Boundary cases include a single token, which has no previous/following neighbor.
    adjoint_errors = []
    for length in (1, 2, 7):
        x = torch.randn(2, length, 16, dtype=torch.float64)
        error = torch.randn_like(x)
        raw = F.linear(x, layer.weight).view(2, length, 2, 8)
        wanted = raw.clone()
        for lo, hi in ((2, 4), (6, 8)):
            wanted[:, 0, :, lo:hi] = 0
            wanted[:, 1:, :, lo:hi] = raw[:, :-1, :, lo:hi]
        torch.testing.assert_close(layer(x), wanted.reshape_as(x), atol=0, rtol=0)
        left = (layer(x)*error).sum()
        right = (raw.reshape_as(x)*layer.unshift_credit(error)).sum()
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
        adjoint_errors.append(float((left-right).abs()))
    direction = torch.randn_like(layer.weight)
    original = layer.weight.clone()
    eps = 1e-4
    linear_credit = layer.unshift_credit(error).reshape(-1, 16).T @ x.reshape(-1, 16)
    layer.weight.copy_(original+eps*direction)
    plus = (layer(x)*error).sum()
    layer.weight.copy_(original-eps*direction)
    minus = (layer(x)*error).sum()
    measured, expected = (plus-minus)/(2*eps), (linear_credit*direction).sum()
    torch.testing.assert_close(measured, expected, atol=1e-8, rtol=1e-8)

    cfg = architecture.GPTConfig(n_layer=2, n_embd=128, n_head=2, n_kv_head=2,
                                  ngram_rows=128, ngram_dim=32, partial_rope=True, key_offset=True)
    settings = training.Settings(model=cfg, key_path_draws=0, chain_tokens=64)
    for change in ({'key_path_draws': 1}, {'future_draws': 1}, {'switch': .1}):
        try:
            replace(settings, **change)
        except ValueError:
            pass
        else:
            raise AssertionError(('unsupported probe combination accepted', change))
    try:
        replace(cfg, partial_rope=False)
    except ValueError:
        pass
    else:
        raise AssertionError('Shift without stationary RoPE lanes accepted')
    model = training.make_model(torch.device('cuda'), 0, cfg, settings.projection_std, settings.head_std)
    assert (model.cos[..., 16:] == 1).all() and (model.sin[..., 16:] == 0).all()
    tokens = torch.randint(0, cfg.vocab_size, (1, 65), device='cuda')
    changed = tokens.clone()
    changed[:, 32:] = (changed[:, 32:]+1) % cfg.vocab_size
    with torch.autocast('cuda', dtype=torch.bfloat16):
        original, intervened = model(tokens), model(changed)
    causal_error = (original[:, :32]-intervened[:, :32]).abs().max().item()
    assert causal_error == 0
    print(json.dumps(dict(adjoint_errors=adjoint_errors,
                         key_weight_directional_fd_error=float((measured-expected).abs()),
                         future_intervention_max_error=causal_error,
                         unsupported_probe_combinations_rejected=True, no_autograd=True)))


if __name__ == '__main__':
    main()
