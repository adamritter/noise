"""Small-backbone, forward-only key-offset pilot; not a leaderboard submission.

Defaults reproduce the two-seed full-size-hash comparison. Run each arm/seed in
its own process. The exported model.py must accompany the final checkpoint.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
SOURCE_TEXT = Path(__file__).read_text()
sys.path.insert(0, str(ROOT))
import run as reference
import model as architecture
from model import GPT
from run import Cut
from utils.prepare_data import load_rows, TRAIN_ROWS
from utils.eval import evaluate as official_evaluate


@torch.no_grad()
def evaluate(model, rows):
    total = 0.
    for batch in rows.split(1):
        batch = batch.cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            total += model(batch[:, :-1], batch[:, 1:], loss_reduction='sum').item()
    return total / (len(rows) * (rows.shape[1] - 1))


def forward_macs(cfg, batch=1):
    # Dense-equivalent causal attention count (both QK and AV); no scalar ops.
    t, d, l, v = cfg.sequence_len, cfg.n_embd, cfg.n_layer, cfg.vocab_size
    attention_macs = 0
    for i in range(l):
        w = t if i == l-1 or cfg.window_pattern[i % len(cfg.window_pattern)] == 'L' else t//2
        w = min(w, t-1)
        pairs = t*(w+1) - w*(w+1)//2
        attention_macs += 2*pairs*d
    return batch * (l * (12*t*d*d + t*cfg.ngram_dim*d) + attention_macs + t*d*v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['base', 'partial', 'shift'], required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--rows', type=int, default=512)
    ap.add_argument('--width', type=int, default=128)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--ngram-rows', type=int, default=1048576)
    ap.add_argument('--ngram-dim', type=int, default=128)
    ap.add_argument('--fresh-offset', type=int, default=192,
                    help='Start 32 confirmation rows this far from the end of the fixed permutation.')
    ap.add_argument('--data-dir', type=Path, default=ROOT/'data')
    ap.add_argument('--official-validation', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--confirm', action='store_true', help='Evaluate frozen final variant on reserved fresh rows.')
    ap.add_argument('--count-macs', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision('high')
    cfg = architecture.GPTConfig(n_layer=args.layers, n_embd=args.width,
        n_head=args.width//64, n_kv_head=args.width//64, ngram_rows=args.ngram_rows, ngram_dim=args.ngram_dim, partial_rope=args.arm != 'base', key_offset=args.arm == 'shift')
    settings = reference.Settings(model=cfg, cut_batch=1, route_k=min(192, 4*args.width),
        key_path_draws=0, chain_tokens=256, key_path_tokens=64)
    def make():
        torch.manual_seed(args.seed)
        model = GPT(cfg).cuda()
        model.init_weights()
        # Preserve exactly the reference backbone initialization for every arm.
        for name, module in model.named_modules():
            if 'c_proj' in name and isinstance(module, nn.Linear):
                scale = settings.projection_std * (3/module.weight.shape[1])**.5
                nn.init.uniform_(module.weight, -scale, scale)
        nn.init.normal_(model.lm_head.weight, std=settings.head_std)
        return model.requires_grad_(False)
    # Construct shared-reference state once to remove RNG-consumption confounds.
    torch.manual_seed(args.seed)
    reference_model = reference.make_model(torch.device('cuda'), args.seed, cfg,
                                           settings.projection_std, settings.head_std)
    initial = {k: v.clone() for k, v in reference_model.state_dict().items()}
    del reference_model
    def initialized():
        model = make()
        model.load_state_dict(initial, strict=True)
        return model
    model = initialized()
    estimator = Cut(model, settings, args.seed, 0, 1, 1, None)
    noise = torch.randint(0, 4096, (1, 2049), device='cuda')
    print('Warming forward-only estimator', flush=True)
    started = time.perf_counter()
    warm_gradients = estimator.estimate(noise[:, :-1], noise[:, 1:])
    warm_optimizer = reference.make_optimizer(model, settings, args.rows)
    for name, parameter in model.named_parameters():
        parameter.grad = (warm_gradients[name]/2048).to(parameter.dtype)
    warm_optimizer.step()
    del warm_optimizer, warm_gradients
    torch.cuda.synchronize()
    print('Warmup seconds', time.perf_counter()-started, flush=True)
    counted_macs = None
    if args.count_macs:
        from torch.utils.flop_counter import FlopCounterMode
        # Counter dispatch cannot enter the FlexAttention higher-order op.
        # For this untimed arithmetic census only, materialize the same masked
        # attention using ATen matmuls. No training state survives the census.
        def counting_flex(q, k, v, block_mask=None, return_lse=False, enable_gqa=False, **kw):
            if q.shape[1] != k.shape[1]:
                repeat = q.shape[1]//k.shape[1]
                k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
            score = (q.float() @ k.float().transpose(-1,-2)) / q.shape[-1]**.5
            qi = torch.arange(q.shape[-2], device=q.device)[:,None]
            ki = torch.arange(k.shape[-2], device=k.device)[None,:]
            if block_mask is not None:
                score = score.masked_fill(~block_mask.mask_mod(0,0,qi,ki), -float('inf'))
            out = (score.softmax(-1) @ v.float()).to(q.dtype)
            return (out, score.logsumexp(-1)) if return_lse else out
        old_arch, old_ref = architecture.flex, reference.flex
        architecture.flex = reference.flex = counting_flex
        try:
            with FlopCounterMode(display=False) as counter:
                estimator.estimate(noise[:, :-1], noise[:, 1:])
        finally:
            architecture.flex, reference.flex = old_arch, old_ref
        counted_macs = counter.get_total_flops() // 2
        print('ATen estimator MAC estimate', counted_macs, flush=True)
    estimator.close()
    del estimator, model
    model = initialized()
    del initial
    estimator = Cut(model, settings, args.seed, 0, 1, 1, None)
    optimizer = reference.make_optimizer(model, settings, args.rows)
    rows = load_rows(args.data_dir/'train.pt', TRAIN_ROWS)
    permutation = torch.randperm(len(rows), generator=torch.Generator().manual_seed(1))
    training = rows[permutation[:args.rows]]
    # Disjoint, frozen before any observation. Confirmation is not a tuning set.
    dev = rows[permutation[-96:-80]]
    fresh_ids = permutation[-args.fresh_offset:][:32]
    assert args.fresh_offset >= 32 and len(fresh_ids) == 32
    assert not set(fresh_ids.tolist()) & set(permutation[:args.rows].tolist())
    assert not set(fresh_ids.tolist()) & set(permutation[-96:-80].tolist())
    fresh = rows[fresh_ids]
    config = dict(architecture=asdict(cfg), settings=asdict(settings), arm=args.arm,
        seed=args.seed, train_ids=permutation[:args.rows].tolist(), dev_ids=permutation[-96:-80].tolist(),
        fresh_ids=fresh_ids.tolist(), upstream='48e9ea73a7d74c6811a52eaa3deb4c2d76977367',
        parameters=sum(p.numel() for p in model.parameters()), torch=torch.__version__,
        hardware=torch.cuda.get_device_name(), official_score=False)
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    (args.output/'train.py.source').write_text(SOURCE_TEXT)
    (args.output/'estimator.py.source').write_text(Path(__file__).with_name('run.py').read_text())
    # Standalone architecture accepted by the unmodified official evaluator.
    exported = Path(__file__).with_name('model.py').read_text()
    exported += '\nBaseConfig = GPTConfig\n'
    exported += '\ndef GPTConfig():\n    return BaseConfig(**' + repr(asdict(cfg)) + ')\n'
    (args.output/'model.py').write_text(exported)
    losses = []
    torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    for step, row in enumerate(training):
        batch = row[None].cuda()
        x, y = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        reference.set_rate(optimizer, step+1, args.rows, settings.warmdown, settings.final)
        gradients = estimator.estimate(x, y)
        loss = estimator.capture.clean_loss.mean().item()
        optimizer.zero_grad(set_to_none=True)
        for name, parameter in model.named_parameters():
            parameter.grad = (gradients[name]/y.numel()).to(parameter.dtype)
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(name)
        optimizer.step()
        estimator.maps.clear()
        losses.append(loss)
        if step % 32 == 0 or step+1 == args.rows:
            print(json.dumps(dict(step=step+1, loss=loss, seconds=time.perf_counter()-begin)), flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-begin
    peak_memory_allocated = torch.cuda.max_memory_allocated()
    peak_memory_reserved = torch.cuda.max_memory_reserved()
    torch.save(model.state_dict(), args.output/'final.pt')
    # Reload final weights before either held-out evaluation.
    restored = make()
    restored.load_state_dict(torch.load(args.output/'final.pt', weights_only=True))
    original_ce, reloaded_ce = evaluate(model, dev[:1]), evaluate(restored, dev[:1])
    if abs(original_ce-reloaded_ce) > 1e-6:
        raise AssertionError('Final checkpoint forward mismatch')
    dev_ce = evaluate(restored, dev)
    fresh_ce = evaluate(restored, fresh) if args.confirm else None
    result = dict(teacher_forced_dev_ce=dev_ce, teacher_forced_fresh_ce=fresh_ce,
        final_checkpoint_parity_error=abs(original_ce-reloaded_ce),
        seconds=seconds, steps=len(training), prediction_tokens=len(training)*2048,
        peak_memory_allocated=peak_memory_allocated, peak_memory_reserved=peak_memory_reserved,
        probe_full_forwards=0, probe_forward_macs=0,
        clean_training_forward_macs=len(training)*forward_macs(cfg),
        aten_estimator_macs_per_step=counted_macs,
        estimated_accumulated_forward_macs=(len(training)*counted_macs) if counted_macs else None,
        estimator_macs_status='ATen FLOP counter divided by two; excludes custom Triton operations and optimizer; probes use analytical dense causal forward MACs. Approximation, not hardware measurement.',
        losses=losses)
    if args.official_validation:
        result['official_full_validation_ce'] = official_evaluate(torch.load(args.output/'final.pt', map_location='cpu', weights_only=True), args.data_dir, batch_size=1, model_file=args.output/'model.py')
    (args.output/'metrics.jsonl').write_text(''.join(json.dumps(dict(step=i+1, teacher_forced_training_ce=loss))+'\n' for i, loss in enumerate(losses)))
    (args.output/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('losses','probes')}), flush=True)
    estimator.close()


if __name__ == '__main__':
    main()
