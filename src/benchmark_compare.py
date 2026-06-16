"""
Comprehensive training-speed benchmark for the SINDy-Autoencoder, producing
two plots in output/:

  1) benchmark_methods.png  -- CPU vs GPU-resident vs GPU-streaming across
     batch sizes (on a subsample that fits in VRAM, float32).

  2) benchmark_dtype.png    -- effect of storage dtype (float32/float16/uint8)
     on the FULL dataset: memory footprint, whether it fits resident in VRAM,
     and ms/epoch for each feasible placement.

Lower-precision storage matters because the dataset bytes are what overflow the
2 GiB MX150: the b&w images (a Gaussian blob on a near-zero background) tolerate
float16/uint8 storage, which shrinks the data enough to live fully in VRAM and
avoid the ~40% per-batch-transfer penalty entirely.

Compute is always float32; the storage dtype is decoded to float32 per batch.
"""
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import network
from data_generator import generate_dataset, _cast_storage
from network import Autoencoder

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'output')
WARMUP = int(os.environ.get('BENCH_WARMUP', 2))
EPOCHS = int(os.environ.get('BENCH_EPOCHS', 6))
DIM = 2601


# ── decoders: storage tensor -> float32 compute tensor ──────────────────────
def decoder_for(storage):
    if storage == 'uint8':
        return lambda t: t.float() / 255.0
    return lambda t: t.float()          # float16 / float32 -> float32


def build_model(device):
    network.device = device
    m = Autoencoder(DIM, 2).to(device)
    return m, torch.optim.Adam(m.parameters(), lr=1e-4)


def run_epoch(model, opt, loader, device, decode):
    model.train()
    nb = device == 'cuda'
    for X_b, Xd_b in loader:
        X_b  = decode(X_b.to(device, non_blocking=nb))
        Xd_b = decode(Xd_b.to(device, non_blocking=nb))
        out = model(X_b, Xd_b)
        tot, _ = model.loss_function(X_b, Xd_b, out[0], out[1], out[3], out[4],
                                     model.XI, 1e-2, 1e-3, 1e-5)
        opt.zero_grad(); tot.backward(); opt.step()


def time_loader(loader, device, decode):
    model, opt = build_model(device)
    for _ in range(WARMUP):
        run_epoch(model, opt, loader, device, decode)
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        run_epoch(model, opt, loader, device, decode)
    if device == 'cuda':
        torch.cuda.synchronize()
    sec = (time.perf_counter() - t0) / EPOCHS
    del model, opt
    if device == 'cuda':
        torch.cuda.empty_cache()
    return sec


def make_loader(Xnp, Xdnp, bs, device, resident, pin):
    tX, tXd = torch.from_numpy(Xnp), torch.from_numpy(Xdnp)
    if resident:
        tX, tXd = tX.to(device), tXd.to(device)
    ds = TensorDataset(tX, tXd)
    return DataLoader(ds, shuffle=True, batch_size=bs, pin_memory=pin)


# ── plot 1: CPU vs GPU-resident vs GPU-streaming (subsample, float32) ───────
def bench_methods(X32, Xd32):
    n_sub = 20000
    idx = torch.randperm(X32.shape[0])[:n_sub].numpy()
    Xs, Xds = X32[idx], Xd32[idx]
    batches = [128, 256, 512, 1024, 2048]
    dec = decoder_for('float32')
    res = {'cpu': [], 'gpu_resident': [], 'gpu_streaming': []}

    print(f"\n[methods] {n_sub:,} samples (float32), batches={batches}")
    for bs in batches:
        ld = make_loader(Xs, Xds, bs, 'cpu', resident=False, pin=False)
        res['cpu'].append(time_loader(ld, 'cpu', dec) * 1e3)
        if torch.cuda.is_available():
            ld = make_loader(Xs, Xds, bs, 'cuda', resident=True, pin=False)
            res['gpu_resident'].append(time_loader(ld, 'cuda', dec) * 1e3)
            ld = make_loader(Xs, Xds, bs, 'cuda', resident=False, pin=True)
            res['gpu_streaming'].append(time_loader(ld, 'cuda', dec) * 1e3)
        print(f"  bs={bs:5d}  cpu={res['cpu'][-1]:8.1f}  "
              f"gpu_res={res['gpu_resident'][-1]:7.1f}  "
              f"gpu_str={res['gpu_streaming'][-1]:7.1f} ms")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(batches, res['cpu'], 'o-', label='CPU')
    ax.plot(batches, res['gpu_resident'], 's-', label='GPU resident (all in VRAM)')
    ax.plot(batches, res['gpu_streaming'], '^-', label='GPU streaming (per-batch)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xticks(batches); ax.set_xticklabels(batches)
    ax.set_xlabel('batch size'); ax.set_ylabel('ms / epoch (log)')
    ax.set_title(f'Training speed by method  ({n_sub:,} samples, float32)\n'
                 f'{torch.cuda.get_device_name(0)}')
    ax.grid(True, which='both', alpha=0.3); ax.legend()
    fig.tight_layout()
    path = os.path.join(OUT, 'benchmark_methods.png')
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  saved {path}")


# ── plot 2: storage dtype trade-off ────────────────────────────────────────
# Footprint/fit is computed analytically for the FULL dataset (the real
# scenario); timing is measured on a light subsample to stay within the
# laptop's limited RAM (full float16+uint8 copies would exhaust it).
def bench_dtype(X32, Xd32, n_full):
    if not torch.cuda.is_available():
        print("\n[dtype] no GPU, skipping")
        return
    bs = 1024
    n_time = X32.shape[0]                       # subsample size used for timing
    free, _ = torch.cuda.mem_get_info()
    reserve = 350 * 1024**2
    sizes = {'float32': 4, 'float16': 2, 'uint8': 1}
    print(f"\n[dtype] full n={n_full:,} (footprint), timing on {n_time:,} samples, "
          f"bs={bs}, free VRAM={free/1024**2:.0f} MiB")

    mem_mib, fits, t_res, t_str = {}, {}, {}, {}
    for name, bsize in sizes.items():
        need = 2 * n_full * DIM * bsize        # analytic full-dataset footprint
        mem_mib[name] = need / 1024**2
        fits[name] = (need + reserve) < free
        dec = decoder_for(name)
        Xn  = _cast_storage(X32, name)         # small (subsample only)
        Xdn = _cast_storage(Xd32, name)
        ld = make_loader(Xn, Xdn, bs, 'cuda', resident=False, pin=True)
        t_str[name] = time_loader(ld, 'cuda', dec) * 1e3
        ld = make_loader(Xn, Xdn, bs, 'cuda', resident=True, pin=False)
        t_res[name] = time_loader(ld, 'cuda', dec) * 1e3
        del Xn, Xdn
        print(f"  {name:8s} full_data={mem_mib[name]:7.0f} MiB  fits={fits[name]!s:5s}  "
              f"resident={t_res[name]:7.1f}  streaming={t_str[name]:7.1f} ms")

    names = list(sizes.keys())
    x = np.arange(len(names)); w = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # left: full-dataset memory footprint vs VRAM budget
    bars = ax1.bar(x, [mem_mib[k] for k in names],
                   color=['#d62728', '#1f77b4', '#2ca02c'])
    budget = (free - reserve) / 1024**2
    ax1.axhline(budget, ls='--', color='k',
                label=f'resident budget ≈ {budget:.0f} MiB')
    for b, k in zip(bars, names):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height(),
                 'fits' if fits[k] else 'too big', ha='center', va='bottom',
                 fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(names)
    ax1.set_ylabel('dataset size (MiB, X+Xdot)')
    ax1.set_title(f'Full-dataset footprint by dtype (n={n_full:,})')
    ax1.legend(); ax1.grid(True, axis='y', alpha=0.3)

    # right: ms/epoch resident vs streaming (measured on subsample)
    ax2.bar(x - w / 2, [t_res[k] for k in names], w, label='resident')
    ax2.bar(x + w / 2, [t_str[k] for k in names], w, label='streaming')
    ax2.set_xticks(x); ax2.set_xticklabels(names)
    ax2.set_ylabel('ms / epoch')
    ax2.set_title(f'Speed by dtype & placement\n(measured on {n_time:,} samples, bs={bs})')
    ax2.legend(); ax2.grid(True, axis='y', alpha=0.3)

    fig.suptitle(f'Storage dtype trade-off  ({torch.cuda.get_device_name(0)})')
    fig.tight_layout()
    path = os.path.join(OUT, 'benchmark_dtype.png')
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  saved {path}")


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Generating full dataset (float32)...")
    X32, Xd32 = generate_dataset(n_ics=50, dtype='float32')
    print(f"  X: {X32.shape}")
    n_full = X32.shape[0]
    sections = sys.argv[1:] or ['methods', 'dtype']
    if 'methods' in sections:
        bench_methods(X32, Xd32)
    if 'dtype' in sections:
        # time on a light subsample; free the full arrays first (limited RAM)
        idx = torch.randperm(n_full)[:20000].numpy()
        Xs, Xds = X32[idx].copy(), Xd32[idx].copy()
        del X32, Xd32
        bench_dtype(Xs, Xds, n_full)
    print("\nDone.")


if __name__ == '__main__':
    main()
