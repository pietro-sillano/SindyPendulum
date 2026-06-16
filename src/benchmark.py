"""
CPU vs GPU training-speed benchmark for the SINDy-Autoencoder.

Times a few full training epochs (warmup excluded) for several batch sizes on
each available device. Data is kept on CPU and each batch is moved to the device
inside the loop -- required here because the full dataset (~1.4 GiB per tensor)
does not fit in the 2 GiB MX150, and is the realistic streaming pattern anyway.

Usage:
    python src/benchmark.py [--n_ics 50] [--epochs 3] [--warmup 1]
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import network
from data_generator import generate_dataset
from network import Autoencoder

BATCH_SIZES = [128, 256, 512, 1024, 2048]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_ics',      type=int, default=50)
    p.add_argument('--epochs',     type=int, default=2, help='measured epochs')
    p.add_argument('--warmup',     type=int, default=1, help='warmup epochs (untimed)')
    p.add_argument('--max_samples', type=int, default=20000,
                   help='subsample dataset to this many rows (0 = use all)')
    p.add_argument('--latent_dim', type=int, default=2)
    p.add_argument('--lr',         type=float, default=1e-4)
    return p.parse_args()


def build_model(device, input_size, latent_dim, lr):
    """Build an Autoencoder fully placed on `device`, working around the
    module-level device hardcoding in network.py."""
    network.device = device                      # used by Autoencoder.__init__
    model = Autoencoder(input_size, latent_dim).to(device)
    model.SINDyLibrary.device = device           # plain attr, not moved by .to()
    model.XI_coefficient_mask = model.XI_coefficient_mask.to(device)  # plain tensor
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def run_epoch(model, optimizer, loader, device):
    model.train()
    non_block = (device == 'cuda')
    for X_b, Xd_b in loader:
        X_b  = X_b.to(device, non_blocking=non_block)
        Xd_b = Xd_b.to(device, non_blocking=non_block)
        xtilde, xtildedot, z, zdot, zdot_hat = model(X_b, Xd_b)
        tot, _ = model.loss_function(
            X_b, Xd_b, xtilde, xtildedot, zdot, zdot_hat, model.XI,
            alpha1=1e-2, alpha2=1e-3, alpha3=1e-5,
        )
        optimizer.zero_grad()
        tot.backward()
        optimizer.step()


def benchmark(device, ds, input_size, batch_size, args):
    """Return sec_per_epoch, or raise on OOM."""
    torch.manual_seed(0)
    loader = DataLoader(ds, shuffle=True, batch_size=batch_size,
                        pin_memory=(device == 'cuda'))
    model, optimizer = build_model(device, input_size, args.latent_dim, args.lr)

    for _ in range(args.warmup):
        run_epoch(model, optimizer, loader, device)
    if device == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.epochs):
        run_epoch(model, optimizer, loader, device)
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    del model, optimizer, loader
    if device == 'cuda':
        torch.cuda.empty_cache()
    return elapsed / args.epochs


def main():
    args = parse_args()
    print(f"Generating data (n_ics={args.n_ics})...")
    X, Xdot = generate_dataset(n_ics=args.n_ics)
    if args.max_samples and X.shape[0] > args.max_samples:
        idx = torch.randperm(X.shape[0])[:args.max_samples].numpy()
        X, Xdot = X[idx], Xdot[idx]
        print(f"  subsampled to {args.max_samples:,} rows for benchmarking")
    n, input_size = X.shape
    print(f"  X: {X.shape}  ({n:,} samples, dim {input_size})")

    # CPU-resident dataset; batches are streamed to the device inside the loop.
    ds = TensorDataset(torch.from_numpy(X).float(),
                       torch.from_numpy(Xdot).float())

    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Measured epochs: {args.epochs} (+{args.warmup} warmup)\n")

    results = {}
    for device in devices:
        for bs in BATCH_SIZES:
            try:
                sec = benchmark(device, ds, input_size, bs, args)
                results[(device, bs)] = sec
                print(f"  {device:4s}  bs={bs:5d}  "
                      f"{sec*1e3:8.1f} ms/epoch  {n/sec:10,.0f} samples/s")
            except RuntimeError as e:
                results[(device, bs)] = None
                tag = 'OOM' if 'out of memory' in str(e).lower() else 'ERR'
                print(f"  {device:4s}  bs={bs:5d}  {tag}: {str(e)[:55]}")
                if device == 'cuda':
                    torch.cuda.empty_cache()

    if 'cuda' in devices:
        print("\n  Batch  | CPU ms  | GPU ms  | Speedup")
        print("  -------|---------|---------|--------")
        for bs in BATCH_SIZES:
            c, g = results.get(('cpu', bs)), results.get(('cuda', bs))
            cs = f"{c*1e3:7.1f}" if c else "   n/a "
            gs = f"{g*1e3:7.1f}" if g else "   OOM "
            sp = f"{c/g:6.2f}x" if (c and g) else "    -  "
            print(f"  {bs:6d} | {cs} | {gs} | {sp}")


if __name__ == '__main__':
    main()
