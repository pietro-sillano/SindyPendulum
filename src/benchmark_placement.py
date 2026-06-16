"""
GPU: resident (all data in VRAM) vs streaming (batches moved per iteration).

Uses a dataset that fits in VRAM so both placements are possible, and times
identical training epochs for each across several batch sizes.
"""
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import network
from data_generator import generate_dataset
from network import Autoencoder

N_SAMPLES = 20000
BATCH_SIZES = [256, 1024, 2048]
WARMUP, EPOCHS = 2, 6
DEV = 'cuda'


def build_model():
    network.device = DEV
    m = Autoencoder(2601, 2).to(DEV)
    return m, torch.optim.Adam(m.parameters(), lr=1e-4)


def run_epoch(model, opt, loader):
    model.train()
    for X_b, Xd_b in loader:
        X_b  = X_b.to(DEV, non_blocking=True)
        Xd_b = Xd_b.to(DEV, non_blocking=True)
        out = model(X_b, Xd_b)
        tot, _ = model.loss_function(X_b, Xd_b, out[0], out[1], out[3], out[4],
                                     model.XI, 1e-2, 1e-3, 1e-5)
        opt.zero_grad(); tot.backward(); opt.step()


def time_it(ds, bs, pin):
    loader = DataLoader(ds, shuffle=True, batch_size=bs, pin_memory=pin)
    model, opt = build_model()
    for _ in range(WARMUP):
        run_epoch(model, opt, loader)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(EPOCHS):
        run_epoch(model, opt, loader)
    torch.cuda.synchronize()
    sec = (time.perf_counter() - t0) / EPOCHS
    del model, opt, loader
    torch.cuda.empty_cache()
    return sec


def main():
    X, Xdot = generate_dataset(n_ics=50)
    idx = torch.randperm(X.shape[0])[:N_SAMPLES].numpy()
    Xc  = torch.from_numpy(X[idx]).float()
    Xdc = torch.from_numpy(Xdot[idx]).float()
    print(f"\nGPU: {torch.cuda.get_device_name(0)}  |  {N_SAMPLES:,} samples, dim {Xc.shape[1]}")
    print(f"epochs={EPOCHS} (+{WARMUP} warmup)\n")

    # resident: data preloaded to GPU once
    ds_res = TensorDataset(Xc.to(DEV), Xdc.to(DEV))
    # streaming: data stays on (pinned) CPU
    ds_str = TensorDataset(Xc, Xdc)

    print("  Batch  | Resident ms | Streaming ms | Overhead")
    print("  -------|-------------|--------------|---------")
    for bs in BATCH_SIZES:
        r = time_it(ds_res, bs, pin=False)
        s = time_it(ds_str, bs, pin=True)
        print(f"  {bs:6d} | {r*1e3:11.1f} | {s*1e3:12.1f} | {(s/r-1)*100:+6.1f}%")


if __name__ == '__main__':
    main()
