"""
SINDy-Autoencoder training pipeline for the pendulum system.

Two-phase curriculum:
  Phase 1 (epoch < phase2_start): pure reconstruction, SINDy losses off
  Phase 2 (epoch >= phase2_start): SINDy losses active with alpha1/alpha2

Usage:
    python src/train.py [--n_ics 50] [--epochs 500] [--latent_dim 2]
                        [--batch_size 1024] [--lr 1e-4] [--out output/]
                        [--seq_thres_every 100] [--save_every 100]
                        [--phase2_start 100] [--alpha1 1e-2] [--alpha2 1e-3]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_generator import generate_dataset
from network import Autoencoder

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser(description="Train SINDy-AE on pendulum images")
    p.add_argument('--n_ics',           type=int,   default=50)
    p.add_argument('--epochs',          type=int,   default=500)
    p.add_argument('--latent_dim',      type=int,   default=2)
    p.add_argument('--batch_size',      type=int,   default=1024)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--seq_thres_every', type=int,   default=100,
                   help='apply sequential thresholding every N epochs (0 = off)')
    p.add_argument('--save_every',      type=int,   default=100,
                   help='save checkpoint every N epochs (0 = final only)')
    p.add_argument('--out',             type=str,   default='output/')
    # two-phase curriculum
    p.add_argument('--phase2_start',    type=int,   default=100,
                   help='epoch at which SINDy losses activate (0 = from start)')
    p.add_argument('--alpha1',          type=float, default=1e-2,
                   help='weight for sindy_loss_x in phase 2')
    p.add_argument('--alpha2',          type=float, default=1e-3,
                   help='weight for sindy_loss_z in phase 2')
    p.add_argument('--alpha3',          type=float, default=1e-5,
                   help='weight for L1 sparsity regularisation (both phases)')
    return p.parse_args()


def make_loaders(X, Xdot, batch_size, val_frac=0.1):
    X_t  = torch.from_numpy(X).float().to(device)
    Xd_t = torch.from_numpy(Xdot).float().to(device)
    ds   = TensorDataset(X_t, Xd_t)
    n_val   = round(len(ds) * val_frac)
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
    return (DataLoader(train_ds, shuffle=True,  batch_size=batch_size),
            DataLoader(val_ds,   shuffle=False, batch_size=batch_size))


def train_epoch(model, optimizer, loader, alpha1, alpha2, alpha3):
    model.train()
    totals = {}
    for X_b, Xd_b in loader:
        xtilde, xtildedot, z, zdot, zdot_hat = model(X_b, Xd_b)
        tot, ld = model.loss_function(
            X_b, Xd_b, xtilde, xtildedot, zdot, zdot_hat, model.XI,
            alpha1=alpha1, alpha2=alpha2, alpha3=alpha3,
        )
        optimizer.zero_grad()
        tot.backward()
        optimizer.step()
        for k, v in ld.items():
            totals[k] = totals.get(k, 0.0) + v.item()
    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def val_epoch(model, loader, alpha1, alpha2, alpha3):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for X_b, Xd_b in loader:
            xtilde, xtildedot, z, zdot, zdot_hat = model(X_b, Xd_b)
            tot, _ = model.loss_function(
                X_b, Xd_b, xtilde, xtildedot, zdot, zdot_hat, model.XI,
                alpha1=alpha1, alpha2=alpha2, alpha3=alpha3,
            )
            total += tot.item()
            n += 1
    return total / n


def save_checkpoint(model, optimizer, epoch, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'sindy_coefficients': model.XI.detach().cpu(),
        'coefficient_mask':   model.XI_coefficient_mask.cpu(),
    }, os.path.join(out_dir, f'model_{epoch}epochs.pt'))
    np.save(os.path.join(out_dir, f'model_{epoch}epochs.npy'),
            model.XI.detach().cpu().numpy())


def print_equations(model, out_dir, epoch):
    mask  = model.XI_coefficient_mask.cpu().detach().numpy()
    XI    = model.XI.cpu().detach().numpy()
    names = model.SINDyLibrary.get_feature_names()
    lines = []
    for j in range(XI.shape[1]):
        for i, name in enumerate(names):
            coeff = XI[i, j] * mask[i, j]
            if abs(coeff) >= 0.1:
                lines.append(f"dz{j}/dt = {coeff:+.4f} * {name}")
    text = "\n".join(lines) if lines else "(no active terms above threshold)"
    print(f"  Equations at epoch {epoch}:\n    " + text.replace("\n", "\n    "))
    path = os.path.join(out_dir, f'equations_{epoch}.txt')
    with open(path, 'w') as f:
        f.write(text + "\n")


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"\nGenerating data (n_ics={args.n_ics})...")
    X, Xdot = generate_dataset(n_ics=args.n_ics)
    print(f"  X: {X.shape}  Xdot: {Xdot.shape}")

    train_loader, val_loader = make_loaders(X, Xdot, args.batch_size)
    input_size = X.shape[1]
    del X, Xdot

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nBuilding model (latent_dim={args.latent_dim})...")
    model     = Autoencoder(input_size, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params  = sum(p.numel() for p in model.parameters())
    features  = model.SINDyLibrary.get_feature_names()
    print(f"  Parameters: {n_params:,}")
    print(f"  SINDy library ({len(features)} functions): {features}")
    print(f"\n  Phase 1: epochs 1–{args.phase2_start - 1}  (recon only)")
    print(f"  Phase 2: epochs {args.phase2_start}+  "
          f"(alpha1={args.alpha1}, alpha2={args.alpha2}, alpha3={args.alpha3})")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs...")
    loss_log = {k: [] for k in
                ['recon_loss', 'sindy_loss_x', 'sindy_loss_z', 'sindy_regular_loss', 'tot']}
    val_log  = []

    for epoch in range(1, args.epochs + 1):
        # curriculum: zero SINDy weights until phase 2
        if epoch < args.phase2_start:
            a1, a2 = 0.0, 0.0
        else:
            a1, a2 = args.alpha1, args.alpha2

        train_ld = train_epoch(model, optimizer, train_loader, a1, a2, args.alpha3)
        val_loss = val_epoch(model, val_loader, a1, a2, args.alpha3)

        for k in loss_log:
            loss_log[k].append(train_ld[k])
        val_log.append(val_loss)

        if epoch % 10 == 0 or epoch == args.epochs or epoch == args.phase2_start:
            phase = "P1" if epoch < args.phase2_start else "P2"
            print(f"  [{phase}] epoch {epoch:4d} | "
                  f"recon={train_ld['recon_loss']:.3e} | "
                  f"sindy_z={train_ld['sindy_loss_z']:.3e} | "
                  f"sindy_x={train_ld['sindy_loss_x']:.3e} | "
                  f"val={val_loss:.3e}")

        # sequential thresholding (phase 2 only, so latent space is stable first)
        if (args.seq_thres_every > 0
                and epoch >= args.phase2_start
                and (epoch - args.phase2_start) % args.seq_thres_every == 0
                and epoch > args.phase2_start):
            model.XI_coefficient_mask = (model.XI.abs() > 0.1).float().detach()
            active = int(model.XI_coefficient_mask.sum().item())
            print(f"  [epoch {epoch}] sequential threshold applied — {active} active terms")

        # checkpoint
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, args.out)
            print_equations(model, args.out, epoch)
            df = pd.DataFrame(loss_log)
            df['val'] = val_log
            df.to_csv(os.path.join(args.out, f'loss_{epoch}epochs.csv'), index=False)

    # ── Final save ────────────────────────────────────────────────────────────
    save_checkpoint(model, optimizer, args.epochs, args.out)
    print_equations(model, args.out, args.epochs)
    df = pd.DataFrame(loss_log)
    df['val'] = val_log
    df.to_csv(os.path.join(args.out, 'loss_final.csv'), index=False)
    print(f"\nDone. Outputs saved to {args.out}")


if __name__ == '__main__':
    main()
