"""
SINDy-Autoencoder training pipeline for the pendulum system.

Usage:
    python src/train.py [--n_ics 50] [--epochs 100] [--latent_dim 2]
                        [--batch_size 1024] [--lr 1e-4] [--out output/]
                        [--seq_thres_every 100] [--save_every 50]
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
    p.add_argument('--n_ics',            type=int,   default=50)
    p.add_argument('--epochs',           type=int,   default=100)
    p.add_argument('--latent_dim',       type=int,   default=2)
    p.add_argument('--batch_size',       type=int,   default=1024)
    p.add_argument('--lr',               type=float, default=1e-4)
    p.add_argument('--seq_thres_every',  type=int,   default=100,
                   help='apply sequential thresholding every N epochs (0 = off)')
    p.add_argument('--save_every',       type=int,   default=50,
                   help='save checkpoint every N epochs (0 = final only)')
    p.add_argument('--out',              type=str,   default='output/')
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


def train_epoch(model, optimizer, loader):
    model.train()
    totals = {}
    for X_b, Xd_b in loader:
        xtilde, xtildedot, z, zdot, zdot_hat = model(X_b, Xd_b)
        tot, ld = model.loss_function(X_b, Xd_b, xtilde, xtildedot, zdot, zdot_hat, model.XI)
        optimizer.zero_grad()
        tot.backward()
        optimizer.step()
        for k, v in ld.items():
            totals[k] = totals.get(k, 0.0) + v.item()
    n = len(loader)
    return {k: v / n for k, v in totals.items()}


def val_epoch(model, loader):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for X_b, Xd_b in loader:
            xtilde, xtildedot, z, zdot, zdot_hat = model(X_b, Xd_b)
            tot, _ = model.loss_function(X_b, Xd_b, xtilde, xtildedot, zdot, zdot_hat, model.XI)
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

    # ── Data ─────────────────────────────────────────────────────────────────
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

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs...")
    loss_log = {k: [] for k in
                ['recon_loss', 'sindy_loss_x', 'sindy_loss_z', 'sindy_regular_loss', 'tot']}
    val_log  = []

    for epoch in range(1, args.epochs + 1):
        train_ld = train_epoch(model, optimizer, train_loader)
        val_loss = val_epoch(model, val_loader)

        for k in loss_log:
            loss_log[k].append(train_ld[k])
        val_log.append(val_loss)

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:4d} | "
                  f"recon={train_ld['recon_loss']:.3e} | "
                  f"sindy_z={train_ld['sindy_loss_z']:.3e} | "
                  f"sindy_x={train_ld['sindy_loss_x']:.3e} | "
                  f"val={val_loss:.3e}")

        # sequential thresholding
        if args.seq_thres_every > 0 and epoch % args.seq_thres_every == 0:
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
