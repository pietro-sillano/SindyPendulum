"""
Diagnostic figure: loss curves, Xi coefficient heatmap, phase portrait comparison.

Layout (2×2):
  top-left   — training loss curves (semilogy)
  top-right  — Xi coefficient matrix as annotated heatmap
  bottom-left  — true pendulum phase portrait
  bottom-right — SINDy-reconstructed phase portrait (integrated in latent space)

Xi are the SINDy sparse coefficients: dz/dt = Theta(z) @ Xi.
Each column corresponds to one latent variable; each row to one library function.
Large values → that term matters; near-zero values → pruned by thresholding.

Usage:
    python scripts/plot_diagnostics.py \
        --checkpoint output/model_500epochs.pt \
        --csv output/loss_final.csv \
        [--phase2_start 100] \
        [--out diagnostics.png]
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.integrate import odeint

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',   type=str, required=True,
                   help='path to model_Nepochs.pt checkpoint')
    p.add_argument('--csv',          type=str, required=True,
                   help='path to loss CSV (e.g. loss_final.csv)')
    p.add_argument('--phase2_start', type=int, default=None,
                   help='epoch where SINDy losses activated (draws a vertical line)')
    p.add_argument('--out',          type=str, default=None,
                   help='save figure here; omit to show interactively')
    return p.parse_args()


# ── Loss panel ────────────────────────────────────────────────────────────────

def plot_loss(ax, csv_path, phase2_start):
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    colors = {'recon_loss': 'tab:blue', 'sindy_loss_x': 'tab:orange',
              'sindy_loss_z': 'tab:green', 'sindy_regular_loss': 'tab:red',
              'tot': 'black', 'val': 'gray'}
    styles = {'tot': '--', 'val': ':'}

    for col in df.columns:
        ax.semilogy(df[col], label=col,
                    color=colors.get(col, None),
                    linestyle=styles.get(col, '-'),
                    linewidth=1.5 if col in ('tot', 'val') else 1.0)

    if phase2_start and phase2_start <= len(df):
        ax.axvline(phase2_start, color='purple', linestyle='--', linewidth=0.8,
                   label=f'phase 2 start (ep {phase2_start})')

    ax.set_xlabel('epoch')
    ax.set_ylabel('loss (log scale)')
    ax.set_title('Training losses')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)


# ── Xi heatmap panel ──────────────────────────────────────────────────────────

def plot_xi(ax, XI, mask, feature_names):
    """
    Annotated heatmap of Xi (n_funcs × latent_dim).
    Masked-out coefficients (after sequential thresholding) are shown faded.
    """
    Xi_eff = XI * mask          # what the model actually uses
    n_funcs, latent_dim = XI.shape

    vmax = max(np.abs(Xi_eff).max(), 0.1)
    im = ax.imshow(Xi_eff, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8, label='coefficient value')

    # annotate each cell
    for i in range(n_funcs):
        for j in range(latent_dim):
            val   = Xi_eff[i, j]
            alive = mask[i, j] > 0.5
            txt   = f'{val:.2f}' if alive else '×'
            color = 'white' if abs(val) > 0.4 * vmax else 'black'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=8, color=color,
                    alpha=1.0 if alive else 0.4)

    ax.set_xticks(range(latent_dim))
    ax.set_xticklabels([f'dz{j}/dt' for j in range(latent_dim)])
    ax.set_yticks(range(n_funcs))
    ax.set_yticklabels(feature_names, fontsize=9)
    ax.set_title('Xi coefficients  (× = masked out)')


# ── Phase portrait panels ─────────────────────────────────────────────────────

def pend_true(y, t):
    theta, omega = y
    return [omega, -np.sin(theta)]


def make_sindy_ode(XI, mask, feature_fn):
    import torch
    Xi_masked = (XI * mask).astype(np.float32)
    def ode(z, t):
        z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
        theta = feature_fn(z_tensor).detach().numpy()
        return (theta @ Xi_masked).ravel().tolist()
    return ode


def plot_portraits(ax_true, ax_sindy, XI, mask, feature_fn):
    ta, tb, dt = 0., 25., 0.05
    t = np.arange(ta, tb, dt)
    ics = [(0.7, 1.8), (0.2, 1.6), (-0.5, 1.2), (0.9, 0.8), (-1.2, 0.5)]

    sindy_ode = make_sindy_ode(XI, mask, feature_fn)

    for ic in ics:
        sol = odeint(pend_true, list(ic), t)
        ax_true.plot(sol[:, 0], sol[:, 1], lw=1)

        try:
            sol_r = odeint(sindy_ode, list(ic), t, rtol=1e-4, atol=1e-6)
            ax_sindy.plot(sol_r[:, 0], sol_r[:, 1], lw=1)
        except Exception:
            pass  # diverged ODE — skip this IC

    for ax, title, xl, yl in [
        (ax_true,  'True pendulum',           'theta', 'omega'),
        (ax_sindy, 'SINDy ODE (latent space)', 'z0',   'z1'),
    ]:
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(title)
        ax.set_xlim(-3.5, 3.5); ax.set_ylim(-2.5, 2.5)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    import torch
    from network import Autoencoder

    # load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    enc_w      = ckpt['model_state_dict']['encoder.fc1.weight']
    input_size = enc_w.shape[1]

    # infer latent_dim from decoder output weight shape
    dec_last_w  = ckpt['model_state_dict']['decoder.fc4.weight']
    latent_dim  = ckpt['model_state_dict']['decoder.fc1.weight'].shape[1]

    model = Autoencoder(input_size, latent_dim)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    XI   = ckpt['sindy_coefficients'].numpy()
    mask = ckpt['coefficient_mask'].numpy()
    feature_names = model.SINDyLibrary.get_feature_names()
    epoch = ckpt.get('epoch', '?')

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9), dpi=150)
    gs  = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.35)

    ax_loss  = fig.add_subplot(gs[0, :2])   # top-left (wide)
    ax_xi    = fig.add_subplot(gs[0, 2])    # top-right
    ax_true  = fig.add_subplot(gs[1, :2])   # bottom-left (wide)  → split below
    ax_sindy = None                         # will be carved out

    # split bottom row into two equal axes manually
    ax_true.remove()
    ax_true  = fig.add_subplot(gs[1, 0:1])
    ax_sindy = fig.add_subplot(gs[1, 1:3])

    plot_loss(ax_loss, args.csv, args.phase2_start)
    plot_xi(ax_xi, XI, mask, feature_names)
    plot_portraits(ax_true, ax_sindy, XI, mask, model.SINDyLibrary.transform)

    fig.suptitle(
        f'SINDy-AE diagnostics — epoch {epoch} — {os.path.basename(args.checkpoint)}',
        fontsize=11, y=1.01,
    )

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches='tight')
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
