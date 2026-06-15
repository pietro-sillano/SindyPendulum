"""
Compare true pendulum phase portrait against the SINDy-reconstructed dynamics.

Loads a trained model checkpoint, extracts the learned Xi coefficients, and
integrates the identified ODE to produce the reconstructed phase portrait.

Usage:
    python scripts/plot_trajectories.py --checkpoint output/model_100epochs.pt
                                        [--latent_dim 2] [--out trajectories.png]
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--latent_dim', type=int, default=2)
    p.add_argument('--out',        type=str, default=None)
    return p.parse_args()


def pend_true(y, t):
    theta, omega = y
    return [omega, -np.sin(theta)]


def make_sindy_ode(XI, mask, feature_fn):
    """Return an ODE function dz/dt = Theta(z) @ (XI * mask)."""
    XI_masked = (XI * mask)
    def ode(z, t):
        z_t = np.array(z, dtype=np.float32)
        import torch
        z_tensor = torch.tensor(z_t).unsqueeze(0)
        theta = feature_fn(z_tensor).detach().numpy()  # (1, n_funcs)
        dzdt  = theta @ XI_masked                       # (1, latent_dim)
        return dzdt.ravel().tolist()
    return ode


def main():
    args = parse_args()
    import torch
    from network import Autoencoder

    device = 'cpu'
    ckpt   = torch.load(args.checkpoint, map_location=device)

    # reconstruct model to get SINDy library
    # input_size is unknown from checkpoint alone — infer from encoder weight shape
    enc_w = ckpt['model_state_dict']['encoder.fc1.weight']
    input_size = enc_w.shape[1]

    model = Autoencoder(input_size, args.latent_dim)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    XI   = ckpt['sindy_coefficients'].numpy()       # (n_funcs, latent_dim)
    mask = ckpt['coefficient_mask'].numpy()

    feature_fn = model.SINDyLibrary.transform

    # ── Phase portrait settings ───────────────────────────────────────────────
    ta, tb, dt = 0., 20., 0.05
    t = np.arange(ta, tb, dt)
    init_conditions = [(0.7, 1.8), (0.2, 1.6), (0.4, 1.0), (0.1, 0.6)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    sindy_ode = make_sindy_ode(XI, mask, feature_fn)

    for ic in init_conditions:
        # true pendulum
        sol = odeint(pend_true, list(ic), t)
        ax1.plot(sol[:, 0], sol[:, 1], lw=1)

        # reconstructed (SINDy ODE in latent space, starting from same ic)
        sol_r = odeint(sindy_ode, list(ic), t)
        ax2.plot(sol_r[:, 0], sol_r[:, 1], lw=1)

    ax1.set_xlabel('theta'); ax1.set_ylabel('omega')
    ax1.set_title('True pendulum attractor')
    ax1.set_xlim(-3.5, 3.5); ax1.set_ylim(-2.5, 2.5)

    ax2.set_xlabel('z0'); ax2.set_ylabel('z1')
    ax2.set_title('SINDy-reconstructed dynamics')
    ax2.set_xlim(-3.5, 3.5); ax2.set_ylim(-2.5, 2.5)

    plt.suptitle(f'Checkpoint: {os.path.basename(args.checkpoint)}')
    plt.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches='tight')
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
