"""
Plot the distribution of SINDy coefficients Xi from saved checkpoints.

Usage:
    python scripts/plot_xi_distribution.py --path output/ --epoch 100 [--out xi.png]
    python scripts/plot_xi_distribution.py --path output/ --all    [--out_dir output/plots/]
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--path',    type=str, required=True, help='directory containing model_Nepochs.npy files')
    p.add_argument('--epoch',   type=int, default=None,  help='specific epoch to plot')
    p.add_argument('--all',     action='store_true',     help='plot all available checkpoints')
    p.add_argument('--out',     type=str, default=None,  help='save single figure here')
    p.add_argument('--out_dir', type=str, default=None,  help='save all figures here (for --all)')
    return p.parse_args()


def plot_epoch(xi_flat, epoch, out_path=None):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.hist(xi_flat, bins=40, log=True)
    ax.set_xlabel('Xi coefficient value')
    ax.set_ylabel('count (log scale)')
    ax.set_title(f'Xi coefficient distribution — epoch {epoch}')
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def main():
    args = parse_args()

    if args.all:
        files = sorted(glob.glob(os.path.join(args.path, 'model_*epochs.npy')))
        if not files:
            print(f"No checkpoint .npy files found in {args.path}")
            return
        out_dir = args.out_dir or args.path
        os.makedirs(out_dir, exist_ok=True)
        for fpath in files:
            basename = os.path.basename(fpath)
            epoch = int(basename.replace('model_', '').replace('epochs.npy', ''))
            xi = np.load(fpath).ravel()
            out = os.path.join(out_dir, f'xi_dist_{epoch}.png')
            plot_epoch(xi, epoch, out_path=out)
            print(f"  saved {out}")
    else:
        epoch = args.epoch
        fpath = os.path.join(args.path, f'model_{epoch}epochs.npy')
        if not os.path.exists(fpath):
            print(f"File not found: {fpath}")
            return
        xi = np.load(fpath).ravel()
        plot_epoch(xi, epoch, out_path=args.out)


if __name__ == '__main__':
    main()
