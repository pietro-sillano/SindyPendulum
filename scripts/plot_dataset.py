"""
Visualise a few frames of the pendulum image dataset.

Usage:
    python scripts/plot_dataset.py --x X.npy [--xdot Xdot.npy]
                                   [--nx 51] [--frames 3] [--out dataset.png]
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--x',      type=str, required=True)
    p.add_argument('--xdot',   type=str, default=None)
    p.add_argument('--nx',     type=int, default=51)
    p.add_argument('--frames', type=int, default=3, help='number of frames to show')
    p.add_argument('--start',  type=int, default=78, help='starting frame index')
    p.add_argument('--out',    type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    X = np.load(args.x)
    show_dot = args.xdot is not None
    Xdot = np.load(args.xdot) if show_dot else None

    rows = 2 if show_dot else 1
    fig, axes = plt.subplots(rows, args.frames, figsize=(4 * args.frames, 4 * rows), dpi=150)
    if rows == 1:
        axes = axes[np.newaxis, :]

    indices = [args.start + i * 7 for i in range(args.frames)]
    for col, idx in enumerate(indices):
        img = X[idx].reshape(args.nx, args.nx)
        axes[0, col].imshow(img, cmap='plasma', interpolation='none')
        axes[0, col].set_title(f'X[{idx}]')
        axes[0, col].axis('off')
        if show_dot:
            dimg = Xdot[idx].reshape(args.nx, args.nx)
            axes[1, col].imshow(dimg, cmap='coolwarm', interpolation='none')
            axes[1, col].set_title(f'Xdot[{idx}]')
            axes[1, col].axis('off')

    axes[0, 0].set_ylabel('X', rotation=0, labelpad=30, va='center')
    if show_dot:
        axes[1, 0].set_ylabel('Xdot', rotation=0, labelpad=40, va='center')

    plt.suptitle('Pendulum image dataset samples')
    plt.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches='tight')
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
