"""
Plot training loss curves from a saved CSV.

Usage:
    python scripts/plot_loss.py --csv output/loss_final.csv [--out loss.png]
"""
import argparse
import matplotlib.pyplot as plt
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=str, required=True, help='path to loss CSV')
    p.add_argument('--out', type=str, default=None,  help='save figure to this path')
    return p.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    # drop index column if present
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for col in df.columns:
        ax.semilogy(df[col], label=col)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss (log scale)')
    ax.legend(loc='best')
    ax.set_title(f'Training losses — {args.csv}')
    plt.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches='tight')
        print(f"Saved to {args.out}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
