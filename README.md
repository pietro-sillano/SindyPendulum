# SindyPendulum

PyTorch implementation of the **SINDy-Autoencoder** architecture from
[Champion et al. (2019)](https://arxiv.org/abs/1904.02107), applied to a
pendulum observed as high-dimensional image sequences.

Read about it on my [blog](https://pietro-sillano.github.io/projects/2.SINDY/)!

---

## What it does

The model learns a low-dimensional latent representation of pendulum video frames
and simultaneously identifies the governing equations of motion in that latent
space using the Sparse Identification of Nonlinear Dynamics (SINDy) framework.

**Architecture:**

```
pixel image x (51×51 = 2601)
        │
    Encoder  (2601 → 128 → 64 → 32 → latent_dim)
        │
    latent z  ──► SINDy library Θ(z) ──► Ξ (sparse coefficients)
        │                                  │
    Decoder  (latent_dim → 32 → 64 → 128 → 2601)
        │
  reconstructed x̃
```

**Loss** (three terms):

| Term | Description |
|---|---|
| `recon_loss` | MSE between input and reconstruction |
| `sindy_loss_z` | consistency of ż = Θ(z)Ξ with the encoder Jacobian |
| `sindy_loss_x` | consistency of ẋ̃ with the decoder Jacobian applied to ż |
| `sindy_regular_loss` | L1 regularisation on Ξ to promote sparsity |

---

## Project structure

```
src/
  data_generator.py   — pendulum ODE integration + Gaussian image generation
  sindy_library.py    — SINDy feature library (states, sin, cos, poly, …)
  network.py          — Encoder, Decoder, Autoencoder modules
  train.py            — training pipeline (CLI)

scripts/
  plot_loss.py              — plot training loss from saved CSV
  plot_xi_distribution.py   — histogram of SINDy coefficients Ξ
  plot_trajectories.py      — true vs SINDy-reconstructed phase portrait
  plot_dataset.py           — visualise dataset image frames
  test_improvements.py      — smoke tests for all src/ modules
  check_backprop.py         — verifies t_derivative against PyTorch autograd

data/
  sample.mp4          — example pendulum video

paper/
  1904.02107.pdf      — Champion et al. (2019) reference paper
```

---

## Setup

```bash
micromamba create -n sindy python=3.11 pytorch=2.2.2 torchvision cpuonly \
  numpy=1.26 scipy matplotlib pandas -c pytorch -c conda-forge
micromamba activate sindy
```

Or with pip (after creating a Python 3.11 environment):

```bash
pip install -r requirements.txt
```

---

## Usage

### Train

```bash
python src/train.py \
  --n_ics 50 \          # grid size for initial conditions (50×50 grid, ~1400 valid ICs)
  --epochs 500 \
  --latent_dim 2 \      # pendulum has 2 physical degrees of freedom
  --batch_size 1024 \
  --lr 1e-4 \
  --seq_thres_every 100 \   # apply sequential thresholding every 100 epochs
  --save_every 100 \
  --out output/
```

Outputs written to `--out`:
- `model_Nepochs.pt` — full checkpoint (weights + optimizer state + Ξ)
- `model_Nepochs.npy` — Ξ coefficients only
- `loss_Nepochs.csv` — loss history
- `equations_N.txt` — active SINDy terms above threshold

### Diagnostic plots

```bash
# Loss curves
python scripts/plot_loss.py --csv output/loss_final.csv --out loss.png

# Xi coefficient distribution at a checkpoint
python scripts/plot_xi_distribution.py --path output/ --epoch 500 --out xi.png
# or for all checkpoints at once:
python scripts/plot_xi_distribution.py --path output/ --all --out_dir output/plots/

# Phase portrait: true pendulum vs SINDy-identified dynamics
python scripts/plot_trajectories.py --checkpoint output/model_500epochs.pt --out trajectories.png

# Dataset sample images
python scripts/plot_dataset.py --x X.npy --xdot Xdot.npy --out dataset.png
```

### Generate dataset only

```bash
python src/data_generator.py --n_ics 100
# writes X.npy and Xdot.npy to the current directory
```

---

## Key implementation notes

- **`t_derivative`**: propagates `ẋ` through the network via the chain rule
  (`da_l/dt = relu'(z_l) · W_l · da_{l-1}/dt`) to obtain `ż` without
  autograd, keeping the SINDy consistency loss differentiable w.r.t. Ξ.
- **`Xdot` formula**: exact time derivative of the Gaussian image blob,
  `dI/dt = I · 40ω · [-(x−cₓ)·sin(θ+π/2) + (y−cᵧ)·cos(θ+π/2)]`.
- **SINDy library**: `poly_deg_2` generates squared terms only (`zᵢ²`);
  cross terms (`zᵢzⱼ`) come from `multiply_pairs`, so both can be enabled
  simultaneously without duplicating features.
