"""
Diagnostic script: checks the t_derivative implementation against
PyTorch autograd (ground truth) and surfaces any discrepancy.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from network import Encoder

torch.manual_seed(42)

input_size = 16
latent_dim = 4
enc = Encoder(input_size, latent_dim)
enc.eval()

batch = 8
x    = torch.rand(batch, input_size, requires_grad=False)
xdot = torch.rand(batch, input_size)

# ── Ground truth via autograd (Jacobian-vector product) ──────────────────────
# We want dz/dt = J_enc(x) @ xdot  for each sample in the batch.
# PyTorch's vmap + jacrev gives us the per-sample Jacobian.
def encoder_fn(x_single):
    return enc(x_single.unsqueeze(0)).squeeze(0)

J_all = torch.vmap(torch.func.jacrev(encoder_fn))(x)   # (batch, latent, input)
zdot_autograd = torch.einsum('bli,bi->bl', J_all, xdot)  # (batch, latent)

# ── Manual t_derivative (current code) ───────────────────────────────────────
enc_params = list(enc.parameters())
weights = [w for w in enc_params if len(w.shape) == 2]
biases  = [b for b in enc_params if len(b.shape) == 1]

a    = x
dadt = xdot

for i in range(len(weights) - 1):          # <-- current: skips last relu
    z = torch.matmul(a, weights[i].T) + biases[i]
    a = torch.relu(z)
    dadt = (z > 0).float() * torch.matmul(dadt, weights[i].T)
dadt_current = torch.matmul(dadt, weights[-1].T)

# ── Fixed t_derivative (include last relu mask) ───────────────────────────────
a2    = x
dadt2 = xdot
for i in range(len(weights)):              # <-- fix: include ALL layers
    z2 = torch.matmul(a2, weights[i].T) + biases[i]
    a2 = torch.relu(z2)
    dadt2 = (z2 > 0).float() * torch.matmul(dadt2, weights[i].T)
dadt_fixed = dadt2

# ── Compare ───────────────────────────────────────────────────────────────────
err_current = (dadt_current - zdot_autograd).abs().mean().item()
err_fixed   = (dadt_fixed   - zdot_autograd).abs().mean().item()

print("=== t_derivative correctness check ===")
print(f"  Mean abs error (current code, missing last relu): {err_current:.6f}")
print(f"  Mean abs error (fixed code,  all relu masks):     {err_fixed:.6f}")

if err_fixed < err_current:
    print("\n  CONFIRMED BUG: current t_derivative is missing the relu mask")
    print("  on the last encoder layer. The fixed version matches autograd.")
else:
    print("\n  Current and fixed are equally accurate.")

# ── Check duplicate SINDy features ───────────────────────────────────────────
print("\n=== SINDyLibrary duplicate-feature check ===")
from sindy_library import SINDyLibrary
lib = SINDyLibrary(
    latent_dim=3, include_biases=True, include_states=True,
    include_sin=True, include_cos=True,
    include_multiply_pairs=True, poly_order=2,   # both enabled = potential overlap
    include_sqrt=False, include_inverse=False,
    include_sign_sqrt_of_diff=False, device='cpu')

names = lib.get_feature_names()
dupes = [n for n in names if names.count(n) > 1]
if dupes:
    print(f"  DUPLICATE features found: {sorted(set(dupes))}")
    print(f"  Full list: {names}")
else:
    print("  No duplicates found.")
