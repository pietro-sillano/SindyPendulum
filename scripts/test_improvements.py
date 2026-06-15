"""
Quick smoke-test for the refactored src/ modules.
Runs on CPU, uses tiny inputs so it finishes in seconds.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch

# ── 1. SINDyLibrary ──────────────────────────────────────────────────────────
print("=== SINDyLibrary ===")
from sindy_library import SINDyLibrary

latent_dim = 3
lib = SINDyLibrary(
    latent_dim=latent_dim,
    include_biases=True,
    include_states=True,
    include_sin=True,
    include_cos=True,
    include_multiply_pairs=True,
    poly_order=2,
    include_sqrt=False,
    include_inverse=False,
    include_sign_sqrt_of_diff=False,
    device='cpu',
)
print(f"  feature names ({lib.number_candidate_functions}): {lib.get_feature_names()}")

z = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 0.0]])
theta = lib.transform(z)
print(f"  transform output shape: {theta.shape}  (expected [2, {lib.number_candidate_functions}])")
assert theta.shape == (2, lib.number_candidate_functions), "SINDyLibrary shape mismatch"
print("  PASS")

# ── 2. Autoencoder (forward + loss) ──────────────────────────────────────────
print("\n=== Autoencoder forward + loss ===")
from network import Autoencoder

input_size = 64   # tiny: 8x8 flat image
latent_dim = 4
model = Autoencoder(input_size, latent_dim)
model.eval()

batch = 16
x    = torch.rand(batch, input_size)
xdot = torch.rand(batch, input_size)

with torch.no_grad():
    xtilde, xtildedot, z, zdot, zdot_hat = model(x, xdot)

print(f"  xtilde shape:    {xtilde.shape}")
print(f"  xtildedot shape: {xtildedot.shape}")
print(f"  z shape:         {z.shape}")
print(f"  zdot shape:      {zdot.shape}")
print(f"  zdot_hat shape:  {zdot_hat.shape}")

assert xtilde.shape    == (batch, input_size),  "xtilde shape wrong"
assert xtildedot.shape == (batch, input_size),  "xtildedot shape wrong"
assert z.shape         == (batch, latent_dim),  "z shape wrong"
assert zdot.shape      == (batch, latent_dim),  "zdot shape wrong"
assert zdot_hat.shape  == (batch, latent_dim),  "zdot_hat shape wrong"

tot, loss_dict = model.loss_function(x, xdot, xtilde, xtildedot, zdot, zdot_hat, model.XI)
print(f"  loss keys:  {list(loss_dict.keys())}")
print(f"  total loss: {tot.item():.4f}")
assert tot.item() >= 0, "negative loss"
print("  PASS")

# ── 3. Backward pass (gradients flow) ────────────────────────────────────────
print("\n=== Backward pass ===")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

x    = torch.rand(batch, input_size)
xdot = torch.rand(batch, input_size)

xtilde, xtildedot, z, zdot, zdot_hat = model(x, xdot)
tot, _ = model.loss_function(x, xdot, xtilde, xtildedot, zdot, zdot_hat, model.XI)
tot.backward()

grads_ok = all(p.grad is not None for p in model.parameters())
print(f"  all parameters received gradients: {grads_ok}")
assert grads_ok, "some parameters have no gradient"

optimizer.step()
optimizer.zero_grad()
print("  PASS")

# ── 4. data_generator (vectorised image gen, small scale) ─────────────────────
print("\n=== data_generator vectorised image_gen ===")
import numpy as np
from scipy.integrate import odeint

NX, NY = 16, 16
ta, tb, dt = 0., 0.5, 0.05
t_arr = np.arange(ta, tb, dt)   # 10 time steps

def pend(y, t):
    theta, omega = y
    return [omega, -np.sin(theta)]

x_grid = np.linspace(-1.5, 1.5, NX)
y_grid = np.linspace(-1.5, 1.5, NY)
xx, yy = np.meshgrid(x_grid, y_grid)
xx2d, yy2d = xx[None], yy[None]

ics = [(0.5, 0.3), (-0.5, 0.1)]
data  = np.empty([len(ics), len(t_arr), NX, NY], dtype=np.float32)
data2 = np.empty([len(ics), len(t_arr), NX, NY], dtype=np.float32)

for idx, (th0, om0) in enumerate(ics):
    sol   = odeint(pend, [th0, om0], t_arr)
    theta = sol[:, 0]
    omega = sol[:, 1]

    cos_th = np.cos(theta + np.pi/2)[:, None, None]
    sin_th = np.sin(theta + np.pi/2)[:, None, None]
    z_all  = np.exp(-20 * ((xx2d - cos_th)**2 + (yy2d - sin_th)**2))
    z_min  = z_all.min(axis=(1, 2), keepdims=True)
    z_max  = z_all.max(axis=(1, 2), keepdims=True)
    data[idx] = ((z_all - z_min) / (z_max - z_min)).astype(np.float32)

    cos_om   = np.cos(omega + np.pi/2)[:, None, None]
    sin_om   = np.sin(omega + np.pi/2)[:, None, None]
    gauss    = np.exp(-20 * ((xx2d - cos_om)**2 + (yy2d - sin_om)**2))
    cos_th2  = np.cos(theta - np.pi/2)[:, None, None]
    sin_th2  = np.sin(theta - np.pi/2)[:, None, None]
    om_t     = omega[:, None, None]
    dz       = -20 * (2*(xx2d - cos_th2)*sin_th2*om_t + 2*(yy2d - sin_th2)*(-cos_th2)*om_t) * gauss
    dz_min   = dz.min(axis=(1, 2), keepdims=True)
    dz_max   = dz.max(axis=(1, 2), keepdims=True)
    data2[idx] = ((dz - dz_min) / (dz_max - dz_min)).astype(np.float32)

print(f"  data shape:  {data.shape}   (expected [{len(ics)}, {len(t_arr)}, {NX}, {NY}])")
print(f"  data2 shape: {data2.shape}")
assert data.shape  == (len(ics), len(t_arr), NX, NY)
assert data2.shape == (len(ics), len(t_arr), NX, NY)
assert np.isfinite(data).all(),  "NaN/Inf in data"
assert np.isfinite(data2).all(), "NaN/Inf in data2"
print("  PASS")

print("\n=== All tests passed ===")
