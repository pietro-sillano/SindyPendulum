import argparse
import numpy as np
from scipy.integrate import odeint


def pend(y, t):
    theta, omega = y
    return [omega, -np.sin(theta)]


def pend_damp(y, t):
    theta, omega = y
    return [omega, -np.sin(theta) - 0.1 * omega]


def select_ics(theta0, omega0):
    ics = []
    for th in theta0:
        for om in omega0:
            if np.abs(om**2 / 2 - np.cos(th)) < 0.99:
                ics.append((th, om))
    return ics


def image_gen(ics, t, NX=51, NY=51):
    x = np.linspace(-1.5, 1.5, NX)
    y = np.linspace(-1.5, 1.5, NY)
    xx, yy = np.meshgrid(x, y)
    xx2d, yy2d = xx[None], yy[None]  # (1, NX, NY) for broadcasting

    data  = np.empty([len(ics), len(t), NX, NY], dtype=np.float32)
    data2 = np.empty([len(ics), len(t), NX, NY], dtype=np.float32)

    for idx, (th0, om0) in enumerate(ics):
        if idx % 100 == 0:
            print(idx, ' su ', len(ics))
        sol   = odeint(pend, [th0, om0], t)
        theta = sol[:, 0]
        omega = sol[:, 1]

        # Gaussian blob at pendulum tip: cx=cos(theta+pi/2), cy=sin(theta+pi/2)
        cx = np.cos(theta + np.pi / 2)[:, None, None]  # (T,1,1)
        cy = np.sin(theta + np.pi / 2)[:, None, None]
        img = np.exp(-20 * ((xx2d - cx)**2 + (yy2d - cy)**2))  # (T, NX, NY)
        i_min = img.min(axis=(1, 2), keepdims=True)
        i_max = img.max(axis=(1, 2), keepdims=True)
        data[idx] = ((img - i_min) / (i_max - i_min)).astype(np.float32)

        # Correct time derivative:
        # dI/dt = I * 40 * omega * (-(x-cx)*sin(theta+pi/2) + (y-cy)*cos(theta+pi/2))
        # sin(theta+pi/2) == cy,  cos(theta+pi/2) == cx
        om_t = omega[:, None, None]
        dimg = img * 40 * om_t * (-(xx2d - cx) * cy + (yy2d - cy) * cx)
        d_min  = dimg.min(axis=(1, 2), keepdims=True)
        d_max  = dimg.max(axis=(1, 2), keepdims=True)
        d_range = np.where(d_max == d_min, 1.0, d_max - d_min)
        data2[idx] = ((dimg - d_min) / d_range).astype(np.float32)

    return data, data2


def generate_dataset(n_ics=50, ta=0., tb=5., dt=0.05, NX=51, NY=51):
    t      = np.arange(ta, tb, dt)
    theta0 = np.linspace(-np.pi, np.pi, n_ics)
    omega0 = np.linspace(-2.1, 2.1, n_ics)
    ics    = select_ics(theta0, omega0)
    print(f"Valid initial conditions: {len(ics)}")
    data, data2 = image_gen(ics, t, NX=NX, NY=NY)
    X    = data.reshape(len(ics) * len(t), NX * NY)
    Xdot = data2.reshape(len(ics) * len(t), NX * NY)
    return X, Xdot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--n_ics', type=int, default=100,
                        help='grid size for initial conditions')
    args = parser.parse_args()

    X, Xdot = generate_dataset(n_ics=args.n_ics)
    print(f"X: {X.shape}  Xdot: {Xdot.shape}")
    np.save('X.npy', X)
    np.save('Xdot.npy', Xdot)
