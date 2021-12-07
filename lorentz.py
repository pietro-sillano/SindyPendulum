import os
import argparse, sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def lorentz(X, t):
    # LORENZ CONSTANTS
    sigma = 10
    rho = 28
    beta = 8 / 3

    x = X[0]
    y = X[1]
    z = X[2]

    dx_dt = sigma * (y-x)
    dy_dt = x * (rho - z) - y
    dz_dt = x*y - beta*z
    return dx_dt, dy_dt, dz_dt

def initial_conditions(n_ics):
    #x0 = np.random.uniform(0.1,1,n_ics)
    #y0 = np.random.uniform(0.1,1,n_ics)
    #z0 = np.random.uniform(0.1,1,n_ics)
    
    ic_means = np.array([0,0,25])
    ic_widths = 2*np.array([36,48,41])
    ics = ic_widths*(np.random.rand(n_ics, 3)-.5) + ic_means    
    return ics


def data_gen(ics,t):
    data = np.empty([ics.shape[0], len(t), 3])
    for idx in range(ics.shape[0]):
        if(idx%10==0): print(idx,' su ', len(ics))
        y0 = [ics[idx][0], ics[idx][1], ics[idx][2]]

        sol = odeint(lorentz, y0,t)
        data[idx] = sol
    return data


def compute_derivatives(x, dt):

  """
  First order forward difference (forward difference)
  TODO: Find out how the pysindy authors, came up with the formula for the start and end points
  controllare anche che razza di formula é questa
  """

  # Uniform timestep (assume t contains dt)

  x_dot = np.full_like(x, fill_value=np.nan)
  x_dot[:-1, :] = (x[1:, :] - x[:-1, :]) / dt
  x_dot[-1, :] = (3 * x[-1, :] / 2 - 2 * x[-2, :] + x[-3, :] / 2) / dt 
  return x_dot


def unpacking(X):
    x = X[:,:,0]
    y = X[:,:,1]
    z = X[:,:,2]
    return x,y,z

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--initialconditions', help='number of initial conditions', required=True)

    args = vars(parser.parse_args())
    n_ics=int(args["initialconditions"])
    
    
    ics = initial_conditions(n_ics)

    ta = 0.
    tb = 5
    dt = 0.02
    t = np.arange(ta, tb ,dt)

    X = data_gen(ics,t)
    Xdot = compute_derivatives(X,dt)

    #x,y,z = unpacking(X)
    #xdot,ydot,zdot = unpacking(Xdot)


    with open('X.npy', 'wb') as f:
        np.save(f, X)

    with open('Xdot.npy', 'wb') as f2:
        np.save(f2, Xdot)

