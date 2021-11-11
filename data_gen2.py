#TODO
# 1) implementare un diretto to device giá qua, se serve
# 3) aggiungere un file con qualche funzione di altri sistemi dinamici
# 5) fare una funzione simulate system dynamics generate

import os
import argparse, sys
import numpy as np
from scipy.integrate import odeint

def pend(y, t,):
    theta, omega = y
    dydt = [omega, - np.sin(theta)]
    return dydt

def pend_damp(y, t,):
    theta, omega = y
    dydt = [omega, - np.sin(theta)-0.1*omega]
    return dydt

def select_ics(theta0,omegaics0):
    ics = []
    for i in range(n_ics):
        for j in range(n_ics):
            lim = (np.abs((omega0[j]**2)/2 - np.cos(theta0[i])))
            if lim <  0.99 :
                ics.append((theta0[i],omega0[j]))
    return ics

def image_gen(ics):
    x = np.linspace(-1.5, 1.5, NX)
    y = np.linspace(-1.5, 1.5, NY)
    xx,yy = np.meshgrid(x, y)

    data = np.empty([len(ics), len(t), len(x), len(y)],dtype = np.uint8)
    #data = np.empty([len(ics), len(t), len(x), len(y)])

    for idx in range(len(ics)):
        if(idx%100==0): print(idx,' su ', len(ics))
        y0 = [ics[idx][0], ics[idx][1]]
        sol = odeint(pend, y0,t)
        theta = sol[:,0]
        omega = sol[:,1]

        temp = []
        for i in range(len(theta)):
            z = np.exp(- 20 *((xx - np.cos(theta[i] + np.pi/2))*(xx - 
                np.cos(theta[i] +np.pi/2))) - 20 * ((yy -np.sin(theta[i]+np.pi/2))*(yy -np.sin(theta[i]+np.pi/2))))
            z = ((z - np.min(z))/(np.max(z)-np.min(z))) * 255
            temp.append(z)
        data[idx] = np.array(temp)
        
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--initialconditions', help='number of initial conditions', required=True)

    args = vars(parser.parse_args())
    n_ics=int(args["initialconditions"])
    
    #COSTANTI e PARAMETRI
    #n_ics = 100
    ta = 0.
    tb = 10.
    dt = 0.1 
    NX = 51
    NY = 51


    t = np.arange(ta, tb ,dt)
    theta0 = np.linspace(-np.pi,np.pi,n_ics)
    omega0 = np.linspace(-2.1, 2.1,n_ics)

    ics = select_ics(theta0,omega0)
    data = image_gen(ics)

#questo reshape serve per mandare al autoencoder delle immagini flat
#TODO verifica che sia corretto questo rehsape --> dovrebb essere ok fatto prova su colab
    data = data.reshape((len(ics) * len(t),NX * NY))
    print(data.shape)
    print("condizioni iniziali valide: ",len(ics))

    #SCALING TO 0-255 AND CONVERTING TO UINT8
    #data = ((data - np.min(data))/(np.max(data)-np.min(data))) * 255
    #data = np.uint8(data)
    with open('data.npy', 'wb') as f:
        np.save(f, data)

