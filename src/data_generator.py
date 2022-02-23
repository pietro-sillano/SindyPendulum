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


    data = np.empty([len(ics), len(t), len(x), len(y)],dtype = np.float32)
    data2 = np.empty([len(ics), len(t), len(x), len(y)],dtype = np.float32)


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
            z = ((z - np.min(z))/(np.max(z)-np.min(z)))

            temp.append(z)
        data[idx] = np.array(temp)
        
        temp = []
        for i in range(len(omega)):
            exp = np.exp(- 20 *((xx - np.cos(omega[i] + np.pi/2))*(xx - 
                np.cos(omega[i] +np.pi/2))) - 20 * ((yy -np.sin(omega[i]+np.pi/2))*(yy -np.sin(omega[i]+np.pi/2))))

            z = -20*(2*(xx - np.cos(theta[i]-np.pi/2))*np.sin(theta[i]-np.pi/2)*omega[i] 
                        + 2*(yy - np.sin(theta[i]-np.pi/2))*(-np.cos(theta[i]-np.pi/2))*omega[i])
            z = z*exp
            z = ((z - np.min(z))/(np.max(z)-np.min(z)))

            temp.append(z)
        data2[idx] = np.array(temp)
        
    return data,data2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--initialconditions', help='number of initial conditions', required=False ,type=int, default=100)

    args = vars(parser.parse_args())
    n_ics=int(args["initialconditions"])
    
    #COSTANTI e PARAMETRI
    ta = 0.
    tb = 5.
    dt = 0.05  
    # tb = 10
    # dt = 0.1
    NX = 51
    NY = 51


    t = np.arange(ta, tb ,dt)
    theta0 = np.linspace(-np.pi,np.pi,n_ics)
    omega0 = np.linspace(-2.1, 2.1,n_ics)

    ics = select_ics(theta0,omega0)
    data,data2 = image_gen(ics)

    data = data.reshape((len(ics) * len(t),NX * NY))
    data2 = data2.reshape((len(ics) * len(t),NX * NY))

    print(data.shape)
    print("condizioni iniziali valide: ",len(ics))


    with open('X.npy', 'wb') as f:
        np.save(f, data)
    del data
    with open('Xdot.npy', 'wb') as f2:
        np.save(f2, data2)
    del data2
