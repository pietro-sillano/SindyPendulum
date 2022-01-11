import numpy as np
from scipy.integrate import odeint
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset



#COSTANTI e PARAMETRI
ta = 0.
tb = 5.
dt = 0.05  
NX = 51
NY = 51



def pend(y, t,):
    theta, omega = y
    dydt = [omega, - np.sin(theta)]
    return dydt

def select_ics(theta0,omega0):
    ics = []
    for i in range(len(theta0)):
        for j in range(len(omega0)):
            lim = (np.abs((omega0[j]**2)/2 - np.cos(theta0[i])))
            if lim <  0.99 :
                ics.append((theta0[i],omega0[j]))
    return ics

def image_gen(ics,t):
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

            exp = -20 * 2 * omega[i]*(np.cos(theta[i]+ np.pi/2) - np.sin(theta[i] + np.pi/2))
            
            z = np.exp(- 20 *((xx - np.cos(theta[i] + np.pi/2))*(xx - 
                np.cos(theta[i] +np.pi/2))) - 20 * ((yy -np.sin(theta[i]+np.pi/2))*(yy -np.sin(theta[i]+np.pi/2))))
            z = z * exp
            #print(z.max())
            z = ((z - np.min(z))/(np.max(z)-np.min(z)+1e-26))
            temp.append(z)
        data2[idx] = np.array(temp)
        
    return data,data2



def create_data(n_ics):
    
    t = np.arange(ta, tb ,dt)
    
    theta0 = np.linspace(-np.pi,np.pi,n_ics)
    omega0 = np.linspace(-2.1, 2.1,n_ics)

    ics = select_ics(theta0,omega0)
    data,data2 = image_gen(ics,t)
    
    X = data.reshape((len(ics) * len(t),NX * NY))
    Xdot = data2.reshape((len(ics) * len(t),NX * NY))
    
    return X,Xdot


def create_dataset(X,Xdot, device,batch_size = 1024):
    
    X = torch.from_numpy(X).float().to(device)
    Xdot = torch.from_numpy(Xdot).float().to(device)
    print(X.shape, Xdot.shape, X.dtype)
    
    
    val_size = round(X.shape[0] * 0.1)
    train_size = X.shape[0] - val_size
    print(train_size, val_size)
    
    my_dataset = TensorDataset(X,Xdot)
    
    train_subset, val_subset = torch.utils.data.random_split(my_dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, shuffle=True, batch_size=batch_size)
    val_loader   = DataLoader(val_subset, shuffle=True, batch_size=batch_size)
    return train_loader, val_loader