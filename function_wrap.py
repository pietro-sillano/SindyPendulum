def pend(y, t,):
    theta, omega = y
    dydt = [omega, - np.sin(theta)]
    return dydt

def select_ics(theta0,omegaics0):
    ics = []
    for i in range(n_ics):
        for j in range(n_ics):
            lim = (np.abs((omega0[j]**2)/2 - np.cos(theta0[i])))
            if lim <  0.99 :
                ics.append((theta0[i],omega0[j]))
    return ics

def wrap_to_pi(z):
    z_mod = z % (2*np.pi)
    subtract_m = (z_mod > np.pi) * (-2*np.pi)
    return z_mod + subtract_m

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
            #exp = np.exp(- 20 *((xx - np.cos(omega[i] + np.pi/2))*(xx - 
            #    np.cos(omega[i] +np.pi/2))) - 20 * ((yy -np.sin(omega[i]+np.pi/2))*(yy -np.sin(omega[i]+np.pi/2))))

            #z = -20*(2*(xx - np.cos(theta[i]-np.pi/2))*np.sin(theta[i]-np.pi/2)*omega[i] 
            #            + 2*(yy - np.sin(theta[i]-np.pi/2))*(-np.cos(theta[i]-np.pi/2))*omega[i])
            #z = z*exp
            #z = ((z - np.min(z))/(np.max(z)-np.min(z)))
            
            exp = 2 * omega[i]*(np.cos(theta[i]+ np.pi/2) - np.sin(theta[i] + np.pi/2))
            z = np.exp(- 20 *((xx - np.cos(theta[i] + np.pi/2))*(xx - 
                np.cos(theta[i] +np.pi/2))) - 20 * ((yy -np.sin(theta[i]+np.pi/2))*(yy -np.sin(theta[i]+np.pi/2))))
            z = z * exp
            z = ((z - np.min(z))/(np.max(z)-np.min(z)))
            temp.append(z)
        data2[idx] = np.array(temp)
        
    return data,data2