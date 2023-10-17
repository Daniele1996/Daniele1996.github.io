import numpy as np
import utils_parameters as param
import matplotlib.pyplot as plt

R = param.R
d = param.d
dt = param.dt
v0 = param.v0
tf = param.tf
TT = param.TT

def skidpad(tt):

    """
    Returns the desired states for a smooth transition

    Args
        - sampled time instant t

    Return 
        - states \in \R^6

    """

    # 1st circumference parametrization
    if tt <= int(TT/2): 
        
        # theta_0 = pi
        theta = np.pi - (v0/R)*dt*tt
        xx = R*np.cos(theta) + d/2
        yy = R*np.sin(theta) 
        psi = np.pi/2 - (v0/R)*tt*dt

        if psi<0:
            psi_new = psi + 2*np.pi
        else:
            psi_new = psi
        
        x_dot = R*np.sin(theta) * (v0/R)
        y_dot = - R*np.cos(theta) * (v0/R)
        psid = -(v0/R)

        Vx = x_dot*np.cos(psi_new) + y_dot*np.sin(psi_new)
        Vy = y_dot*np.cos(psi_new) - x_dot*np.sin(psi_new)
        
    
    # 2nd circumference parametrization
    else:

        # theta_0 = 0
        theta = (v0/R)*dt*(tt-TT/2)
        xx = R*np.cos(theta) - d/2
        yy = R*np.sin(theta)
        psi_new = np.pi/2 + (v0/R)*(tt-TT/2)*dt
        psi = (np.pi/2 - (v0/R)*TT/2*dt) + (v0/R)*(tt-TT/2)*dt

        x_dot = - R*np.sin(theta) * (v0/R)
        y_dot = R*np.cos(theta) * (v0/R)
        psid = (v0/R)

        Vx = x_dot*np.cos(psi_new) + y_dot*np.sin(psi_new)
        Vy = y_dot*np.cos(psi_new) - x_dot*np.sin(psi_new)

    states = np.array([xx, yy, psi, Vx, Vy, psid])

    return states

