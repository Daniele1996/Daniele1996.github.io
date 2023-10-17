import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

m = 1480
a = 1.421
b = 1.029
h = 0.42
g = 9.81
Izz = 1950
Ixz = -50


bfx = 8.22
cfx = 1.65
dfx = 1.688
efx = -10
r1fx = 13.476
r2fx = 11.354

brx = 8.22
crx = 1.65
drx = 1.688
erx = -10
r1rx = 13.476
r2rx = 11.354

bfy = 12.848
cfy = 1.79
dfy = 1.688
efy = -1.206
r1fy = 7.7856
r2fy = 8.1697

bry = 8.822
cry = 1.79
dry = 1.688
ery = -2.02
r1ry = 7.7856
r2ry = 8.1697

cxbeta = 1.1231
cysigma = 1.0533

mu = 1
coefficient = bfx*cfx*dfx

def non_linear_function(xx_in, uu_in, b, c):

    x = xx_in[0]
    y = xx_in[1]
    psi = xx_in[2]
    vx = xx_in [3]
    vy = xx_in[4]
    psi_dot = xx_in[5]

    delta = uu_in[0]
    sigma_f = uu_in[1]
    sigma_r = uu_in[2]
  
    psi_dot_skew = np.array([[0, -psi_dot, 0], [psi_dot, 0, 0], [0, 0, 0]])
    v_contact_front = np.array([[vx], [vy], [0]]) + psi_dot_skew*np.array([[a+b], [0], [0]])
    v_contact_rear = np.array([[vx], [vy], [0]])

    rot = np.array([[np.cos(delta), np.sin(delta), 0], [-np.sin(delta), np.cos(delta), 0], [0, 0, 1]])
    v_contact_front = rot@v_contact_front

    beta_f = np.arctan2(v_contact_front[1], v_contact_front[0])
    beta_f = - beta_f + delta

    beta_r = np.arctan2(v_contact_rear[1], v_contact_rear[0])

    mu_fx = dfx*np.sin(cfx*np.arctan(bfx*sigma_f-efx*(bfx*sigma_f-np.arctan(bfx*sigma_f)))) * np.cos(cxbeta*np.arctan(beta_f*r1fx/(1+(r2fx*sigma_f)**2)))
    mu_rx = drx*np.sin(crx*np.arctan(brx*sigma_r-erx*(brx*sigma_r-np.arctan(brx*sigma_r)))) * np.cos(cxbeta*np.arctan(beta_r*r1rx/(1+(r2rx*sigma_r)**2)))
    mu_fy = dfy*np.sin(cfy*np.arctan(bfy*beta_f-efy*(bfy*beta_f-np.arctan(bfy*beta_f)))) * np.cos(cysigma*np.arctan(sigma_f*r1fy/(1+(r2fy*beta_f)**2)))
    mu_ry = dry*np.sin(cry*np.arctan(bry*beta_r-ery*(bry*beta_r-np.arctan(bry*beta_r)))) * np.cos(cysigma*np.arctan(sigma_r*r1ry/(1+(r2ry*beta_r)**2)))

    mu_fx = mu_fx*np.cos(delta) - mu_fy*np.sin(delta)
    mu_rx = mu_rx
    mu_fy = mu_fx*np.sin(delta) + mu_fy*np.cos(delta)
    mu_ry = mu_ry

    return mu_fx

x = np.linspace(-.4, .4, 100)  # Sample x values
y = non_linear_function(x, 2, 0.5, 1)  # Sample y values

popt, pcov = curve_fit(non_linear_function, x, y)
