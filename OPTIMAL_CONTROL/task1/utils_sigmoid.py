import numpy as np
import utils_parameters as param

dt = param.dt
TT = param.TT
v0 = param.v0
tf = param.tf

def sigmoid(tt):
  
  """
  Returns the desired states for a smooth transition

  Args
    - sampled time instant t

  Return 
    - states \in \R^6

  """

  kk = param.kk
  ll = param.ll
  mid = param.mid
  acc = param.acc
  time = tt*dt

  xx = v0*time + 0.5*acc*(time)**2
  yy = ll/(1 + np.exp(-kk*(xx-mid)))
  psi = np.arctan(kk * ll * np.exp(-kk * (-mid + xx)) / (1 + np.exp(-kk * (-mid + xx))) ** 2)
  x_dot = v0 + acc*(tt*dt)
  y_dot = kk * ll * (1.0 * acc * time + v0) * np.exp(-kk * (0.5 * acc * time ** 2 - mid + time * v0)) / (1 + np.exp(-kk * (0.5 * acc * time ** 2 - mid + time * v0))) ** 2
  psid = (-kk**2*ll*(1.0*acc*time + v0)*np.exp(-kk*(0.5*acc*time**2 - mid + time*v0))/(1 + np.exp(-kk*(0.5*acc*time**2 - mid + time*v0)))**2 + 2*kk**2*ll*(1.0*acc*time + v0)*np.exp(-2*kk*(0.5*acc*time**2 - mid + time*v0))/(1 + np.exp(-kk*(0.5*acc*time**2 - mid + time*v0)))**3)/(kk**2*ll**2*np.exp(-2*kk*(0.5*acc*time**2 - mid + time*v0))/(1 + np.exp(-kk*(0.5*acc*time**2 - mid + time*v0)))**4 + 1)

  Vx = x_dot*np.cos(psi) + y_dot*np.sin(psi)
  Vy = y_dot*np.cos(psi) - x_dot*np.sin(psi)

  states = np.array([xx, yy, psi, Vx, Vy, psid])

  return states
