import numpy as np
import dynamics as dyn
import utils_parameters as param

dt = param.dt
ns = param.ns
ni = param.ni
TT = param.TT

mm = dyn.m
IIz = dyn.Izz
aa = dyn.a
bb = dyn.b
mu = dyn.mu
gg = dyn.g

def equilibria(uu, *data):

  """
  Return 
  - equation #4 of the dynamic with Vxp == Vx:
    - f1(u) = psid*Vy + (Fx/mm)*cos(delta) - ((mu * ((mm*gg*bb)/(aa+bb)) * (delta - (Vy+aa*psid)/Vx))/mm)*np.sin(delta)
  - equation #5 of the dynamic with Vyp == Vy
    - f2(u) = psid*Vx + (Fx/mm)*np.sin(delta) + ((mu * ((mm*gg*bb)/(aa+bb)) * (delta - (Vy+aa*psid)/Vx))/mm)*np.cos(delta) + ((mu * (mm*gg*aa)/(aa*bb) * (- ((Vy-bb*psid/Vx))))/mm)

  """
  (Vx, Vy, psid) = data

  return np.array([(psid*Vy + (uu[1]/mm)*np.cos(uu[0]) - ((mu * ((mm*gg*bb)/(aa+bb)) * (uu[0] - (Vy+aa*psid)/Vx))/mm)*np.sin(uu[0])),
                  (-psid*Vx + (uu[1]/mm)*np.sin(uu[0]) + ((mu * ((mm*gg*bb)/(aa+bb)) * (uu[0] - (Vy+aa*psid)/Vx))/mm)*np.cos(uu[0]) + ((mu * (mm*gg*aa)/(aa*bb) * (- ((Vy-bb*psid/Vx))))/mm))])
