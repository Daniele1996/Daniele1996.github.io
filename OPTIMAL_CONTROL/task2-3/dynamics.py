#
# Dynamics of the Vehicle
# 

import numpy as np
import utils_parameters as param

ns = param.ns # Number of state variable
ni = param.ni # Number of input variables

dt = param.dt # discretization step - Forward Euler

#####  Dynamics msms parameters ##### 
mm = 1480 #Kg
hh = 1
IIz = 1950 #Kgm^2
IIxz = -15
aa = 1.421 #m
bb = 1.029 #m
mu = 1 
gg = 9.81 #m/s^2

#####  Dynamics parameters ##### 
mm = 1480 #Kg
IIz = 1950 #Kgm^2
aa = 1.421 #m
bb = 1.029 #m
mu = 1 
gg = 9.81 #m/s^2

def dynamics_msms(xx_in,uu_in):

  """
    Dynamics of a discrete-time vehicle system

    Args
      - xx_in \in \R^8 state at time t
      - uu_in \in \R^3 input at time t

    Return 
      - next state xx_{t+1}: xxp
      - gradient of f wrt x, at xx,uu: dfx
      - gradient of f wrt u, at xx,uu: dfu
      
  """

  xx_in = xx_in.squeeze()
  uu_in = uu_in.squeeze()

  xx = xx_in[0]
  yy = xx_in[1]
  psi = xx_in[2]
  Vx = xx_in [3]
  Vy = xx_in[4]
  psid = xx_in[5]
  ffz = xx_in[6]
  frz = xx_in[7]

  delta = uu_in[0]
  Fx = uu_in[1]
  # Fy = uu_in[2]

  ##### Discretization of the CT dynamics via forward Euler ##### 

  xxp = xx + dt * (Vx * np.cos(psi) - Vy * np.sin(psi))
  yyp = yy + dt * (Vx * np.sin(psi) + Vy * np.cos(psi))
  psip = psi + dt * psid
  Vxp = Vx + dt * (-1/mm)*(Fx*np.cos(delta)-mm*bb*psid**2-mm*Vy*psid) 
  Vyp = Vy + dt * (-1/mm)*(Fx*np.sin(delta)+mm*bb*psidp+mm*Vx*psid)
  psidp = psid + dt * (-1/(IIz+mm*bb**2))*(mm*bb*Vyp+mm*bb*Vx*psid)
  
  ffz = -(1/aa+bb)*(-mm*hh*Vxp+(IIxz+mm*bb*hh)*psip**2+mm*hh*Vy*psip+mm*gg*bb)
  frz = - ffz - mm*gg

  xxp = np.array([[xxp],[yyp],[psip],[Vxp],[Vyp],[psidp], [ffz], [frz]])

  ##### Gradient ##### 

  At = np.array([[1, 0, dt*(-Vx*np.sin(psi) - Vy*np.cos(psi)), dt*np.cos(psi), -dt*np.sin(psi), 0, 0, 0],
                [0, 1, dt*(Vx*np.cos(psi) - Vy*np.sin(psi)), dt*np.sin(psi), dt*np.cos(psi), 0, 0, 0],
                [0, 0, 1, 0, 0, dt, 0, 0],
                [0, 0, 0, 1 - bb*dt*gg*mu*(Vy + aa*psid)*np.sin(delta)/(Vx**2*(aa + bb)), dt*(psid + bb*gg*mu*np.sin(delta)/(Vx*(aa + bb))), dt*(Vy + aa*bb*gg*mu*np.sin(delta)/(Vx*(aa + bb)))],
                [0, 0, 0, dt*(-psid + bb*gg*mu*(Vy + aa*psid)*np.cos(delta)/(Vx**2*(aa + bb)) - gg*mu*psid/Vx**2), dt*(-gg*mu/bb - bb*gg*mu*np.cos(delta)/(Vx*(aa + bb))) + 1, dt*(-Vx - aa*bb*gg*mu*np.cos(delta)/(Vx*(aa + bb)) + gg*mu/Vx)],
                [0, 0, 0, aa*dt*(bb*gg*mm*mu*psid/Vx**2 + bb*gg*mm*mu*(Vy + aa*psid)*np.cos(delta)/(Vx**2*(aa + bb)))/IIz, aa*dt*(gg*mm*mu - bb*gg*mm*mu*np.cos(delta)/(Vx*(aa + bb)))/IIz, 1 + aa*dt*(-aa*bb*gg*mm*mu*np.cos(delta)/(Vx*(aa + bb)) - bb*gg*mm*mu/Vx)/IIz]
                ])

  dfx = At.T
  
  Bt = np.array([[0, 0],
                 [0, 0],
                 [0, 0],
                 [dt*(-Fx*np.sin(delta)/mm - bb*gg*mu*(delta - (Vy + aa*psid)/Vx)*np.cos(delta)/(aa + bb) - bb*gg*mu*np.sin(delta)/(aa + bb)), dt*np.cos(delta)/mm],
                 [dt*(Fx*np.cos(delta)/mm - bb*gg*mu*(delta - (Vy + aa*psid)/Vx)*np.sin(delta)/(aa + bb) + bb*gg*mu*np.cos(delta)/(aa + bb)), dt*np.sin(delta)/mm],
                 [aa*dt*(Fx*np.cos(delta) - bb*gg*mm*mu*(delta - (Vy + aa*psid)/Vx)*np.sin(delta)/(aa + bb) + bb*gg*mm*mu*np.cos(delta)/(aa + bb))/IIz, aa*dt*np.sin(delta)/IIz]
                 ])

  dfu = Bt.T

  return xxp, dfx, dfu

def dynamics(xx_in,uu_in):

  """
    Dynamics of a discrete-time vehicle system

    Args
      - xx_in \in \R^6 state at time t
      - uu_in \in \R^2 input at time t

    Return 
      - next state xx_{t+1}: xxp
      - gradient of f wrt x, at xx,uu: dfx
      - gradient of f wrt u, at xx,uu: dfu
      - hessian of f wrt x-x, at xx,uu: HHxx
      - hessian of f wrt x-u, at xx,uu: HHxu
      - hessian of f wrt u-x, at xx,uu: HHux
      - hessian of f wrt u-u, at xx,uu: HHuu
      
  """

  xx_in = xx_in.squeeze()
  uu_in = uu_in.squeeze()

  xx = xx_in[0]
  yy = xx_in[1]
  psi = xx_in[2]
  Vx = xx_in [3]
  Vy = xx_in[4]
  psid = xx_in[5]

  delta = uu_in[0]
  Fx = uu_in[1]

  ##### Discretization of the CT dynamics via forward Euler ##### 

  xxp = xx + dt * (Vx * np.cos(psi) - Vy * np.sin(psi))
  yyp = yy + dt * (Vx * np.sin(psi) + Vy * np.cos(psi))
  psip = psi + dt * psid
  Vxp = Vx + dt * (psid*Vy + (Fx/mm)*np.cos(delta) - ((mu * ((mm*gg*bb)/(aa+bb)) * (delta - (Vy+aa*psid)/Vx))/mm)*np.sin(delta))
  Vyp = Vy + dt * (-psid*Vx + (Fx/mm)*np.sin(delta) + ((mu * ((mm*gg*bb)/(aa+bb)) * (delta - (Vy+aa*psid)/Vx))/mm)*np.cos(delta) + ((mu * (mm*gg*aa)/(aa*bb) * (- ((Vy-bb*psid/Vx))))/mm))
  psidp = psid + dt * (aa/IIz) * (Fx*np.sin(delta)+(mu * ((mm*gg*bb)/(aa+bb)) * (delta - (Vy+aa*psid)/Vx))*np.cos(delta) - ((mu * (mm*gg*aa)/(aa*bb) * (- (Vy-bb*psid/Vx)))*bb))

  xxp = np.array([[xxp],[yyp],[psip],[Vxp],[Vyp],[psidp]])

  ##### Gradient ##### 

  At = np.array([[1, 0, dt*(-Vx*np.sin(psi) - Vy*np.cos(psi)), dt*np.cos(psi), -dt*np.sin(psi), 0],
                [0, 1, dt*(Vx*np.cos(psi) - Vy*np.sin(psi)), dt*np.sin(psi), dt*np.cos(psi), 0],
                [0, 0, 1, 0, 0, dt], 
                [0, 0, 0, 1 - bb*dt*gg*mu*(Vy + aa*psid)*np.sin(delta)/(Vx**2*(aa + bb)), dt*(psid + bb*gg*mu*np.sin(delta)/(Vx*(aa + bb))), dt*(Vy + aa*bb*gg*mu*np.sin(delta)/(Vx*(aa + bb)))],
                [0, 0, 0, dt*(-psid + bb*gg*mu*(Vy + aa*psid)*np.cos(delta)/(Vx**2*(aa + bb)) - gg*mu*psid/Vx**2), dt*(-gg*mu/bb - bb*gg*mu*np.cos(delta)/(Vx*(aa + bb))) + 1, dt*(-Vx - aa*bb*gg*mu*np.cos(delta)/(Vx*(aa + bb)) + gg*mu/Vx)],
                [0, 0, 0, aa*dt*(bb*gg*mm*mu*psid/Vx**2 + bb*gg*mm*mu*(Vy + aa*psid)*np.cos(delta)/(Vx**2*(aa + bb)))/IIz, aa*dt*(gg*mm*mu - bb*gg*mm*mu*np.cos(delta)/(Vx*(aa + bb)))/IIz, 1 + aa*dt*(-aa*bb*gg*mm*mu*np.cos(delta)/(Vx*(aa + bb)) - bb*gg*mm*mu/Vx)/IIz]
                ])

  dfx = At.T
  
  Bt = np.array([[0, 0],
                 [0, 0],
                 [0, 0],
                 [dt*(-Fx*np.sin(delta)/mm - bb*gg*mu*(delta - (Vy + aa*psid)/Vx)*np.cos(delta)/(aa + bb) - bb*gg*mu*np.sin(delta)/(aa + bb)), dt*np.cos(delta)/mm],
                 [dt*(Fx*np.cos(delta)/mm - bb*gg*mu*(delta - (Vy + aa*psid)/Vx)*np.sin(delta)/(aa + bb) + bb*gg*mu*np.cos(delta)/(aa + bb)), dt*np.sin(delta)/mm],
                 [aa*dt*(Fx*np.cos(delta) - bb*gg*mm*mu*(delta - (Vy + aa*psid)/Vx)*np.sin(delta)/(aa + bb) + bb*gg*mm*mu*np.cos(delta)/(aa + bb))/IIz, aa*dt*np.sin(delta)/IIz]
                 ])

  dfu = Bt.T

  ##### Hessian ##### 

  Ht = np.array([[
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,dt*(-Vx*np.cos(psi) + Vy*np.sin(psi)),-dt*np.sin(psi),-dt*np.cos(psi),0,0,0],
    [0,0,-dt*np.sin(psi),0,0,0,0,0],
    [0,0,-dt*np.cos(psi),0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0]],
    [
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,dt*(-Vx*np.sin(psi) - Vy*np.cos(psi)),dt*np.cos(psi), -dt*np.sin(psi),0,0,0],
    [0,0,dt*np.cos(psi),0,0,0,0,0],
    [0,0,-dt*np.sin(psi),0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0]],
    [
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0]],
    [
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,2*bb*dt*gg*mu*(Vy + aa*psid)*np.sin(delta)/(Vx**3*(aa + bb)),-bb*dt*gg*mu*np.sin(delta)/(Vx**2*(aa + bb)),-aa*bb*dt*gg*mu*np.sin(delta)/(Vx**2*(aa + bb)),-bb*dt*gg*mu*(Vy + aa*psid)*np.cos(delta)/(Vx**2*(aa + bb)),0],
    [0,0,0,-bb*dt*gg*mu*np.sin(delta)/(Vx**2*(aa + bb)),0,dt,bb*dt*gg*mu*np.cos(delta)/(Vx*(aa + bb)),0],
    [0,0,0,-aa*bb*dt*gg*mu*np.sin(delta)/(Vx**2*(aa + bb)),dt,0,aa*bb*dt*gg*mu*np.cos(delta)/(Vx*(aa + bb)),0],
    [0,0,0,-bb*dt*gg*mu*(Vy + aa*psid)*np.cos(delta)/(Vx**2*(aa + bb)),bb*dt*gg*mu*np.cos(delta)/(Vx*(aa + bb)),aa*bb*dt*gg*mu*np.cos(delta)/(Vx*(aa + bb)),dt*(-Fx*np.cos(delta)/mm + bb*gg*mu*(delta - (Vy + aa*psid)/Vx)*np.sin(delta)/(aa + bb) - 2*bb*gg*mu*np.cos(delta)/(aa + bb)),-dt*np.sin(delta)/mm],
    [0,0,0,0,0,0,-dt*np.sin(delta)/mm,0]],
    [
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,dt*(-2*bb*gg*mu*(Vy + aa*psid)*np.cos(delta)/(Vx**3*(aa + bb)) + 2*gg*mu*psid/Vx**3),bb*dt*gg*mu*np.cos(delta)/(Vx**2*(aa + bb)),dt*(-1 + aa*bb*gg*mu*np.cos(delta)/(Vx**2*(aa + bb)) - gg*mu/Vx**2),-bb*dt*gg*mu*(Vy + aa*psid)*np.sin(delta)/(Vx**2*(aa + bb)),0],
    [0,0,0,bb*dt*gg*mu*np.cos(delta)/(Vx**2*(aa + bb)),0,0,bb*dt*gg*mu*np.sin(delta)/(Vx*(aa + bb)),0],
    [0,0,0,dt*(-1 + aa*bb*gg*mu*np.cos(delta)/(Vx**2*(aa + bb)) - gg*mu/Vx**2),0,0,aa*bb*dt*gg*mu*np.sin(delta)/(Vx*(aa + bb)),0],
    [0,0,0,-bb*dt*gg*mu*(Vy + aa*psid)*np.sin(delta)/(Vx**2*(aa + bb)),bb*dt*gg*mu*np.sin(delta)/(Vx*(aa + bb)),aa*bb*dt*gg*mu*np.sin(delta)/(Vx*(aa + bb)),dt*(-Fx*np.sin(delta)/mm - bb*gg*mu*(delta - (Vy + aa*psid)/Vx)*np.cos(delta)/(aa + bb) - 2*bb*gg*mu*np.sin(delta)/(aa + bb)),dt*np.cos(delta)/mm],
    [0,0,0,0,0,0,dt*np.cos(delta)/mm,0]],
    [
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,aa*dt*(-2*bb*gg*mm*mu*psid/Vx**3 - 2*bb*gg*mm*mu*(Vy + aa*psid)*np.cos(delta)/(Vx**3*(aa + bb)))/IIz,aa*bb*dt*gg*mm*mu*np.cos(delta)/(IIz*Vx**2*(aa + bb)),aa*dt*(aa*bb*gg*mm*mu*np.cos(delta)/(Vx**2*(aa + bb)) + bb*gg*mm*mu/Vx**2)/IIz,-aa*bb*dt*gg*mm*mu*(Vy + aa*psid)*np.sin(delta)/(IIz*Vx**2*(aa + bb)),0],
    [0,0,0,aa*bb*dt*gg*mm*mu*np.cos(delta)/(IIz*Vx**2*(aa + bb)),0,0,aa*bb*dt*gg*mm*mu*np.sin(delta)/(IIz*Vx*(aa + bb)),0],
    [0,0,0,aa*dt*(aa*bb*gg*mm*mu*np.cos(delta)/(Vx**2*(aa + bb)) + bb*gg*mm*mu/Vx**2)/IIz,0,0,aa**2*bb*dt*gg*mm*mu*np.sin(delta)/(IIz*Vx*(aa + bb)),0],
    [0,0,0,-aa*bb*dt*gg*mm*mu*(Vy + aa*psid)*np.sin(delta)/(IIz*Vx**2*(aa + bb)),aa*bb*dt*gg*mm*mu*np.sin(delta)/(IIz*Vx*(aa + bb)),aa**2*bb*dt*gg*mm*mu*np.sin(delta)/(IIz*Vx*(aa + bb)),aa*dt*(-Fx*np.sin(delta) - bb*gg*mm*mu*(delta - (Vy + aa*psid)/Vx)*np.cos(delta)/(aa + bb) - 2*bb*gg*mm*mu*np.sin(delta)/(aa + bb))/IIz, aa*dt*np.cos(delta)/IIz],
    [0,0,0,0,0,0,aa*dt*np.cos(delta)/IIz,0]]
    ])

  HHxx = []
  HHxu = []
  HHux = []
  HHuu = []

  for m in Ht:
      HHxx_temp = []
      HHxu_temp = []
      HHux_temp = []
      HHuu_temp = []

      for r in range(8):
          if r<6:
              HHxx_temp.append(m[r][:6])
              HHxu_temp.append(m[r][6:])
          else:
              HHux_temp.append(m[r][:6])
              HHuu_temp.append(m[r][6:])
      
      HHxx.append(HHxx_temp)
      HHxu.append(HHxu_temp)
      HHux.append(HHux_temp)
      HHuu.append(HHuu_temp)

  HHxx = np.asarray(HHxx)
  HHxu = np.asarray(HHxu)
  HHux = np.asarray(HHux)
  HHuu = np.asarray(HHuu)

  return xxp, dfx, dfu, HHxx, HHxu, HHux, HHuu
