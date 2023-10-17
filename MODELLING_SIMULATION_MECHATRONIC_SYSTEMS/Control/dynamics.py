#
# Dynamics of the Vehicle
# 

from numpy import *
import utils_parameters as param

ns = param.ns # Number of state variable
ni = param.ni # Number of input variables

dt = param.dt # discretization step - Forward Euler

#####  Dynamics parameters ##### 

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

def dynamics_msms(xx_in,uu_in):

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

  x = xx_in[0]
  y = xx_in[1]
  psi = xx_in[2]
  vx = xx_in [3]
  vy = xx_in[4]
  psi_dot = xx_in[5]

  delta = uu_in[0]
  sigma_f = uu_in[1]
  sigma_r = uu_in[2]
  
  psi_dot_skew = array([[0, -psi_dot, 0], [psi_dot, 0, 0], [0, 0, 0]], dtype=float64)
  v_contact_front = array([[vx], [vy], [0]]) + psi_dot_skew@array([[a+b], [0], [0]], dtype=float64)
  v_contact_rear = array([[vx], [vy], [0]], dtype=float64)

  beta_f = delta - arctan2(v_contact_front[1], v_contact_front[0])

  beta_r = - arctan2(v_contact_rear[1], v_contact_rear[0])

  mu_fx = dfx*sin(cfx*arctan(bfx*sigma_f-efx*(bfx*sigma_f-arctan(bfx*sigma_f)))) * cos(cxbeta*arctan(beta_f*r1fx/(1+(r2fx*sigma_f)**2)))
  mu_rx = drx*sin(crx*arctan(brx*sigma_r-erx*(brx*sigma_r-arctan(brx*sigma_r)))) * cos(cxbeta*arctan(beta_r*r1rx/(1+(r2rx*sigma_r)**2)))
  mu_fy = dfy*sin(cfy*arctan(bfy*beta_f-efy*(bfy*beta_f-arctan(bfy*beta_f)))) * cos(cysigma*arctan(sigma_f*r1fy/(1+(r2fy*beta_f)**2)))
  mu_ry = dry*sin(cry*arctan(bry*beta_r-ery*(bry*beta_r-arctan(bry*beta_r)))) * cos(cysigma*arctan(sigma_r*r1ry/(1+(r2ry*beta_r)**2)))

  mu_fx_T = mu_fx*cos(delta) - mu_fy*sin(delta)
  mu_rx_T = mu_rx
  mu_fy_T = mu_fx*sin(delta) + mu_fy*cos(delta)
  mu_ry_T = mu_ry
  
  M = array([[m, 0, 0, mu_fx_T, mu_rx_T],
             [0, m, m*b, mu_fy_T, mu_ry_T],
             [0, m*b, Izz+m*b**2, (a+b)*mu_fy_T, 0],
             [0, 0, 0, -1, -1],
             [-m*h, 0 ,0, a+b, 0]],
            dtype=float64) #matrix M
  
  K = array([[-m*b*psi_dot**2 - m*vy*psi_dot],
             [m*vx*psi_dot],
             [m*b*vx*psi_dot],
             [-m*g],
             [m*g*b+m*h*vy*psi_dot+(Ixz+m*h*b)*psi_dot**2]],
            dtype=float64)
  
  x_dot = - linalg.inv(M) @ K
  
  ##### Discretization of the CT dynamics via forward Euler ##### 

# discretizing with Euler

  x_p = x + dt * (vx * cos(psi) - vy * sin(psi))

  y_p = y + dt * (vx * sin(psi) + vy * cos(psi))

  psi_p = psi + dt * psi_dot

  vx_p = vx + dt * x_dot[0]
  
  vy_p = vy +  dt * x_dot[1]
  
  psi_dot_p =  psi_dot + dt * x_dot[2]
    
  xxp = array([[x_p],[y_p],[psi_p],[vx_p],[vy_p],[psi_dot_p]], dtype=float64)

  return xxp

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
  m = 1480
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

  xxp = xx + dt * (Vx * cos(psi) - Vy * sin(psi))
  yyp = yy + dt * (Vx * sin(psi) + Vy * cos(psi))
  psip = psi + dt * psid
  Vxp = Vx + dt * (psid*Vy + (Fx/m)*cos(delta) - ((mu * ((m*g*b)/(a+b)) * (delta - (Vy+a*psid)/Vx))/m)*sin(delta))
  Vyp = Vy + dt * (-psid*Vx + (Fx/m)*sin(delta) + ((mu * ((m*g*b)/(a+b)) * (delta - (Vy+a*psid)/Vx))/m)*cos(delta) + ((mu * (m*g*a)/(a*b) * (- ((Vy-b*psid/Vx))))/m))
  psidp = psid + dt * (a/Izz) * (Fx*sin(delta)+(mu * ((m*g*b)/(a+b)) * (delta - (Vy+a*psid)/Vx))*cos(delta) - ((mu * (m*g*a)/(a*b) * (- (Vy-b*psid/Vx)))*b))

  xxp = array([[xxp],[yyp],[psip],[Vxp],[Vyp],[psidp]])

  ##### Gradient ##### 

  At = array([[1, 0, dt*(-Vx*sin(psi) - Vy*cos(psi)), dt*cos(psi), -dt*sin(psi), 0],
                [0, 1, dt*(Vx*cos(psi) - Vy*sin(psi)), dt*sin(psi), dt*cos(psi), 0],
                [0, 0, 1, 0, 0, dt], 
                [0, 0, 0, 1 - b*dt*g*mu*(Vy + a*psid)*sin(delta)/(Vx**2*(a + b)), dt*(psid + b*g*mu*sin(delta)/(Vx*(a + b))), dt*(Vy + a*b*g*mu*sin(delta)/(Vx*(a + b)))],
                [0, 0, 0, dt*(-psid + b*g*mu*(Vy + a*psid)*cos(delta)/(Vx**2*(a + b)) - g*mu*psid/Vx**2), dt*(-g*mu/b - b*g*mu*cos(delta)/(Vx*(a + b))) + 1, dt*(-Vx - a*b*g*mu*cos(delta)/(Vx*(a + b)) + g*mu/Vx)],
                [0, 0, 0, a*dt*(b*g*m*mu*psid/Vx**2 + b*g*m*mu*(Vy + a*psid)*cos(delta)/(Vx**2*(a + b)))/Izz, a*dt*(g*m*mu - b*g*m*mu*cos(delta)/(Vx*(a + b)))/Izz, 1 + a*dt*(-a*b*g*m*mu*cos(delta)/(Vx*(a + b)) - b*g*m*mu/Vx)/Izz]
                ])

  dfx = At.T
  
  Bt = array([[0, 0],
                 [0, 0],
                 [0, 0],
                 [dt*(-Fx*sin(delta)/m - b*g*mu*(delta - (Vy + a*psid)/Vx)*cos(delta)/(a + b) - b*g*mu*sin(delta)/(a + b)), dt*cos(delta)/m],
                 [dt*(Fx*cos(delta)/m - b*g*mu*(delta - (Vy + a*psid)/Vx)*sin(delta)/(a + b) + b*g*mu*cos(delta)/(a + b)), dt*sin(delta)/m],
                 [a*dt*(Fx*cos(delta) - b*g*m*mu*(delta - (Vy + a*psid)/Vx)*sin(delta)/(a + b) + b*g*m*mu*cos(delta)/(a + b))/Izz, a*dt*sin(delta)/Izz]
                 ])

  dfu = Bt.T

  ##### Hessian ##### 

  Ht = array([[
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,dt*(-Vx*cos(psi) + Vy*sin(psi)),-dt*sin(psi),-dt*cos(psi),0,0,0],
    [0,0,-dt*sin(psi),0,0,0,0,0],
    [0,0,-dt*cos(psi),0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0]],
    [
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,dt*(-Vx*sin(psi) - Vy*cos(psi)),dt*cos(psi), -dt*sin(psi),0,0,0],
    [0,0,dt*cos(psi),0,0,0,0,0],
    [0,0,-dt*sin(psi),0,0,0,0,0],
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
    [0,0,0,2*b*dt*g*mu*(Vy + a*psid)*sin(delta)/(Vx**3*(a + b)),-b*dt*g*mu*sin(delta)/(Vx**2*(a + b)),-a*b*dt*g*mu*sin(delta)/(Vx**2*(a + b)),-b*dt*g*mu*(Vy + a*psid)*cos(delta)/(Vx**2*(a + b)),0],
    [0,0,0,-b*dt*g*mu*sin(delta)/(Vx**2*(a + b)),0,dt,b*dt*g*mu*cos(delta)/(Vx*(a + b)),0],
    [0,0,0,-a*b*dt*g*mu*sin(delta)/(Vx**2*(a + b)),dt,0,a*b*dt*g*mu*cos(delta)/(Vx*(a + b)),0],
    [0,0,0,-b*dt*g*mu*(Vy + a*psid)*cos(delta)/(Vx**2*(a + b)),b*dt*g*mu*cos(delta)/(Vx*(a + b)),a*b*dt*g*mu*cos(delta)/(Vx*(a + b)),dt*(-Fx*cos(delta)/m + b*g*mu*(delta - (Vy + a*psid)/Vx)*sin(delta)/(a + b) - 2*b*g*mu*cos(delta)/(a + b)),-dt*sin(delta)/m],
    [0,0,0,0,0,0,-dt*sin(delta)/m,0]],
    [
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,dt*(-2*b*g*mu*(Vy + a*psid)*cos(delta)/(Vx**3*(a + b)) + 2*g*mu*psid/Vx**3),b*dt*g*mu*cos(delta)/(Vx**2*(a + b)),dt*(-1 + a*b*g*mu*cos(delta)/(Vx**2*(a + b)) - g*mu/Vx**2),-b*dt*g*mu*(Vy + a*psid)*sin(delta)/(Vx**2*(a + b)),0],
    [0,0,0,b*dt*g*mu*cos(delta)/(Vx**2*(a + b)),0,0,b*dt*g*mu*sin(delta)/(Vx*(a + b)),0],
    [0,0,0,dt*(-1 + a*b*g*mu*cos(delta)/(Vx**2*(a + b)) - g*mu/Vx**2),0,0,a*b*dt*g*mu*sin(delta)/(Vx*(a + b)),0],
    [0,0,0,-b*dt*g*mu*(Vy + a*psid)*sin(delta)/(Vx**2*(a + b)),b*dt*g*mu*sin(delta)/(Vx*(a + b)),a*b*dt*g*mu*sin(delta)/(Vx*(a + b)),dt*(-Fx*sin(delta)/m - b*g*mu*(delta - (Vy + a*psid)/Vx)*cos(delta)/(a + b) - 2*b*g*mu*sin(delta)/(a + b)),dt*cos(delta)/m],
    [0,0,0,0,0,0,dt*cos(delta)/m,0]],
    [
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,a*dt*(-2*b*g*m*mu*psid/Vx**3 - 2*b*g*m*mu*(Vy + a*psid)*cos(delta)/(Vx**3*(a + b)))/Izz,a*b*dt*g*m*mu*cos(delta)/(Izz*Vx**2*(a + b)),a*dt*(a*b*g*m*mu*cos(delta)/(Vx**2*(a + b)) + b*g*m*mu/Vx**2)/Izz,-a*b*dt*g*m*mu*(Vy + a*psid)*sin(delta)/(Izz*Vx**2*(a + b)),0],
    [0,0,0,a*b*dt*g*m*mu*cos(delta)/(Izz*Vx**2*(a + b)),0,0,a*b*dt*g*m*mu*sin(delta)/(Izz*Vx*(a + b)),0],
    [0,0,0,a*dt*(a*b*g*m*mu*cos(delta)/(Vx**2*(a + b)) + b*g*m*mu/Vx**2)/Izz,0,0,a**2*b*dt*g*m*mu*sin(delta)/(Izz*Vx*(a + b)),0],
    [0,0,0,-a*b*dt*g*m*mu*(Vy + a*psid)*sin(delta)/(Izz*Vx**2*(a + b)),a*b*dt*g*m*mu*sin(delta)/(Izz*Vx*(a + b)),a**2*b*dt*g*m*mu*sin(delta)/(Izz*Vx*(a + b)),a*dt*(-Fx*sin(delta) - b*g*m*mu*(delta - (Vy + a*psid)/Vx)*cos(delta)/(a + b) - 2*b*g*m*mu*sin(delta)/(a + b))/Izz, a*dt*cos(delta)/Izz],
    [0,0,0,0,0,0,a*dt*cos(delta)/Izz,0]]
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

  HHxx = asarray(HHxx)
  HHxu = asarray(HHxu)
  HHux = asarray(HHux)
  HHuu = asarray(HHuu)

  return xxp, dfx, dfu, HHxx, HHxu, HHux, HHuu
