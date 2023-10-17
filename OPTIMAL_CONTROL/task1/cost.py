import numpy as np
import utils_parameters as param

ns = param.ns
ni = param.ni
dt = param.dt

v0 = param.v0
TT = param.TT

##### Weight Matrices #####

QQt = np.eye(ns)
RRt = np.eye(ni)

QQt[0,0] = 1e-1
QQt[1,1] = 1e1
QQt[2,2] = 1e3
QQt[3,3] = 1e1
QQt[4,4] = 1e2
QQt[5,5] = 1e3

RRt[0,0] = 1e3
RRt[1,1] = 1e-1

QQT = QQt

QQT[1,1] = 1e3

######################################
# Cost Function
######################################


def stagecost(xx, uu, xx_ref, uu_ref):

  """
    Stage-cost 

    Quadratic cost function 
    l(x,u) = 1/2 (x - x_ref)^T Qt (x - x_ref) + 1/2 (u - u_ref)^T Rt (u - u_ref)

    Args
      - xx \in \R^6 state at time t
      - xx_ref \in \R^6 state reference at time t

      - uu \in \R^2 input at time t
      - uu_ref \in \R^2 input reference at time t


    Return 
      - cost at xx,uu: ll
      - gradient of l wrt x, at xx,uu: dlx
      - gradient of l wrt u, at xx,uu: dlu
      - hessian of l wrt x-x, at xx,uu: dlxx
      - hessian of l wrt u-u, at xx,uu: dluu
      - hessian of l wrt x-u, at xx,uu: dlxu

  """

  ll = 0.5*(xx - xx_ref).T@QQt@(xx - xx_ref) + 0.5*(uu - uu_ref).T@RRt@(uu - uu_ref)

  ##### Gradients of stage cost ##### 

  dlx = QQt@(xx - xx_ref)
  dlu = RRt@(uu - uu_ref)

  ##### Hessians of stage cost ##### 

  dlxx = QQt
  dluu = RRt
  dlxu = np.zeros((ni,ns))

  return ll, dlx, dlu, dlxx, dluu, dlxu


def termcost(xx, xx_ref_T):

  """
    Terminal-cost

    Quadratic cost function l_T(x) = 1/2 (x - x_ref)^T Q_T (x - x_ref)

    Args
      - xx \in \R^2 state at time t
      - xx_ref \in \R^2 state reference at time t

    Return 
      - cost at xx,uu: llT
      - gradient of l wrt x, at xx,uu: dlTx
      - hessian of l wrt x-x, at xx,uu: dlTxx
  
  """
  llT = 0.5*(xx - xx_ref_T).T@QQT@(xx - xx_ref_T)

  ##### Gradient ##### 

  dlTx = QQT@(xx - xx_ref_T)

  ##### Hessian ##### 
  
  dlTxx = QQT

  return llT, dlTx, dlTxx
