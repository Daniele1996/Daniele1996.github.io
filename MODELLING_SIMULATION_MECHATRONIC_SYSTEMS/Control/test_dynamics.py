import numpy as np
import dynamics as dyn
import utils_parameters as param
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def test_dyn(xxt_init, inputs):

    """
    Return the states obtained by applying given inputs to the system in the whole time horizon

    Args
      - xxt_init \in \R^6 states at time 0
      - inputs \in \R^2 inputs from time 0 to T-2

    Return 
      - xxt states from time 0 to T-1
  
    """

    ns = param.ns
    TT = 10000
    # TT = param.TT

    xxt = np.zeros((ns, TT))
    xxt[:,0] = xxt_init


    for tt in range(TT-1):

        xxt[:,tt+1] = dyn.dynamics(xxt[:,tt], inputs[:,tt])[0].squeeze()

    return xxt

def test_dyn_msms(xxt_init, inputs):

    """
    Return the states obtained by applying given inputs to the system in the whole time horizon

    Args
      - xxt_init \in \R^6 states at time 0
      - inputs \in \R^2 inputs from time 0 to T-2

    Return 
      - xxt states from time 0 to T-1
  
    """

    ns = param.ns
    # TT = 10000
    TT = param.TT

    xxt = np.zeros((ns, TT),dtype=np.float64)
    xxt[:,0] = xxt_init

    for tt in range(TT-1):

        xxt[:,tt+1] = dyn.dynamics_msms(xxt[:,tt], inputs[:,tt]).squeeze()

    return xxt

# TT = 10000
# xxt_init = np.zeros(6)

# xxt_init[3] = 5

# qs_inputs = np.zeros((2, TT-1))
# qs_inputs_msms = np.zeros((3, TT-1))

# qs_inputs[0,:] = 5*0.04 #delta  
# qs_inputs[1,:] = 0 #force  

# qs_inputs_msms[0,:] = 0.3 #delta  
# qs_inputs_msms[1,:] = 0 #sigma_f 
# qs_inputs_msms[2,:] = 0 #sigma_r 

# xxt = test_dyn(xxt_init,qs_inputs)
# xxt_msms = test_dyn_msms(xxt_init,qs_inputs_msms)

# def scale_x_axis(x, pos):
#     return x / 100
# formatter = FuncFormatter(scale_x_axis)

# fig1 = plt.figure('OPTCON-states')
# legends = ['x', 'y', 'psi', 'vx', 'vy', 'psi_dot']
# for i in range(6):
#   plt.subplot(2, 3, i+1)
#   plt.plot(range(TT), xxt[i,:], linewidth=2)
#   plt.gca().xaxis.set_major_formatter(formatter)
#   plt.title(legends[i])
#   plt.grid()

# fig2= plt.figure('OPTCON-plane')
# plt.plot(xxt[0,:], xxt[1,:], linewidth=2)
# plt.title('XY plane')
# plt.grid()

# fig3 = plt.figure('MSMS-states')
# legends = ['x', 'y', 'psi', 'vx', 'vy', 'psi_dot']
# for i in range(6):
#   plt.subplot(2, 3, i+1)
#   plt.plot(range(TT), xxt_msms[i,:], linewidth=2)
#   plt.gca().xaxis.set_major_formatter(formatter)
#   plt.title(legends[i])
#   plt.grid()

# fig4= plt.figure('MSMS-plane')
# plt.plot(xxt_msms[0,:], xxt_msms[1,:], linewidth=2)
# plt.title('XY plane')
# plt.grid()
# plt.show()
