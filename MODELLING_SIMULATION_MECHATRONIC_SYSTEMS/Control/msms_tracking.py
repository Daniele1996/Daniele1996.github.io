import numpy as np
import dynamics as dyn
import cost as cst 
import utils_parameters as param
import skidpad
import equilibria as eq
from scipy.optimize import fsolve
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from test_dynamics import test_dyn
from solver_ltv_LQR import ltv_LQR
from matplotlib.ticker import FuncFormatter

#######################################
# Algorithm parameters
#######################################

ns = param.ns
ni = param.ni
mm = dyn.m
gg = dyn.g
coefficient = dyn.coefficient
# reference curve parameters 
radius = param.R
d = param.d
v0 = param.v0
aa = dyn.a
bb = dyn.b

# trajectory parameters
dt = param.dt
tf = param.tf
TT = param.TT

max_iters_newt = param.max_iters_newt
stop_criterion = param.stop_criterion

# armijo parameters
stepsize_0 = param.stepsize_0
cc = param.cc
beta = param.beta
armijo_max_iters = param.armijo_max_iters
armijo_plot = False

car_animation = True


#######################################
# Tracking
#######################################

Q = np.zeros((ns, ns))
R = np.zeros((ni, ni))

Q[0,0] = 5e2
Q[1,1] = 5e2

R[0,0] = 1e4
R[1,1] = 1e-3

QT = Q


AAt = np.zeros((ns, ns, TT))
BBt = np.zeros((ns, ni, TT))
QQt = np.zeros((ns, ns, TT))
RRt = np.zeros((ni, ni, TT))

xx_opt = np.load('xx_opt.npy')
uu_opt = np.load('uu_opt.npy')

for tt in range(TT-1):
    fx, fu = dyn.dynamics(xx_opt[:, tt], uu_opt[:, tt])[1:3]
    AAt[:, :, tt] = fx.T
    BBt[:, :, tt] = fu.T
    QQt[:, :, tt] = Q
    RRt[:, :, tt] = R

#######################################
# Main
#######################################
KK, PP = ltv_LQR(AAt,BBt,QQt,RRt,QT,TT)

xx = np.zeros((ns,TT))
xx[:, 0] = np.array([0, 0, np.pi/2, 5, 0, 0])
uu = np.zeros((ni,TT-1))
uu_msms = np.zeros((3,TT-1))
coefficient = 21 # estimated linear pacejka coefficient

for tt in range(TT-1):

    uu[:,tt] = uu_opt[:, tt] + KK[:,:,tt]@(xx[:, tt] - xx_opt[:, tt])
    uu_msms[0,tt] = uu[0,tt]
    uu_msms[1,tt] = uu[1,tt]*(aa+bb)/(coefficient*mm*gg*bb)
    uu_msms[2,tt] = 0
    xx[:,tt+1] = dyn.dynamics_msms(xx[:, tt], uu_msms[:, tt]).squeeze()

#######################################
# Plots
#######################################

tt_hor = np.linspace(0,tf,TT)

def scale_x_axis(x, pos):
    return x / 1000
formatter = FuncFormatter(scale_x_axis)

fig0 = plt.figure('States')
legends = ['x', 'y', 'psi', 'vx', 'vy', 'psi_dot']
for i in range(6):
  plt.subplot(2, 3, i+1)
  plt.plot(range(TT), xx[i,:], linewidth=1)
  plt.gca().xaxis.set_major_formatter(formatter)
  plt.title(legends[i])
  plt.grid()

fig1 = plt.figure('input')
plt.plot(tt_hor[:-1], uu[0,:],'r', linewidth=2)
plt.plot(tt_hor[:-1], uu_opt[0,:],'b', linewidth=1)
plt.grid()
plt.ylabel('$delta$')

############ plotting animation
X1= xx[0,:] #
Y1= xx[1,:] #
delta = uu[0,:]
psi1 = xx[2,:]

X_f = X1 + aa*np.cos(psi1)
X_r = X1 - bb*np.cos(psi1)

Y_f = Y1 + aa*np.sin(psi1)
Y_r = Y1 - bb*np.sin(psi1)

X_t_f = X_f[:-1] + aa/3*np.cos(delta+psi1[:-1])
X_t_r = X_f[:-1] - aa/3*np.cos(delta+psi1[:-1])

Y_t_f = Y_f[:-1] + aa/3*np.sin(delta+psi1[:-1])
Y_t_r = Y_f[:-1] - aa/3*np.sin(delta+psi1[:-1])

X_t_f = np.append(X_t_f, X_t_f[-1])
Y_t_f = np.append(Y_t_f, Y_t_f[-1])

X_t_r = np.append(X_t_r, X_t_r[-1])
Y_t_r = np.append(Y_t_r, Y_t_r[-1])

theta = np.linspace(0, 2*np.pi, 361)
centre_r = d/2
centre_l = -d/2
r_int = 7.625
r_ext = 10.625
alfa = np.arccos(.5*d/r_ext)

def circle(ro, theta, centre): 
    xx = ro*np.cos(theta) + centre
    yy = ro*np.sin(theta)
    return list(xx), list(yy)

xx1, yy1 = circle(r_ext, np.linspace(0, np.pi - alfa, 181), centre_r)
xx2, yy2 = circle(r_ext, np.linspace(alfa, 2*np.pi - alfa, 361), centre_l)
xx3, yy3 = circle(r_ext, np.linspace(np.pi + alfa, 2*np.pi, 181), centre_r)

xx_ext = xx1+xx2+xx3
yy_ext = yy1+yy2+yy3

circle_int_l = circle(r_int, theta, centre_l)
circle_int_r = circle(r_int, theta, centre_r)

states = np.zeros((2,TT))

for tt in range(int(TT-1)):
    states[:,tt] = skidpad.skidpad(tt)[:2]


# set up the figure and subplot
fig = plt.figure('Animation')
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-22,22), ylim=(-12,12))
ax.plot(states[0,:], states[1,:], 'b--', linewidth=1, label="Reference")
ax.plot(xx[0,:], xx[1,:], 'r', linewidth=0.5, label="Tracking")
ax.plot(circle_int_l[0], circle_int_l[1], 'k', linewidth=1)
ax.plot(circle_int_r[0], circle_int_r[1], 'k', linewidth=1)
ax.plot(xx_ext, yy_ext, 'k', linewidth=1)
ax.set_aspect('equal', adjustable='box')
ax.grid()
ax.set_title('Car Animation')
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
path, = ax.plot(xx[0,:1], xx[1,:1], 'b', linewidth=2, label='Path')
line, = ax.plot([], [], 'o-', lw=3.5, color='r', label='Chassis')
steering, = ax.plot([], [], '.-k', linewidth=3, label='Stering wheel')
# initialization function
def init():
    line.set_data([], [])
    return line,

# animation function
def animate(i):
    x_points = [X_r[i], X_f[i]]
    y_points = [Y_r[i], Y_f[i]]

    x_points_steering = [X_t_r[i], X_t_f[i]]
    y_points_steering = [Y_t_r[i], Y_t_f[i]]

    line.set_data(x_points, y_points)
    steering.set_data(x_points_steering, y_points_steering)
    path.set_data(xx[0,:i], xx[1,:i])
    return  path, line, steering,

# call the animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(X1), interval=1, blit=True, repeat=False)

#show the animation
fig.legend(loc='lower center', ncol = 5, fontsize = 10)
plt.show()
