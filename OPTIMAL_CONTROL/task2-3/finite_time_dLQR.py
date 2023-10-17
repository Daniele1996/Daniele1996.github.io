import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import skidpad as skid
import dynamics as dyn
import utils_parameters as param
from solver_ltv_LQR import ltv_LQR

#######################################
# Parameters
#######################################
ns = param.ns
ni = param.ni
v0 = param.v0
dt = param.dt
tf = param.tf
TT = param.TT
radius = param.R
aa = dyn.aa
bb = dyn.bb
d = param.d
#######################################
# Dynamics
Q = np.zeros((ns, ns))
R = np.zeros((ni, ni))

Q[0,0] = 0.5e3
Q[1,1] = 0.5e3

R[0,0] = 1e4
R[1,1] = 1e-3

QT = Q
QT[0,0] = 1e4
QT[1,1] = 1e4
QT[2,2] = 1e4

AAt = np.zeros((ns, ns, TT))
BBt = np.zeros((ns, ni, TT))
QQt = np.zeros((ns, ns, TT))
RRt = np.zeros((ni, ni, TT))

xx_opt = np.load('C:\\Users\\danis\\OneDrive - Alma Mater Studiorum Università di Bologna\\Desktop\\OPTCON-PROJECT\\Final_project\\Task3\\xx_opt.npy')
uu_opt = np.load('C:\\Users\\danis\\OneDrive - Alma Mater Studiorum Università di Bologna\\Desktop\\OPTCON-PROJECT\\Final_project\\Task3\\uu_opt.npy')

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
xx[:, 0] = np.array([2.5, 2.5, np.pi/2, 0.1, 0, 0])
uu = np.zeros((ni,TT-1))

for tt in range(TT-1):
  uu[:,tt] = uu_opt[:, tt] + KK[:,:,tt]@(xx[:, tt] - xx_opt[:, tt])
  xx[:,tt+1] = dyn.dynamics(xx[:, tt], uu[:, tt])[0].squeeze()

#######################################
# Plots
#######################################

tt_hor = np.linspace(0,tf,TT)

plt.figure()
plt.plot(xx[0,:], xx[1,:], 'r', linewidth=2)
plt.plot(xx_opt[0,:], xx_opt[1,:], 'b', linewidth=1)
plt.axis("equal")
plt.grid()

fig1, axs1 = plt.subplots(2, 1, sharex='all')
axs1[0].plot(tt_hor[:-1], uu[0,:],'r', linewidth=2)
axs1[0].plot(tt_hor[:-1], uu_opt[0,:],'b', linewidth=1)
axs1[0].grid()
axs1[0].set_ylabel('$delta$')

axs1[1].plot(tt_hor[:-1], uu[1,:],'r', linewidth=2)
axs1[1].plot(tt_hor[:-1], uu_opt[1,:],'b', linewidth=1)
axs1[1].grid()
axs1[1].set_ylabel('$Fx$')
axs1[1].set_xlabel('time')

fig1.align_ylabels(axs1)

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
    states[:,tt] = skid.skidpad(tt)[:2]


# set up the figure and subplot
fig = plt.figure()
fig.canvas.set_window_title('Matplotlib Animation')
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-22,22), ylim=(-12,12))
ax.plot(states[0,:], states[1,:], 'b--', linewidth=1, label="Reference")
ax.plot(xx[0,:], xx[1,:], 'r', linewidth=1, label="Tracking")
ax.plot(circle_int_l[0], circle_int_l[1], 'k', linewidth=1, label="Reference")
ax.plot(circle_int_r[0], circle_int_r[1], 'k', linewidth=1, label="Reference")
ax.plot(xx_ext, yy_ext, 'k', linewidth=1, label="Reference")
ax.set_aspect('equal', adjustable='box')
ax.grid()
ax.set_title('Car Animation')
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
line, = ax.plot([], [], 'o-', lw=5, color='#de2d26')

# initialization function
def init():
    line.set_data([], [])
    return line,

# animation function
def animate(i):
    x_points = [X_r[i], X_f[i], X_t_r[i], X_t_f[i]]
    y_points = [Y_r[i], Y_f[i], Y_t_r[i], Y_t_f[i]]

    line.set_data(x_points, y_points)

    return line,

# call the animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(X1), interval=38, blit=True, repeat=False)
## to save animation, uncomment the line below:
## ani.save('offset_piston_motion_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

#show the animation
plt.show()







