import numpy as np

##### Parameters ##### 

ns = 6 # Number of state variable
ni = 2 # Number of input variables

dt = 1e-2 # discretization step

##### skidpad parameters ##### 

d = 18.25 # distance b/w circles' centers [m]
R = d/2 # radius of the circle [m]

##### Reference parameters ##### 

v0 = 5 # velocity [m/s]

tf = (4*np.pi*R)/v0 # final time [s]
TT = int(tf/dt) # Discrete-time samples

##### Algorithm parameters ##### 

max_iters_newt = 500
stop_criterion = 1e-1

##### Armijo parameters #####

stepsize_0 = 1
cc = 0.5
beta = 0.7
armijo_max_iters = 20
