import numpy as np

##### Parameters ##### 

ns = 6 # Number of state variable
ni = 2 # Number of input variables

dt = 1e-2 # discretization step

##### Reference parameters ##### 

v0 = 20 # velocity [m/s]
acc = 0 # accelleration [m/s^2]

tf = 5 # final time [s]
TT = int(tf/dt) # Discrete-time samples

##### sigmoid parameters ##### 

kk = 0.1 
ll = 4 # amplitude [m] 
mid = (v0*tf + 0.5*acc*(tf**2))/2 # half of the traveled space 

##### Algorithm parameters ##### 

max_iters_newt = 500
stop_criterion = 1e-2

##### Armijo parameters #####

stepsize_0 = 1
cc = 0.5
beta = 0.7
armijo_max_iters = 20
