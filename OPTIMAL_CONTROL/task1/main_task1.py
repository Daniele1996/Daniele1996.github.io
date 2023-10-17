import numpy as np
import dynamics as dyn
import cost as cst 
import utils_parameters as param
import utils_sigmoid as sigm
import equilibria as eq
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from test_dynamics import test_dyn

#######################################
# Algorithm parameters
#######################################

ns = param.ns
ni = param.ni

# reference curve parameters 
mm = dyn.mm
v0 = param.v0
acc = param.acc
mid = param.mid

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
armijo_plot = True 

#######################################
# Initialization via quasi-static
#######################################

# Arrays to store data
xxt = np.zeros((ns,TT,max_iters_newt)) # state seq.
uut = np.zeros((ni,TT-1,max_iters_newt)) # input seq.

qs_states = np.zeros((ns, TT)) # quasi static states
qs_inputs = np.zeros((ni, TT-1)) # quasi static inputs

# Initial guess for the inputs obtained via root finding function fsolve
for tt in range(TT):

    states = sigm.sigmoid(tt)
    qs_states[:, tt] = states
    data=(states[3],states[4],states[5])
    fsolve_init = np.zeros(2)

    if tt <= TT - 2:
        qs_inputs[:, tt] = fsolve(eq.equilibria, fsolve_init, args=data)

uut[:,:,0] = qs_inputs

# Initial guess for the states 
xx0 = test_dyn(sigm.sigmoid(0), qs_inputs)
xxt[:,:,0] = xx0

# Set the same initial condition (at t = 0) for every iteration of newton
for kk in range(max_iters_newt):
    xxt[:,0,kk] = xx0[:,0] 

#######################################
# Reference trajectory
#######################################

xx_ref = np.zeros((ns,TT))
uu_ref = np.zeros((ni,TT-1))

xx_ref[0,:] = [v0*tt*dt + .5*acc*(tt*dt)**2 for tt in range(TT)]

for tt in range(TT):
    if v0*tt*dt + .5*acc*(tt*dt)**2 > mid:
        xx_ref[1, tt] = 4
    if tt <= TT-2:
        uu_ref[1,tt] = mm*acc

xx_ref[3,:] = [v0 + acc*(tt*dt) for tt in range(TT)]

#######################################
# Main
#######################################

# Arrays to store data
JJ = np.zeros(max_iters_newt) # collect cost
descent = np.zeros(max_iters_newt) # collect descent direction

grad = np.zeros((ni,TT,max_iters_newt)) # gradient of J in u 
lmbd = np.zeros((ns,TT,max_iters_newt)) # lambdas - costate seq.

for kk in range(max_iters_newt):

    # Arrays to store data 
    A_matrix = np.zeros((ns,ns,TT))
    B_matrix = np.zeros((ns,ni,TT))
    K_matrix = np.zeros((ni,ns,TT))
    sigma_matrix = np.zeros((ni,TT))
    ppt = np.zeros((ns,TT))
    PPt = np.zeros((ns,ns,TT))

    delta_xt = np.zeros((ns,TT))
    delta_ut = np.zeros((ni,TT-1))

    ##################################
    # Descent direction calculation
    ##################################

    # initialize final terms
    ppt[:,TT-1], PPt[:,:,TT-1] = cst.termcost(xxt[:,TT-1,kk],xx_ref[:,TT-1])[1:3]
    ppt_temp = ppt[:,TT-1].reshape(6,1)
    
    lmbd[:,TT-1,kk] = cst.termcost(xxt[:,TT-1,kk], xx_ref[:,TT-1])[1]
    lmbd_temp = lmbd[:,TT-1,kk].reshape(6,1)

    for tt in reversed(range(TT-1)): # integration backward in time

        qq_temp, rr_temp, QQ_temp, RR_temp, SS_temp = cst.stagecost(xxt[:,tt,kk],uut[:,tt,kk],xx_ref[:,tt],uu_ref[:,tt])[1:]
        qq_temp = qq_temp.reshape(6,1)
        rr_temp = rr_temp.reshape(2,1)

        dfx, dfu = dyn.dynamics(xxt[:,tt,kk], uut[:,tt,kk])[1:3]

        At = dfx.T
        Bt = dfu.T

        # Introduce the update of Q-R-S matrices thanks to hessian terms
        if False:

            HHxx, HHxu, HHux, HHuu = dyn.dynamics(xxt[:,tt,kk], uut[:,tt,kk])[3:]

            M0 = QQ_temp + np.tensordot(HHxx,lmbd[:,tt+1,kk], axes=((0),(0)))
            M1 = RR_temp + np.tensordot(HHuu,lmbd[:,tt+1,kk], axes=((0),(0)))
            M2 = SS_temp + np.tensordot(HHux,lmbd[:,tt+1,kk], axes=((0),(0)))

            if np.all(np.linalg.eigvals(M0)) > 0 and np.all(np.linalg.eigvals(M1)) > 0:

                QQ_temp = M0
                RR_temp = M1
                SS_temp = M2

        # update terms that will be needed to compute optimal solution
        Kt = - (np.linalg.inv(RR_temp + Bt.T@PPt[:,:,tt+1]@Bt)@(Bt.T@PPt[:,:,tt+1]@At))
        sigmat = - (np.linalg.inv(RR_temp + Bt.T@PPt[:,:,tt+1]@Bt)@(rr_temp + Bt.T@ppt_temp))

        K_matrix[:,:,tt] = Kt
        sigma_matrix[:,tt] = sigmat.squeeze()
        A_matrix[:,:,tt] = At
        B_matrix[:,:,tt] = Bt

        PPt[:,:,tt] = QQ_temp + At.T@PPt[:,:,tt+1]@At - Kt.T@(RR_temp + Bt.T@PPt[:,:,tt+1]@Bt)@Kt
        ppt_temp = qq_temp + At.T@ppt_temp - Kt.T@(RR_temp + Bt.T@PPt[:,:,tt+1]@Bt)@sigmat
        ppt[:,tt] = ppt_temp.squeeze()

        # computate necessary terms for armijo 
        lmbd_temp = At.T@lmbd[:,tt+1,kk].reshape(6,1) + qq_temp # costate equation
        grad_temp = - Bt.T@lmbd[:,tt+1,kk].reshape(6,1) - rr_temp

        lmbd[:,tt,kk] = lmbd_temp.squeeze()
        grad[:,tt,kk] = grad_temp.squeeze()

    for tt in range(TT-1):

        # optimal solution
        delta_ut[:,tt] = (K_matrix[:,:,tt]@(delta_xt[:,tt].reshape(6,1)) + (sigma_matrix[:,tt].reshape(2,1))).squeeze()
        delta_xt[:,tt+1] = (A_matrix[:,:,tt]@(delta_xt[:,tt].reshape(6,1)) + B_matrix[:,:,tt]@(delta_ut[:,tt].reshape(2,1))).squeeze()

        # evaluate stage-cost and add it to the total cost 
        cost_temp = cst.stagecost(xxt[:,tt,kk], uut[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt])[0]
        JJ[kk] += cost_temp
        # evaluate descent for each time instant
        descent[kk] += grad[:,tt,kk].T@delta_ut[:,tt]

    # add term-cost to the total cost
    cost_temp = cst.termcost(xxt[:,TT-1,kk], xx_ref[:,TT-1])[0]
    JJ[kk] += cost_temp

    ###################################
    # Stepsize selection - ARMIJO
    ###################################
    
    stepsizes = [] # list of stepsizes
    costs_armijo = [] # list of costs associated to the stepsizes

    stepsize = stepsize_0

    for ii in range(armijo_max_iters):

        # temp solution update
        xx_temp = np.zeros((ns,TT))
        uu_temp = np.zeros((ni,TT-1))

        xx_temp[:,0] = xxt[:,0,kk]

        for tt in range(TT-1):
                uu_temp[:,tt] = uut[:,tt,kk] + stepsize*delta_ut[:,tt]
                xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0].squeeze()

        # temp cost calculation
        JJ_temp = 0

        for tt in range(TT-1):
            temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
            JJ_temp += temp_cost

        temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
        JJ_temp += temp_cost

        stepsizes.append(stepsize) # save the stepsize
        costs_armijo.append(JJ_temp) # save the cost associated to the stepsize

        if JJ_temp > JJ[kk] - cc*stepsize*descent[kk]:
            # update the stepsize
            stepsize = beta*stepsize

        else:
            break
        ############################
        
    print('Iteration: {}\t Cost: {:10.3f}\t Armijo stepsize #{}: {:01.4f}\t Descent: {:10.6f}\t '.format(kk, JJ[kk],ii+1, stepsize, descent[kk]))
    
    ############################
    # Armijo plot
    ############################

    if armijo_plot:

        steps = np.linspace(0,1,int(1e1))
        costs = np.zeros(len(steps))

        for ii in range(len(steps)):

            step = steps[ii]

            # temp solution update
            xx_temp = np.zeros((ns,TT))
            uu_temp = np.zeros((ni,TT-1))

            xx_temp[:,0] = xxt[:,0,kk]
            
            for tt in range(TT-1):
                uu_temp[:,tt] = uut[:,tt,kk] + step*delta_ut[:,tt]
                xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0].squeeze()

            # temp cost calculation
            JJ_temp = 0

            for tt in range(TT-1):
                temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
                JJ_temp += temp_cost

            temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
            JJ_temp += temp_cost

            costs[ii] = JJ_temp

        plt.figure(1)
        plt.clf()

        plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - stepsize*d^k)$')
        plt.plot(steps, JJ[kk] - descent[kk]*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
        plt.plot(steps, JJ[kk] - cc*descent[kk]*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')

        plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize

        plt.grid()
        plt.xlabel('stepsize')
        plt.legend()
        plt.draw()

        plt.show()
        # plt.show(block=False)
        # plt.pause(3)

    #############################
    # Update the current solution
    #############################

    for tt in range(TT - 1):

        uut[:, tt, kk + 1] = uut[:, tt, kk] + stepsize * delta_ut[:, tt]
        xxt[:,tt+1,kk+1] = dyn.dynamics(xxt[:,tt,kk+1], uut[:,tt,kk+1])[0].squeeze()

    #############################
    # Termination condition
    #############################

    if kk == max_iters_newt - 1:

        break

    if descent[kk]<= stop_criterion:

        max_iters_newt = kk+1

        break

#######################################
# Plots
#######################################

# cost and descent

plt.figure('Task1 - Descent direction')
plt.plot(np.arange(max_iters_newt), descent[:max_iters_newt])
plt.xlabel('$k$')
plt.ylabel('$descent$')
plt.yscale('log')
plt.grid()
plt.show(block=False)

plt.figure('Task1 - Cost')
plt.plot(np.arange(max_iters_newt), JJ[:max_iters_newt])
plt.xlabel('$k$')
plt.ylabel('$J(\\mathbf{u}^k)$')
plt.yscale('log')
plt.grid()
plt.show(block=False)

tt_hor = np.linspace(0,tf,TT)
tt_hor_input = tt_hor[:-1]

# evolution of y over k iterations
plt.figure('Task1 - Evolution of y in the iterations')
plt.plot(xxt[0,:,0], xxt[1,:,0], 'g', linewidth=2, label="Iter1 Newton")
plt.plot(xxt[0,:,1], xxt[1,:,1], 'm', linewidth=2, label="Iter2 Newton")
plt.plot(xxt[0,:,2], xxt[1,:,2], 'c', linewidth=2, label="Iter3 Newton")
plt.plot(xxt[0,:,max_iters_newt-1], xxt[1,:,max_iters_newt-1], 'b', linewidth=2, label="Last iter Newton")
plt.plot(xx_ref[0,:], xx_ref[1,:], 'b--', linewidth=2)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='upper left',ncol=4, fontsize=12)
plt.grid()
plt.show(block=False)

# quasi static and reference plots
fig, axs = plt.subplots(3, 1, sharex='all')
fig.suptitle('Task1 - Initial guess for the states $(x,\:,y\:,\psi)$', fontsize=15)

axs[0].plot(tt_hor, xxt[0,:,0],'r', linewidth=2, label="Initial guess")
axs[0].plot(tt_hor, xx_ref[0,:],'b--', linewidth=2, label="Reference")
axs[0].plot(tt_hor, qs_states[0,:],'g-.', linewidth=2, label="Quasi/static")
axs[0].grid()
axs[0].set_ylabel('$x$')

axs[1].plot(tt_hor, xxt[1,:,0], 'r',linewidth=2)
axs[1].plot(tt_hor, xx_ref[1,:], 'b--',linewidth=2)
axs[1].plot(tt_hor, qs_states[1,:], 'g-.',linewidth=2)
axs[1].grid()
axs[1].set_ylabel('$y$')

axs[2].plot(tt_hor, xxt[2,:,0],'r',linewidth=2)
axs[2].plot(tt_hor, xx_ref[2,:],'b--',linewidth=2)
axs[2].plot(tt_hor, qs_states[2,:],'g-.',linewidth=2)
axs[2].grid()
axs[2].set_ylabel('$\psi$')
axs[2].set_xlabel('time')

fig.align_ylabels(axs)
fig.legend(loc='lower center', ncol=3, fontsize=12)

fig1, axs1 = plt.subplots(3, 1, sharex='all')
fig1.suptitle('Task1 - Initial guess for the states $(V_x,\:V_y,\:\dot{\psi})$', fontsize=15)

axs1[0].plot(tt_hor, xxt[3,:,0],'r',linewidth=2, label="Initial guess")
axs1[0].plot(tt_hor, xx_ref[3,:],'b--',linewidth=2, label="Reference")
axs1[0].plot(tt_hor, qs_states[3,:],'g-.',linewidth=2, label="Quasi/static")
axs1[0].grid()
axs1[0].set_ylabel('$V_x$')

axs1[1].plot(tt_hor, xxt[4,:,0],'r',linewidth=2)
axs1[1].plot(tt_hor, xx_ref[4,:],'b--',linewidth=2)
axs1[1].plot(tt_hor, qs_states[4,:],'g-.',linewidth=2)
axs1[1].grid()
axs1[1].set_ylabel('$V_y$')

axs1[2].plot(tt_hor, xxt[5,:,0],'r',linewidth=2)
axs1[2].plot(tt_hor, xx_ref[5,:],'b--',linewidth=2)
axs1[2].plot(tt_hor, qs_states[5,:],'g-.',linewidth=2)
axs1[2].grid()
axs1[2].set_ylabel('$\dot{\psi}$')
axs1[2].set_xlabel('time')

fig1.align_ylabels(axs1)
fig1.legend(loc='lower center', ncol=3, fontsize=12)

fig2, axs2 = plt.subplots(2, 1, sharex='all')
fig2.suptitle('Task1 - Initial guess for the inputs  $(\delta,\:F_x)$', fontsize=15)

axs2[0].plot(tt_hor_input, uut[0,:,0],'r',linewidth=2, label='Initial guess')
axs2[0].plot(tt_hor_input, uu_ref[0,:],'b--',linewidth=2,  label='Reference')
axs2[0].plot(tt_hor_input, qs_inputs[0,:],'g-.',linewidth=2,  label='Quasi/static')
axs2[0].grid()
axs2[0].set_ylabel('$\delta$')

axs2[1].plot(tt_hor_input, uut[1,:,0],'r',linewidth=2)
axs2[1].plot(tt_hor_input, uu_ref[1,:],'b--',linewidth=2)
axs2[1].plot(tt_hor_input, qs_inputs[1,:],'g-.',linewidth=2)
axs2[1].grid()
axs2[1].set_ylabel('$F_x$')
axs2[1].set_xlabel('time')
fig2.align_ylabels(axs2)
fig2.legend(loc='lower center',ncol=3, fontsize=12)
##### end #####

# final results Newton
fig3, axs3 = plt.subplots(3, 1, sharex='all')
fig3.suptitle('Task1 - Results of Newton Algorithm for the states $(x,\:y,\:\psi)$', fontsize=15)

axs3[0].plot(tt_hor, xxt[0,:,max_iters_newt-1],'r',linewidth=2, label="Last iter Newton")
axs3[0].plot(tt_hor, xx_ref[0,:],'b--', linewidth=2, label="Reference")
axs3[0].grid()
axs3[0].set_ylabel('$x$')

axs3[1].plot(tt_hor, xxt[1,:,1], 'm', linewidth=2, label="Iter2 Newton")
axs3[1].plot(tt_hor, xxt[1,:,2], 'c', linewidth=2, label="Iter3 Newton")
axs3[1].plot(tt_hor, xxt[1,:,max_iters_newt-1], 'r', linewidth=2)
axs3[1].plot(tt_hor, xx_ref[1,:], 'b--', linewidth=2)
axs3[1].grid()
axs3[1].set_ylabel('$y$')

axs3[2].plot(tt_hor, xxt[2,:,max_iters_newt-1], 'r', linewidth=2)
axs3[2].plot(tt_hor, xx_ref[2,:], 'b--', linewidth=2)
axs3[2].grid()
axs3[2].set_ylabel('$\psi$')
axs3[2].set_xlabel('time')
fig3.align_ylabels(axs3)
fig3.legend(loc='lower center', ncol=2, fontsize=12)

fig4, axs4 = plt.subplots(3, 1, sharex='all')
fig4.suptitle('Task1 - Results of Newton Algorithm for the states $(V_x,\:V_y\:,\dot{\psi})$', fontsize=15)
axs4[0].plot(tt_hor, xxt[3,:,max_iters_newt-1], 'r', linewidth=2, label="Last iter Newton")
axs4[0].plot(tt_hor, xx_ref[3,:], 'b--', linewidth=2, label="Reference")
axs4[0].grid()
axs4[0].set_ylabel('$V_x$')

axs4[1].plot(tt_hor, xxt[4,:,max_iters_newt-1], 'r', linewidth=2)
axs4[1].plot(tt_hor, xx_ref[4,:], 'b--', linewidth=2)
axs4[1].grid()
axs4[1].set_ylabel('$V_y$')

axs4[2].plot(tt_hor, xxt[5,:,max_iters_newt-1], 'r', linewidth=2)
axs4[2].plot(tt_hor, xx_ref[5,:], 'b--', linewidth=2)
axs4[2].grid()
axs4[2].set_ylabel('$\dot{\psi}$')
axs4[2].set_xlabel('time')
fig4.align_ylabels(axs4)
fig4.legend(loc='lower center', ncol=2, fontsize=12)

fig5, axs5 = plt.subplots(2, 1, sharex='all')
fig5.suptitle('Task1 - Results of Newton Algorithm for the inputs $(\delta,\:F_x)$', fontsize=15)

axs5[0].plot(tt_hor_input, uut[0,:,max_iters_newt-1],'r', linewidth =2, label="Last iter Newton")
axs5[0].plot(tt_hor_input, uu_ref[0,:],'b--', linewidth=2, label="Reference")
axs5[0].grid()
axs5[0].set_ylabel('$\delta$')

axs5[1].plot(tt_hor_input, uut[1,:,max_iters_newt-1],'r', linewidth=2)
axs5[1].plot(tt_hor_input, uu_ref[1,:],'b--', linewidth=2)
axs5[1].grid()
axs5[1].set_ylabel('$F_x$')
axs5[1].set_xlabel('time')

fig5.align_ylabels(axs5)
fig5.legend(loc='lower center', ncol=2, fontsize=12)

plt.show()

