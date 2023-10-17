#
# Distributed Gradient Method for QP
# Group9
#
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import networkx as nx

np.random.seed(0)
######### parameters ##########
MAXITERS = int(1e6) # Explicit Casting
stepsize0 = 1e-1

NN = 6 #number of agents
ns = 5 # state dimensionality


######### quadratic loss function ##########
def quadratic_fi(xx,QQ,rr):
	
	fval = 0.5*(xx.T@QQ@xx)+rr.T@xx
	fgrad = QQ@xx+rr

	return fval, fgrad

# Declare Cost Variables
QQ = np.zeros((ns,ns,NN)) # positive definite
for ii in range(NN):
	T = scipy.linalg.orth(np.random.rand(ns,ns))
	D = np.diag(np.random.rand(ns))*10
	QQ[:, :, ii] = T.T@D@T

RR = 10*(np.random.rand(ns,NN)-1)


######### star graph ##########
I_NN = np.identity(NN, dtype=int)
while 1:
	graph = nx.star_graph(NN-1) # this function takes as input the number of nodes to connect to a centra node, if NN = 10 then we should give as input 9
	Adj = nx.to_numpy_array(graph)

	I_NN = np.identity(NN, dtype=int)
	test = np.linalg.matrix_power((I_NN+Adj),NN)

	if np.all(test>0): #check on primitivity
		print("\n1_ The graph is connected\n")
		break
	else:
		print("\n1_ The graph is NOT connected\n")

degree = np.sum(Adj, axis=0)
nx.draw(nx.from_numpy_array(I_NN+Adj))	


######### weight matrix ##########
threshold = 1e-10
WW = np.zeros((NN,NN))
for ii in range(NN):
   neigh_ii = np.nonzero(Adj[ii,:])[0]

   print(f"Agent_{ii} has neighbours: {neigh_ii}, Adj_mat row: {Adj[ii,:]}")
   for jj in neigh_ii:
      WW[ii,jj] = 1/(1+np.max([degree[ii],degree[jj]]))

   WW[ii,ii] = 1-np.sum(WW[ii,:])

check_row_stochasticity = np.all(np.sum(WW, axis = 0))
check_col_stochasticity = np.all(np.sum(WW, axis = 1))
print(f"\n2_ WW is row-stochastic: {check_row_stochasticity}, WW is col-stochastic: {check_col_stochasticity}\n")


######### distributed alg ##########
# Compute Optimal Solution for reference
QQ_temp = np.sum(QQ, axis=2)
RR_temp = np.sum(RR, axis=1).reshape((ns, 1))

xopt = -np.linalg.inv(QQ_temp)@RR_temp.reshape((ns, 1))
fopt = 0.5*xopt.T@QQ_temp@xopt + RR_temp.T@xopt
fgradopt = QQ_temp@xopt + RR_temp

print(f"3_ xopt = {xopt.squeeze()}, fopt = {fopt.squeeze()}, fgrad at xopt = {fgradopt.squeeze()}")

# Declare Algorithmic Variables
XX = np.zeros((ns, NN, MAXITERS))
VV = np.zeros((ns, NN, MAXITERS))
SS = np.zeros((ns, NN, MAXITERS))
FF = np.zeros((MAXITERS))
FF_grad = np.zeros((NN, MAXITERS))
SS_grad = np.zeros((NN, MAXITERS))
distances = np.zeros(shape = (NN, MAXITERS-1))

XX_init = 10*np.random.rand(ns,NN)
XX[:, :, 0] = XX_init

for ii in range(NN):
    _, SS_ii_0 = quadratic_fi(XX[:, ii, 0].reshape((ns, 1)), QQ[:, :, ii], RR[:, ii].reshape((ns, 1)))
    SS[:, ii, 0] = SS_ii_0.squeeze()

# GO!
for kk in range (MAXITERS-1):
	stepsize = stepsize0/(kk+1) # Diminishing stepsize

	for agent in range(NN):
		for ll in range(NN):
			distances[agent, kk] += np.linalg.norm(XX[:,agent,kk].flatten() - XX[:,ll,kk].flatten())

	if (kk % 100) == 0:
		print(f"Iteration: {kk:3d}")

	for ii in range (NN):
		neigh_ii = np.nonzero(Adj[ii, :])[0]

		f_ii, _ = quadratic_fi(XX[:, ii, kk].reshape((ns, 1)), QQ[:, :, ii], RR[:, ii].reshape((ns, 1))) # ,_ ignores the second output
		FF[kk] += f_ii

		VV[:, ii, kk+1] = WW[ii, ii]*XX[:, ii, kk]
		for jj in neigh_ii:
			VV[:, ii, kk+1] += WW[ii, jj]*XX[:, jj, kk]

		dXX = - stepsize*SS[:, ii, kk]
		XX[:,ii, kk+1] = VV[:, ii, kk+1] + dXX.squeeze()

		SS[:, ii, kk+1] = WW[ii, ii]*SS[:, ii, kk]
		for jj in neigh_ii:
			SS[:, ii, kk+1] += WW[ii, jj]*SS[:, jj, kk]

		_, grad_fii_k = quadratic_fi(XX[:, ii, kk].reshape((ns, 1)), QQ[:, :, ii], RR[:, ii].reshape((ns, 1))) # _, ignores the first output
		_, grad_fii_kplus1 = quadratic_fi(XX[:, ii, kk+1].reshape((ns, 1)), QQ[:, :, ii], RR[:, ii].reshape((ns, 1)))

		FF_grad_ii_kk = QQ_temp@XX[:,ii, kk].reshape((ns,1)) + RR_temp #for graphic comparison
		FF_grad[ii, kk] = np.linalg.norm(FF_grad_ii_kk.squeeze()) #for graphic comparison
		SS_grad[ii, kk] = np.linalg.norm(NN*SS[:, ii, kk])

		SS[:, ii, kk+1] += grad_fii_kplus1.squeeze() - grad_fii_k.squeeze()

# Terminal iteration
FF_temp = np.zeros((MAXITERS))

for ii in range (NN):
	f_ii, _ = quadratic_fi(XX[:,ii,-1].reshape((ns,1)),QQ[:,:,ii],RR[:,ii].reshape(ns,1))  
	FF[-1] += f_ii # Last entry

	FF_grad_ii_last = QQ_temp@XX[:,ii, -1].reshape((ns,1)) + RR_temp #for graphic comparison
	FF_grad[ii, -1] = np.linalg.norm(FF_grad_ii_last.squeeze())


for ii in range (NN):
	f_ii, _ = quadratic_fi(xopt.reshape((ns,1)), QQ[:,:,ii], RR[:,ii].reshape((ns,1))) 
	FF_temp[-1] += f_ii # Last entry

print('xopt: {} \nfopt: {}\n'.format(xopt,fopt))

#Last iter
for ii in range(NN):
	_, FF_grad_lastiter = quadratic_fi(XX[:, ii, -1].reshape((ns,1)), QQ[:,:,ii], RR[:,ii].reshape((ns,1))) 
	print('Agent: {}, last iter XX: {}, last iter FF: {}, last iter FFgrad: {}\n'.format(ii, XX[:,ii,-1].squeeze(), FF_temp[-1], FF_grad_lastiter.squeeze()))


######### plots ##########
#Figure 1 : Evolution of the local estimates
if 1:
	plt.figure()
	counter = 1
	for _ in range(NN):
		plt.subplot(2,3,counter)
		
		for ii in range(ns):
			plt.plot(np.arange(MAXITERS), np.repeat(xopt[ii], MAXITERS), '--', linewidth=3, color='r')

		for jj in range(ns):
			plt.plot(np.arange(MAXITERS), XX[jj,ii,:], color='b')     
		
		if counter == 1:
			plt.xlabel(r"iterations")
			plt.ylabel(r"$x_i^k$")
			plt.title(r"Evolution of the local estimates")
		
		counter+=1
		plt.grid()

#Figure 2 : Visualize consensus on x_opt
if 1:
	colors = {}
	for ii in range(NN):
		colors[ii] = np.random.rand(3)

	plt.figure()
	for ii in range(NN):
		plt.semilogy(np.arange(MAXITERS-1), distances[ii], color=colors[ii])     

	plt.xlabel(r"iterations")
	plt.ylabel(r"$\sum_{j=1}^{N} | \ || x^{k}_i|| - || x^{k}_j|| \ |$")
	plt.title(r"Visualize consensus on $\mathbf{x}_{opt}$")
	plt.grid()

	# Adding legend
	legend_labels = [f"agent {ii}" for ii in range(NN)]  # Replace with your actual legend labels
	plt.legend(legend_labels)

#Figure 3 : Cost Evolution
if 1:
	plt.figure()
	plt.plot(np.arange(MAXITERS), np.repeat(fopt,MAXITERS), '--', linewidth=3, color='r')
	plt.plot(np.arange(MAXITERS), FF, color='b')
	plt.xlabel(r"iterations")
	plt.ylabel(r"$\sum_{i=1}^N f_i(x_i^k)$, $f^\star$")
	plt.title(r"Evolution of the cost")
	plt.grid()

#Figure 4 : Cost Error Evolution
if 1:
	plt.figure()
	plt.semilogy(np.arange(MAXITERS), np.abs(FF-np.repeat(fopt,MAXITERS)), '--', linewidth=3, color='b')
	plt.xlabel(r"iterations")
	plt.ylabel(r"$|| \sum_{i=1}^N f_i(x_i^k) - f^\star ||$")
	plt.title(r"Evolution of cost error")
	plt.grid()

#Figure 5: Evolution of the grad norm
if 1:
	colors = {}
	for ii in range(NN):
		colors[ii] = np.random.rand(3)

	plt.figure()
	for ii in range(NN):
		plt.semilogy(np.arange(MAXITERS-1), SS_grad[ii,:-1], color=colors[ii])     

	plt.xlabel(r"iterations")
	plt.ylabel(r"$|| \sum_{i=1}^N f_i(x_i^k) ||$")
	plt.title(fr"Evolution of the local gradient norm estimate")
	plt.grid()

	# Adding legend
	legend_labels = [f"agent {ii}" for ii in range(NN)]  # Replace with your actual legend labels
	plt.legend(legend_labels)

plt.show()
