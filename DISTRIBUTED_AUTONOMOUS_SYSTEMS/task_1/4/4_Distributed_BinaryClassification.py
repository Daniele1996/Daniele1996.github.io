import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
import sys  
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

np.random.seed(0)
######### Loss function ##########
##
def categorical_crossentropy(y_true, y_pred):
  epsilon = 1e-7  #small constant to avoid division by zero
  y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)  #clip predictions to prevent log(0) errors
  return -np.sum(y_true * np.log(y_pred))

##
def categorical_crossentropy_derivative(y_true, y_pred):
  epsilon = 1e-7  #small constant to avoid division by zero
  clipped_y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
  return -(y_true / clipped_y_pred)


########## Activation Functions and Other ##########
##
def softmax(xt):
  sum = np.sum(np.exp(xt))
  e_x = np.exp(xt)
  return e_x / sum

##
def softmax_jacob_computation(z):
  l = z.shape[0]
  softmax_jacob = np.zeros(shape=(l, l))
  softmax_z = softmax(z)
  for ii in range(l):
    for jj in range(l):
      if ii == jj:
        softmax_jacob[ii, ii] = softmax_z[ii] * (1 - softmax_z[ii])
      else:
        softmax_jacob[ii, jj] = - softmax_z[ii] * softmax_z[jj]
  
  return softmax_jacob

##
def df_du_computation(st, xt, ut):
  l = st.shape[0]
  d_current = ut.shape[0]
  d_previous = ut.shape[1] - 1

  softmax_jacob = softmax_jacob_computation(st)

  Mat = np.zeros(shape = (l, (d_previous+1)*d_current))

  for jj in range(l):
    insert_vector = np.insert(xt, 0, 1)
    Mat[jj, jj*(d_previous+1) : (jj + 1)*(d_previous+1)] = insert_vector

  return np.transpose(softmax_jacob @ Mat)


########## Utils for learning alg ##########
##
def inference_dynamics(xt,ut):
  """
    input: 
              xt current state (size=d)
              ut current input (size=(d, d+1)), including bias)
    output: 
              xt_plus next state
  """
  d_current = ut.shape[0] # number of neurons in current layer

  xt_plus = np.zeros((d_current, 1))

  temp = ut[:, 1:] @ xt + ut[:, 0]

  xt_plus = softmax(temp)

  return xt_plus

##
def forward_pass(uu,x_input):
  """
  input: 
            uu input trajectory: u[0],u[1],..., u[T-1]
            x_input image in input
  output: 
            xx state trajectory: x[1],x[2],..., x[T]
  """
  xx = []

  for tt in range(2):
    xx.append(np.zeros(shape = (d[tt], )))

  xx[0] = x_input
  xx[1] = inference_dynamics(xx[0], uu)

  return xx

##
def adjoint_dynamics(ltp,xt,ut):
  """
    input: 
              llambda_tp current co-state
              xt current state
              ut current input
    output: 
              llambda_t next co-state
              Delta_ut loss gradient wrt u_t
  """
  d_current = ut.shape[0]
  d_previous = ut.shape[1] - 1

  temp = ut[:, 1:] @ xt + ut[:, 0]
  df_du = df_du_computation(temp, xt, ut) #B.T
  
  Delta_ut_vec = df_du@ltp 
  Delta_ut = np.reshape(Delta_ut_vec,(d_current, d_previous+1))

  return Delta_ut

##
def backward_pass(xx,uu,llambdaT):
  """
  input: 
            xx state trajectory: x[1],x[2],..., x[T]
            uu input trajectory: u[0],u[1],..., u[T-1]
            llambdaT terminal condition
  output: 
            llambda costate trajectory
            Delta_u costate output, i.e., the loss gradient rearranged in a matrix
  """
  llambda = []
  for tt in range(T):
    llambda.append(np.zeros(shape = (d[tt], )))
  llambda[-1] = llambdaT

  Delta_u = np.zeros(shape=(d[tt], d[tt-1]+1))

  Delta_u = adjoint_dynamics(llambda[1], xx[0], uu)

  return Delta_u


######### NN parameters ##########
MAXITERS = int(200) #number of epochs
pixels = 28*28
d = [int(pixels), 2]
T = len(d)


######### training parameters ##########
BATCHSIZE = 128
factor = BATCHSIZE/128
BATCHES_x_AGENT = int(20/factor)
N_AGENTS = 5
N_SAMPLES = int(BATCHSIZE*BATCHES_x_AGENT*N_AGENTS)
stepsize0 = 1e-3
target_num = 7

########## train data set ##########
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = np.reshape(x_train,(60000,28*28))
x_test = np.reshape(x_test,(10000,28*28))

y_train_bin = np.zeros(shape = (60000, 2))
y_test_bin = np.zeros(shape = (10000, 2))
for ii in range(60000):
  if y_train[ii] == target_num:
    y_train_bin[ii, :] = np.array([1, 0])
  else:
    y_train_bin[ii, :] = np.array([0, 1])

for ii in range(10000):
  if y_test[ii] == target_num:
    y_test_bin[ii, :] = np.array([1, 0])
  else:
    y_test_bin[ii, :] = np.array([0, 1])

image_and_label = []
for ii in range(N_SAMPLES):
  image_and_label.append((x_train[ii], y_train_bin[ii]))

dataset_temp = []
for ii in range(N_AGENTS):
  dataset_temp.append(image_and_label[ii*(BATCHSIZE*BATCHES_x_AGENT) : (ii+1)*(BATCHSIZE*BATCHES_x_AGENT)])

dataset = []
for agent in range(N_AGENTS):
  dataset.append([dataset_temp[agent][jj*BATCHSIZE : (jj+1)*BATCHSIZE] for jj in range(BATCHES_x_AGENT) ])


########## binomial graph ##########
num_nodes = N_AGENTS #Define the number of nodes in the graph

probability_edge_creation = 0.5 #Define the probability of creating an edge

while 1: #Create a binomial connected graph
  graph = nx.binomial_graph(num_nodes, probability_edge_creation)
  Adj = nx.to_numpy_array(graph)

  I_NN = np.identity(N_AGENTS, dtype=int)
  test = np.linalg.matrix_power((I_NN+Adj),N_AGENTS)

  if np.all(test>0): #check on primitivity
    print("\n1_ The graph is connected\n")
    break
  else:
    print("\n1_ The graph is NOT connected\n")

degree = np.sum(Adj, axis=0)


########## weight matrix ##########
threshold = 1e-10
WW = np.zeros((N_AGENTS, N_AGENTS))
for ii in range(N_AGENTS):
  neigh_ii = np.nonzero(Adj[ii,:])[0]

  print(f"Agent_{ii} has neighbours: {neigh_ii}, Adj_mat row: {Adj[ii,:]}")
  for jj in neigh_ii:
    WW[ii,jj] = 1/(1+np.max([degree[ii],degree[jj]]))

  WW[ii,ii] = 1-np.sum(WW[ii,:])

check_row_stochasticity = np.all(np.sum(WW, axis = 0))
check_col_stochasticity = np.all(np.sum(WW, axis = 1))
print(f"\n2_ WW is row-stochastic: {check_row_stochasticity}, WW is col-stochastic: {check_col_stochasticity}\n")


########## training alg ##########
# UU Initialization
UU = np.random.randn(N_AGENTS, d[1], d[0]+1)*np.sqrt(2/(d[0]))
    
# feedforward for SS initialisation
SS = np.zeros_like(UU)
for agent in range(N_AGENTS):
  sys.stdout.write(f"\rInitializing agent n°{agent+1}")
  sys.stdout.flush()

  kk=0 #initialize with only the first mini-batch
  for bb in range(BATCHSIZE):
    x_input = dataset[agent][kk][bb][0]
    y_true = dataset[agent][kk][bb][1]

    xx = forward_pass(UU[agent], x_input)
    y_pred = xx[-1]
    
    llambdaT = np.zeros((d[-1], ))
    llambdaT = categorical_crossentropy_derivative(y_true, y_pred) # Gradient of the loss function J at x_T

    DeltaU_bb = backward_pass(xx, UU[agent], llambdaT) 
    SS[agent] += DeltaU_bb
print("\n")

# Distributed Learning
JJ = np.zeros((N_AGENTS, MAXITERS, BATCHES_x_AGENT)) # Loss for plot
grad_norm_estimation = np.zeros((N_AGENTS, MAXITERS, BATCHES_x_AGENT)) 
distances = np.zeros(shape = (N_AGENTS, MAXITERS, BATCHES_x_AGENT))
counter = 0

for epoch in range(MAXITERS):
    for kk in range(BATCHES_x_AGENT):
        DeltaU_previous = np.zeros((N_AGENTS, d[1], d[0]+1))
        DeltaU_current = np.zeros((N_AGENTS, d[1], d[0]+1))

        #feedforward for gradient computation
        for agent in range(N_AGENTS):
          for bb in range(BATCHSIZE):
            x_input = dataset[agent][kk][bb][0]
            y_true = dataset[agent][kk][bb][1]

            xx = forward_pass(UU[agent], x_input)
            y_pred = xx[-1]
            
            llambdaT = np.zeros((d[-1], ))
            llambdaT = categorical_crossentropy_derivative(y_true, y_pred) # Gradient of the loss function J at x_T

            DeltaU_bb = backward_pass(xx, UU[agent], llambdaT) 
            DeltaU_previous[agent] += DeltaU_bb

        stepsize = stepsize0/(counter+1) #decrease stepsize
        counter += 1

        #UU update
        for agent in range(N_AGENTS):
            UU_temp = np.zeros((d[1], d[0]+1))
            for neigh, w in enumerate(WW[agent,:].tolist()):
                UU_temp += UU[neigh]*w

            grad_norm_estimation[agent][epoch][kk] = np.linalg.norm(SS[agent])
            UU[agent] = UU_temp + - stepsize*SS[agent]
        
        #distances for plot purposes 
        for agent in range(N_AGENTS):
          for jj in range(N_AGENTS):
            distances[agent, epoch, kk] += np.linalg.norm(UU[agent].flatten() - UU[jj].flatten())

        #distributed alg
        for agent in range(N_AGENTS):
            sys.stdout.flush()
            sys.stdout.write(f"\rEpoch n°{epoch+1}, batch n°{kk+1} processing agent n°{agent+1}")

            for bb in range(BATCHSIZE):
                x_input = dataset[agent][kk][bb][0]
                y_true = dataset[agent][kk][bb][1]

                xx = forward_pass(UU[agent], x_input)
                y_pred = xx[-1]

                JJ_bb = categorical_crossentropy(y_true, y_pred)
                JJ[agent][epoch][kk] += JJ_bb/BATCHSIZE
                
                llambdaT = np.zeros((d[-1], ))
                llambdaT = categorical_crossentropy_derivative(y_true, y_pred) # Gradient of the loss function J at x_T

                DeltaU_bb = backward_pass(xx, UU[agent], llambdaT) 
                DeltaU_current[agent] += DeltaU_bb
  
        #SS update
        for agent in range(N_AGENTS):
            SS_temp = np.zeros((d[1], d[0]+1))
            for neigh, w in enumerate(WW[agent,:].tolist()):
                SS_temp += SS[neigh]*w

            SS[agent] = SS_temp + DeltaU_current[agent] - DeltaU_previous[agent]
    
    #shuffle the dataset and create new batches
    np.random.shuffle(image_and_label)

    dataset_temp = []
    for ii in range(N_AGENTS):
      dataset_temp.append(image_and_label[ii*(BATCHSIZE*BATCHES_x_AGENT) : (ii+1)*(BATCHSIZE*BATCHES_x_AGENT)])

    dataset = []
    for agent in range(N_AGENTS):
      dataset.append([dataset_temp[agent][jj*BATCHSIZE : (jj+1)*BATCHSIZE] for jj in range(BATCHES_x_AGENT) ])


######### test alg ##########
accuracy = 0
for ii in range(10000):
    x_input = x_test[ii]
    y_true = y_test_bin[ii]

    xx = forward_pass(UU[agent], x_input)
    y_pred = xx[-1]

    if np.argmax(y_pred) == np.argmax(y_true):
      accuracy += 1/10000

colors = {}
for ii in range(N_AGENTS):
    colors[ii] = np.random.rand(3)

plt.figure()
plt.title(rf'Consensus on weights and biases, accuracy = {accuracy}')
plt.xlabel(r"batches")
plt.ylabel(r"$\sum_{j=1}^{N} | \ || u^{k}_i|| - || u^{k}_j|| \ |$")

for agent in range(N_AGENTS):
  plt.semilogy(np.arange(MAXITERS*BATCHES_x_AGENT), distances[agent].flatten(), linestyle="-", color = colors[agent])

# Adding legend
legend_labels = [f"agent {ii}" for ii in range(N_AGENTS)]  # Replace with your actual legend labels
plt.legend(legend_labels)


plt.figure()
plt.title(fr'Estimation of the gradient norm, accuracy = {accuracy}')
plt.xlabel(r"batches")
plt.ylabel(r"$|| \sum_{i=1}^N J_i(u_i^k) ||$")

for agent in range(N_AGENTS):
  plt.semilogy(np.arange(MAXITERS*BATCHES_x_AGENT), grad_norm_estimation[agent].flatten(), linestyle="-", color = colors[agent])

# Adding legend
legend_labels = [f"agent {ii}" for ii in range(N_AGENTS)]  # Replace with your actual legend labels
plt.legend(legend_labels)

plt.figure()
plt.title(f'Mean-Classification-Error, accuracy = {accuracy}')
plt.xlabel("batches")
plt.ylabel('MCE')

for agent in range(N_AGENTS):
  plt.semilogy(np.arange(MAXITERS*BATCHES_x_AGENT), JJ[agent].flatten(), linestyle="-", color = colors[agent])

# Adding legend
legend_labels = [f"agent {ii}" for ii in range(N_AGENTS)]  # Replace with your actual legend labels
plt.legend(legend_labels)
plt.show()
