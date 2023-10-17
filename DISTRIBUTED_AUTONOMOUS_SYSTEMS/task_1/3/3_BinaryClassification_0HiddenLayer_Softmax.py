import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
import sys  
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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


########## Activation Function  ##########
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


########## NN parameters ##########
EPOCHS = 10
pixels = 28*28
d = [int(pixels), 2]
T = len(d) # number of layers, input layer and output layer included


########## training parameters ##########
N_SAMPLE = 6400
batch_size = 128
batches_per_epoch = int(N_SAMPLE/batch_size)
target_num = 5
stepsize = 1e-3 


########## train data set ##########
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = np.reshape(x_train,(60000,28*28))
x_test = np.reshape(x_test,(10000,28*28))

y_train_cat = np.zeros(shape = (60000, 2))
y_test_cat = np.zeros(shape = (10000, 2))

for ii in range(60000):
  if y_train[ii] == target_num:
    y_train_cat[ii, :] = np.array([1, 0])
  else:
    y_train_cat[ii, :] = np.array([0, 1])

for ii in range(10000):
  if y_test[ii] == target_num:
    y_test_cat[ii, :] = np.array([1, 0])
  else:
    y_test_cat[ii, :] = np.array([0, 1])

image_and_label = []
for ii in range(N_SAMPLE):
  image_and_label.append((x_train[ii], y_train_cat[ii]))


########## training alg ##########
J = np.zeros((EPOCHS, batches_per_epoch)) #Loss for plot
uu = np.random.randn(d[1], d[0]+1)*np.sqrt(2/(d[0]))

for epoch in range(EPOCHS):
  np.random.shuffle(image_and_label) #to avoid patterns in the training

  batches = []
  for ii in range(batches_per_epoch):
    batches.append(image_and_label[ii*batch_size : (ii+1)*batch_size])

  for kk in range(batches_per_epoch):
    DeltaU = np.zeros(shape=(d[1], d[0]+1))

    for bb in range(batch_size):
      x_input = batches[kk][bb][0]
      y_true = batches[kk][bb][1]

      xx = forward_pass(uu, x_input)
      y_pred = xx[-1]

      J_bb = categorical_crossentropy(y_true, y_pred)
      J[epoch][kk] += J_bb
      
      llambdaT = np.zeros((d[-1], ))
      llambdaT = categorical_crossentropy_derivative(y_true, y_pred) #Gradient of the loss function J at x_T

      DeltaU_bb = backward_pass(xx, uu, llambdaT) 
      DeltaU += DeltaU_bb

      sys.stdout.write(f"\rProcessing sample n°{bb+1}")
      sys.stdout.flush()

    #uu update
    uu = uu - stepsize*DeltaU
    
    # file_path = os.path.join("mnist_test", "_1st_attempt", "weights_and_biases.npy")
    # np.save(file_path, uu)

    J[epoch][kk] = J[epoch][kk]/batch_size
    print(f'\nMean classification error at epoch = {epoch + 1}, batch n = {kk + 1} is J = {J[epoch][kk]}\n\n')


########## test alg ##########
accuracy = 0
for ii in range(10000):
    x_input = x_test[ii]
    y_true = y_test_cat[ii]

    xx = forward_pass(uu, x_input)
    y_pred = xx[-1]

    if np.argmax(y_pred) == np.argmax(y_true):
      accuracy += 1/10000

J_plot = []
for epoch in range(EPOCHS):
  J_plot += J[epoch].tolist() 

plt.figure()
plt.semilogy([i for i in range(len(J_plot))], J_plot, linestyle='-', color='b')
plt.xlabel('batches')
plt.ylabel('MCE')
plt.title(f'Mean-Classification-Error\n(final achieved accuracy = {accuracy})')
plt.show()