import csv
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

path_X = sys.argv[1];
path_Y = sys.argv[2];
tau = float(sys.argv[3]);

# Read the CSV files to create X_weighted and Y_weighted

read_1 = [];
with open(path_X) as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        read_1.append(row)

X_rawest = np.asarray(read_1)
mu = np.mean(X_rawest);
sigma = np.std(X_rawest);
X_raw = (X_rawest - mu)/sigma;
X_weighted = np.c_[ np.ones((X_raw.shape[0],1)), X_raw]

read_2 = [];
with open(path_Y) as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        read_2.append(row)

Y_weighted = np.asarray(read_2)

m = Y_weighted.shape[0];

theta_linear = np.matmul(np.linalg.pinv(np.matmul(X_weighted.T,X_weighted)),np.matmul(X_weighted.T,Y_weighted))

fig1,ax1 = plt.subplots()
plt.plot(X_rawest,Y_weighted,'ro');
x1 = np.arange(np.min(X_rawest), np.max(X_rawest), 0.01)
theta_final = np.zeros((2,1));
theta_final[0] = theta_linear[0]-mu*theta_linear[1]/sigma;
theta_final[1] = theta_linear[1]/sigma;
y1 = theta_final[0] + theta_final[1]*x1;
plt.plot(x1,y1);
plt.xlabel('Inputs')
plt.ylabel('Outputs')
plt.title('Unweighted Linear Regression')
plt.show(block=False);
input("Press Enter to go to part (b)")
plt.close(fig1)

def normal_gradient(X,Y,tau,i):
    theta_w = np.zeros((2,1));
    W = np.zeros((m,m));
    for j in range(m):
        W[j,j] = np.exp(-1.0* np.square(X[i,1] - X[j,1])/(2*tau*tau));

    inv_mat = np.linalg.pinv(np.matmul(np.matmul(X.T,W.T + W),X));
    other_mat = np.matmul(np.matmul(Y.T,W.T + W),X);
    theta_w = np.matmul(inv_mat,other_mat.T);
    return np.matmul(X[i],theta_w)

def weighted_reg(X,Y,tau):
  Y_pred = np.zeros(Y.shape);
  for i in range (Y.shape[0]):
    Y_pred[i] = normal_gradient(X, Y, tau, i);

  X_total = np.c_[X_rawest, Y_pred];
  c = np.rec.fromarrays([X_rawest, Y, Y_pred]);
  c.sort(axis=0);
  fig,ax = plt.subplots()
  plt.plot(c.f0,c.f1,'ro');
  plt.plot(c.f0, c.f2, 'b--');
  plt.xlabel('Inputs')
  plt.ylabel('Outputs')
  plt.title('Weighted Linear Regression (tau = %.2f)' %tau)
  plt.show();

weighted_reg(X_weighted,Y_weighted,tau);

for i in [0.1,0.3,2,10]:
  weighted_reg(X_weighted,Y_weighted,i);
