import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

path_X = sys.argv[1];
path_Y = sys.argv[2];
learning_rate = float(sys.argv[3]);
time_gap = float(sys.argv[4]);
# Read the CSV files to create X_linear and Y_linear

read_1 = [];
with open(path_X) as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        read_1.append(row)

X_rawest = np.asarray(read_1)
mu = np.mean(X_rawest);
sigma = np.std(X_rawest);
X_raw = (X_rawest - mu)/sigma;
X_linear = np.c_[ np.ones((X_raw.shape[0],1)), X_raw]

read_2 = [];
with open(path_Y) as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        read_2.append(row)

Y_linear = np.asarray(read_2)

m = Y_linear.shape[0];
#Function to compute cost

def compute_cost(X,Y,theta):
    h = np.matmul(X,theta);
    J = np.sum(np.square(h-Y))/(2*m);
    return float(J);

#Function for Gradient Descent

def grad_descent(X,Y,theta,alpha):
    h = np.matmul(X,theta);
    d_theta = np.zeros(theta.shape);
    d_theta[0] = np.matmul((h-Y).T,X[:,0])/m;
    d_theta[1] = np.matmul((h-Y).T,X[:,1])/m;
    theta = theta - alpha * d_theta;
    return theta;

#Function to do the regression

def linear_reg(X,Y,theta,alpha, max_iter):
    J = [];
    theta0_iter = [];
    theta1_iter = [];
    J.append(compute_cost(X,Y,theta));
    theta0_iter.append(theta[0]);
    theta1_iter.append(theta[1]);
    for i in range (max_iter):
        theta = grad_descent(X,Y,theta,alpha);
        J.append(compute_cost(X,Y,theta));
        theta0_iter.append(theta[0]);
        theta1_iter.append(theta[1]);
        delta = np.abs(J[i+1] - J[i])/alpha;
        if(delta<np.power(10.0,-10)):
            #print("Number of iterations ", i);
            #print("Final cost ", J[i]);
            break;
    return theta,J, theta0_iter, theta1_iter;

def plot_contour(X,Y,eta,time_gap):
  fig, ax = plt.subplots()
  plt.figure(fig.number)
  X_3d = np.arange(-1, 3, 0.05)
  Y_3d = np.arange(-1, 1, 0.05)


  Z_3d = np.zeros((X_3d.shape[0],Y_3d.shape[0]))
  for i in range(X_3d.shape[0]):
      for j in range(Y_3d.shape[0]):
          Z_3d[i,j] = compute_cost(X_linear, Y_linear,np.array([X_3d[i], Y_3d[j]]))/m;
  X_3d, Y_3d = np.meshgrid(X_3d, Y_3d)

  plt.contour(X_3d,Y_3d,Z_3d.T,10);
  plt.title('Learning Rate = %.2f' %eta);
  theta = np.zeros((2,1));
  [theta,J, t0_iter, t1_iter] = linear_reg(X_linear,Y_linear,theta,eta, max_iter);
  rounder = 1;
  for i in range(round(len(J)/rounder)):
    x0 = t0_iter[i*rounder];
    x1 = t1_iter[i*rounder];
    if(x0*x0 + x1*x1 > 100):
      break;
    scat = ax.scatter(x0,x1,c='r');
    fig.canvas.draw();
    plt.pause(time_gap)
  return fig;


theta = np.zeros((2,1));
alpha = learning_rate;
max_iter = np.power(10,3);
[theta,J, t0_iter, t1_iter] = linear_reg(X_linear,Y_linear,theta,alpha, max_iter);
t0_iter = np.array(t0_iter);
t1_iter = np.array(t1_iter);
theta_final = np.zeros((2,1));
theta_final[0] = theta[0]-mu*theta[1]/sigma;
theta_final[1] = theta[1]/sigma;
print("Theta 0 : ", theta_final[0,0]);
print("Theta 1 : ", theta_final[1,0]);
input("Press Enter to go to part (b)")
fig,ax = plt.subplots()
plt.plot(X_rawest,Y_linear,'ro');
x1 = np.arange(np.min(X_rawest)-5, np.max(X_rawest)+5, 0.01)
y1 = theta_final[0] + theta_final[1]*x1;
plt.plot(x1,y1,'k',linewidth=2.0,);
plt.xlabel('Inputs')
plt.ylabel('Outputs')
plt.title('Linear Regression')
plt.show(block=False);
input("Press Enter to go to part (c)")
plt.close(fig)
plt.ion();
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')

X_3d = np.arange(0, 2, 0.05)
Y_3d = np.arange(-1, 1, 0.05)
Z_3d = np.zeros((X_3d.shape[0],Y_3d.shape[0]))
for i in range(X_3d.shape[0]):
    for j in range(Y_3d.shape[0]):
        Z_3d[i,j] = compute_cost(X_linear, Y_linear,np.array([X_3d[i], Y_3d[j]]))/m;
X_3d, Y_3d = np.meshgrid(X_3d, Y_3d)

ax1.plot_surface(X_3d, Y_3d, Z_3d, cmap='magma',alpha=0.7)
plt.title('3D Plot for Error Function vs Theta');
ax1.view_init(elev=45,azim=20);
J_array=[]
rounder=1;
for i in range(round(len(J)/rounder)):
  x0 = t0_iter[i*rounder,0];
  x1 = t1_iter[i*rounder,0];
  J_cost = J[i*rounder];
  ax1.scatter(xs=x0,ys=x1,zs=J_cost,c='r');
  fig1.canvas.draw();
  plt.pause(time_gap);
input("Press Enter to go to part (d)")
plt.close(fig1);
plt.ioff()
figx = plot_contour(X_linear,Y_linear,alpha,time_gap);
input("Press Enter to go to part (e)")
plt.close(figx);
eta_array = [0.1, 0.5, 0.9, 1.3, 1.7, 2.1, 2.5];
for eta in eta_array:
  figy = plot_contour(X_linear,Y_linear,eta,time_gap);
  time.sleep(2);
  plt.close(figy);