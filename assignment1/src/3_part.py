import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

path_X = "logisticX.csv";
path_Y = "logisticY.csv";

read_1 = [];
with open(path_X) as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    m = 0;
    for row in reader: # each row is a list
        read_1.append(row)
        m+=1;

X_rawest = np.asarray(read_1)
mu = np.mean(X_rawest);
sigma = np.std(X_rawest);
X_raw = (X_rawest - mu)/sigma;
X_log = np.c_[ np.ones((m,1)), X_raw]

read_2 = [];
with open(path_Y) as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        read_2.append(row)

Y_log = np.asarray(read_2)

def sigmoid(z):
    a = 1.0/(1+ np.exp(-1.0*z));
    return a;

def compute_cost(X,Y,theta):
    m = Y.shape[0];
    h = sigmoid(np.matmul(X,theta));
    J =  (-1.0/m) * np.sum(Y * np.log(h) + (1-Y)* np.log(1- h));
    return J;

def der_cal(X,Y,theta,alpha):
    f1 = np.zeros(theta.shape);
    for i in range(theta.shape[0]):
        theta_plus = np.copy(theta);
        theta_minus = np.copy(theta);
        theta_plus[i] = theta_plus[i] + alpha;
        theta_minus[i] = theta_minus[i] - alpha;
        J_plus = compute_cost(X,Y,theta_plus);
        J = compute_cost(X,Y,theta);
        f1[i] = (J_plus - J)/(alpha);
    return f1;

def hessian_cal(X,Y,theta, alpha):
    H = np.zeros([theta.shape[0],theta.shape[0]]);
    for i in range (theta.shape[0]):
        theta_plus = np.copy(theta);
        theta_minus = np.copy(theta);
        theta_plus[i] = theta_plus[i] + alpha;
        theta_minus[i] = theta_minus[i] - alpha;
        J_plus = der_cal(X,Y,theta_plus,alpha);
        J = der_cal(X,Y,theta,alpha);
        H[i] = np.reshape((J_plus - J)/(alpha),3);
    return H;

def newton_method( X, Y, theta, alpha, num_iter):
    theta = np.zeros((X_log.shape[1],1));
    for i in range(num_iter):
        f1 = der_cal(X_log, Y_log, theta, alpha);
        H = hessian_cal(X_log, Y_log, theta, alpha);
        delta = np.matmul(np.linalg.pinv(H),f1);
        theta = theta - delta;
        if(np.linalg.norm(delta)<np.power(10.0,-5)):
            break;
    return theta;

alpha = 0.1;
num_iter = 500;
theta = np.zeros((X_log.shape[1],1));

theta_final = newton_method(X_log, Y_log, theta, alpha, num_iter);
print("Theta0 = ",theta_final[0,0]);
print("Theta1 = ",theta_final[1,0]);

input("Press Enter to go to part (b)")

plt.scatter(X_log[:,1], X_log[:,2], c=Y_log.reshape(-1))
ax = plt.gca()
x1 = np.arange(np.min(X_log[:,1]), np.max(X_log[:,1]), 0.01);
y1 = -(theta_final[0] + theta_final[1]*x1)/theta_final[2];
plt.plot(x1, y1, '-', c="red", linewidth=2.0)
plt.title('Logistical Regression');
plt.xlabel('x1');
plt.ylabel('x2');
plt.show();