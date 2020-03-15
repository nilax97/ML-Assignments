import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

path_X = sys.argv[1];
path_Y = sys.argv[2];
quad = int(sys.argv[3]);

x_df = pd.read_csv(path_X, sep="\s+", skiprows=0, header=None)
X_gda = np.asarray(x_df.values)

y_df = pd.read_csv(path_Y, sep="\s+", skiprows=0, header=None)
Y_gda1 = np.asarray(y_df.values)

mu_0 = np.zeros(X_gda[0].shape);
mu_1 = np.zeros(X_gda[0].shape);

Y_gda = np.zeros(Y_gda1.shape)
for i in range(Y_gda1.shape[0]):
    if (Y_gda1[i]=='Alaska'):
        Y_gda[i]=0
    else:
        Y_gda[i]=1

m = Y_gda.shape[0];
n = X_gda.shape[1];
phi = np.sum(Y_gda)/m;

mu_0 = np.zeros((X_gda[0].shape[0],1));
mu_1 = np.zeros((X_gda[0].shape[0],1));
for i in range(m):
    if(Y_gda[i]==0):
        mu_0 = mu_0 + np.reshape(X_gda[i],mu_0.shape);
    else:
        mu_1 = mu_1 + np.reshape(X_gda[i],mu_0.shape);
mu_0 = mu_0/(m*(1-phi));
mu_1 = mu_1/(m*phi);

sigma = np.zeros((X_gda.shape[1],X_gda.shape[1]));
for i in range(m):
    if(Y_gda[i]==0):
        temp = np.reshape(X_gda[i],mu_0.shape) - mu_0;
    else:
        temp = np.reshape(X_gda[i],mu_0.shape) - mu_1;
    sigma = sigma + temp * temp.T;
sigma = sigma/m;

sigma0 = np.zeros((X_gda.shape[1],X_gda.shape[1]));
sigma1 = np.zeros((X_gda.shape[1],X_gda.shape[1]));

for i in range(m):
    if(Y_gda[i]==0):
        temp = np.reshape(X_gda[i],mu_0.shape) - mu_0;
        sigma0 = sigma0 + temp * temp.T;
    else:
        temp = np.reshape(X_gda[i],mu_0.shape) - mu_1;
        sigma1 = sigma1 + temp * temp.T;
sigma1 = sigma1/(phi*m)
sigma0 = sigma0/((1-phi)*m)

if(quad==0):
    print("mu0 = ",mu_0);
    print("mu1 = ",mu_1);
    print("sigma = ",sigma);
    input("Press Enter to go to part (b)")
    fig,ax = plt.subplots()
    leg = ax.scatter(X_gda[:,0],X_gda[:,1], c=Y_gda.ravel(), cmap='RdBu')
    plt.title('Training Data')
    patch1 = mpatches.Patch(color='Blue', label='Alaska');
    patch2 = mpatches.Patch(color='Red', label='Canada');
    plt.legend(handles=[patch1,patch2]);
    plt.show()
    theta = -2 * np.matmul((mu_1.T - mu_0.T),np.linalg.pinv(sigma))
    c = np.matmul(np.matmul(mu_1.T, np.linalg.pinv(sigma)),mu_1) - np.matmul(np.matmul(mu_0.T, np.linalg.pinv(sigma)),mu_0) - 2 * np.log(phi/(1-phi));
    fig2,ax2 = plt.subplots()
    x1 = np.arange(np.min(X_gda[:,0])-5, np.max(X_gda[:,0])+5, 0.01)
    x2 = x1 * (-theta[0,0]/theta[0,1]) - c[0,0]/theta[0,1];
    leg = ax2.scatter(X_gda[:,0],X_gda[:,1], c=Y_gda.ravel(), cmap='RdBu')
    x = np.linspace(np.min(X_gda[:,0]), np.max(X_gda[:,0]), 400)
    y = np.linspace(np.min(X_gda[:,1]), np.max(X_gda[:,1]), 400)
    x, y = np.meshgrid(x, y)
    ax2.contour(x, y, (theta[0,0]*x + theta[0,1]*y + c[0,0]), [0], colors='k')
    patch1 = mpatches.Patch(color='Blue', label='Alaska');
    patch2 = mpatches.Patch(color='Red', label='Canada');
    plt.legend(handles=[patch1,patch2]);
    plt.title('Linear Seperator in GDA')
    plt.show();
else:
    print("mu0 = ",mu_0);
    print("mu1 = ",mu_1);
    print("sigma0 = ",sigma0);
    print("sigma1 = ",sigma1);
    input("Press Enter to go to part (e)")
    a = np.linalg.pinv(sigma1) - np.linalg.pinv(sigma0);
    b = -2 * (np.matmul(mu_1.T,np.linalg.pinv(sigma1)) - np.matmul(mu_0.T,np.linalg.pinv(sigma0)))
    c = np.matmul(np.matmul(mu_1.T, np.linalg.pinv(sigma1)),mu_1) - np.matmul(np.matmul(mu_0.T, np.linalg.pinv(sigma0)),mu_0) - 2 * np.log(phi/(1-phi)) - np.log(np.linalg.det(sigma0)/np.linalg.det(sigma1));
    a1 = a[0,0]
    a2 = a[0,1] + a[1,0]
    a3 = a[1,1]
    a4 = b[0,0]
    a5 = b[0,1]
    a6 = c[0,0]
    x = np.linspace(np.min(X_gda[:,0]), np.max(X_gda[:,0]), 400)
    y = np.linspace(np.min(X_gda[:,1]), np.max(X_gda[:,1]), 400)
    x, y = np.meshgrid(x, y)
    fig2,ax2 = plt.subplots()
    ax2.contour(x, y, (a1*x**2 + a2*x*y + a3*y**2 + a4*x + a5*y + a6), [0], colors='k')
    leg = ax2.scatter(X_gda[:,0],X_gda[:,1], c=Y_gda.ravel(), cmap='RdBu')
    patch1 = mpatches.Patch(color='Blue', label='Alaska');
    patch2 = mpatches.Patch(color='Red', label='Canada');
    plt.legend(handles=[patch1,patch2]);
    plt.title('Quadratic Seperator in GDA')
    plt.show();