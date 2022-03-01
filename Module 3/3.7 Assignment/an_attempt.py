# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


# true signal curve
x = np.arange(0, 2, 10**-4)
y = .2 * np.ones(len(x)) - x + .9 * x**2 + .7 * x**3 - .2 * x**5
y_dict = {round(x[i],4): y[i] for i in range(len(x))}

# plot the true curve
#plt.plot(x,y)

# training samples
N = 50

# linear coefficients
K = 5

# sample interval [a b]
a = 0
b = 2

# generate samples
x1 = np.arange(a, b, (b-a)/N)
y1 = np.asarray([y_dict[round(x,4)] for x in x1])

# noise generation
var_eta = 0.05
n = np.sqrt(var_eta) * np.random.normal(0,1,N)

# use true parameter theta
theta_true = np.asarray([.2, -1, .9, .7, -.2])

var_theta = 100

# compute the measurement matrix
Phi = np.transpose(np.vstack([np.ones(N), x1, x1**2, x1**3, x1**5]))
Phi_gram = np.matmul(np.transpose(Phi),Phi) # NR_phi

# generate noisy observations using the linear model
Sigma_theta_y = np.linalg.inv((1/var_theta) * np.identity(K) + (1/var_eta)*Phi_gram)
term_1 = (1/var_eta)*np.linalg.inv((1/var_theta)*np.identity(K)+(1/var_eta)*Phi_gram)
term_2 = np.transpose(Phi)
term_3 = np.transpose(y1)-np.matmul(Phi,theta_true)
mu_theta_y = np.matmul(theta_true + np.matmul(np.matmul(term_1,term_2),term_3)

mu_y = np.mean(np.matmul(Phi,mu_theta_y))
var_y = np.mean(np.diag(var_eta + np.matmul(np.matmul(Phi,Sigma_theta_y),np.transpose(Phi))))

# perform prediction on new samples
Np = 20

# generate prediction samples
x2 = (b-a) * np.random.uniform(0,1,Np)

# compute prediction measurement matrix
Phip = np.transpose(np.vstack([np.ones(Np), x2, x2**2, x2**3, x2**5]))

# compute the predicted mean and variance




#plt.plot(x, y, color='k', linewidth=0.5)
#plt.plot(x1, y_hat, 'x', color='k', markersize=6)
#plt.errorbar(x2, y_pred, yerr=y_pred_var, fmt='none', color='r', linewidth=0.5, capsize=3) 
# plt.xlabel('x')
# plt.ylabel('y')
# plt.axis([a,b,-0.2,1.8])
# plt.show()







