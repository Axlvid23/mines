import numpy as np

import matplotlib.pyplot as plt

# question 1 part A, for the n = 10
n = 10
p = np.arange(0,1,1/n)
print(p)
# MSE for the posterior mean based on a uniform prior
MSE_Pml = (p*(1-p))/n
MSE_Pb = (((1-2*p)**2)/((n+2)**2)) + ((n*p*(1-p))/(n+2)**2)
print('The MSE_Pml for n=10 is %s, ' %MSE_Pml)
print('The MSE_Pb for n=10 is %s, '%MSE_Pb)
M_rat = MSE_Pml/MSE_Pb
plt.plot(p,M_rat)
plt.title('MSE Estimator Ratio vs. p, X~ Bernoulli(p) for n=10')
plt.xlabel('p')
plt.ylabel('MSE Estimator Ratio')
plt.savefig("/home/alexander/PycharmProjects/StatisticalMethodsHW/Images/HW4Q1PTA.png")
plt.close()


# question 2 part B, for the n = 60
n=60
p = np.arange(0,1,1/n)
print(p)
# MSE for the posterior mean based on a uniform prior
MSE_Pml = p*(1-p)*(1/n)
MSE_Pb = (((1-2*p)**2)/((n+2)**2)) + ((n*p*(1-p))/(n+2)**2)
print('The MSE_Pml for n=60 is %s, ' %MSE_Pml)
print('The MSE_Pb for n=60 is %s, ' %MSE_Pb)
M_rat = MSE_Pml/MSE_Pb
plt.plot(p,M_rat)
plt.title('MSE Estimator Ratio vs. p, X~ Bernoulli(p) for n=60')
plt.xlabel('p')
plt.ylabel('MSE Estimator Ratio')
plt.savefig("/home/alexander/PycharmProjects/StatisticalMethodsHW/Images/HW4Q1PTB.png")
plt.close()

#Question 3.  Parametric bootstrap part i.
mu_x = 1.0
mu_y = 2.0
var_x = 0.05
var_y = 0.1
n=10000
X = np.random.normal(mu_x,var_x, n)
Y = np.random.normal(mu_y,var_y, n)
d_hat = np.sqrt(X**2+Y**2)
theta_bar = np.sqrt(mu_x**2+mu_y**2)
std_d = np.std(d_hat)
bias_d = (1/n)*(np.sum(theta_bar - d_hat))
var_d = (1/(n-1))*np.sum((d_hat-theta_bar)**2)
print('The bias of dhat is %s' %bias_d)
print('The variance of dhat is %s,' %var_d)

#question 3.  non-parametric bootstrap iii.
A = np.array([[1.015,1.981],[0.933,2.073],[1.034,1.941],[1.081,2.218],[0.965,1.986],
              [1.043,2.011],[1.063,2.107],[0.920,2.006],[0.928,1.990],[1.029,1.917]])

mu_x = 1.0
mu_y = 2.0
var_x = 0.05
var_y = 0.1
X = A[:,0]
Y = A[:,1]
n=10000
Xbar = np.random.choice(X,n,replace=True)
Ybar = np.random.choice(Y,n,replace=True)
d_hat = np.sqrt(X**2+Y**2)
theta_bar = np.mean(d_hat)

print(theta_bar)
print(std_d)
theta_hat = np.sqrt(Xbar**2+Ybar**2)
Var_d1 = (1/(n-1))*np.sum((theta_hat-theta_bar)**2)
Bias_d1 = (1/n)*(np.sum(theta_bar - theta_hat))
print("The variance of dhat using nonparametric bootstrap is %s" %Var_d1)
print('The bias of dhat using non-parametric bootstrap is %s' %Bias_d1)

#question 4 part a bruhhhh
M = np.array([[1.6,100,100,0,240,600],[0.1,0.1,1,1,10,1],[-0.249,0.199,-0.00199,0.00199,0.000825,0.000332]])

lambda_var = 0
for i in range(0,len(M[0,:])):
    lambda_var = (M[2,i]**2)*(M[1,i]**2)+lambda_var
lambda_std  = np.sqrt(lambda_var)
print('The standard deviation of Lambda is: %s ' %lambda_std)

i=0
lambda_var = (M[2,i]**2)*(M[1,i]**2)
lambda_std  = np.sqrt(np.abs(lambda_var))
print('The D component of the standard deviation: %s' %lambda_std)

i=1
lambda_var = (M[2,i]**2)*(M[1,i]**2)
lambda_std  = np.sqrt(np.abs(lambda_var))
print('The L component of the standard deviation: %s' %lambda_std)

i=2
lambda_var = (M[2,i]**2)*(M[1,i]**2)
lambda_std  = np.sqrt(np.abs(lambda_var))
print('The T1 component of the standard deviation: %s' %lambda_std)

i=3
lambda_var = (M[2,i]**2)*(M[1,i]**2)
lambda_std  = np.sqrt(np.abs(lambda_var))
print('The T2 component of the standard deviation: %s' %lambda_std)

i=4
lambda_var = (M[2,i]**2)*(M[1,i]**2)
lambda_std  = np.sqrt(np.abs(lambda_var))
print('The Q component of the standard deviation: %s' %lambda_std)

i=5
lambda_var = (M[2,i]**2)*(M[1,i]**2)
lambda_std  = np.sqrt(np.abs(lambda_var))
print('The t component of the standard deviation: %s' %lambda_std)

