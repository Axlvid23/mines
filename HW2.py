import numpy as np
import matplotlib.pyplot as plt



#1 Simulation Exercise:  IN each case normalize the samples to mean zero and a standard deviation of one.  All plots
# Should have the same limits in the axes (see Figures 2.9 and 2.10 in the lecture notes).

#a) plot 100 samples from N(0,1)
mu = 0
sigma = 1
X_n = np.random.normal(mu, sigma, 100)
plt.plot(X_n)
plt.title('Part A')
plt.xlim(xmin = 0, xmax = 100)
plt.ylim(ymin = -10, ymax = 10)
plt.xlabel('Realizations')
plt.ylabel('N(0,1)')
plt.show()
#b) plot 100 samples from LN(0,1)
X_ln = np.random.lognormal(mu,sigma,100)
plt.plot(X_ln)
plt.title('Part B')
plt.xlim(xmin = 0, xmax = 100)
plt.ylim(ymin = -10, ymax = 10)
plt.xlabel('Realizations')
plt.ylabel('LN(0,1)')
plt.show()
#c) plot 100 samples from the Gaussian mixture 0.8N(0,1) + 0.1N(0,4) + 0.1N(0,10)
sigma2 = 4
sigma3 = 10
X_n2 = np.random.normal(mu,sigma2,100)
X_n3 = np.random.normal(mu,sigma3,100)
GMix = 0.8*X_n + 0.1*X_n2 + 0.1*X_n3
plt.plot(GMix)
plt.title('Part C')
plt.xlim(xmin = 0, xmax = 100)
plt.ylim(ymin = -10, ymax = 10)
plt.xlabel('Realizations')
plt.ylabel('0.8N(0,1)+0.1N(0,4)+0.1N(0,10)')
plt.show()

#2.  Determine the function f(x) = k(5-x) for 0<x<1 and f(x) for all other x
#a. find the value of K so that f is a PDF and make a plot of it.

x = np.random.random(20)
k = 2/9
f = k*(5-x)

plt.plot(x,f)
plt.title('Part A')
plt.xlim(xmin=0,xmax=1)
plt.ylim(ymin=0,ymax =1.2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

CDF = k*(5*x-((x**2)/2))
plt.plot(x, CDF)
plt.title('Part C')
plt.xlabel('x')
plt.ylabel('CDF')
plt.show()
print('x is %s' %x)
print('f(x) is %s' %f)
print('CDF is %s'%CDF)

Y = np.array([0,1,2,3,4])
X = np.array([0,1,2,0,0])
mu_x = 0.44
mu_y = 1.86

Z_V = np.array = ([0.15,0.05,0.01,0,0])
O_V = np.array = ([0.10,0.08,0.01,0,0])
T_V = np.array = ([0.10,0.14,0.02,0,0])
Tr_V = np.array = ([0.10,0.08,0.03,0,0])
F_V = np.array = ([0.05,0.05,0.03,0,0])

Z_C = sum((X-mu_x)*(0-mu_y)*Z_V)
O_C = sum((X-mu_x)*(1-mu_y)*O_V)
T_C = sum((X-mu_x)*(2-mu_y)*T_V)
Tr_C = sum((X-mu_x)*(3-mu_y)*Tr_V)
F_C = sum((X-mu_x)*(4-mu_y)*F_V)
COV = Z_C+O_C+T_C+Tr_C+F_C
print('the COV is: %s' %COV)



