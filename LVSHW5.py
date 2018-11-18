import numpy as np
import matplotlib.pyplot as plt

#Y = np.arange(1960,2016,4)
T = np.array([10.32,10.06,9.95,10.14,10.06,10.25,9.99,9.92,9.96,9.84,9.87,9.85,9.69,9.63])
s = np.linspace(0,1,14)
s = np.reshape(s, [1,14])
A = np.array([[np.ones([14,3])],[s],[np.power(s,2)]])
print(T)
print(s)
print(A)
Q,R = np.linalg.qr(A)
c=(np.transpose(Q))*T
beta = np.linalg.solve(R,c)
y_ls = A*beta
plt.scatter(s,T)
plt.plot(s,y_ls)
plt.xlabel('Year')
plt.ylabel('Gold Medal Winning Time (s)')
plt.show()

y_2010 = [1, 1.2, 1.2**2, 1.2**3]*beta
y_true = 1960
rel_err = np.abs(y - y_true)/52

"""
%% REDUCED MODEL (Linear in time)
y = [150.697;
179.323;
203.212;
226.505;
249.633;
281.422];
s = [0:0.2:1]';
A2 = [ones(size(s)), s];
[Q2, R2] = qr(A2);
c2 = Q2'*y;
beta2 = R2\c2;
y2_ls = A2*beta2;
plot(s,y, 'ob', s, y2_ls, 'r');
y2_2010 = [1, 1.2]*beta2
rel_err = abs(y2_2010 - y_true)/y_true
Annotations
"""
