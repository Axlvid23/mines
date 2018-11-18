import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# number 2:  i.  Do you think the data are reasonably normal?  Use code to conduct a normality test

data = np.sort(np.array([28.57,44.82,12.20,68.47,20.10,77.37,22.55,42.64,69.05,24.67,
                 14.30,16.70,65.40,65.01,31.79,86.27,29.59,9.99,11.60,76.73]))
res = stats.probplot(data, plot=plt)
plt.title('White Collar Fathers Normality Test')
plt.savefig("/home/alexander/PycharmProjects/StatisticalMethodsHW/Images/HW5Q2PTA.png")
plt.close()

statistic, pvalue = stats.normaltest(data)
print('the statistic is %s and the pvalue is %s' % (statistic, pvalue))

zalph2 = 1.96
n = 20
phat = np.mean(data)/100
LLp = phat - zalph2*np.sqrt(phat*(1-phat)/n)
ULp = phat + zalph2*np.sqrt(phat*(1-phat)/n)
print('The 95 CI for white collar job dad percentage, the standard way, is (%s,%s)' % (LLp, ULp))

nsqig = n + zalph2**2
print(nsqig)
psqig = (n/nsqig)*phat + ((zalph2**2)/(2*nsqig))
print(psqig)

LLpsqig = psqig - zalph2*np.sqrt(psqig*(1-psqig)/nsqig)
ULpsqig = psqig + zalph2*np.sqrt(psqig*(1-psqig)/nsqig)
print('The 95 CI for white collar job dad percentage using the Agresti Coull method is (%s,%s)' % (LLpsqig, ULpsqig))

#comute the p-value for a null hypothesis that the pop proportion is less than 50%

stat, pval = stats.ttest_1samp(data, 50)
print('The P-value for the null hypothesis that the sampling proportion is less than 50 is %s' %pval)

# number 3: i.  Construct the 95% confidence interval for the two time measurements:

ground = np.array([4.46, 3.99, 3.73, 3.29, 4.82, 6.71, 4.61, 3.87, 3.17, 4.42, 3.76, 3.30])
satellite =  np.array([4.08, 3.94, 5.00, 5.20, 3.92, 6.21, 5.95, 3.07, 4.76, 3.25, 4.89, 4.80])

xbar = np.mean(ground)
ybar = np.mean(satellite)
n=12
nullhyp = xbar-ybar
talph2 = 2.20
Sx = np.std(ground)
Sy = np.std(satellite)

LLx = xbar - talph2*np.sqrt((Sx**2)/n)
ULx = xbar + talph2*np.sqrt((Sx**2)/n)

LLy = ybar - talph2*np.sqrt((Sy**2)/n)
ULy = ybar + talph2*np.sqrt((Sy**2)/n)

LL = nullhyp - talph2*np.sqrt((Sx**2)/n + (Sy**2)/n)
UL = nullhyp + talph2*np.sqrt((Sx**2)/n + (Sy**2)/n)

print('The value of xbar is %s' %xbar)
print('the value of Sx is %s' %Sx)
print('the value of ybar is %s' %ybar)
print('the value of Sy is %s' %Sy)
print('xbar-ybar is %s' %nullhyp)
print('the 95 CI for xbar is (%s,%s)' %(LLx,ULx))
print('the 95 CI for ybar is (%s,%s)' %(LLy,ULy))
print('the 95 CI for the difference between ground and satellite measured times is (%s,%s)' % (LL,UL))

#questoin 4: i. construct 95% CI for competence and interesting presentation for each of the types of clothing.

#formal professional:
#Competence
xbar_formal_C= 4.29
s_formal_C = 0.55
n_formal_C =125
zalph2 = 1.96

LL_formal_C = xbar_formal_C - zalph2*np.sqrt(s_formal_C**2/n_formal_C)
UL_formal_C = xbar_formal_C + zalph2*np.sqrt(s_formal_C**2/n_formal_C)

print('The 95 CI for "formal professional - competence" is (%s,%s)' % (LL_formal_C,UL_formal_C))

#formal professional
#interesting presentation
xbar_formal_I = 3.90
s_formal_I = 1.08
n_formal_I = 125

LL_formal_I = xbar_formal_I - zalph2*np.sqrt(s_formal_I**2/n_formal_I)
UL_formal_I = xbar_formal_I + zalph2*np.sqrt(s_formal_I**2/n_formal_I)

print('The 95 CI for "formal professional - interesting presentation" is (%s,%s)' % (LL_formal_I,UL_formal_I))

#casual professional
#competence
xbar_casual_C = 4.24
s_casual_C = 0.60
n_casual_C = 144

LL_casual_C = xbar_casual_C - zalph2*np.sqrt(s_casual_C**2/n_casual_C)
UL_casual_C = xbar_casual_C + zalph2*np.sqrt(s_casual_C**2/n_casual_C)

print('The 95 CI for "casual professional - competence" is (%s,%s)' % (LL_casual_C, UL_casual_C))

#casual professional
#interesting presentation
xbar_casual_I = 3.93
s_casual_I = 0.98
n_casual_I = 143

LL_casual_I = xbar_casual_I - zalph2*np.sqrt(s_casual_I**2/n_casual_I)
UL_casual_I = xbar_casual_I + zalph2*np.sqrt(s_casual_I**2/n_casual_I)

print('The 95 CI for "casual professional - interesting presentation" is (%s,%s)' % (LL_casual_I, UL_casual_I))

#casual
#competence
xbar_cash_C = 4.11
s_cash_C = 0.62
n_cash_C = 132

LL_cash_C = xbar_cash_C-zalph2*np.sqrt(s_cash_C**2/n_cash_C)
UL_cash_C = xbar_cash_C+zalph2*np.sqrt(s_cash_C**2/n_cash_C)

print('The 95 CI for "casual - competence" is (%s,%s)' % (LL_cash_C, UL_cash_C))

#casual
#interesting presentation
xbar_cash_I = 4.30
s_cash_I = 0.89
n_cash_I = 132

LL_cash_I = xbar_cash_I-zalph2*np.sqrt(s_cash_I**2/n_cash_I)
UL_cash_I = xbar_cash_I+zalph2*np.sqrt((s_cash_I**2/n_cash_I))

print('The 95 CI for "casual - interesting presentation" is (%s,%s)' % (LL_cash_I, UL_cash_I))

#competence CI plots
fig, ax = plt.subplots()
plt.errorbar(1, xbar_formal_C, yerr = (UL_formal_C-LL_formal_C)/2, fmt = ".k")
plt.errorbar(2, xbar_casual_C, yerr = (UL_casual_C-LL_casual_C)/2, fmt = ".k")
plt.errorbar(3, xbar_cash_C, yerr =(UL_cash_C-LL_cash_C)/2, fmt = ".k")
ax.set_xticks([1,2,3])
plt.title("Clothing vs. Competence (95% CI)")
plt.xlabel('Clothing Type')
plt.ylabel('Student ratings on 5-point scale')
plt.xlim(xmin=.5,xmax=3.5)
plt.savefig("/home/alexander/PycharmProjects/StatisticalMethodsHW/Images/HW5Q4PTA.png")
plt.close()

fig, ax = plt.subplots()
plt.errorbar(1, xbar_formal_I, yerr = (UL_formal_I-LL_formal_I)/2, fmt = ".k")
plt.errorbar(2, xbar_casual_I, yerr = (UL_formal_I-LL_formal_I)/2, fmt = ".k")
plt.errorbar(3, xbar_cash_I, yerr = (UL_cash_I-LL_cash_I)/2, fmt = ".k")
ax.set_xticks([1,2,3])
plt.title("Clothing vs. Interesting Presentation (95% CI)")
plt.xlabel("Clothing Type")
plt.ylabel("Student ratings on 5-point scale")
plt.xlim(xmin=.5,xmax = 3.5)
plt.savefig("/home/alexander/PycharmProjects/StatisticalMethodsHW/Images/HW5Q4PTB.png")
plt.close()
