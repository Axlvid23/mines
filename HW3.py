import scipy.misc
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import statistics



#question 1
nsample = 20
np.random.seed(7654321)
z = stats.t.rvs(3, size=nsample)
res = stats.probplot(z, plot=plt)
plt.title('Normal Probability Plot')
plt.show(res)

x = stats.norm.rvs(3, size=nsample)
res2 = stats.probplot(x, plot=plt)
plt.title('Normal Probability Plot')
plt.show(res2)

nsample2 = 100
np.random.seed(7654321)
z1 = stats.t.rvs(3, size=nsample2)
res3 = stats.probplot(z1, plot=plt)
plt.title('Normal Probability Plot')
plt.show(res3)

x1 = stats.norm.rvs(3, size=nsample2)
res4 = stats.probplot(x1, plot=plt)
plt.title('Normal Probability Plot')
plt.show(res4)

#question 2
#imports and cleans the datasets for
gey1 = np.genfromtxt('/home/alexander/Downloads/gey1.dat',
                     skip_header=0,
                     skip_footer=0,
                     dtype=None,
                     delimiter='')



plt.boxplot(gey1[:,1])
plt.title('Interruption Time Boxplot')
plt.ylabel('Time (minutes)')
plt.show()

x1=gey1[:,0]
y1=gey1[:,1]
plt.hist(y1)
plt.title('Interruption Time Histogram')
plt.ylabel('Time (minutes)')
plt.show()

CCOEF=np.corrcoef(gey1[:,1])
print('the correlation coefficent of the gy1 data is:  %s' %CCOEF)
print('the mean of duration time is %s, ' %np.median(y1))
print('the mean of interruption time is %s, ' %np.mean(y1))

plt.scatter(x1,y1)
plt.title('Training Data:  Interruption time vs. duration time scatterplot')
plt.ylabel('Interruption Time (minutes)')
plt.xlabel('Duration Time (minutes)')
slope, intercept, r_value, p_value, std_err = stats.linregress(x1,y1)
line = slope*x1+intercept
plt.plot(x1,line)
plt.show()
r_sqrd = r_value**2
print('The slope of the best fit line for the training data is: %s' %slope)
print('The intercept of the best fit line for the training data is: %s,' %intercept)
print('The R^2 value of the best fit line for the training data is: %s' %r_sqrd)


gey2 = np.genfromtxt('/home/alexander/Downloads/gey2.dat',
                     skip_header=0,
                     skip_footer=0,
                     dtype=None,
                     delimiter='')

x2 = gey2[:,0]
y2 = gey2[:,1]
plt.scatter(x2,y2)
plt.title('Validation Data vs. Training Data')
plt.ylabel('Interruption Time (minutes)')
plt.xlabel('Duration Time (minutes)')
plt.plot(x1,line, label = 'Training')
slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x2,y2)
line2 = slope2*x2+intercept2
plt.plot(x2,line2,'r-', label = 'Validation')
plt.legend()
plt.show()
r_sqrd2 = r_value2**2
print('The slope of the best fit line for the training data is: %s' %slope2)
print('The intercept of the best fit line for the training data is: %s,' %intercept2)
print('The R^2 value of the best fit line for the training data is: %s' %r_sqrd2)
#plt.boxplot(gey2)
#plt.show()


# Number 3
LW = np.array([1.15,0.84,0.88,0.91,0.86,0.88,0.92,0.87,0.93,0.95])
CW = np.array([0.89,0.69,0.46,0.85,0.73,0.67,0.78,0.77,0.80,0.79])
LW_x_bar = statistics.mean(LW)
CW_x_bar = statistics.mean(CW)
LW_med = statistics.median(LW)
CW_med = statistics.median(CW)
LW_std = statistics.stdev(LW,LW_x_bar)
CW_std = statistics.stdev(CW,CW_x_bar)
LW_1st = np.percentile(LW,25)
LW_3rd = np.percentile(LW,75)
CW_1st = np.percentile(CW,25)
CW_3rd = np.percentile(CW,75)
def mad(a, axis=None):
    med = np.median(a, axis=axis, keepdims=True)
    mad = np.median(np.absolute(a - med), axis=axis)
    return mad

print(mad(LW))

LW_mad = mad(LW)
CW_mad = mad(CW)
sum_titles = np.array(['x bar','median','standard deviation', '1st quartile', '3rd quartile','MAD'])
LW_summary = np.array([LW_x_bar,LW_med,LW_std,LW_1st,LW_3rd,LW_mad])
CW_summary = np.array([CW_x_bar,CW_med,CW_std,CW_1st,CW_3rd,CW_mad])
print(sum_titles)
print(LW_summary)
print(CW_summary)
plt.boxplot(LW)
plt.title('Boxplot of Lengthwise Cutting Technique')
plt.ylabel('Impact Strength (Ft-lb')
plt.ylim(ymin= .4,ymax = 1.15)
plt.show()

plt.boxplot(CW)
plt.title('Boxplot of Crosswise Cutting Technique')
plt.ylabel('Impact Strength (Ft-lb)')
plt.ylim(ymin = .4,ymax= 1.15)
plt.show()

stats.probplot(LW,plot=plt)
plt.title('Lengthwise Cutting Technique QQ Plot')
plt.show()

stats.probplot(CW,plot=plt)
plt.title("Crosswise Cutting Technique QQ Plot")
plt.show()


import numpy as np
import pylab


#Calculate quantiles
LW.sort()
quantile_levels1 = np.arange(len(LW),dtype=float)/len(LW)

CW.sort()
quantile_levels2 = np.arange(len(CW),dtype=float)/len(CW)

#Use the smaller set of quantile levels to create the plot
quantile_levels = quantile_levels2

#We already have the set of quantiles for the smaller data set
quantiles2 = CW

#We find the set of quantiles for the larger data set using linear interpolation
quantiles1 = np.interp(quantile_levels,quantile_levels1,LW)

#Plot the quantiles to create the qq plot
pylab.plot(quantiles1,quantiles2)

#Add a reference line
maxval = max(LW[-1],CW[-1])
minval = min(LW[0],CW[0])
pylab.plot([minval,maxval],[minval,maxval],'k-')
plt.title('QQ Plot Comparing Lengthwise and Crosswise Cutting Techniques')
plt.xlabel('Lengthwise Cutting Technique Quartiles')
plt.ylabel('Crosswise Cutting Technique Quartiles')
pylab.show()
