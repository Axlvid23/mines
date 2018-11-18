import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import statistics


# Question # 1
# A test to detect certain bacteria in water has the following characteristics: if the water is
# free from the bacteria, which happens with probability 0.99, the test indicates the the water
# has the bacteria with probability 0.05. If the water is contaminated by the bacteria, the test
# correctly detects it with probability 0.99. Determine the probability that the water is not
# contaminated by the bacteria when the test indicates that it is.

P_F = 0.99 # Probability that water is free from bacteria
P_TposF = 0.05 # Probability that test is positive given free from bacteria
P_TposB = 0.99 # Probability that if water is contaminated, test is correct
#Using bayes theorem for conditional probability
P_FTpos = (P_TposF*P_F)/((P_TposF*P_F)+((1-P_TposF)*(1-P_F))) # Probability that water is not contaminated when test indicates it is contaminated
print("Q1: Probability that water is not contaminated when test indicates that it is contaminated: %s" %P_FTpos)

#Question #2
# (a) A discrete random variable X has the following probability mass function:
x = np.array([0,1,2,3,4])
Px = np.array([0.1,0.2,0.3,0.3,0.1])
CDFx = (np.cumsum(x*Px))/(np.sum(x*Px))
#plt.step(x,Px, where='post')
plt.step(x,CDFx, 'r--')
plt.title('CDF Q2 PT A')
plt.xlabel('x')
plt.ylabel('Px')
plt.xlim(xmin = 0,xmax = 5)
plt.ylim(ymin = 0, ymax = 1.2)
plt.show()


# (b) Determine the probability that 0 < X ≤ 2 and the probability that 0 < X < 2
# a = P(0<x<=2)= Fx(x<=2) - Fx(x<=0) where Fx(x<=b)=P(x<=b)
resulta = .3 + .2
resultb = .2
print("Q2, PT B: Probability that X is greater than 0 but less than or equal to 2 is: %s" %resulta)
print("Q2, PT B: Probability that X is greater than 0 but less than 2: %s" %resultb)
# (c) Find the probability that X = 0 given that X is less than 2
# Since X = 0 is contained in the set X is less than 2:
#  c = (P(X=0)P(X<2))/(P(X<2) = P(X=0)
resultc = 1/3
print("Q2, PT C: Probability of X=0 given that X is less than 2 is: %s"% resultc)

# (d) Let Y = X^2 + 1.  Find the mean of X and Y
X_bar = sum(Px*x)
print("Q2, PT D: X_bar is: %s" %X_bar)

Y_bar = (X_bar)**2 +1
print("Q2, PT D: Y_bar is: %s" %Y_bar)

# (e) Find the joint PMF of X and Y and the PMF of Y:

Py = ((Px)**2 + 1)
y = (x)**2 + 1

PMF_Y = np.row_stack(([Px], [y]))
print("Q2, PT E: PMF of Y is: %s" %PMF_Y)

PMF_XY = np.row_stack(([Px],[x],[y]))
print("Q2 PT E: Joint PMF of X and Y is: %s" %PMF_XY)

# (f) Find the standard deviation of X and y

sigma_x = statistics.stdev(x,X_bar)
#sigma_x = np.sqrt(np.mean(abs(x - X_bar)**2))
print("Q2 PT F: Standard deviation of X: %s" %sigma_x)

sigma_y = statistics.stdev(y,Y_bar)
#sigma_y = np.sqrt(np.mean(abs(y-Y_bar)**2))
print("Q2 PT F: Standard deviation of Y: %s" %sigma_y)

# Question #3
# Ten subjects are asked to taste wine from three different glasses. Two of the glasses contain
# Spanish wine and the other a Californian wine. Let X be the number of subjects that
# correctly identify the Californian wine.
# (a) Assuming that the Spanish and Californian wine taste exactly the same, what is the
# distribution of X?
P_Success = 10/30 # Probability wine selected is Californian variety
n = 10
k = np.array([1,2,3,4,5,6,7,8,9,10]) # Number of subjects in the experiment

# 10 subjects and the probability that the subject will correctly identify is random if the taste is the same
noverk= scipy.misc.factorial(n)/(scipy.misc.factorial(n-k)*scipy.misc.factorial(k))
P_Binom = (noverk)*(P_Success**k)*(1-P_Success)**(n-k) # Probability of success in 10 trials using Poisson distribution
print("Q3, PT A: The distribution of X is: %s" %P_Binom)
np.histogram2d(k,P_Binom)
plt.xlabel('# of successes')
plt.ylabel('Probability')
plt.title('Binomial Distribution P3 PT A')
plt.show()

# (b) Do you think there would be evidence of difference in taste of the two wines if 7 out of
# the 10 subjects correctly identify the Californian wine? What if it is 8 out of the 10?
print("Q3, PT B: The probability 7 out of 10 subjects correctly identify is: %s" %P_Binom[6])
print("Q3 PT B: The probability 8 out of 10 subjects correctly identify is %s" %P_Binom[7])

# Question #4
#  Use simulations to approximate the value of E( sin( X) ) when X ∼ B(10, 2/3)

n_4 = 10
P_4 = 2/3
X_4 = np.random.binomial(n_4, P_4, 10000)
Approx_val = np.sin(np.sqrt(X_4))
Y_bar_4 = sum(Approx_val)/len(Approx_val)
print("Q4:  The expected value Y_bar is: %s" %Y_bar_4)
plt.plot(Approx_val)
plt.title("Simulation of the value of sin(sqrt(x)) when X ~ B(10,2/3), P4")
plt.xlabel("Simulations")
plt.ylabel("X")
plt.show()

# Question #5
# Interruptions in a telecommunications network occur at an average rate of one per day. Let
# X be the number of interruptions in the next five-day work week, and let Y be the number
# of weeks in the next four in which there are no interruptions.
# (a) Model the distribution of X with a Poisson distribution and determine P(X = 0)
n_5 = 5
P_5 = 1
lambd_5 = n_5*P_5
nseq_5 = np.arange(0,20,1)
P_Poisson_5 = ((lambd_5**nseq_5)/scipy.misc.factorial(nseq_5))*np.exp(-lambd_5)
plt.plot(nseq_5,P_Poisson_5)
plt.title("Poisson distribution of telecommunication interruptions, P5 PT A")
plt.xlabel("n")
plt.ylabel("Probability")
plt.show()
print("Q5 PT A: P(X=0) is: %s" %P_Poisson_5[0])

# (b) What is a reasonable distribution to model Y ? Using such distribution, determine
# P (Y = 2)

P_Success_5b = P_Poisson_5[0]
n_5b = 4
k_5b = np.array([1,2,3,4])
noverk_5b= scipy.misc.factorial(n_5b)/(scipy.misc.factorial(n_5b-k_5b)*scipy.misc.factorial(k_5b))
P_Binom_5b = (noverk_5b)*(P_Success_5b**k_5b)*(1-P_Success_5b)**(n_5b-k_5b)

print("Q5 PT B: P(X=2) is: %s" %P_Binom_5b[2])
