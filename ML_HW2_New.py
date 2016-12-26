import matplotlib.pyplot as pyplot
import numpy as np
import scipy.optimize as optimize
import time as time

np.random.seed(int(time.time()))
N = 1000
x = np.zeros(shape=(N,1))
x = np.random.uniform(-100,100,N)
fx = np.zeros(shape=(N,1))
t = np.zeros(shape=(N,1))
y = np.zeros(shape=(N,1))
smat = np.zeros(shape=(N,1))
alpha = 1
M = 4
meanarray = np.zeros(shape=(N,1))
vararray = np.zeros(shape=(N,1))
mean_plus_var = np.zeros(shape=(N,1))
mean_minus_var = np.zeros(shape=(N,1))

# function to generate labels for a given array of data points
def func(r):
    return np.random.normal(0.1+(2.0*r)+(r*r)+(3.0*np.power(r,3)),1)

#function used to fit the dataset
def curfunc(x,w0,w1,w2,w3):
    return w0+(w1*x)+(w2*np.power(x,2))+(w3*np.power(x,3))

#function to transform the new datapoint to a Mx1 vector
def phi(x):
    xphi = np.zeros(shape=(M,1))
    for i in range(0,M):
        xphi[i]=np.power(x,i)
    return xphi

#Function to generate the S matrix used in the mean calculation
def gen_s_mat(x):
    sinvmat = np.zeros(shape=(M,M))
    for i in range(0,len(x)):
       sinvmat= sinvmat+np.dot(phi(x[i]),phi(x[i]).transpose())
    sinvmat = np.multiply(beta,sinvmat) + np.dot(alpha,np.identity(M))
    return np.linalg.inv(sinvmat)

#Function to calculate posterior mean for any new data point
def gen_mean(x,xn,smat):
    tempmat = np.zeros(shape=(M,1))
    for i in range(0,N):
        tempmat = tempmat + np.multiply(phi(x[i]),t[i,0])
    tempmat2 = np.multiply(beta,np.dot(phi(xn).transpose(),np.dot(smat,tempmat)))
    return tempmat2

#function to calculate the posterior varaince for any new data point
def gen_var(xn,smat):
    var = (1/beta) + np.dot(phi(xn).transpose(),np.dot(smat,phi(xn)))
    return var
#generate the labels for input data points
t = func(x)
wopt,wcov = optimize.curve_fit(curfunc,x,t)

# Reshape t and x arrays to M dimensional vectors
t = t.reshape(N,1)
x = x.reshape(N,1)

#Generate the new set labels with the weight vector obtained from curve fitting
for i in range(0,N):
    y[i] = curfunc(x[i],wopt[0],wopt[1],wopt[2],wopt[3])

#Calculate the precision of curve fitting
diffmat = np.subtract(y,t)
dotmat = np.dot(diffmat.transpose(),diffmat)
beta = N/dotmat

print("the number of samples :%d" %N)
print("the weight vector is : ")
print(wopt)
print("the beta value is : ")
print(beta)

#Generate S matrix for mean calculation
smat = gen_s_mat(x)
#Generate posterior mean and variance for all datapoints
for i in range(0,N):
    meanarray[i] = gen_mean(x,x[i],smat)
    vararray[i] =  gen_var(x[i],smat)

vararray = np.sqrt(vararray)
mean_plus_var = np.add(meanarray,vararray)
mean_minus_var = np.subtract(meanarray,vararray)

print("the mean of the variance is :")
print(np.mean(vararray))
print("the value of alpha is:")
print(alpha)

fig = pyplot.figure()
ax1 = fig.add_subplot(221)
ax1.scatter(x,t)
pyplot.xlabel('X vs T data')
pyplot.ylabel('Labels')
pyplot.grid()

ax2 = fig.add_subplot(222)
ax2.scatter(x,y)
pyplot.xlabel('X vs Estimated Mean(from CurveFit)')
pyplot.ylabel('Labels')
pyplot.grid()

ax3 = fig.add_subplot(223)
sc1 = ax3.scatter(x,meanarray,c='b',)
sc2 = ax3.scatter(x,mean_plus_var, c='g')
sc3 = ax3.scatter(x,mean_minus_var, c='r')
ax3.legend((sc1,sc2,sc3),('mean','mean+SD','mean-SD'),loc='upper left',fontsize = 12)
pyplot.xlabel('X vs Posterior Mean and Variance')
pyplot.ylabel('Labels')
pyplot.grid()


pyplot.show()








