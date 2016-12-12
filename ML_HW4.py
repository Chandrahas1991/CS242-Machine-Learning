import numpy as np
import matplotlib.pyplot as plt
from svmutil import *

''' Practice with LibSVM '''
mean_plus = [1, 1]
cov_plus = [[1, 0], [0, 1]]
data_plus = np.random.multivariate_normal(mean_plus, cov_plus, 1000)
#print(data_plus[:1,:])
#print "Data_Plus data type is :" + str(type(data_plus))
#print "Data_plus size is: " + str(data_plus.shape)

mean_minus = [-1,-1]
cov_minus = [[3, 0],[0, 3]]
data_minus = np.random.multivariate_normal(mean_minus,cov_minus,1000)
#print "Data_minus data type is :" + str(type(data_minus))

data_train = np.concatenate((data_plus[:900,:],data_minus[:900,:]),axis=0)
data_test  = np.concatenate((data_plus[900:,:],data_minus[900:,:]),axis=0)

print "Size of data_train :" + str(data_train.size)
print "Shape of data_train :" + str(data_train.shape)


#x_train = [i{"j" : i[j] for j in range(0,2)} for i in range(0,1800)]
x_train = []
x_test = []

for sample in range(0,1800):
    temp = {
    1: data_train[sample,0],
    2: data_train[sample,1]
    }
    x_train.append(temp)

for sample in range(0,200):
    temp ={
        1: data_test[sample,0],
        2: data_test[sample,1]
    }
    x_test.append(temp)

#print x_train[:1]

y_train  = np.concatenate((np.ones(shape=(900,1)),-np.ones(shape=(900,1))),axis = 0).tolist()
y_test   = np.concatenate((np.ones(shape=(100,1)),-np.ones(shape=(100,1))),axis = 0).tolist()
y_train  = [i[0] for i in y_train]
y_test   = [i[0] for i in y_test]
print "y == " + str (y_train[1790:])

#print "DataX type is " + str(type(dataX_train))
prob = svm_problem(y_train,x_train )

#print "The second type of DataX is :" + str(len(dataX_train))

model_ptr = svm_train(prob,'-c 1 -g 1000')
model = toPyModel(model_ptr)
p = model.SVs
w = np.dot(model.SVs.transpose(),model.sv_coef)
b = -model.rho

print "W ===="
print w
print "b ==="
print b

p_label, p_acc, p_val = svm_predict(y_test,x_test,model_ptr)
plt.plot(data_plus[:,0],data_plus[:,1], 'x')
plt.axis('equal')
plt.show()
