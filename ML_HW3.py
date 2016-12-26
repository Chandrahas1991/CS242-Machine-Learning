import numpy as np
import scipy.optimize as opti

'''
age = np.array([17,22,27,32,37,42,47])
ster = np.array([3,80,216,268,197,150,91])
other = np.array([61,137,131,76,50,24,10])
none = np.array([232,400,301,203,188,164,183])
labels = np.array([2,2,2,0,0,2,2])
'''

age = np.array([17,22,27,32,37,42,47])
test_age = np.array([17,22,35,32,37,42,47])
#test_age = np.array([32,33,34,35,36,37,38])
ster = np.array([3,80,216,268,197,150,91])
other = np.array([61,137,131,76,50,24,10])
none = np.array([232,400,301,203,188,164,183])
labels = np.array([2,2,0,0,0,2,2])
#labels = np.array([0,0,0,0,0,0,0])
age_new = np.array([])
labels_new = np.array([])

for i in range(0,age.__len__()):
    for j in range(0,ster[i]):
      age_new = np.append(age_new,age[i])
      labels_new = np.append(labels_new,0)
    for j in range(0,other[i]):
      age_new = np.append(age_new, age[i])
      labels_new = np.append(labels_new,1)
    for j in range (0,none[i]):
      age_new = np.append(age_new, age[i])
      labels_new = np.append(labels_new,2)



test_age = test_age.reshape(test_age.__len__(),1)
age = age.reshape(age.__len__(),1)
age_new = np.asarray(age_new)
labels_new = np.asarray(labels_new)
age_new = age_new.reshape(age_new.__len__(),1)
labels_new = labels_new.reshape(labels_new.__len__(),1)
age_new = age_new.astype(int)
labels_new = labels_new.astype(int)

num_classes = 3
in_size = 3


def feature_func(x):
    m,n = x.shape
    x = x.reshape(m,n)
    phi = np.zeros(shape=(3,m))
    phi[0,:] = np.ones(shape=(1,n))
    phi[1,:] = x.transpose()
    phi[2,:] = np.power(x.transpose(),2)
    #phi = np.array([x, np.power(x,2), np.power(x,3)])
    phi.reshape(3,m)
    return phi



def cost_func(weight_mat,phi_mat ,labels_cost):
    weight_mat = weight_mat.reshape(num_classes,in_size)
    prob_data = (weight_mat).dot(phi_mat)
    soft_data = np.exp(prob_data)/np.sum(np.exp(prob_data),axis=0)
    tags = np.zeros(shape=(soft_data.shape))
    for i in range(0,labels_cost.__len__()):
        tags[labels_cost[i],i]= 1
    soft_sum = np.sum(np.multiply(np.log(soft_data),tags)) #+ np.sum(np.multiply(weight_mat,weight_mat))
    grad_jat = np.subtract(soft_data,tags).dot(phi_mat.transpose())
    grad_mat = np.subtract(soft_data,tags)
    grad_vec = np.zeros(shape=(3,3))


    for i in range(0,grad_mat.shape[0]):
        grad = np.zeros(shape=(3, 1))
        for j in range(0,grad_mat.shape[1]):
            temp = np.asarray(age_new[j,0])
            temp =temp.reshape(1,1)
            dummy = np.multiply(grad_mat[i,j],feature_func(temp))
            grad = np.add(grad,dummy)
        grad_vec[i,:]= np.divide(grad.transpose(),grad_mat.shape[1])

    soft_sum = -soft_sum #/grad_mat.shape[1]
    print(soft_sum)
    return soft_sum,-grad_vec.flatten()
    #print(-soft_sum)
    #return soft_sum, -grad_jat.flatten()

def predict(opt_weight, data_mat):
    prod = opt_weight.dot(data_mat)
    pred = np.exp(prod) / np.sum(np.exp(prod), axis=0)
    print pred.shape
    print pred
    pred = pred.argmax(axis=0)

    return pred

weight_init = 0.05*np.random.randn(num_classes*in_size)
#weight_init = [0,0,0,0,0,0,0,0,0]
#phi_mat = feature_func(age)
phi_mat = feature_func(age_new)

J = lambda x: cost_func(x,phi_mat,labels_new)
options = {'maxiter': 400, 'disp': True}
opt_weight = opti.minimize(J,weight_init, method='SLSQP',jac= True, options= options)
print("the opt weight is")
optimal_weight = opt_weight.x
optimal_weight = optimal_weight.reshape(3,3)
print(optimal_weight)
print("the prediction is :")
print(predict(optimal_weight,feature_func(test_age)))


print "Accuracy: {0:.2f}%".format(100 * np.sum(predict(optimal_weight,feature_func(test_age)) == labels, dtype=np.float64) / labels.shape[0])

