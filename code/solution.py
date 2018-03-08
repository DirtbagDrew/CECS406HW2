import numpy as np 
from helper import *
import math
import pdb
'''
Homework2: logistic regression classifier
'''


def logistic_regression(data, label, max_iter, learning_rate):
    '''
	The logistic regression classifier function.

	Args:
	data: train data with shape (1561, 3), which means 1561 samples and 
		  each sample has 3 features.(1, symmetry, average internsity)
	label: train data's label with shape (1561,1). 
		   1 for digit number 1 and -1 for digit number 5.
	max_iter: max iteration numbers
	learning_rate: learning rate for weight update
	
	Returns:
		w: the seperater with shape (3, 1). You must initilize it with w = np.zeros((d,1))
	''' 
    
    feature_count=data.shape[1]
    w = np.zeros(feature_count) # initialize the weights at time step t=0 to w(0)
    n, _ = data.shape
    g=0
    for i in range (max_iter):  # for t=0,1,2....max_iter do
        for j in range(n):      # compute the gradient: 1/N(the sum from 1 to N(ln(1+e^(-yn*w^t*xn))))
            W=np.transpose(w)
            g=g+(data[j]*label[j])/(1+np.exp(label[j]*W*data[j]))
        g=g*(-1/n)
        v=-g    #set the direction to move
        w=w+learning_rate*v     #update the weights
    return w    #return the final weights
    


def thirdorder(data):
    '''
	This function is used for a 3rd order polynomial transform of the data.
	Args:
	data: input data with shape (:, 3) the first dimension represents 
		  total samples (training: 1561; testing: 424) and the 
		  second dimesion represents total features.

	Return:
		result: A numpy array format new data with shape (:,10), which using 
		a 3rd order polynomial transformation to extend the feature numbers 
		from 3 to 10. 
		The first dimension represents total samples (training: 1561; testing: 424) 
		and the second dimesion represents total features.
    '''
    #third order transformation(x) = (1, x1, x2, x1^2, x1x2, x2^2, x1^3, x1^2 * x2, x1 * x2^2, x2^3) 
    n, _= data.shape
    result = [[0 for x in range(10)] for y in range(n)] 
    for i in range(n):
        result[i][0]=1
        result[i][1]=data[i][0]
        result[i][2]=data[i][1]
        result[i][3]=data[i][0]**2
        result[i][4]=data[i][0]*data[i][1]
        result[i][5]=data[i][1]**2
        result[i][6]=data[i][0]**3
        result[i][7]=(data[i][0]**2)*data[i][1]
        result[i][8]=data[i][0]*(data[i][1]**2)
        result[i][9]=data[i][1]**3
    return np.array(result)        

def accuracy(x, y, w): #x is data y is w
    '''
    This function is used to compute accuracy of a logsitic regression model.
    
    Args:
    x: input data with shape (n, d), where n represents total data samples and d represents
        total feature numbers of a certain data sample.
    y: corresponding label of x with shape(n, 1), where n represents total data samples.
    w: the seperator learnt from logistic regression function with shape (d, 1),
        where d represents total feature numbers of a certain data sample.

    Return 
        accuracy: total percents of correctly classified samples. Set the threshold as 0.5,
        which means, if the predicted probability > 0.5, classify as 1; Otherwise, classify as -1.
    '''
    n, _ = x.shape
    threshold=.5
    numCorr=0
    for i in range(n):  
        sgmd=1/(1+np.exp(-1*y[i]*np.dot(w,x[i])))   #1/(1+e^(-1*yn*wt*xn))
        if sgmd >threshold: 
            cls=1
        else:
            cls=-1
        if cls==y[i]: # if the sigmoid classification is equal to the label at to numCorr counter
            numCorr=numCorr+1
    accuracy=numCorr/n 
    return accuracy
    
    
    


