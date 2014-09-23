# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 17:59:37 2014

@author: tomek
"""

import numpy as np
from StringIO import StringIO
from pylab import *
import scipy.linalg as linalg

'''
--------- Preprocessing data ---------
TODO :
    - Optimize and refactor
'''
# open file and read data
data = open('Data.data', 'r').read()
# read data from string and put in columns as strings
array_of_strings = np.genfromtxt(StringIO(data),dtype='|S11', delimiter=",")

# create lists with attributes values
make = sorted(set(array_of_strings[:,2]))
fuel_type = sorted(set(array_of_strings[:,3]))
aspiration = sorted(set(array_of_strings[:,4]))
num_of_doors = sorted(set(array_of_strings[:,5])-set('?'))
body_style = sorted(set(array_of_strings[:,6]))
drive_wheels = sorted(set(array_of_strings[:,7]))
engine_location = sorted(set(array_of_strings[:,8]))
engine_type = sorted(set(array_of_strings[:,14]))
num_of_cylinders = sorted(set(array_of_strings[:,15]))
fuel_system = sorted(set(array_of_strings[:,17]))

array_of_data = np.genfromtxt(StringIO(data),
                              converters = {
                    2: lambda x: float(make.index(x)),
                    3: lambda x: float(fuel_type.index(x)),
                    4: lambda x: float(aspiration.index(x)),
                    5: lambda x: float(num_of_doors.index(x)),
                    6: lambda x: float(body_style.index(x)),
                    7: lambda x: float(drive_wheels.index(x)),
                    8: lambda x: float(engine_location.index(x)),
                    14: lambda x: float(engine_type.index(x)),
                    15: lambda x: float(num_of_cylinders.index(x)),
                    17: lambda x: float(fuel_system.index(x))},
                    usecols = (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                               20, 21, 22, 23, 24, 25), 
                    delimiter=",")

_X = [item.tolist() for item in array_of_data]
index = [ _X.index(item) for item in _X for _item in item if np.isnan(_item)]
_index = sorted(set(index))

iter = 0

for i in _index:
    del _X[i-iter]
    iter += 1    

'''
-------- End of data preprocessing -----------
'''
'''
------- PCA ---------
'''
Num_of_attributes = 25
Num_of_instances = len(_X)

# Create matrix X with data
X = np.mat(np.empty((Num_of_instances, Num_of_attributes)))
for i in range(Num_of_instances):
    X[i, :] = np.mat(_X[i])

Y = X - np.ones((Num_of_instances, 1))*X.mean(axis=0)
    
# PCA by computing SVD of Y
U,S,V = linalg.svd(Y,full_matrices=False)

#Z = Y * V
#
i = 0
j = 1

# Plot PCA of the data
#f = figure()
#f.hold()
#Z = array(Z)
#
#plot(Z[:,i], Z[:,j], 'o')

## Compute variance explained by principal components
#rho = (S*S) / (S*S).sum() 
##
#figure()
#plot(range(1,len(rho)+1),rho,'o-')
#title('Variance explained by principal components');
#xlabel('Principal component');
#ylabel('Variance explained');
#show()
#
#i = 0
#j = 1
#
###
## Make a simple plot of the i'th attribute against the j'th attribute
## Notice that X is of matrix type and need to be cast to array. 
#figure()
#X = array(X)
#plot(X[:,i], X[:,j], 'o');

'''
------- End of PCA ------
'''
