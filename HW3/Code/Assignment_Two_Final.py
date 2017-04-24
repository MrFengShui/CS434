
# coding: utf-8

# In[1]:

# Plotting library
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec

# Numerical Library
import numpy as np
from numpy import genfromtxt
import math


# In[20]:

def func_read_data():
    raw_train = genfromtxt('knn_train.csv', delimiter=',')
    raw_test = genfromtxt('knn_test.csv', delimiter=',')
    return raw_train, raw_test

def func_filter_data(src_data):
    true_data = src_data[:,0]
    src_data = np.delete(src_data, 0, axis=1)
    temp_data = np.ones(len(src_data))
    dummy_data = temp_data[np.newaxis].T
    feature_data = np.concatenate((dummy_data, src_data), axis=1)
    return true_data, feature_data

raw_train, raw_test = func_read_data()
# print len(raw_train)
true_data, feature_data = func_filter_data(raw_train)
print true_data, len(true_data), '\n', feature_data, len(feature_data), len(feature_data[0])


# In[ ]:



