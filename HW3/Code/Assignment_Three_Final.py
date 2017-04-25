
# coding: utf-8

# In[ ]:

# Plotting library
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import gridspec

# Numerical Library
import numpy as np
from numpy import genfromtxt
import collections, math, operator

from scipy.special import expit

# Limit printout to 3 decimal places
np.set_printoptions(precision=3,suppress=True)

# Allows for printing inline for jupyter notebook
get_ipython().magic(u'matplotlib inline')


# In[ ]:

def func_read_data():
    raw_train = genfromtxt('knn_train.csv', delimiter=',')
    raw_test = genfromtxt('knn_test.csv', delimiter=',')
    return raw_train, raw_test

def func_filter_data(src_data):
    true_data = src_data[:,0]
    feature_data = np.delete(src_data, 0, axis=1)
#     src_data = np.delete(src_data, 0, axis=1)
#     temp_data = np.ones(len(src_data))
#     dummy_data = temp_data[np.newaxis].T
#     feature_data = np.concatenate((dummy_data, src_data), axis=1)
    return true_data, feature_data


# In[ ]:

## Part 1
Neigbor = collections.namedtuple('Neigbor', 'distance_data true_data')

def func_norm_data(feature_data):
    feature_max, feature_min = np.amax(feature_data), np.amin(feature_data)
    feature_range, feature_avg = feature_max - feature_min, np.average(feature_data)
    feature_norm = (feature_data - feature_avg) / feature_range
    return np.absolute(feature_norm)

def func_calc_dist(norm_data, test_data):
    diff_sqrt = (test_data - norm_data) ** 2
    distance_data = np.sqrt(np.sum(diff_sqrt, axis=1))
    return distance_data

def func_data_class(norm_data, true_data, test_data, k = 1):
    distance_data = func_calc_dist(norm_data, test_data)
    neigbor_data = map(Neigbor, distance_data, true_data)
    neigbor_data = sorted(neigbor_data, key=operator.attrgetter('distance_data'))
    neigbor_data_sum = np.sum([neigbor.true_data for neigbor in neigbor_data[:k]])
    return -1 if neigbor_data_sum < 0 else 1

def func_data_fmt(norm_data, true_data, k):
    return map(lambda classifier: func_data_class(norm_data, true_data, classifier, k), norm_data)

def func_calc_error(norm_data, true_data, k=1):
    temp_data = func_data_fmt(norm_data, true_data, k)
    return np.sum(np.abs(temp_data - true_data)) / float(2 * len(true_data))

def func_calc_cross_valid_error(norm_data, true_data, k = 1):
    error = 0
    for i in range(len(norm_data)):
        temp_data = func_data_class(norm_data, true_data, norm_data[i], k)
        error += float(np.absolute(true_data[i] - temp_data) / 2)
    return float(error / len(norm_data))
        


# In[ ]:

def func_test_data():
    raw_train, raw_test = func_read_data()
    true_data, feature_data = func_filter_data(raw_train)
    print true_data, '\n', feature_data
    print len(true_data), ',', len(feature_data), ',', len(feature_data[0])
    norm_data = func_norm_data(feature_data)
    print norm_data, len(norm_data), len(norm_data[0])    

def func_test_numpy():
    data = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    print np.sum(data, axis=0)
    print map(lambda x: x ** 2, data[0])

def func_plot_data(x_list, y_list, label):
    
    plt.plot(x_list, y_list, label=label)
    plt.legend()
    fig.suptitle(label + ' Error', fontsize=16)
    plt.xlabel('K-Value', fontsize=16)
    plt.ylabel('Error', fontsize=16)
    
if __name__ == '__main__':
    raw_train, raw_test = func_read_data()
    true_data, feature_data = func_filter_data(raw_train)
    test_true_data, test_feature_data = func_filter_data(raw_test)
    norm_data, test_norm_data = func_norm_data(feature_data), func_norm_data(test_feature_data)
    k_list, train_error_list, cross_valid_list, test_error_list = range(1, 52, 2), [], [], []
    
    for k in k_list:
        train_error = func_calc_error(norm_data, true_data, k)
        train_error_list.append(train_error)
        cross_valid = func_calc_cross_valid_error(norm_data, true_data, k)
        cross_valid_list.append(cross_valid)
        test_error = func_calc_error(test_norm_data, test_true_data, k)
        test_error_list.append(test_error)

    print 'Training:', train_error_list
    print 'Cross Valid:', cross_valid_list
    print 'Testing:', test_error_list
    fig = plt.figure(figsize=(9,6))
    func_plot_data(k_list, train_error_list, 'Training Data')
    func_plot_data(k_list, cross_valid_list, 'Cross Validation')
    func_plot_data(k_list, test_error_list, 'Testing Data')


# In[ ]:



