# Numerical Library
import numpy as np
from numpy import genfromtxt
import collections, math, operator
from scipy.special import expit
# Limit printout to 3 decimal places
np.set_printoptions(precision=3,suppress=True)
## Part 1
DistanceFlagPair = collections.namedtuple('Distance', 'distance flag')

class DataFormat():

    def __init__(self, name):
        self.name = name

    def func_norm_data(self):
        src_data = genfromtxt(self.name, delimiter=',')
        flag_data, feature_data = src_data[:,0], np.delete(src_data, 0, axis=1)
        feature_max, feature_min = np.amax(feature_data, axis=0), np.amin(feature_data, axis=0)
        feature_range, feature_avg = feature_max - feature_min, np.average(feature_data)
        norm_data = (feature_data - feature_avg) / feature_range
        self.norm_data, self.flag_data = norm_data, flag_data
        return norm_data, flag_data

    def func_leave_out(self, index):
        return self.norm_data[index], self.flag_data[index]

class KNN():

    def __init__(self, norm_data, flag_data):
        self.norm_data, self.flag_data = norm_data, flag_data

    def func_classify(self, test_data, k):
        neigbor_pair = self.func_build_neigbor(test_data)
        flag_sum = np.sum([neigbor.flag for neigbor in neigbor_pair[:k]])
        return -1 if flag_sum < 0 else 1
        
    def func_build_neigbor(self, test_data):
        diff_sqrt = (test_data - self.norm_data) ** 2
        distance_data = np.sqrt(np.sum(diff_sqrt, axis=1))
        neigbor_pair = []
        for i in range(len(distance_data)):
            neigbor = DistanceFlagPair(distance_data[i], self.flag_data[i])
            neigbor_pair.append(neigbor)
        return sorted(neigbor_pair, key=operator.attrgetter('distance'))        

def func_calc_error(data, knn, k = 1):
    temp_data = [knn.func_classify(classifier, k) for classifier in data.norm_data]
    return np.sum(np.abs(temp_data - data.flag_data)) / float(2 * len(data.flag_data))
    
def func_cross_valid_error(data, knn, k = 1):
    error = 0
    for i in range(len(data.norm_data)):
        norm, flag = data.func_leave_out(i)
        flag_sum = knn.func_classify(norm, k)
        flag_error = np.abs(flag - flag_sum) / 2
        error += flag_error
    return float(error) / (len(data.flag_data) + 1) 

def func_part_one():
    train_data, test_data = DataFormat('knn_train.csv'), DataFormat('knn_test.csv')
    train_norm_data, train_flag_data = train_data.func_norm_data()
    test_norm_data, test_flag_data = test_data.func_norm_data()
    knn_train, knn_test = KNN(train_norm_data, train_flag_data), KNN(test_norm_data, test_flag_data)
    k_list, train_error_list, cross_valid_list, test_error_list = range(1, 52, 2), [], [], []
    for k in k_list:
        train_error = func_calc_error(train_data, knn_train, k)
        train_error_list.append(train_error)
        cross_valid = func_cross_valid_error(train_data, knn_train, k)
        cross_valid_list.append(cross_valid)
        test_error = func_calc_error(test_data, knn_test, k)
        test_error_list.append(test_error)

    print 'Training:', train_error_list
    print 'Cross Valid:', cross_valid_list
    print 'Testing:', test_error_list    
   
if __name__ == '__main__':
    func_part_one()