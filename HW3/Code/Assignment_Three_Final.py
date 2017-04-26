import numpy as np
from numpy import genfromtxt
import collections, math, operator

from scipy.special import expit

np.set_printoptions(precision=3,suppress=True)

<<<<<<< 33e1e6747ad2d09b701afb0ff03fd9fed8e45099
## Part 1
class DataFormat():

=======
	
## Part 1
class DataFormat():
	
>>>>>>> d226101e70227611927aea4dd9d7e82c8c0f7957
	def __init__(self, name):
		self.name = name

	def func_norm_data(self):
		src_data = genfromtxt(self.name, delimiter=',')
		true_data, feature_data = src_data[:,0], np.delete(src_data, 0, axis=1)
<<<<<<< 33e1e6747ad2d09b701afb0ff03fd9fed8e45099

		feature_max, feature_min = np.amax(feature_data), np.amin(feature_data)
		feature_range, feature_avg = feature_max - feature_min, np.average(feature_data)
		feature_norm = (feature_data - feature_avg) / feature_range

		self.norm_data, self.true_data = np.absolute(feature_norm), true_data
		return np.absolute(feature_norm), true_data

=======
		
		feature_max, feature_min = np.amax(feature_data), np.amin(feature_data)
		feature_range, feature_avg = feature_max - feature_min, np.average(feature_data)
		feature_norm = (feature_data - feature_avg) / feature_range
		
		self.norm_data, self.true_data = np.absolute(feature_norm), true_data
		return np.absolute(feature_norm), true_data
		
>>>>>>> d226101e70227611927aea4dd9d7e82c8c0f7957
	def func_leave_out(self, index):
		return self.norm_data[index], self.true_data[index]

class KNN():
<<<<<<< 33e1e6747ad2d09b701afb0ff03fd9fed8e45099

	def __init__(self, norm_data):
		self.norm_data = norm_data

	def func_classify(test_data):
        diff_sqrt = (test_data - self.norm_data) ** 2
        distance_data = np.sqrt(np.sum(diff_sqrt, axis=1))
        print '+++', distance_data
=======
	
	def __init__(self, norm_data):
		self.norm_data = norm_data
		
	def func_classify():
>>>>>>> d226101e70227611927aea4dd9d7e82c8c0f7957
# Neigbor = collections.namedtuple('Neigbor', 'distance_data true_data')

# def func_data_class(norm_data, true_data, test_data, k = 1):
	# diff_sqrt = (test_data - norm_data) ** 2
	# distance_data = np.sqrt(np.sum(diff_sqrt, axis=1))
	# neigbor_data = map(Neigbor, distance_data, true_data)
	# neigbor_data = sorted(neigbor_data, key=operator.attrgetter('distance_data'))
	# neigbor_data_sum = np.sum([neigbor.true_data for neigbor in neigbor_data[:k]])
	# return -1 if neigbor_data_sum < 0 else 1

# def func_calc_error(norm_data, true_data, k = 1):
	# temp_data = map(lambda classifier: func_data_class(norm_data, true_data, classifier, k), norm_data)
	# return np.sum(np.abs(temp_data - true_data)) / float(2 * len(true_data))

# def func_calc_cross_valid_error(norm_data, true_data, k = 1):
	# error = 0
	# for i in range(len(norm_data)):
		# temp_data = func_data_class(norm_data, true_data, norm_data[i], k)
		# error += float(np.absolute(true_data[i] - temp_data) / 2)
	# return float(error / len(norm_data))

if __name__ == '__main__':
	train_data = DataFormat('knn_train.csv')
	train_norm_data, true_data = train_data.func_norm_data()
	# train_leave_out = train_data.func_leave_out(3)
	# print train_norm_data, train_leave_out
<<<<<<< 33e1e6747ad2d09b701afb0ff03fd9fed8e45099
    knn_train = KNN(train_norm_data)
    knn_train.func_classify(train_norm_data[0])
    
=======
>>>>>>> d226101e70227611927aea4dd9d7e82c8c0f7957
	# raw_train, raw_test = func_read_data()
	# true_data, feature_data = func_filter_data(raw_train)
	# test_true_data, test_feature_data = func_filter_data(raw_test)
	# norm_data, test_norm_data = func_norm_data(feature_data), func_norm_data(test_feature_data)
	# k_list, train_error_list, cross_valid_list, test_error_list = range(1, 52, 2), [], [], []
<<<<<<< 33e1e6747ad2d09b701afb0ff03fd9fed8e45099

=======
	
>>>>>>> d226101e70227611927aea4dd9d7e82c8c0f7957
	# for k in k_list:
		# train_error = func_calc_error(norm_data, true_data, k)
		# train_error_list.append(train_error)
		# cross_valid = func_calc_cross_valid_error(norm_data, true_data, k)
		# cross_valid_list.append(cross_valid)
		# test_error = func_calc_error(test_norm_data, test_true_data, k)
		# test_error_list.append(test_error)

	# print 'Training:', train_error_list
	# print 'Cross Valid:', cross_valid_list
<<<<<<< 33e1e6747ad2d09b701afb0ff03fd9fed8e45099
	# print 'Testing:', test_error_list
=======
	# print 'Testing:', test_error_list
>>>>>>> d226101e70227611927aea4dd9d7e82c8c0f7957
