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

    def __init__(self, name = None, norm_data = None, flag_data = None):
        self.name = name
        self.norm_data = np.array(norm_data)
        self.flag_data = np.array(flag_data)

    def data_set(self):
        return genfromtxt(self.name, delimiter=',')

    def norm_flag_data(self):
        src_data = genfromtxt(self.name, delimiter=',')
        flag_data, feature_data = src_data[:,0], np.delete(src_data, 0, axis=1)
        feature_max, feature_min = np.amax(feature_data, axis=0), np.amin(feature_data, axis=0)
        feature_range, feature_avg = feature_max - feature_min, np.average(feature_data)
        norm_data = (feature_data - feature_avg) / feature_range
        self.norm_data, self.flag_data = norm_data, flag_data
        return norm_data, flag_data

    def leave_out(self, index):
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
        norm, flag = data.leave_out(i)
        flag_sum = knn.func_classify(norm, k)
        flag_error = np.abs(flag - flag_sum) / 2
        error += flag_error
    return float(error) / (len(data.flag_data) + 1)

## Part 2
class DecisionStump():

    def __init__(self, norm_data, flag_data):
        self.norm_data, self.flag_data = norm_data, flag_data

    def build_stump(self):
        row, col = self.norm_data.shape
        cur_info_gain, feature_num, split_num = 0, 0, 0
        for index in range(col):
            for value in self.norm_data.T[index]:
                tmp_info_gain = self.info_gain(index, value)
                if cur_info_gain < tmp_info_gain:
                    cur_info_gain = tmp_info_gain
                    feature_num, split_num = index, value
        return cur_info_gain, feature_num, split_num

    def info_gain(self, index, value):
        pos, neg = func_count_one(self.flag_data)
        init_entropy = func_calc_entropy(pos, neg)
        row_count, col_count = self.norm_data.shape
        upper_pos_count, upper_neg_count, lower_pos_count, lower_neg_count = self.count_one(index, value)
        upper_entropy = func_calc_entropy(upper_pos_count, upper_neg_count)
        upper_entropy = float(upper_pos_count + upper_neg_count) * upper_entropy / row_count
        lower_entropy = func_calc_entropy(lower_pos_count, lower_neg_count)
        lower_entropy = float(lower_pos_count + lower_neg_count) * lower_entropy / row_count
        return init_entropy - upper_entropy - lower_entropy

    def count_one(self, index, value):
        upper_pos_count, upper_neg_count = 0, 0
        lower_pos_count, lower_neg_count = 0, 0
        norm_data_t = self.norm_data.T[index]
        for i in range(len(norm_data_t)):
            tmp_value = norm_data_t[i]
            if tmp_value > value:
                if self.flag_data[i] > 0: upper_pos_count += 1
                else: upper_neg_count += 1
            else:
                if self.flag_data[i] > 0: lower_pos_count += 1
                else: lower_neg_count += 1
        return upper_pos_count, upper_neg_count, lower_pos_count, lower_neg_count

    def calc_error_rate(self, feature_num, split_num):
        predict, true, false = 0, 0, 0
        for i in range(len(self.norm_data)):
            predict = -1 if self.norm_data[i][feature_num] < split_num else 1
            if predict == self.flag_data[i]: true += 1
            else: false += 1
        return float(true) / (true + false)

class TreeNode():

    def __init__(self, column = -1, value = None, rs = None, left = None, right = None):
        self.column, self.value = column, value
        self.rs, self.left, self.right = rs, left, righ

class DecisionTree():

    def __init__(self, norm_data, flag_data, depth, height):
        self.norm_data, self.flag_data = norm_data, flag_data
        self.depth, self.height = depth, height
        self.cur_info_gain, self.feature_num, self.split_num = 0, 0, 0
        if depth > 0:
            self.cur_info_gain, self.feature_num, self.split_num = self.format_data()
            if self.cur_info_gain > 0:
                data_left, data_right = self.split_data(self.norm_data, self.flag_data, self.feature_num, self.split_num)
                #LHS
                left_norm_data, left_flag_data = data_left.norm_data, data_left.flag_data
                self.left_child = DecisionTree(left_norm_data, left_flag_data, self.depth - 1, self.height + 1)
                #RHS
                right_norm_data, right_flag_data = data_right.norm_data, data_right.flag_data
                self.right_child = DecisionTree(right_norm_data, right_flag_data, self.depth - 1, self.height + 1)
            else:
                self.depth = 0

    def format_data(self):
        row, col = self.norm_data.shape
        cur_info_gain, feature_num, split_num = 0, 0, 0
        for index in range(col):
            for value in self.norm_data.T[index]:
                tmp_info_gain = self.info_gain(index, value)
                if cur_info_gain < tmp_info_gain:
                    cur_info_gain = tmp_info_gain
                    feature_num, split_num = index, value
        return cur_info_gain, feature_num, split_num

    def split_data(self, norm_data, flag_data, feature_num, split_num):
        row, col = norm_data.shape
        left_norm, left_flag = [], []
        right_norm, right_flag = [], []
        for i in range(col):
            item_data = norm_data[i]
            if item_data[i] > split_num:
                right_norm.append(item_data[i])
                right_flag.append(flag_data[i])
            else:
                left_norm.append(item_data[i])
                left_flag.append(flag_data[i])
        return DataFormat(norm_data = left_norm, flag_data = left_flag), DataFormat(norm_data = right_norm, flag_data = right_flag)

    def info_gain(self, index, value):
        pos, neg = func_count_one(self.flag_data)
        init_entropy = func_calc_entropy(pos, neg)
        row_count, col_count = self.norm_data.shape
        upper_pos_count, upper_neg_count, lower_pos_count, lower_neg_count = self.count_one(index, value)
        upper_entropy = func_calc_entropy(upper_pos_count, upper_neg_count)
        upper_entropy = float(upper_pos_count + upper_neg_count) * upper_entropy / row_count
        lower_entropy = func_calc_entropy(lower_pos_count, lower_neg_count)
        lower_entropy = float(lower_pos_count + lower_neg_count) * lower_entropy / row_count
        return init_entropy - upper_entropy - lower_entropy

    def count_one(self, index, value):
        upper_pos_count, upper_neg_count = 0, 0
        lower_pos_count, lower_neg_count = 0, 0
        norm_data_t = self.norm_data.T[index]
        for i in range(len(norm_data_t)):
            tmp_value = norm_data_t[i]
            if tmp_value > value:
                if self.flag_data[i] > 0: upper_pos_count += 1
                else: upper_neg_count += 1
            else:
                if self.flag_data[i] > 0: lower_pos_count += 1
                else: lower_neg_count += 1
        return upper_pos_count, upper_neg_count, lower_pos_count, lower_neg_count

    def calc_error_rate(self, feature_num, split_num):
        predict, true, false = 0, 0, 0
        for i in range(len(self.norm_data)):
            predict = -1 if self.norm_data[i][feature_num] < split_num else 1
            if predict == self.flag_data[i]: true += 1
            else: false += 1
        return float(true) / (true + false)

    def print_tree(self):
        if self.depth > 0:
            print '+' * self.height, self.feature_num, self.split_num
            self.left_child.print_tree()
            print '+' * self.height, self.feature_num, self.split_num
            self.right_child.print_tree()
            print '+' * self.height, self.cur_info_gain
        else:
            print None

def func_count_one(flag_data):
    pos, neg = 0, 0
    for data in flag_data:
        if data == 1: pos += 1
        if data == -1: neg += 1
    return pos, neg

def func_calc_entropy(pos, neg):
    try:
        lhs_prob, rhs_prob = float(pos) / (pos + neg), float(neg) / (pos + neg)
        entropy = lhs_prob * np.log(lhs_prob) + rhs_prob * np.log(rhs_prob)
        return -entropy
    except ZeroDivisionError:
        return 0

## Main Part
def func_part_one():
    train_data, test_data = DataFormat('knn_train.csv'), DataFormat('knn_test.csv')
    train_norm_data, train_flag_data = train_data.norm_flag_data()
    test_norm_data, test_flag_data = test_data.norm_flag_data()
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

def func_part_two():
    train_data, test_data = DataFormat('knn_train.csv'), DataFormat('knn_test.csv')
    train_norm_data, train_flag_data = train_data.norm_flag_data()
    test_norm_data, test_flag_data = test_data.norm_flag_data()
    ''' Decision Stump '''
    # train_stump, test_stump = DecisionStump(train_norm_data, train_flag_data), DecisionStump(test_norm_data, test_flag_data)
    # train_info_gain, train_feature, train_split = train_stump.build_stump()
    # test_info_gain, test_feature, test_split = test_stump.build_stump()
    # # Training Data Outputs
    # print 'Training Information Gain Value:', train_info_gain
    # print 'Training Feature Value:', train_feature
    # # print 'Training Splitted Value:', train_split
    # print 'Training Error Rate:', train_stump.calc_error_rate(train_feature, train_split)
    # print
    # # Testing Data outputs
    # print 'Testing Information Gain Value:', test_info_gain
    # print 'Testing Feature Value:', test_feature
    # # print 'Testing Splitted Value:', test_split
    # print 'Testing Error Rate:', test_stump.calc_error_rate(test_feature, test_split)
    ''' Decision Tree '''
    train_tree = DecisionTree(train_norm_data, train_flag_data, 5, 0)
    train_tree.print_tree()

if __name__ == '__main__':
    # func_part_one()
    func_part_two()
