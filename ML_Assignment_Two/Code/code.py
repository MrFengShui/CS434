import time
import numpy as np
from numpy import *

def func_read_data():
    raw_train = genfromtxt('usps-4-9-train.csv', delimiter=',')
    raw_test = genfromtxt('usps-4-9-test.csv', delimiter=',')
    return raw_train, raw_test

def func_fmt_data(src_data):
    data_y = src_data[:, 256]
    data_y = data_y[np.newaxis].T

    data_x = np.delete(src_data, 256, axis = 1)
    dummy_x = np.ones(len(data_x))[np.newaxis].T
    data_x = np.concatenate((dummy_x, src_data), axis = 1)
    return data_x, data_y

def batch_learning(src_data_x, src_data_y, n, eta, epsilon = 0.001):
    size = len(src_data_x[0])
    weight, old_d = np.zeros(size), np.zeros(size)
    while True:
        new_d = np.zeros(size)
        for i in range(n):
            expo = np.dot(weight.T, src_data_x[i])
            y_hat_i = 1 / (1 + np.exp(-expo))
            error = src_data_y[i] - y_hat_i
            new_d = np.add(new_d, error * src_data_x[i])
        d_norm = np.linalg.norm(old_d - new_d)
        if d_norm < epsilon: break
        weight, old_d = np.add(weight, eta * new_d), new_d
    return weight

def func_calc_loss(x, y, weight):
    expo = np.dot(weight, x)
    print expo,
    sigmoid = 1 / (1 + np.exp(-expo))
    if y == 1:
        return -np.log(sigmoid)
    else:
        return -np.log(1 - sigmoid)

def func_log_reg(src_data_x, src_data_y, weight, lambda_value):
    fst = 0
    for i in range(len(src_data_x)):
        log_reg = func_calc_loss(src_data_x[i], src_data_y[i], weight.T)
        fst += log_reg

    snd = (lambda_value * (np.linalg.norm(weight, 2) ** 2)) / 2
    return fst + snd

if __name__ == '__main__':
    train_data, test_data = func_read_data()
    train_x, train_y = func_fmt_data(train_data)
    weight_vectors, weight_norms, learning_rates, lambdas = [], [], [
        # 0.00000001, 0.00000005,
        # 0.0000001, 0.0000005,
        # 0.000001, 0.000005,
        0.00001, 0.00005,
        0.0001, 0.0005,
        0.001, 0.005,
        0.01, 0.05,
        0.1, 0.5,
        1, 5
    ], [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

    for rate in learning_rates:
        tick = time.clock()
        batch = batch_learning(train_x, train_y, 1400, rate)
        weight_vectors.append(batch)
        norm = np.linalg.norm(batch)
        weight_norms.append(norm)
        print '%f <---> %f :: %.3f(s)' % (rate, norm, time.clock() - tick)

    for weight in weight_vectors:
        print '+++',
        for value in lambdas:
            log_reg = func_log_reg(train_x, train_y, weight, value)
            print log_reg,
        print
