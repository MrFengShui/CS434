import numpy as np
# Part I
fp = open('housing_train.txt','r')
X = np.loadtxt(fp)
# print X
Y = X[:,13]
# print Y, len(Y)
# Remove column 13 from X
temp = X
temp = np.delete(temp, np.s_[13::], 1)
print '+++', temp.shape
print '***', X

# array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print array
# array = np.delete(array, np.s_[2::], 1)
# print array