###################################################################
#	CS 434: Linear Regression Model Experimentation
#   		Nathaniel Whitlock, Songjian Luan, and Raja Petroff
###################################################################

# Import third party libraries

# Numerical library
import numpy as np

# Used for matrix inversion
from numpy.linalg import inv

# Plotting library -- Commented out due to no display on flip server
#import matplotlib.pyplot as plt
#from matplotlib import gridspec

# Used for sorting dictionary by key
import collections

# For printing neatly
np.set_printoptions(precision=3)

## Functions
# Create array of dummy ones and returned columnar vector
def make_dummy_vector(target):
    temp = np.ones(len(target))
    return temp[np.newaxis].T
	
# Compute optimal weight vector w = (X^T * X)^-1 (X^T * Y)
def calc_w_vector(X, y):
    return np.dot(inv(np.dot(X.T,X)), np.dot(X.T,y))
	
# Compute SSE with matrix formula
def calc_sse(X, y, w):
    return np.dot(np.subtract(y, np.dot(X, w)).T, np.subtract(y,np.dot(X, w)))

# Generate uniformly random distribution in range
def generate_uniform_feat(low,high,length):
    return np.random.uniform(low,high,length)[np.newaxis].T
	
# Compute optimal weight vector w -- (X^T * X + lamda * I)^-1 (X^T * Y)
def calc_w_vector_identity(X, y, lamda):
    I = np.identity(len(np.dot(X.T,X)))
    return np.dot(inv(np.dot(X.T,X) + lamda * I), np.dot(X.T,y))
	
# Compute SSE with regularization term
def calc_sse_reg(X, y, w, lamda):
    return np.add(np.dot(np.subtract(y, np.dot(X, w)).T, np.subtract(y,np.dot(X, w))),np.dot(lamda,np.linalg.norm(w, ord=2) ** 2))
	
# Print formatting
def sep_line():
	return '**********************************************************'


## PART 1
# Load datasets and store in ndarray
training_data = open('housing_train.txt','r')
X_train = np.loadtxt(training_data)

testing_data = open('housing_test.txt', 'r')
X_test = np.loadtxt(testing_data)

# Split off known target values
y_train = X_train[:,13]
y_test = X_test[:,13]

# Add dimension to y_train and transpose
y_train = y_train[np.newaxis].T
y_test = y_test[np.newaxis].T

# Remove column 13 from X
X_train = np.delete(X_train, 13, axis=1)
X_test = np.delete(X_test, 13, axis=1)

# Create dummy 1 values
dummy_train = make_dummy_vector(X_train)
dummy_test = make_dummy_vector(X_test)

# Add dummy data to feature matrices
X_train = np.concatenate((dummy_train, X_train), axis=1)
X_test = np.concatenate((dummy_test, X_test), axis=1)

## PART 2
# Caculate w vectors
w_train = calc_w_vector(X_train,y_train)
w_test = calc_w_vector(X_test,y_test)

# Print both weight vectors to console
print(sep_line())
print 'PART 2\r\n'
print 'w_train vector:'
print('\n'.join('{}: {}'.format(*k) for k in enumerate(w_train)))

print ' \r\nw_test vector:'
print('\n'.join('{}: {}'.format(*k) for k in enumerate(w_test)))
print(sep_line())

## PART 3
# Apply learned weight vectors
target_func_train = np.dot(X_train, w_train)
target_func_test = np.dot(X_test, w_test)

# Print SSE values
print 'PART 3\r\n'
print 'Training Model: \r\nSSE: %.2f \r\n' % calc_sse(X_train, y_train, w_train)
print 'Testing Model: \r\nSSE: %.2f' % calc_sse(X_test, y_test, w_test)
print(sep_line())

## PART 4
# Remove dummy column from both tables
X_train_no_dummy = X_train[:, (1,2,3,4,5,6,7,8,9,10,11,12,13)]
X_test_no_dummy = X_test[:, (1,2,3,4,5,6,7,8,9,10,11,12,13)]

# Caculate w vectors
w_train_no_dummy = calc_w_vector(X_train_no_dummy,y_train)
w_test_no_dummy = calc_w_vector(X_test_no_dummy,y_test)

# Print both weight vectors to console
print 'PART 4\r\n'
print 'w_train_no_dummy vector:'
print('\n'.join('{}: {}'.format(*k) for k in enumerate(w_train_no_dummy)))

print ' \r\nw_test_no_dummy vector:'
print('\n'.join('{}: {}'.format(*k) for k in enumerate(w_test_no_dummy)))

# Apply learned weight vectors
target_func_train_no_dummy = np.dot(X_train_no_dummy, w_train_no_dummy)
target_func_test_no_dummy = np.dot(X_test_no_dummy, w_test_no_dummy)

# Print SSE values
print '\r\nTraining Model without Dummy: \r\nSSE: %.2f \r\n' % calc_sse(X_train_no_dummy, y_train, w_train_no_dummy)
print 'Testing Model without dummy: \r\nSSE: %.2f \r\n' % calc_sse(X_test_no_dummy, y_test, w_test_no_dummy)
print(sep_line())

# Set up values and experiment names
a_vals = [10,100,200,400,600,800,1000,1200,1400,1600]
experiments = np.arange(10)
inst = ["train", "test"]

# Dictionary to hold new features
new_features = {}

# Loop through and create feature for train and test
ctr = 0
for exp in experiments:
    for item in inst:
        if item == 'train':
            new_features["f_{0}_{1}".format(item,exp)] = generate_uniform_feat(0, a_vals[ctr], 433)
        else:
            new_features["f_{0}_{1}".format(item,exp)] = generate_uniform_feat(0, a_vals[ctr], 74)
    ctr += 1

## PART 5
# Set up cases for 2,4,6,8,10 additional uniformly distributed features
new_datasets = [2,4,6,8,10]

# New test matricies
two_feat_train = np.concatenate((X_train,new_features['f_train_0'],new_features['f_train_1']), axis=1)
two_feat_test = np.concatenate((X_test,new_features['f_test_0'],new_features['f_test_1']), axis=1)

four_feat_train = np.concatenate((two_feat_train,new_features['f_train_2'],new_features['f_train_3']), axis=1)
four_feat_test = np.concatenate((two_feat_test, new_features['f_test_2'],new_features['f_test_3']), axis=1)

six_feat_train = np.concatenate((four_feat_train, new_features['f_train_4'],new_features['f_train_5']), axis=1)
six_feat_test = np.concatenate((four_feat_test, new_features['f_test_4'],new_features['f_test_5']), axis=1)

eight_feat_train = np.concatenate((six_feat_train, new_features['f_train_6'],new_features['f_train_7']), axis=1)
eight_feat_test = np.concatenate((six_feat_test, new_features['f_test_6'],new_features['f_test_7']), axis=1)

ten_feat_train = np.concatenate((eight_feat_train,new_features['f_train_8'],new_features['f_train_9']), axis=1)
ten_feat_test = np.concatenate((eight_feat_test,new_features['f_test_8'],new_features['f_test_9']), axis=1)

# Create weight vector for each test case
w_two_train = calc_w_vector(two_feat_train, y_train)
w_two_test = calc_w_vector(two_feat_test, y_test)

w_four_train = calc_w_vector(four_feat_train, y_train)
w_four_test = calc_w_vector(four_feat_test, y_test)

w_six_train = calc_w_vector(six_feat_train, y_train)
w_six_test = calc_w_vector(six_feat_test, y_test)

w_eight_train = calc_w_vector(eight_feat_train, y_train)
w_eight_test = calc_w_vector(eight_feat_test, y_test)

w_ten_train = calc_w_vector(ten_feat_train, y_train)
w_ten_test = calc_w_vector(ten_feat_test, y_test)

# Store SSE scores in lists
sse_train = []
sse_test = []

sse_train.append(calc_sse(two_feat_train, y_train,w_two_train))
sse_test.append(calc_sse(two_feat_test, y_test,w_two_test))

sse_train.append(calc_sse(four_feat_train, y_train,w_four_train))
sse_test.append(calc_sse(four_feat_test, y_test,w_four_test))

sse_train.append(calc_sse(six_feat_train, y_train,w_six_train))
sse_test.append(calc_sse(six_feat_test, y_test,w_six_test))

sse_train.append(calc_sse(eight_feat_train, y_train,w_eight_train))
sse_test.append(calc_sse(eight_feat_test, y_test,w_eight_test))

sse_train.append(calc_sse(ten_feat_train, y_train,w_ten_train))
sse_test.append(calc_sse(ten_feat_test, y_test,w_ten_test))

## Plot the values gathered above
#fig = plt.figure(figsize=(9,6))
#plt.scatter(new_datasets,sse_train)
#plt.plot(np.resize(new_datasets,(len(new_datasets),1)),np.resize(sse_train,(len(sse_train),1)))
#fig.suptitle('X_train', fontsize=16)
#plt.xlabel('Number of Added Features', fontsize=16)
#plt.ylabel('Sum Squared \nError (SSE)', fontsize=16)

### Plot of effect of alterficial features on X_test
#fig = plt.figure(figsize=(9,6))
#plt.scatter(new_datasets,sse_test)
#plt.plot(np.resize(new_datasets,(len(new_datasets),1)),np.resize(sse_test,(len(sse_test),1)))
#fig.suptitle('X_test', fontsize=16)
#plt.xlabel('Number of Added Features', fontsize=16)
#plt.ylabel('Sum Squared \nError (SSE)', fontsize=16)

# Calculate SSE for each feature experiment
print 'PART 5\r\n'
print 'Two_train features added: \r\nSSE: %.2f \r\n' % calc_sse(two_feat_train, y_train,w_two_train)
print 'Four_train features added: \r\nSSE: %.2f \r\n' % calc_sse(four_feat_train, y_train,w_four_train)
print 'Six_train features added: \r\nSSE: %.2f \r\n' % calc_sse(six_feat_train, y_train,w_six_train)
print 'Eight_train features added: \r\nSSE: %.2f \r\n' % calc_sse(eight_feat_train, y_train,w_eight_train)
print 'Ten_train features added: \r\nSSE: %.2f \r\n' % calc_sse(ten_feat_train, y_train,w_ten_train)

print 'Two_test features added: \r\nSSE: %.2f \r\n' % calc_sse(two_feat_test, y_test,w_two_test)
print 'Four_test features added: \r\nSSE: %.2f \r\n' % calc_sse(four_feat_test, y_test,w_four_test)
print 'Six_test features added: \r\nSSE: %.2f \r\n' % calc_sse(six_feat_test, y_test,w_six_test)
print 'Eight_test features added: \r\nSSE: %.2f \r\n' % calc_sse(eight_feat_test, y_test,w_eight_test)
print 'Ten_test features added: \r\nSSE: %.2f \r\n' % calc_sse(ten_feat_test, y_test,w_ten_test)
print(sep_line())

## PART 6
# Dictionaries to store results
w_train_results, w_test_results = {}, {}

# Set of lambda values
lamdas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 30, 40, 50, 100, 200, 300]

# Caculate w vectors
for lamda in lamdas:
    w_train_results.setdefault(lamda, [])
    w_train = calc_w_vector_identity(X_train,y_train, lamda)
    w_train_results[lamda] = np.asarray(w_train)
    
    w_test_results.setdefault(lamda, [])
    w_test = calc_w_vector_identity(X_test,y_test, lamda)
    w_test_results[lamda] = w_test

sse_train, sse_test = {}, {}

# Print SSSE values
print 'PART 6\r\n'
print 'Training Model:'
for key in w_train_results:
    w_train = w_train_results[key]
    target_func_train = np.dot(X_train, w_train)
    sse = calc_sse(X_train, y_train, w_train)
    sse_train.setdefault(key, sse[0][0])
    print '[%0.2f]-SSE: %.2f \r\n' % (key, sse)

print 'Testing Model:'
for key in w_test_results:
    w_test = w_test_results[key]
    target_func_test = np.dot(X_test, w_test)
    sse = calc_sse(X_test, y_test, w_test)
    sse_test.setdefault(key, sse[0][0])
    print '[%0.2f]-SSE: %.2f \r\n' % (key, sse)
print(sep_line())

# Order dictionary
ordered_sse_train = collections.OrderedDict(sorted(sse_train.items()))

# Get values for plotting
lambda_train = ordered_sse_train.keys()
sse_values_train = ordered_sse_train.values()
w_train_final = w_train_results.values()

## Format and plot figure
#fig = plt.figure(figsize=(9,6))
#fig.suptitle('X_train', fontsize=16)
#plt.xlabel('Lambda values', fontsize=16)
#plt.ylabel('Sum Squared \nError (SSE)', fontsize=16)
#plt.scatter(lambda_train, sse_values_train)
#plt.plot(lambda_train,sse_values_train)

# Order dictionary
ordered_sse_test = collections.OrderedDict(sorted(sse_test.items()))

# Get values for plotting
lambda_test = ordered_sse_test.keys()
sse_values_test = ordered_sse_test.values()
w_test_final = w_test_results.values()

## Format and plot figure
#fig = plt.figure(figsize=(9,6))
#fig.suptitle('X_test', fontsize=16)
#plt.scatter(lambda_test, sse_values_test)
#plt.plot(lambda_test,sse_values_test)
#plt.xlabel('Lambda values', fontsize=16)
#plt.ylabel('Sum Squared \nError (SSE)', fontsize=16)

## PART 8
# Prepare y values
reg_train_sse, reg_test_sse, ctr = [], [], 0

# Loop through and calculated SSE values
for i in range(len(w_train_final)):
    tmp_train_sse, tmp_test_sse = [], []
    for item in lamdas:
        sse = calc_sse_reg(X_train, y_train, w_train_final[i], item)
        tmp_train_sse.append(sse)
        sse = calc_sse_reg(X_test, y_test, w_test_final[i], item)
        tmp_test_sse.append(sse)
    reg_train_sse.append(tmp_train_sse)
    reg_test_sse.append(tmp_test_sse)
	
## Plot the values gathered above
#fig = plt.figure(figsize=(9,6))
#fig.suptitle('Effect of Noramilzation Term of Train', fontsize=16)
#for i in range(14):
#    plt.scatter(lamdas, reg_train_sse[i])
#    plt.plot(lamdas, reg_train_sse[i])    
#    plt.xlabel('Lambda Values', fontsize=16)
#    plt.ylabel('Sum Squared \nError (SSE)', fontsize=16)

## Plot the values gathered above
#fig = plt.figure(figsize=(9,6))
#fig.suptitle('Effect of Noramilzation Term Test', fontsize=16)
#for i in range(14):
#    plt.scatter(lamdas, reg_test_sse[i])
#    plt.plot(lamdas, reg_test_sse[i])    
#    plt.xlabel('Lambda Values', fontsize=16)
#    plt.ylabel('Sum Squared \nError (SSE)', fontsize=16)
    

