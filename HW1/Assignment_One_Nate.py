
# coding: utf-8

# In[32]:

# Import third party libraries

# Numerical library
import numpy as np

# Used for matrix inversion
from numpy.linalg import inv

# Plotting library
import matplotlib.pyplot as plt

# Allows for printing inline for jupyter notebook
get_ipython().magic(u'matplotlib inline')


# In[33]:

# Load datasets and store in ndarray
training_data = open('housing_train.txt','r')
X_init_train = np.loadtxt(training_data)

testing_data = open('housing_test.txt', 'r')
X_init_test = np.loadtxt(testing_data)


# In[34]:

# Split off known target values
y_train = X_init_train[:,13]
y_test = X_init_test[:,13]

# Add dimension to y_train and transpose
y_train = y_train[np.newaxis].T
y_test = y_test[np.newaxis].T


# In[35]:

# Remove column 13 from X
X_train = np.delete(X_init_train, 13, axis=1)
X_test = np.delete(X_init_test, 13, axis=1)

# Function to create array of dummy ones and returned 
# columnar vector
def make_dummy_vector(target):
    temp = np.ones(len(target))
    return temp[np.newaxis].T

# Create dummy 1 values
dummy_train = make_dummy_vector(X_train)
dummy_test = make_dummy_vector(X_test)

# Add dummy data to feature matrices
X_train = np.concatenate((dummy_train, X_train), axis=1)
X_test = np.concatenate((dummy_test, X_test), axis=1)

## WE SHOULD TALK ABOUT THIS AS A GROUP
# Transpose X for further calculations
#X_train = X_train.T
#X_test = X_test.T

print X_train.shape


# In[36]:

## PART 2
# Compute optimal weight vector w -- (X^T * X)^-1 (X^T * Y)
def calc_w_vector(X, y):
    return np.dot(inv(np.dot(X.T,X)), np.dot(X.T,y))

def alt_calc(X,y):
    return np.dot(np.dot(inv(X), inv(X.T), np.dot(X.T,y)))
    
# Limit printout to 3 decimal places
np.set_printoptions(precision=3)

# Caculate w vectors
w_train = calc_w_vector(X_train,y_train)
w_test = calc_w_vector(X_test,y_test)

# Print both weight vectors to console
print 'w_train vector:'
print('\n'.join('{}: {}'.format(*k) for k in enumerate(w_train)))

print ' \r\nw_test vector:'
print('\n'.join('{}: {}'.format(*k) for k in enumerate(w_test)))


# In[37]:

## PART 3
# Functions
def calc_sse(X, y, w):
    return np.dot(np.subtract(y, np.dot(X, w)).T, np.subtract(y,np.dot(X, w)))

# Apply learned weight vectors
target_func_train = np.dot(X_train, w_train)
target_func_test = np.dot(X_test, w_test)

# Print error output, not sure about the 0 values

print 'Training Model: \r\nSSE: %.2f \r\n' % calc_sse(X_train, y_train, w_train)

print 'Testing Model: \r\nSSE: %.2f' % calc_sse(X_test, y_test, w_test)


# In[38]:

## PART 4
# Repeating part 2 and 3 without a dummy features of 1's in X

# Remove dummy column from both tables
X_train_no_dummy = X_train[:, (1,2,3,4,5,6,7,8,9,10,11,12,13)]
X_test_no_dummy = X_test[:, (1,2,3,4,5,6,7,8,9,10,11,12,13)]

# Caculate w vectors
w_train_no_dummy = calc_w_vector(X_train_no_dummy,y_train)
w_test_no_dummy = calc_w_vector(X_test_no_dummy,y_test)

# Print both weight vectors to console
print 'w_train_no_dummy vector:'
print('\n'.join('{}: {}'.format(*k) for k in enumerate(w_train_no_dummy)))

print ' \r\nw_test_no_dummy vector:'
print('\n'.join('{}: {}'.format(*k) for k in enumerate(w_test_no_dummy)))


# <h3>Thoughts about results</h3>
# The above results make it seems like our model will be centered around the orgin beacuse we did not calcuate a true b value in the w vector.

# In[39]:

## PART 4 cont.
# Apply learned weight vectors
target_func_train_no_dummy = np.dot(X_train_no_dummy, w_train_no_dummy)
target_func_test_no_dummy = np.dot(X_test_no_dummy, w_test_no_dummy)

# Print error output, not sure about the 0 values
print 'Training Model without Dummy: \r\nSSE: %.2f \r\n' % calc_sse(X_train_no_dummy, y_train, w_train_no_dummy)

print 'Testing Model without dummy: \r\nSSE: %.2f' % calc_sse(X_test_no_dummy, y_test, w_test_no_dummy)


# In[40]:

# Generate uniform additional uniformly distributed features
feature_one = np.random.uniform(0,10,433)
feature_two = np.random.uniform(0,100,433)
feature_three = np.random.uniform(0,200,433)
feature_four = np.random.uniform(0,400,433)
feature_five = np.random.uniform(0,600,433)
feature_six = np.random.uniform(0,800,433)
feature_seven = np.random.uniform(0,1000,433)
feature_eight = np.random.uniform(0,1200,433)
feature_nine = np.random.uniform(0,1400,433)
feature_ten = np.random.uniform(0,1600,433)


# In[41]:

# Set up cases for 2,4,6,8,10 additional uniformly distributed features
#two_feat = X_train[:, ()]


# In[62]:

# Part 6 cont
# Split off known target values
y_train = X_init_train[:,13]
y_test = X_init_test[:,13]

# Add dimension to y_train and transpose
y_train = y_train[np.newaxis].T
y_test = y_test[np.newaxis].T

# Remove column 13 from X
X_train = np.delete(X_init_train, 13, axis=1)
X_test = np.delete(X_init_test, 13, axis=1)

# Function to create array of dummy ones and returned 
# columnar vector
def make_dummy_vector(target):
    temp = np.ones(len(target))
    return temp[np.newaxis].T

# Create dummy 1 values
dummy_train = make_dummy_vector(X_train)
dummy_test = make_dummy_vector(X_test)

# Add dummy data to feature matrices
X_train = np.concatenate((dummy_train, X_train), axis=1)
X_test = np.concatenate((dummy_test, X_test), axis=1)

# Compute optimal weight vector w -- (X^T * X + lamda * I)^-1 (X^T * Y)
def calc_w_vector(X, y, lamda):
    I = None
    return np.dot(inv(np.dot(X.T,X + lamda * I)), np.dot(X.T,y))

def alt_calc(X,y):
    return np.dot(np.dot(inv(X), inv(X.T), np.dot(X.T,y)))
    
# Limit printout to 3 decimal places
np.set_printoptions(precision=3)

# Caculate w vectors
w_train = calc_w_vector(X_train,y_train)
w_test = calc_w_vector(X_test,y_test)

# Print both weight vectors to console
print 'w_train vector:'
print('\n'.join('{}: {}'.format(*k) for k in enumerate(w_train)))

print ' \r\nw_test vector:'
print('\n'.join('{}: {}'.format(*k) for k in enumerate(w_test)))


# In[43]:

## Extra stuff below ##


# In[44]:

print X_train.shape
print w_train.shape


# In[45]:

print X_train.T


# In[46]:

# Don't show scientific notation
np.set_printoptions(suppress=True)

print "Printing X_train:"
print X_train


# In[47]:

print "Printing y_train:"
print y_train


# In[48]:

# Plot feature 1: Crime rate by town
plt.scatter(X_train[:, 0],y_train)


# In[49]:

# Plot feature 2: Residential land zoned for lots over 25,0000 sq. ft
plt.scatter(X_train[:, 1],y_train)


# In[50]:

# Multiplotting feature 1 & 2
plt.scatter(X_train[:, 0],y_train)
plt.scatter(X_train[:, 1],y_train)


# In[51]:

# Plot feature 3: Proportion of non-retail business acres per town
plt.scatter(X_train[:, 2],y_train)


# In[52]:

# Plot feature 4: Charles River dummy variable (= 1 if tract bounds river, 0 otherwise)
plt.scatter(X_train[:, 3],y_train)


# In[53]:

# Plot feature 5: Nitric oxides concentration (parts per 10 million)
plt.scatter(X_train[:, 4],y_train)


# In[54]:

# Plot feature 6: Average number fo rooms per dwelling
plt.scatter(X_train[:, 5],y_train)


# In[55]:

# Plot feature 7: Porportion of owner-occupied units built prior to 1940
plt.scatter(X_train[:, 6],y_train)


# In[56]:

# Plot feature 8: Weighted distances to five Boston employment centers
plt.scatter(X_train[:, 7],y_train)


# In[57]:

# Plot feature 9: Index of accessability to radial highways
plt.scatter(X_train[:, 8],y_train)


# In[58]:

# Plot feature 10: Full-value property-tax rate per $10,000
plt.scatter(X_train[:, 9],y_train)


# In[59]:

# Plot feature 11: Pupil-teacher ratio by town
plt.scatter(X_train[:, 10],y_train)


# In[60]:

# Plot feature 12: 1000(Bk - 0.63)^2 where Bk is the population fo blacks by town
plt.scatter(X_train[:, 11],y_train)


# In[61]:

# Plot feature 13: % lower status of the population
plt.scatter(X_train[:, 12],y_train)

