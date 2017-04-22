###################################################################
#	CS 434: Logistic Regression Model Experimentation
#   		Nathaniel Whitlock, Songjian Luan, and Raja Petroff
###################################################################
# Plotting library
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#from matplotlib import gridspec

# Numerical Library
import numpy as np
from numpy import genfromtxt

# Image manipulation library
#from PIL import Image

# Limit printout to 3 decimal places
np.set_printoptions(precision=3,suppress=True)

np.seterr(all='ignore')

# Allows for printing inline for jupyter notebook
#get_ipython().magic(u'matplotlib inline')

## FUNCTIONS ##
# Create array of dummy ones and returned columnar vector
def make_dummy_vector(target):
    temp = np.ones(len(target))
    return temp[np.newaxis].T
	
# Displays a single sample for context
def visualize_sample(data,target):
    temp = np.reshape(data[target],(16,16), order='F')
    img = Image.fromarray(temp)
    img.show()

# Calculate sigmoid function
def calc_sigmoid(target_function):
    return (1 / (1 + np.exp(-target_function)))

# Try to predict
def prediction(X,w):
    values = []
    for i in range(len(X)):      
        values.append(calc_sigmoid(np.dot(w.T,X[i])))
        
    fours = values[0:(len(X)/2)]
    nines = values[(len(X)/2):len(X)]
    ctr_four, ctr_nine = 0.0, 0.0
    
    for i in range(len(X)/2):
        if fours[i] == 0.0:
            ctr_four += 1
        if nines[i] == 1.0:
            ctr_nine += 1  
    return  ((ctr_four / (len(X)/2) * 100), (ctr_nine / (len(X)/2) * 100)) 

# Inital batch learnging algorithm with gradient descent -- gathers prediction values    
def batch_learning(X, y, n, eta, epsilon = 0.001):
    accuracy = []
    length = len(X[0])
    w, old_d = np.zeros(length), np.zeros(length)
    ctr = 0
    while True:
        new_d = np.zeros(length)
        for i in range(n):
            y_hat_i = 1 / (1 + np.exp(-np.dot(w.T, X[i])))
            error = y[i] - y_hat_i
            new_d = np.add(new_d, error * X[i])
        d_norm = np.linalg.norm(old_d - new_d, 2)
        #print d_norm
        ctr = ctr + 1
        if d_norm < epsilon:
            break
        else:
            w, old_d = np.add(w, eta * new_d), new_d
            accuracy.append(prediction(X,w))
    return w, ctr, accuracy  
	
# Modified batch learning algorithm with regularization term
def batch_learning_reg(X, y, n, eta, lam,epsilon = 0.001):
    length = len(X[0])
    w, old_d = np.zeros(length), np.zeros(length)
    while True:
        new_d = np.zeros(length)
        for i in range(n):
            y_hat_i = 1 / (1 + np.exp(-np.dot(w.T, X[i])))
            error = y[i] - y_hat_i
            new_d = np.add(new_d, error * X[i])
        d_norm = np.linalg.norm(old_d - new_d, 2)
        if d_norm < epsilon:
            break
        else:
            w, old_d = np.add(np.add(w, eta * new_d), (lam * np.linalg.norm(w,2))) , new_d
    return w


## DATA PREPERATION ##
print 'Preparing data for anaylsis\r\n'

# Load datasets and store in ndarray
raw_train = genfromtxt('usps-4-9-train.csv', delimiter=',')
raw_test = genfromtxt('usps-4-9-test.csv', delimiter=',')

# Split off known target values
y_train = raw_train[:,256]
y_test = raw_test[:,256]

# Add dimension to y_train and transpose
y_train = y_train[np.newaxis].T
y_test = y_test[np.newaxis].T

# Remove column 256 from X
raw_train = np.delete(raw_train, 256, axis=1)
raw_test = np.delete(raw_test, 256, axis=1)

# Create dummy 1 values
dummy_train = make_dummy_vector(raw_train)
dummy_test = make_dummy_vector(raw_test)

# Add dummy data to feature matrices
X_train = np.concatenate((dummy_train, raw_train), axis=1)
X_test = np.concatenate((dummy_test, raw_test), axis=1)


# Example call to function
#visualize_sample(raw_train, 1200)

## PART 1
print 'Beginning part one...\r\n'

# Experimenting with different learning rates
learning_rates = np.linspace(0.0,0.001,100)
ctr_tally = []
acc_tally = []

for i in learning_rates:
    w, ctr, accuracy = batch_learning(X_train,y_train,1400,i)
    ctr_tally.append(ctr)
    acc_tally.append(accuracy)
	

print 'Epoch count for each learning rate:\r\n %s' % ctr_tally

## PART 2
print '\r\nBeginning part two...\r\n'

# Initialize train learning exercise with chosen rate
w, ctr, accuracy = batch_learning(X_train,y_train,1400,0.001)

# Unpack data
four_acc_per_epoch_train = []
nine_acc_per_epoch_train = []
for i,(a, b) in enumerate(accuracy):
    four_acc_per_epoch_train.append(a)
    nine_acc_per_epoch_train.append(b)

# Plot the values gathered above
#fig = plt.figure(figsize=(9,6))
#plt.plot(range(33),four_acc_per_epoch_train,label="Four")
#plt.plot(range(33),nine_acc_per_epoch_train,label="Nine")
#plt.legend()
#fig.suptitle('Training Prediction Precentage', fontsize=16)
#plt.xlabel('Iterations', fontsize=16)
#plt.ylabel('Prediction Percent', fontsize=16)

# Initialize test learning exercise with chosen rate
w, ctr, accuracy = batch_learning(X_test,y_test,800,0.001)

# Unpack data
four_acc_per_epoch_test = []
nine_acc_per_epoch_test = []
for i,(a, b) in enumerate(accuracy):
    four_acc_per_epoch_test.append(a)
    nine_acc_per_epoch_test.append(b)
    
# Plot the values gathered above
#fig = plt.figure(figsize=(9,6))
#plt.plot(range(35),four_acc_per_epoch_test,label="Four")
#plt.plot(range(35),nine_acc_per_epoch_test,label="Nine")
#plt.legend()
#fig.suptitle('Testing Prediction Precentage', fontsize=16)
#plt.xlabel('Iterations', fontsize=16)
#plt.ylabel('Prediction Percent', fontsize=16)

print 'Training accuracy for 4s per epoch:\r\n %s' % four_acc_per_epoch_train
print '\r\nTraining accuracy for 9s per epoch:\r\n %s' % nine_acc_per_epoch_train
## PART 4
print '\r\nBeginning part four...\r\n'

# Experiment with the effect of regularization term
lambdas = [0.00000001,0.0000001,0.000001,0.000001,0.0001,0.001,0.01,0.1,1]
w_temp = []

# Loop through and run batch learning with lambdas
for i in lambdas:
    w  = batch_learning_reg(X_test,y_test,800,0.001,i)
    w_temp.append(w)
    
# Calculate prediction percentage
predictions_train = []
predictions_test = []

for i in range(len(w_temp)):
    predictions_train.append(prediction(X_train,w_temp[i]))
    predictions_test.append(prediction(X_test,w_temp[i]))
	
# Grab prediction values for 4's and 9's
four_acc_per_epoch_train = []
nine_acc_per_epoch_train = []

for i,(a, b) in enumerate(predictions_train):
    four_acc_per_epoch_train.append(a)
    nine_acc_per_epoch_train.append(b)

four_acc_per_epoch_test = []
nine_acc_per_epoch_test = []
for i,(a, b) in enumerate(predictions_test):
    four_acc_per_epoch_test.append(a)
    nine_acc_per_epoch_test.append(b)

# Plot results
#fig = plt.figure(figsize=(9,6))
#plt.plot(lambdas[0:6],four_acc_per_epoch_train[0:6],label="Four")
#plt.plot(lambdas[0:6],nine_acc_per_epoch_train[0:6],label="Nine")
#plt.legend()
#fig.suptitle('Training Prediction Precentage', fontsize=16)
#plt.xlabel('Lambda Values', fontsize=16)
#plt.ylabel('Prediction Percent', fontsize=16)

#fig = plt.figure(figsize=(9,6))
#plt.plot(lambdas[0:6],four_acc_per_epoch_test[0:6],label="Four")
#plt.plot(lambdas[0:6],nine_acc_per_epoch_test[0:6],label="Nine")
#plt.legend()
#fig.suptitle('Testing Prediction Precentage', fontsize=16)
#plt.xlabel('Lambda Values', fontsize=16)
#plt.ylabel('Prediction Percent', fontsize=16)

print '\r\nFinished logistic regression exercise\r\n'
