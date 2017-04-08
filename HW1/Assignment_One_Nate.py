# Import third party libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Allows for printing inline for jupyter notebook
get_ipython().magic(u'matplotlib inline')

# Load dataset and store in ndarray
fp = open('C:\Users\whitlock\Downloads\housing_train.txt','r')
X = np.loadtxt(fp)

# Split off known target values
Y = X[:,13]

# Transpose row vector to columnar
Y = Y[np.newaxis].T

# Remove column 13 from X
X = np.delete(X, 13, axis=1)

# Don't show scientific notation
np.set_printoptions(suppress=True)

print "Printing X:"
print X

print "Printing Y:"
print Y

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X,Y)

# Coefficients
print('Coefficients: \n' , regr.coef_)

# Mean squared error
print("Mean squared error: %.2f" 
      % np.mean((regr.predict(X) - Y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X, Y))


## DISTRIBUTION PLOTS OF EACH FEATURE

# Plot feature 1: Crime rate by town
plt.scatter(X[:, 0],Y)

# Plot feature 2: Residential land zoned for lots over 25,0000 sq. ft
plt.scatter(X[:, 1],Y)

# Plot feature 3: Proportion of non-retail business acres per town
plt.scatter(X[:, 2],Y)

# Plot feature 4: Charles River dummy variable (= 1 if tract bounds river, 0 otherwise)
plt.scatter(X[:, 3],Y)

# Plot feature 5: Nitric oxides concentration (parts per 10 million)
plt.scatter(X[:, 4],Y)

# Plot feature 6: Average number fo rooms per dwelling
plt.scatter(X[:, 5],Y)

# Plot feature 7: Porportion of owner-occupied units built prior to 1940
plt.scatter(X[:, 6],Y)

# Plot feature 8: Weighted distances to five Boston employment centers
plt.scatter(X[:, 7],Y)

# Plot feature 9: Index of accessability to radial highways
plt.scatter(X[:, 8],Y)

# Plot feature 10: Full-value property-tax rate per $10,000
plt.scatter(X[:, 9],Y)

# Plot feature 11: Pupil-teacher ratio by town
plt.scatter(X[:, 10],Y)

# Plot feature 12: 1000(Bk - 0.63)^2 where Bk is the population fo blacks by town
plt.scatter(X[:, 11],Y)

# Plot feature 13: % lower status of the population
plt.scatter(X[:, 12],Y)

