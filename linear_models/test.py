import time
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import argparse

from linear_models import (LinearRegression, LogisticRegression, RidgeRegression,
	LassoRegression, ElasticNet)

# from sklearn.linear_model import ElasticNet

def createDatasetLinearRegression():
	X, Y = datasets.make_regression(n_samples=100, n_features=2, noise=10,
	 random_state=101)
	return X, Y

def MSE(ytrue, ypred):
	return np.mean((ytrue - ypred)**2)

def accuracy(ytrue, ypred):
	return np.sum(ytrue == ypred)/len(ytrue)

if __name__ == "__main__":
	# parser = argparse.ArgumentParser()
	# parser.add_argument(
	# 	"--lr",
	# 	type=float
	# )
	# parser.add_argument(
	# 	"--iter",
	# 	type=int
	# )
	# args = parser.parse_args()

	# 
	# 
	# 
	# UN-COMMENT BELOW LINES FOR TESTING LINEAR REGRESSION.
	# 
	# 
	# 
	print('='*25 + 'LINEAR REGRESSION TESTING:' + '='*25 + '\n\n')
	X, Y = createDatasetLinearRegression()
	xtrain, xtest, ytrain, ytest = train_test_split(X, Y, random_state=101)

	start = time.time()

	model = LinearRegression(learning_rate=0.001, n_iter=10000)
	model.fit(xtrain, ytrain)
	predictions = model.predict(xtest)
	
	print('#'*50 + '\n')
	print(f'mean squared error: {MSE(ytest, predictions)}')
	
	end = time.time()
	
	print(f'time taken: {end - start}')
	print('#'*50 + '\n')
	#
	#
	#
	# 
	# UN-COMMENT BELOW LINES FOR TESTING RIDGE REGRESSION.
	# 
	#
	# 
	# print('='*25 + 'RIDGE REGRESSION TESTING:' + '='*25 + '\n\n')
	# X, Y = createDatasetLinearRegression()
	# xtrain, xtest, ytrain, ytest = train_test_split(X, Y, random_state=101)

	# start = time.time()

	# model = RidgeRegression(learning_rate=0.001, n_iter=10000)
	# model.fit(xtrain, ytrain)
	# predictions = model.predict(xtest)
	
	# print('#'*50 + '\n')
	# print(f'mean squared error: {MSE(ytest, predictions)}')
	
	# end = time.time()
	
	# print(f'time taken: {end - start}')
	# print('#'*50 + '\n')
    # 
	#
	#
	#
    # UN-COMMENT BELOW LINES FOR TESTING LASSO REGRESSION.
	# 
	#
	#
	# 
	# print('='*25 + 'LASSO REGRESSION TESTING:' + '='*25 + '\n\n')
	# X, Y = createDatasetLinearRegression()
	# xtrain, xtest, ytrain, ytest = train_test_split(X, Y, random_state=101)

	# start = time.time()

	# model = LassoRegression(learning_rate=0.001, n_iter=10000)
	# model.fit(xtrain, ytrain)
	# predictions = model.predict(xtest)
	
	# print('#'*50 + '\n')
	# print(f'mean squared error: {MSE(ytest, predictions)}')
	
	# end = time.time()
	
	# print(f'time taken: {end - start}')
	# print('#'*50 + '\n')
	# 
	# 
	# 
	# UN-COMMENT BELOW LINES FOR TESTING ELASTIC REGRESSION.
	# 
	# 
	# print('='*25 + 'ELASTIC-NET TESTING:' + '='*25 + '\n\n')
	# X, Y = createDatasetLinearRegression()
	# xtrain, xtest, ytrain, ytest = train_test_split(X, Y, random_state=101)

	# start = time.time()

	# model = ElasticNet(learning_rate = 0.01, n_iter = 100000,
	# 					 _lambda=0.01, _alpha = 0.5)
	# model.fit(xtrain, ytrain)
	# predictions = model.predict(xtest)
	
	# print('#'*50 + '\n')
	# print(f'mean squared error: {MSE(ytest, predictions)}')
	
	# end = time.time()
	
	# print(f'time taken: {end - start}')
	# print('#'*50 + '\n')
	# 
	# 
	# 
	# 
	# UN-COMMENT BELOW LINES FOR TESTING LOGISTIC REGRESSION.
	# 
	# 
	# 
	# print('='*25 + 'LOGISTIC REGRESSION TESTING:' + '='*25 + '\n\n')
	# breast_cancer = datasets.load_breast_cancer()
	# X, Y = breast_cancer.data, breast_cancer.target
	# xtrain, xtest, ytrain, ytest = train_test_split(X, Y, random_state = 101)

	# start = time.time()
	
	# model = LogisticRegression()
	# model.fit(xtrain, ytrain)
	# predictions = model.predict(xtest)
	
	# end = time.time()
	
	# print('#'*50 + '\n')
	# score = accuracy(ytest, predictions)
	# print(f'accuracy score: {score}')
	# print(f'time taken: {end - start}')
	# print('#'*50 + '\n')