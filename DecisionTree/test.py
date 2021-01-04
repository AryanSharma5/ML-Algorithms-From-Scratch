import time
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from decisionTreeClassifier import DecisionTree

def accuracy(ytrue, ypred):
	return np.sum(ypred == ytrue) / len(ypred)

def makeDataset():
	breastCancer = datasets.load_breast_cancer()
	X, y = breastCancer.data, breastCancer.target
	return X, y

if __name__ == "__main__":
	X, y = makeDataset()
	xtrain, xtest, ytrain, ytest = train_test_split(X, y, random_state = 101)

	print('='*25 + 'DECISION TREE CLASSIFIER' + '='*25 + '\n\n')
	start = time.time()
	treeModel = DecisionTree()
	treeModel.fit(xtrain, ytrain)
	predictions = treeModel.predict(xtest)
	score = accuracy(ytest, predictions)
	end = time.time()
	print(f'accuracy score: {score}\n')
	print(f'time taken: {end - start}\n')