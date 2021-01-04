import time
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from SVM import SVM

def createDataset():
	X, y = datasets.make_classification(n_samples = 100, n_classes = 2, 
		n_features=15, random_state = 101)
	return X, y

def accuracy(ytrue, ypred):
	return np.sum(ytrue == ypred) / len(ytrue)

if __name__ == "__main__":
	X, y = createDataset()
	xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.15,
	 random_state = 101)

	print('='*25 + 'SVM' + '='*25 + '\n\n')
	start = time.time()
	model = SVM()
	model.fit(xtrain, ytrain)
	predictions = model.predict(xtest)
	print(f'accuracy: {accuracy(ytest, predictions)}')
	end = time.time()
	print(f'time taken: {end - start}\n\n')