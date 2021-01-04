import numpy as np
import time
from sklearn import datasets
from sklearn.model_selection import train_test_split

from naive import GaussianNaiveClassifier

def createDataset():
	X, y = datasets.make_classification(n_samples=1000, n_features=10,
		n_classes=2, random_state=101)
	return X, y

def accuracy(ytrue, ypred):
	return np.sum(ypred == ytrue) / len(ytrue)

if __name__ == "__main__":
	X, y = createDataset()
	xtrain, xtest, ytrain, ytest = train_test_split(X, y)

	start = time.time()
	print('='*25 + "GAUSSIAN NAIVE BAYES CLASSIFIER" + "="*25 + "\n\n")
	model = GaussianNaiveClassifier()
	model.fit(xtrain, ytrain)
	predictions = model.predict(xtest)
	print(predictions)
	end = time.time()
	print(f"accuracy: {accuracy(ytest, predictions)}")
	print(f"time taken: {end - start}\n\n")
