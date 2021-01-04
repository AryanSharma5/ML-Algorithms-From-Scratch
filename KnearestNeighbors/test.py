import time
import numpy as np
import argparse
from sklearn import datasets
from sklearn.model_selection import train_test_split

from knn import KNN
from knnUtil import accuracy

iris = datasets.load_iris()
data, target = iris.data, iris.target
xtrain, xtest, ytrain, ytest = train_test_split(data, target)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--k",
		type=int
	)
	args = parser.parse_args()

	start = time.time()

	model = KNN(k=args.k)
	model.fit(xtrain, ytrain)
	predictions = model.predict(xtest)
	print('#'*50)
	print(f'accuracy: {accuracy(predictions, ytest)}')
	end = time.time()
	print(f'time taken: {end - start}s')
	print('#'*50)