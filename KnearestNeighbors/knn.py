import numpy as np
from collections import Counter
from knnUtil import euclid

# K-nearest Neighbors Classifier

class KNN:
	def __init__(self, k=2):
		self.k = k
		
	def fit(self, X, Y):
		self.xtrain = X
		self.ytrain = Y

	def predict(self, X):
		predictedLabels = [self._predictUtil(x) for x in X]
		return predictedLabels

	def _predictUtil(self, x):
		# Euclid distance
		distances = [euclid(x, y) for y in self.xtrain]
		# k-nearest samples
		kIndices = np.argsort(distances)[:self.k]
		kLabels = [self.ytrain[i] for i in kIndices]
		# majority Voting
		majority = Counter(kLabels).most_common(1)
		return majority[0][0]