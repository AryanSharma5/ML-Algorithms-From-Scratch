import numpy as np

class SVM:
	def __init__(self, learning_rate = 0.01, n_iter = 1000, _lambda = 0.01):
		self.learning_rate = learning_rate
		self.n_iter = n_iter
		self.weights = None
		self.bias = None
		self._lambda = _lambda

	def fit(self, X, y):
		n_samples, n_features = X.shape
		self.weights = np.random.rand(n_features)
		self.bias = np.random.rand()
		y_ = np.where(y<=0, -1, 1)
		for _ in range(self.n_iter):
			for idx, x in enumerate(X):
				# yhat calculation
				yhat = y_[idx]*(np.dot(x, self.weights) - self.bias)
				# gradients
				if yhat >= 1:
					dw = 2*self._lambda*(self.weights**2)
					db = 0
				else:
					dw = 2*self._lambda*(self.weights**2) - (1/n_samples)*np.dot(y_[idx], x)
					db = (1/n_samples)*y_[idx]
				# Update rule
				self.weights -= self.learning_rate*dw
				self.bias -= self.learning_rate*db

	def predict(self, X):
		return np.sign(np.dot(X, self.weights) - self.bias)