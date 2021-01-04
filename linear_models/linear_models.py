import numpy as np

class BaseRegression:
	"""
	docstring for BaseRegression
	Base regression class
	"""
	def __init__(self, learning_rate = 0.01, n_iter = 1000):
		self.learning_rate = learning_rate
		self.n_iter = n_iter
		self.weights = None
		self.bias = None

	def fit(self, X, y):
		n_samples, n_features = X.shape
		self.weights = np.random.rand(n_features)
		self.bias = np.random.rand()

		# Gradient Descent
		for _ in range(self.n_iter):
			yhat = self._modelFunction(X, self.weights, self.bias)
			# weight gradient and bias gradient
			dw, db = self._updateRule(n_samples, X, yhat, y, self.bias, self.weights)
			# update rule
			self.weights -= self.learning_rate*dw
			self.bias -= self.learning_rate*db

	def predict(self, X):
		return self._predict(X)

	def predict_proba(self, X):
		return self._predict_proba(X)

	def _predict_proba(self, X):
		raise NotImplementedError()

	def _predict(self, X):
		raise NotImplementedError()

	def _modelFunction(self, X, w, b):
		raise NotImplementedError()

	def _updateRule(self, n_samples, X, yhat, y, bias):
		raise NotImplementedError()

# Linear Regression algorithm implementaion.
class LinearRegression(BaseRegression):
	"""
	docstring for LinearRegression
	linear regression class
	"""
	def __init__(self, learning_rate = None, n_iter = None):
		super(LinearRegression, self).__init__(learning_rate, n_iter)

	def _predict(self, X):
		return np.dot(X, self.weights) + self.bias

	def _updateRule(self, n_samples, X, yhat, y, bias, weights):		
		dw = (1/n_samples)*np.dot(X.T, (yhat - y))
		db = (1/n_samples)*np.sum(yhat - y)
		return dw, db

	def _modelFunction(self, X, w, b):
		yhat = np.dot(X, w) + b
		return yhat

class LassoRegression(BaseRegression):
	"""
	docstring for LassoRegression
	lasso regression class
	"""
	def __init__(self, learning_rate = None, n_iter = None, _lambda=0.01):
		super(LassoRegression, self).__init__(learning_rate, n_iter)
		self._lambda = _lambda

	def _predict(self, X):
		return np.dot(X, self.weights) + self.bias + self._lambda*np.sum(self.weights)

	def _updateRule(self, n_samples, X, yhat, y, bias, weights):		
		dw = (1/n_samples)*np.dot(X.T, (yhat - y)) + self._lambda
		db = (1/n_samples)*np.sum(yhat - y)
		return dw, db

	def _modelFunction(self, X, w, b):
		yhat = np.dot(X, w) + b + self._lambda*np.sum(w)
		return yhat
		
class RidgeRegression(BaseRegression):
	"""
	docstring for RidgeRegression
	ridge regression class
	"""
	def __init__(self, learning_rate = None, n_iter = None, _lambda = 0.01):
		super(RidgeRegression, self).__init__(learning_rate, n_iter)
		self._lambda = _lambda
	
	def _modelFunction(self, X, w, b):
		yhat = np.dot(X, w) + b + self._lambda*np.sum(w**2)
		return yhat

	def _updateRule(self, n_samples, X, yhat, y, bias, weights):		
		dw = (1/n_samples)*np.dot(X.T, (yhat - y)) + 2*self._lambda*np.sum(weights)
		db = (1/n_samples)*np.sum(yhat - y)
		return dw, db

	def _predict(self, X):
		return np.dot(X, self.weights) + self.bias + self._lambda*np.sum(self.weights**2)
		
class ElasticNet(BaseRegression):
	"""docstring for ElasticNet"""
	def __init__(self, learning_rate = None, n_iter = None, _lambda = 0.01, _alpha = 0.5):
		super(ElasticNet, self).__init__(learning_rate, n_iter)
		self._lambda = _lambda
		self._alpha = _alpha
	
	def _modelFunction(self, X, w, b):
		yhat = np.dot(X, w) + b + self._lambda*(((1-self._alpha)/2)*np.sum(w**2) + self._alpha*np.sum(w))
		return yhat

	def _updateRule(self, n_samples, X, yhat, y, bias, weights):
		dw = (1/n_samples)*np.dot(X.T, (yhat - y)) + self._lambda*((1-self._alpha)*np.sum(self.weights) + self._alpha)
		db = (1/n_samples)*np.sum(yhat - y)
		return dw, db

	def _predict(self, X):
		return np.dot(X, self.weights) + self.bias + self._lambda*(((1-self._alpha)/2)*np.sum(self.weights**2) + self._alpha*np.sum(self.weights))


# logistic regression algorithm implementation.
class LogisticRegression(BaseRegression):
	"""
	docstring for LogisticRegression
	logistic regression class
	"""
	def __init__(self, learning_rate = None, n_iter = None):
		super(LogisticRegression, self).__init__(learning_rate, n_iter)
		
	def _predict(self, X):
		yhat = np.dot(X, self.weights) + self.bias
		sigmoid = self._sigmoid(yhat)
		prediction_labels = [1 if x >= 0.5 else 0 for x in sigmoid]
		return prediction_labels

	def _predict_proba(self, X):
		yhat = np.dot(X, self.weights) + self.bias
		sigmoid = self._sigmoid(yhat)
		return sigmoid

	def _updateRule(self, n_samples, X, yhat, y, bias, weights):		
		dw = (1/n_samples)*np.dot(X.T, (yhat - y))
		db = (1/n_samples)*np.sum(yhat - y)
		return dw, db

	def _modelFunction(self, X, w, b):
		yhat = np.dot(X, w) + b
		sigmoid = self._sigmoid(yhat)
		return sigmoid

	def _sigmoid(self, yhat):
		return 1/(1 + np.exp(-yhat))