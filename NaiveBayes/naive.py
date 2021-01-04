import numpy as np

class GaussianNaiveClassifier:

	def fit(self, X, y):
		n_samples, n_features = X.shape
		self.classes = np.unique(y)
		self.n_classes = len(self.classes)

		self._mean = np.zeros((n_samples, n_features), dtype = np.float64)
		self._variance = np.zeros((n_samples, n_features), dtype = np.float64)
		self._priors = np.zeros(self.n_classes)

		for classLabel in self.classes:
			X_classLabel = X[classLabel == y]
			self._mean[classLabel, :] = X_classLabel.mean(axis = 0)
			self._variance[classLabel, :] = X_classLabel.var(axis = 0)
			self._priors[classLabel] = X_classLabel.shape[0]/n_samples

	def predict(self, X):
		yhat = [self._predict(x) for x in X]
		return yhat

	def _predict(self, x):
		posteriors = []
		for idx, classLabel in enumerate(self.classes):
			priors = np.log(self._priors[idx])
			classConditionalProb = np.sum(np.log(self._pdf(idx, x)))
			posterior = priors + classConditionalProb
			posteriors.append(posterior)
		return self.classes[np.argmax(posteriors)]

	def _pdf(self, classIdx, x):
		mean = self._mean[classIdx]
		variance = self._variance[classIdx]
		numerator = np.exp((-(x - mean)**2) / 2*variance)
		denomenator = np.sqrt(2*np.pi*variance)
		return numerator / denomenator