import numpy as np
from collections import Counter

def entropy(y):
	freq = np.bincount(y)
	probs = freq/len(y)
	return -np.sum([p*np.log2(p) for p in probs if p>0])

class Node:
	def __init__(self, feature = None, threshold = None, left = None, right = None, *, value = None):
		self.feature = feature
		self.threshold = threshold
		self.left = left
		self.right = right
		self.value = value

	def isLeafNode(self):
		return self.value is None

class DecisionTree:
	def __init__(self, max_depth = 100, min_sample_split = 2, n_feats = None):
		self.max_depth = max_depth
		self.min_sample_split = min_sample_split
		self.n_feats = n_feats
		self.root = None

	def fit(self, X, y):
		self.n_feats = X.shape[1] if self.n_feats is None else min(self,n_feats, X.shape[1])
		self.root = self._growTree(X, y)

	def _growTree(self, X, y, depth = 0):
		n_sample, n_feature = X.shape
		# Base condition
		if (depth >= self.max_depth) or (n_sample == 1) or (n_sample < self.min_sample_split):
			leaf = self._mostCommonLabel(y)
			return Node(value=leaf)
		# Select some random features from X train input.
		feat_idxs = np.random.choice(n_feature, self.n_feats, replace = False)
		# Select best feat and best threshold using best information gain.
		best_feat, best_threshold = self._bestCriteria(X, y, feat_idxs)
		left_idxs, right_idxs = self._split(X[:, best_feat], best_threshold)
		# Recur for left and right child
		left = self._growTree(X[left_idxs, :], y[left_idxs], depth + 1)
		right = self._growTree(X[right_idxs, :], y[right_idxs], depth + 1)
		return Node(best_feat, best_threshold, left, right)

	def _bestCriteria(self, X, y, feat_idxs):
		best_gain = float('-inf')
		split_idx, split_thresh = None, None
		for feat_idx in feat_idxs:
			X_ = X[:, feat_idx]
			thresholds = np.unique(y)
			for threshold in thresholds:
				gain = self._informationGain(X_, y, threshold)
				if gain > best_gain:
					best_gain = gain
					split_idx = feat_idx
					split_thresh = threshold
		return split_idx, split_thresh

	def _informationGain(self, X, y, threshold):
		# Parent entropy
		parent = entropy(y)
		# split
		left_idxs, right_idxs = self._split(X, threshold)
		if len(left_idxs) == 0 or len(right_idxs) == 0:
			return 0
		# Computing weighted average
		total = len(y)
		leftChildEntropy, rightChildEntropy = entropy(y[left_idxs]), entropy(y[right_idxs])
		# Information gain: parent entropy - weighted average of entropy of left and right childs.
		informationGain = parent - ((len(left_idxs)/total)*leftChildEntropy + (len(right_idxs)/total)*rightChildEntropy)
		return informationGain

	def _split(self, X, best_threshold):
		left_idxs = np.argwhere(X<=best_threshold).flatten()
		right_idxs = np.argwhere(X>best_threshold).flatten()
		return left_idxs, right_idxs

	def _mostCommonLabel(self, y):
		return Counter(y).most_common(1)[0][0]

	def predict(self, X):
		return np.array([self._traverseTree(x, self.root) for x in X])

	def _traverseTree(self, x, rootNode):
		# Base condition
		if rootNode.isLeafNode():
			return rootNode.value
		if x[rootNode.feature] <= rootNode.threshold:
			return self._traverseTree(x, rootNode.left)
		return self._traverseTree(x, rootNode.right)

	def predict_proba(self):
		pass