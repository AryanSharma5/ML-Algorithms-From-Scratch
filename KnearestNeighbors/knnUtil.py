import numpy as np

def euclid(x1, x2):
	return np.sqrt(np.sum((x2-x1)**2))

def accuracy(ypred, ytrue):
	return np.sum(ypred == ytrue)/len(ytrue) 