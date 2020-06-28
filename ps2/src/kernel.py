import numpy as np

def polynomial_kernel(X, X1, order, bias):
    K = (X.dot(X1.T) + bias) ** order
    return K

def gaussian_kernel(X, X1, radius):
    squared_X = np.sum(X * X, axis=1)
    squared_X1 = np.sum(X1 * X1, axis=1)
    gram = X.dot(X1.T)
    K = np.exp(-(squared_X.reshape((1, -1)) + squared_X1.reshape((-1, 1)) - 2 * gram) / (2 * (radius ** 2)) )
    return K