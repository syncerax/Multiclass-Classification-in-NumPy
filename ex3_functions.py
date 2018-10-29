import numpy as np
from scipy.optimize import fmin_cg
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def h(X, theta):
    return sigmoid(X.dot(theta))


def cost_function_reg(theta, X, y, reg_lambda):
    m = len(y)
    y_zero = (1 - y).dot(np.log(1 - h(X, theta)))
    y_one = y.dot(np.log(h(X, theta)))
    reg = (reg_lambda / (2 * m)) * sum(theta[1:] ** 2)
    J = (-1 / m) * (y_zero + y_one) + reg
    return J


def gradient_reg(theta, X, y, reg_lambda):
    m = len(y)
    reg = (reg_lambda / m) * theta
    reg[0] = 0
    return ((h(X, theta) - y).dot(X) / m) + reg


def one_vs_all(X, y, num_labels, reg_lambda):
    m, n = X.shape
    all_theta = np.zeros((num_labels, n + 1))
    X = np.hstack((np.ones((m, 1)), X))
    for c in range(1, num_labels + 1):
        initial_theta = np.zeros((n + 1, 1))
        theta = fmin_cg(f=cost_function_reg, x0=initial_theta, fprime=gradient_reg, args=(X, y == c, reg_lambda), maxiter=100)
        all_theta[c - 1, :] = theta.T

    return all_theta


def predict_one_vs_all(all_theta, X):
    m = len(X)
    X = np.hstack((np.ones((m, 1)), X))
    return np.argmax(h(X, all_theta.T), axis=1) + 1 # +1 for the 0 -> 10 transition


def plot_random_samples(X):
    size = 20
    random_samples = np.random.randint(X.shape[0], size=25)
    plt.figure(figsize=(5, 5))
    for i, sample in enumerate(random_samples):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[sample].reshape(size, size).T, cmap=plt.cm.binary)
    plt.show()