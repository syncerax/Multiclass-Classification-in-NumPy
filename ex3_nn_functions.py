import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(Theta1, Theta2, X):
    m = len(X)
    num_labels = Theta2.shape[0]
    # p = np.zeros((m, 1))
    a1 = np.hstack((np.ones((m, 1)), X)).T
    z2 = Theta1 @ a1
    a2 = sigmoid(z2)
    a2 = np.vstack((np.ones((1, m)), a2))
    z3 = Theta2 @ a2
    a3 = sigmoid(z3)
    p = np.argmax(a3, axis=0).T + 1
    return p