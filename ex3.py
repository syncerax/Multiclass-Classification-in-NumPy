import numpy as np
import matplotlib.pyplot as plt
from ex3_functions import cost_function_reg, gradient_reg, one_vs_all, predict_one_vs_all, plot_random_samples

input_layer_size = 400 # 20 x 20 pixels
num_labels = 10

# Loading training data
X = np.loadtxt("ex3data.csv", delimiter=',')
y = X[:, -1]
X = X[:, 0:-1]
m = len(y)

# Visualising some samples
plot_random_samples(X)

# Testing logistic regression cost function with regularisation
theta_t = np.array([-2, -1, 1, 2]).T
X_t = np.hstack((
    np.ones((5, 1)),
    np.reshape(np.arange(0.1, 1.6, 0.1), (3, 5)).T
))
y_t = np.array([1, 0, 1, 0, 1]).T
lamdba_t = 3

J = cost_function_reg(theta_t, X_t, y_t, lamdba_t)
grad = gradient_reg(theta_t, X_t, y_t, lamdba_t)

print("Cost: {}".format(J))
print("Expected cost: 2.534819")
print("Gradients:")
print(grad)
print("Expected gradients:")
print("[0.146561, -0.548558, 0.724722, 1.398003]")

input("\nProgram paused. Press enter to continue.")

# Training One vs All
reg_lambda = 0.1
all_theta = one_vs_all(X, y, num_labels, reg_lambda)

input("\nProgram paused. Press enter to continue.")

# Predict based on One vs All
pred = predict_one_vs_all(all_theta, X)

print('Training Set Accuracy: {}'.format(np.mean((pred == y)) * 100))