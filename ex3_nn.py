import numpy as np
from ex3_nn_functions import predict

# Set up the parameters you will use for this exercise
input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10

# Loading training data
X = np.loadtxt("ex3data.csv", delimiter=',')
y = X[:, -1]
X = X[:, 0:-1]
m = len(y)

Theta1 = np.loadtxt("Theta1.csv", delimiter=',')
Theta2 = np.loadtxt("Theta2.csv", delimiter=',')

pred = predict(Theta1, Theta2, X)

print("Training Set Accuracy: {}".format(np.mean(pred == y) * 100))