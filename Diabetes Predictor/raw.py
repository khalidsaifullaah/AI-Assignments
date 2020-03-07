import numpy as np
import pandas as pd
from math import exp
from math import log
import matplotlib.pyplot as plt

def scale_data():
    for i in data.columns:
        if i != "Outcome":
            data[i] = data[i] - data[i].mean() / (data[i].max() - data[i].min())

def plot_data():
    # y = target values, last column of the data frame
    y = data.iloc[:, -1]
    # filter out the applicants that got admitted
    has_diabetes = data.loc[y == 1]

    # filter out the applicants that din't get admission
    no_diabetes = data.loc[y == 0]

    # plots
    plt.scatter(has_diabetes.iloc[:, 0], has_diabetes.iloc[:, 1], s=10, label='has_diabetes')
    plt.scatter(no_diabetes.iloc[:, 0], no_diabetes.iloc[:, 1], s=10, label='no_diabetes')
    plt.legend()
    plt.show()

data = pd.read_csv("diabetes_data.csv", index_col = None, header = 0)
# print(data)
plot_data()
scale_data()
data

# 70% of the data assigned to the train set
train_size = int(0.7*len(data))
train_set = data[:train_size]

# 30% of the data assigned to the test set
test_size = int(0.3*len(data))
test_set = data[train_size:len(data)]

X = train_set.iloc[:, :-1].values # Taking out all the features (except for output column) from the data and storing into X
Y = train_set.iloc[:, -1].values  # The column that contains the result/output
Y = Y.reshape((-1, 1)) # making the array n*1 shape

# def sigmoid(z):
#     g = (1 / (1 + np.exp(-z)))
#     return g

def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))     # Define sigmoid function
    sig = np.minimum(sig, 0.9999)  # Set upper bound
    sig = np.maximum(sig, 0.0001)  # Set lower bound
    return sig

def loss_function(hypotheses, Y):
    m = len(X)
    h = hypotheses
#     j = -sum(Y * np.log(h) + (1 - Y) * np.log(1 - h)) / m
    return -np.mean((Y * np.log(h) + (1 - Y) * np.log(1 - h)))

def predict(test_data):
    predictions = []
    Z = np.dot(test_data, theta) + bias
    for i in sigmoid(Z):
        if i > 0.5:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions

# Initialization part
np.random.seed(0) # prevents from having different random nums each time code runs
theta = np.random.uniform(0, 1, size = (X.shape[1], 1)) #this is our weight vectors
bias = 0.5
alpha = 0.05
m = len(X)
j = 1 # initialising the cost with random value
converge_change = 0.005 #this is the least value we want for j/cost to be

# Algorithm in action
# while j > converge_change:
for i in range(100000):
    hypotheses = np.dot(X, theta) + bias # return an array containing our hypotheses
    hypotheses = sigmoid(hypotheses)
    #     print(hypotheses, Y)
    j = loss_function(hypotheses, Y)
    print("------------->",j)
    grad= hypotheses - Y
    grad_weight= np.dot(X.T,grad)/X.shape[0]
    theta=theta-.01*grad_weight
#     theta = theta - (alpha * sum(np.dot(X.T, (hypotheses - Y)))) / m

X_test = test_set.iloc[:, :-1].values
Y_test = predict(X_test)
print(test_set)
print(Y_test)