import numpy as np
import csv

def euclidean_distance(x1, x2):
    #Compute Euclidean distance between two data points.
    distance = 0
    for index, elem in enumerate(x1):
        distance += np.square(x1[index] - x2[index])
    distance = np.sqrt(distance)
    return distance

def get_neighbors(X_train, y_train, x_new, k):
    dist = np.zeros(np.size(y_train))
    for i in range(np.size(y_train)):
        dist[i] = euclidean_distance(X_train[i, :], x_new)
    maxDist = np.amax(dist) + 1.0
    neighbors_labels = np.empty(k)
    neighbors_distances = np.empty(k)
    for i in range(k):
        minInd = np.argmin(dist)
        neighbors_distances[i] = dist[minInd]
        neighbors_labels[i] = y_train[minInd]
        dist[minInd] = maxDist
    return neighbors_labels, neighbors_distances

def get_response(neighbors, distances, num_classes=3):
    sum_dist = np.sum(1/distances)
    y_hat = 0
    y_hat += np.sum((1/distances) * neighbors)
    y_hat /= sum_dist
    return y_hat

def compute_accuracy(y_pred, y_test):
    n_right = 0.0
    for ind, elem in enumerate(y_pred):
        if elem == y_test[ind]:
            n_right += 1.0
    print(y_pred)
    return n_right / np.size(y_pred)

def predict(X_train, y_train, X_test, k):
    y_pred = []
    for x_new in X_test:
        neighbors, distances = get_neighbors(X_train, y_train, x_new, k)
        y_pred.append(get_response(neighbors,distances))
    return y_pred

dataset = np.genfromtxt('01_homework_dataset.csv', delimiter=',', skip_header=1,
                     skip_footer=0)
k = 3
X_train = dataset[:,0:3]
y_train = dataset[:,3]
X_test = np.array([[4.1,-0.1,2.2],[6.1,0.4,1.3]])
y_pred = predict(X_train, y_train, X_test, k)
print("y_pred = ")
print(y_pred)