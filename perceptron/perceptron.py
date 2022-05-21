import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt

m = 20
learning_rate = 0.5

# generate artificial data
X1, Y1 = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, n_classes=2, n_samples=m, class_sep=2)
plt.scatter(X1[:, 0], X1[:,1], marker='x', c=Y1, s=25, edgecolor='k')
# plt.show()

X_train = np.hstack((X1, np.ones((X1.shape[0], 1), dtype=X1.dtype)))
Y_train = Y1

# classifier class
class PerceptronClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def activation(self, X):
        activation_val = np.dot(self.weights, X)
        return 1 if activation_val > 0 else 0

    def fit(self, X, Y):
        self.weights = np.zeros(X.shape[1])
        self.k = 0

        while(self.k < 1000):
            classified_correctly = True
            errors = []

            # check if classified correctly
            for i, x in enumerate(X):
                prediction = self.activation(x)
                err = Y[i] - prediction
                errors.append(err)
                if err != 0:
                    classified_correctly = False
                
            if classified_correctly:
                break
            else:
                # get random x
                index = np.random.randint(len(X))
                # update weights
                for j, w in enumerate(self.weights):
                    self.weights[j] = self.weights[j] + (self.lr * errors[index] * X[index][j])
            self.k += 1

        print(self.k)
        print(self.weights)

        return self

    def predict(self, X):
        pass

est = PerceptronClassifier(learning_rate)
est = est.fit(X_train, Y_train)
