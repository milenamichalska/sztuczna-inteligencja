from sklearn.datasets import fetch_openml
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

X_iris, y_iris = fetch_openml(name="iris", version=1, return_X_y=True)

y = y_iris.values
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
y = y.reshape(-1, 1)
enc = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse= False)
y = enc.fit_transform(y)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X_iris, y, random_state=13)

class MLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden = 10, epochs = 100, eta = 0.1, shuffle = True):
        self.hidden = hidden
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _forward(self, X):
        hidden_sum = np.dot(self.w_h, X) + self.b_h

        hidden_output = []
        for n in hidden_sum:
            hidden_output.append(self._sigmoid(n))

        

    def _compute_cost(self, y, out):
        pass

    def fit(self, X_train, y_train):
        self.w_h = np.random.randn(X_train.shape[0], self.hidden) * 0.1
        self.w_h = np.random.randn(X_train.shape[0], self.hidden) * 0.1
