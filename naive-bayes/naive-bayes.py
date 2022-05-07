import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

# estimator class

class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, laplacian_smoothing = False):
        self.laplacian_smoothing = laplacian_smoothing

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.features_ = X.shape[1]
        print(self.features_)
        self.X_ = X
        self.y_ = y

        # Calculate classes a priori
        priori = []
        for c in self.classes_:
            priori.append((y == c).sum()/len(y))
        self.classes_priori = priori

        # Calculate features likelihood
        likelihood = []
        for f in range(self.features_):
            pass
        self.features_likelihood = likelihood
          
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]

    def predict_proba(self, X):
        pass

# config
n_bins = 10 # number of bins to discretize data

# data loading
data = np.genfromtxt('/home/milena/projects/studia/sztuczna-inteligencja/naive-bayes/wine.data', delimiter=",")
print(data.shape)
X = data[:,:13]
y = data[:,-1:]

# data discretization
discretizer = KBinsDiscretizer(n_bins, encode='ordinal', strategy='uniform')
discretizer.fit(X)
Xt = discretizer.transform(X)

# train test split
X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.33, random_state=42)

# use classifier
est = NaiveBayesClassifier()
est.fit(X_train, y_train)
prediction = est.predict(X_test)
prediction_proba = est.predict_proba(X_test)
