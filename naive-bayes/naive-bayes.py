import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split

# estimator class

class NaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_bins = None, laplacian_smoothing = False):
        self.laplacian_smoothing = laplacian_smoothing
        self.n_bins = n_bins

    def fit(self, X, y):
        # Check that X and y have correct shape
        # X, y = check_X_y(X, y)

        # data discretization
        if self.n_bins != None:
            discretizer = KBinsDiscretizer(self.n_bins, encode='ordinal', strategy='uniform')
            discretizer.fit(X)
            X = discretizer.transform(X)

        # Store the classes seen during fit
        self.classes = unique_labels(y)
        self.features_count = X.shape[1]
        self.X_ = X
        self.y_ = y

        # Calculate classes a priori
        priori = []
        for c in self.classes:
            priori.append((y == c).sum()/len(y))
        self.classes_priori = priori

        # Calculate features likelihood - multinominal distribution
        likelihood = []
        for f in range(self.features_count):
            value_row = []
            for v in range(n_bins):
                classes_row = []
                for c in self.classes:
                    #Calculated for every feature, for every value, for every class
                    class_value_occ = 0
                    for i, row in enumerate(X):
                        if (row[f] == v and y[i][0] == c):
                            class_value_occ += 1
                    # print(class_value_occ)
                    prob = class_value_occ/len(y)
                    if (self.laplacian_smoothing and self.n_bins):
                        prob = (class_value_occ + 1)/(len(y) + self.n_bins)
                    classes_row.append(prob)
                value_row.append(classes_row)
            likelihood.append(value_row)

        likelihood = np.array(likelihood)
        print(likelihood.shape)

        self.features_likelihood = likelihood

        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)

        if self.n_bins != None:
            discretizer = KBinsDiscretizer(self.n_bins, encode='ordinal', strategy='uniform')
            discretizer.fit(X)
            X = discretizer.transform(X)

        results = []
        for row in range(X.shape[0]):
            class_prob = []
            for c in range(len(self.classes)):
                res = 1
                for f in range(self.features_count):
                    res *= self.classes_priori[c] * self.features_likelihood[f][int(X[row][f])][c]
                class_prob.append(res)
            results.append(class_prob)
        # print(results)
        
        assigned_class = []
        for r in results:
            max = np.argmax(r)
            assigned_class.append(max)

        return np.array(assigned_class)

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)

        if self.n_bins != None:
            discretizer = KBinsDiscretizer(self.n_bins, encode='ordinal', strategy='uniform')
            discretizer.fit(X)
            X = discretizer.transform(X)

        results = []
        for row in range(X.shape[0]):
            class_prob = []
            for c in range(len(self.classes)):
                res = 1
                for f in range(self.features_count):
                    res *= self.classes_priori[c] * self.features_likelihood[f][int(X[row][f])][c]
                class_prob.append(res)
            results.append(class_prob)

        return np.array(results)

# find accuracy
def find_accuracy(result, y):
    correct_counter = 0
    for i, r in enumerate(result):
        if (r == int(y[i][0])):
            correct_counter += 1
    return correct_counter / len(result)

# config
n_bins = 2 # number of bins to discretize data

# data loading
data = np.genfromtxt('/home/milena/projects/studia/sztuczna-inteligencja/naive-bayes/wine.data', delimiter=",")
print(data.shape)
X = data[:,1:]
y = data[:,:1]
# print(X.shape)
# print(y.shape)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# use classifier
est = NaiveBayesClassifier(n_bins, laplacian_smoothing=True)
est = est.fit(X_train, y_train)
prediction = est.predict(X_test)
print(prediction)
prediction_proba = est.predict_proba(X_test)
# print(prediction_proba)

print(find_accuracy(prediction, y_test))

# divorce data
data = np.genfromtxt('/home/milena/projects/studia/sztuczna-inteligencja/naive-bayes/divorce.csv', delimiter=";")
data = data[1:]
print(data.shape)
X = data[:,:-1]
y = data[:,-1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
est = NaiveBayesClassifier(n_bins, laplacian_smoothing=True)
est = est.fit(X_train, y_train)
prediction = est.predict(X_test)
print(prediction)
print(find_accuracy(prediction, y_test))