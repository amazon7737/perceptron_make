#from re import I
from ast import increment_lineno
import numpy as np

# X , W -> radom value use

class CustomPerceptron(object):
    def __init__(self, n_iterations=100, random_state=1, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.learning_rate = learning_rate

# Net Input

    def Net_input(self, X):
        weighted_sum = np.dot(X, self.coef_[1:]) + self.coef_[0]

        return weighted_sum

# Activation function

    def activation_function(self, X):
        weighted_sum = self.Net_input(X)
        return np.where(weighted_sum >= 0.0, 1, 0)



# Prediction function -> easy just output activation function

    def predict(self, X):
        return self.activation_function(X)


# Stochastic Gradient Descent -> I don know why learning this 

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.coef_ = rgen.normal(loc=0.0, scale=0.01, size = 1 + X.shape[1])
        self.errors_ = [] 
        for _ in range(self.n_iterations):
            errors = 0 
            for xi, expected_value in zip(X, y):
                predicted_value = self.predict(xi)
                self.coef_[1:] = self.coef_[1:] + self.learning_rate * (expected_value - predicted_value) * xi
                self.coef_[0] = self.coef_[0] + self.learning_rate * (expected_value - predicted_value) * 1
                update = self.learning_rate * (expected_value - predicted_value)
                errors += int(update != 0.0)
                self.errors_.append(errors)





# just use and anal

# Perceptron Implementation

# score -> calcul expected value and prediceted value

    def score(self,X, y):
        misclassified_data_count = 0
        for xi, target in zip(X, y):
            output = self.predict(xi)
            if(target != output):
                misclassified_data_count += 1
        total_data_count = len(X)
        self.score_ = (total_data_count - misclassified_data_count) / total_data_count
        return self.score_




# -- training course #

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load the data set

bc = datasets.load_breast_cancer()
X = bc.data
y = bc.target

# Create training and test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify = y)



# Instantiagte CustomPerceptron

prcptrn = CustomPerceptron(n_iterations = 10)


# Fit the model

prcptrn.fit(X_train, y_train)


# Score the model

prcptrn.score(X_test, y_test),
prcptrn.score(X_train, y_train)

#%matplotlib inline
import matplotlib.pyplot as plt


plt.plot(range(1, len(prcptrn.errors_) + 1), prcptrn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

# plt.savefig('images/02_07.png', dpi=300)

plt.show()







