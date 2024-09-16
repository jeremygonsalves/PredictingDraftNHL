from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.base import clone, BaseEstimator, ClassifierMixin

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm


r""" models taken from Github from Quoc-Huy Nguyen and Ryan DeSalvio and repurposed for this project"""

class LogisticOrdinalRegression(BaseEstimator, ClassifierMixin):
    r"""
    This is a twist on Logistic Regression for ranking. Adapted from StackOverflow post:
    https://stackoverflow.com/questions/57561189/multi-class-multi-label-ordinal-classification-with-sklearn
    The primary changes are simply modifying the method signature for the __init__ method.
    """

    def __init__(self, penalty='l2', *, dual=False, tol=0.001, C=1.0, fit_intercept=True, 
                 intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', 
                 max_iter=10, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, 
                 l1_ratio=None):
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio

    def fit(self, X, y):
        # Initialize the base classifier (Logistic Regression)
        self.clf_ = LogisticRegression(**self.get_params())
        self.clfs_ = {}

        # Get unique classes and set classes_ attribute
        self.classes_ = np.unique(y)  # This is necessary for compatibility with scikit-learn
        self.unique_class_ = np.sort(self.classes_)
        
        # If there are more than two classes, fit ordinal classifiers
        if self.unique_class_.shape[0] > 2:
            for i in tqdm(range(self.unique_class_.shape[0]-1)):
                # For each ordinal value, fit a binary classifier
                binary_y = (y > self.unique_class_[i]).astype(np.uint8)
                clf = clone(self.clf_)
                clf.fit(X, binary_y)
                self.clfs_[i] = clf
        return self

    def predict_proba(self, X):
        clfs_predict = {k: self.clfs_[k].predict_proba(X) for k in self.clfs_}
        predicted = []
        for i, y in enumerate(self.unique_class_):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[i][:, 1])
            elif i in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                predicted.append(clfs_predict[i-1][:, 1] - clfs_predict[i][:, 1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[i-1][:, 1])
        return np.vstack(predicted).T

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y, sample_weight=None):
        _, indexed_y = np.unique(y, return_inverse=True)
        return accuracy_score(indexed_y, self.predict(X), sample_weight=sample_weight)




class RandomForestOrdinalClassifier(BaseEstimator, ClassifierMixin):
    r"""
    This is a twist on Random Forest Classification for ranking. Adapted from StackOverflow post:
    https://stackoverflow.com/questions/57561189/multi-class-multi-label-ordinal-classification-with-sklearn
    The primary changes are simply modifying the method signature for the __init__ method.
    """

    def __init__(self, n_estimators=1000, *, criterion='gini', max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
                 max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, 
                 bootstrap=True, oob_score=False, n_jobs=None, random_state=None, 
                 verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, 
                 max_samples=None):

        self.n_estimators=n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples

    def fit(self, X, y):

        self.clf_ = RandomForestClassifier(**self.get_params())
        self.clfs_ = {}

        self.unique_class_ = np.sort(np.unique(y))
        if self.unique_class_.shape[0] > 2:
            for i in tqdm(range(self.unique_class_.shape[0]-1)):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.unique_class_[i]).astype(np.uint8)
                clf = clone(self.clf_)
                clf.fit(X, binary_y)
                self.clfs_[i] = clf

    def predict_proba(self, X):
        clfs_predict = {k: self.clfs_[k].predict_proba(X) for k in self.clfs_}
        predicted = []
        for i, y in enumerate(self.unique_class_):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[i][:,1])
            elif i in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                 predicted.append(clfs_predict[i-1][:,1] - clfs_predict[i][:,1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[i-1][:,1])
        return np.vstack(predicted).T

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y, sample_weight=None):
        _, indexed_y = np.unique(y, return_inverse=True)
        return accuracy_score(indexed_y, self.predict(X), sample_weight=sample_weight)
    
    
    
    
class SVMOrdinalClassifier(BaseEstimator, ClassifierMixin):
    r"""
    This is a twist on SVM Classification for ranking. Adapted from StackOverflow post:
    https://stackoverflow.com/questions/57561189/multi-class-multi-label-ordinal-classification-with-sklearn
    The primary changes are simply modifying the method signature for the __init__ method.
    """

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True,
                 probability=True, tol=1e-3, cache_size=200, class_weight=None, verbose=False,
                 max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state

    def fit(self, X, y):
        self.clf_ = SVC(probability=True, **self.get_params())
        self.clfs_ = {}

        self.unique_class_ = np.sort(np.unique(y))
        if self.unique_class_.shape[0] > 2:
            for i in tqdm(range(self.unique_class_.shape[0] - 1)):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.unique_class_[i]).astype(np.uint8)
                clf = clone(self.clf_)
                clf.fit(X, binary_y)
                self.clfs_[i] = clf

    def predict_proba(self, X):
        clfs_predict = {k: self.clfs_[k].predict_proba(X) for k in self.clfs_}
        predicted = []
        for i, y in enumerate(self.unique_class_):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[i][:, 1])
            elif i in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                predicted.append(clfs_predict[i - 1][:, 1] - clfs_predict[i][:, 1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[i - 1][:, 1])
        return np.vstack(predicted).T

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y, sample_weight=None):
        _, indexed_y = np.unique(y, return_inverse=True)
        return accuracy_score(indexed_y, self.predict(X), sample_weight=sample_weight)
    
    
    
    
class MLPOrdinalClassifier(BaseEstimator, ClassifierMixin):
        """
        This is a twist on MLP (Multi-Layer Perceptron) Classification for ranking.
        """

        def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, 
                     batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, 
                     max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, 
                     warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, 
                     validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-8, n_iter_no_change=10, 
                     max_fun=15000):
            self.hidden_layer_sizes = hidden_layer_sizes
            self.activation = activation
            self.solver = solver
            self.alpha = alpha
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.learning_rate_init = learning_rate_init
            self.power_t = power_t
            self.max_iter = max_iter
            self.shuffle = shuffle
            self.random_state = random_state
            self.tol = tol
            self.verbose = verbose
            self.warm_start = warm_start
            self.momentum = momentum
            self.nesterovs_momentum = nesterovs_momentum
            self.early_stopping = early_stopping
            self.validation_fraction = validation_fraction
            self.beta_1 = beta_1
            self.beta_2 = beta_2
            self.epsilon = epsilon
            self.n_iter_no_change = n_iter_no_change
            self.max_fun = max_fun

        def fit(self, X, y):
            self.clf_ = MLPClassifier(**self.get_params())
            self.clfs_ = {}

            self.unique_class_ = np.sort(np.unique(y))
            if self.unique_class_.shape[0] > 2:
                for i in tqdm(range(self.unique_class_.shape[0] - 1)):
                    # for each k - 1 ordinal value we fit a binary classification problem
                    binary_y = (y > self.unique_class_[i]).astype(np.uint8)
                    clf = clone(self.clf_)
                    clf.fit(X, binary_y)
                    self.clfs_[i] = clf

        def predict_proba(self, X):
            clfs_predict = {k: self.clfs_[k].predict_proba(X) for k in self.clfs_}
            predicted = []
            for i, y in enumerate(self.unique_class_):
                if i == 0:
                    # V1 = 1 - Pr(y > V1)
                    predicted.append(1 - clfs_predict[i][:, 1])
                elif i in clfs_predict:
                    # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                    predicted.append(clfs_predict[i - 1][:, 1] - clfs_predict[i][:, 1])
                else:
                    # Vk = Pr(y > Vk-1)
                    predicted.append(clfs_predict[i - 1][:, 1])
            return np.vstack(predicted).T

        def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1)

        def score(self, X, y, sample_weight=None):
            _, indexed_y = np.unique(y, return_inverse=True)
            return accuracy_score(indexed_y, self.predict(X), sample_weight=sample_weight)