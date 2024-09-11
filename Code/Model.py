from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.base import clone, BaseEstimator, ClassifierMixin

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression





class LogisticOrdinalRegression(BaseEstimator, ClassifierMixin):
    r"""
    This is a twist on Logistic Regression for ranking. Adapted from StackOverflow post:
    https://stackoverflow.com/questions/57561189/multi-class-multi-label-ordinal-classification-with-sklearn
    The primary changes are simply modifying the method signature for the __init__ method.
    """

    def __init__(self, penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
                 intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', 
                 max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, 
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

        self.clf_ = LogisticRegression(**self.get_params())
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



class OrdinalKNeighborsClassifier(BaseEstimator, ClassifierMixin):
    r"""
    This is a twist on K-Nearest Neighbors Classifier for ranking. Adapted from StackOverflow post:
    https://stackoverflow.com/questions/57561189/multi-class-multi-label-ordinal-classification-with-sklearn
    The primary changes are simply modifying the method signature for the __init__ method.
    """

    def __init__(self, n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, 
                 p=2, metric='minkowski', metric_params=None, n_jobs=None):
        
        self.n_neighbors=n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs      

    def fit(self, X, y):

        self.clf_ = KNeighborsClassifier(**self.get_params())
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



class RandomForestOrdinalClassifier(BaseEstimator, ClassifierMixin):
    r"""
    This is a twist on Random Forest Classification for ranking. Adapted from StackOverflow post:
    https://stackoverflow.com/questions/57561189/multi-class-multi-label-ordinal-classification-with-sklearn
    The primary changes are simply modifying the method signature for the __init__ method.
    """

    def __init__(self, n_estimators=100, *, criterion='gini', max_depth=None, 
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