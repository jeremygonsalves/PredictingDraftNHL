import json
import pickle

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

from sentence_transformers import SentenceTransformer


def _get_text_data(x):
    return x.ravel()

class CustomBertTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        super().__init__()

        self.model = SentenceTransformer('all-mpnet-base-v2')

    def fit(self, X, y=None):
        # no fitting
        return self
    
    def transform(self, X, y=None):
        # return bert embeddings
        embeddings = self.model.encode(X)
        return embeddings

def setup(numeric_cols=None, categorical_cols=None, text_cols=None, func=None, bert=False, *args, **kwargs):
    r"""
    Setup a model for prediction. The idea to organize the pipeline into preprocessing 
    the features based on their data type comes from the course SIADS 696.

    Parameters
    ----------
    numeric_cols : list, default=None
        The numeric columns/features of X.
    categorical_cols : list, default=None
        The categorical columns/features of X, but not including the text columns.
    text_cols : list, default=None
        The text columns/features of X.
    func : sklearn.estimator
        An instance of a classification/regression estimator from scikit-learn.
    bert : bool, default=False.
        Boolean value whether to use the Bert sentence transformers.
        
    Returns
    -------
    model :
        The untrained scikit-learn model.
    """

    if numeric_cols is None:
        numeric_cols = ['Height', 'Weight']
    if categorical_cols is None:
        categorical_cols = ['Position']
    if text_cols is None:
        text_cols = ['all_reports']

    numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scalar', StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            # encode position with OneHotEncoder over LabelEncoder
            #   since LabelEncoder defines an unintentional ordering
            #   (e.g., 0 < 1 < 2)
            ('imputer', SimpleImputer(strategy='constant', fill_value=' ')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]
    )
    
    text_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=' ')),
            ('selector', FunctionTransformer(_get_text_data)),
            ('vectorizer', CustomBertTransformer() if bert else TfidfVectorizer(analyzer='word', max_df=0.5, min_df=0.04, ngram_range=(1, 3)))  
        ]
    )

    feature_transformer = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_cols),
            ('categorical', categorical_transformer, categorical_cols),
            ('text', text_transformer, text_cols)
        ]
    )

    model = Pipeline(
        steps=[
            ('features', feature_transformer),
            ('clf', func)
        ]
    )

    return model
