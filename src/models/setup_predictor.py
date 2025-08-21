"""
Model Pipeline Setup Module

This module provides utilities for setting up scikit-learn pipelines for NHL draft prediction.
It organizes preprocessing based on feature types (numeric, categorical, text) and supports
both traditional TF-IDF and BERT embeddings for text features.

Author: NHL Draft Prediction Team
"""

# Standard library imports
from typing import List, Optional, Union, Any
import warnings

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.svm import SVC
from sentence_transformers import SentenceTransformer

# Configuration constants
DEFAULT_NUMERIC_COLS = ['Height', 'Weight']
DEFAULT_CATEGORICAL_COLS = ['Position']
DEFAULT_TEXT_COLS = ['all_reports']
DEFAULT_BERT_MODEL = 'all-mpnet-base-v2'
DEFAULT_TFIDF_PARAMS = {
    'analyzer': 'word',
    'max_df': 0.5,
    'min_df': 0.04,
    'ngram_range': (1, 3)
}


def _get_text_data(x: np.ndarray) -> np.ndarray:
    """
    Helper function to flatten text data for processing.
    
    Parameters
    ----------
    x : np.ndarray
        Input text data array
        
    Returns
    -------
    np.ndarray
        Flattened text data
    """
    return x.ravel()


class CustomBertTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for BERT sentence embeddings.
    
    This transformer uses the SentenceTransformer library to convert text
    into BERT embeddings for machine learning models.
    """
    
    def __init__(self, model_name: str = DEFAULT_BERT_MODEL):
        """
        Initialize the BERT transformer.
        
        Parameters
        ----------
        model_name : str, default=DEFAULT_BERT_MODEL
            Name of the BERT model to use for embeddings
        """
        super().__init__()
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def fit(self, X: Any, y: Optional[Any] = None) -> 'CustomBertTransformer':
        """
        Fit the transformer (no-op for BERT).
        
        Parameters
        ----------
        X : Any
            Input data (not used)
        y : Optional[Any], default=None
            Target data (not used)
            
        Returns
        -------
        CustomBertTransformer
            Self reference for method chaining
        """
        return self
    
    def transform(self, X: Any, y: Optional[Any] = None) -> np.ndarray:
        """
        Transform text data to BERT embeddings.
        
        Parameters
        ----------
        X : Any
            Input text data
        y : Optional[Any], default=None
            Target data (not used)
            
        Returns
        -------
        np.ndarray
            BERT embeddings
        """
        embeddings = self.model.encode(X)
        return embeddings


def create_numeric_transformer() -> Pipeline:
    """
    Create a pipeline for numeric feature preprocessing.
    
    Returns
    -------
    Pipeline
        Pipeline with imputation and scaling for numeric features
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])


def create_categorical_transformer() -> Pipeline:
    """
    Create a pipeline for categorical feature preprocessing.
    
    Returns
    -------
    Pipeline
        Pipeline with imputation and one-hot encoding for categorical features
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=' ')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])


def create_text_transformer(use_bert: bool = False, tfidf_params: Optional[dict] = None) -> Pipeline:
    """
    Create a pipeline for text feature preprocessing.
    
    Parameters
    ----------
    use_bert : bool, default=False
        Whether to use BERT embeddings instead of TF-IDF
    tfidf_params : Optional[dict], default=None
        Parameters for TF-IDF vectorizer (only used if use_bert=False)
        
    Returns
    -------
    Pipeline
        Pipeline with imputation and vectorization for text features
    """
    if tfidf_params is None:
        tfidf_params = DEFAULT_TFIDF_PARAMS
    
    vectorizer = (CustomBertTransformer() if use_bert 
                 else TfidfVectorizer(**tfidf_params))
    
    return Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=' ')),
        ('selector', FunctionTransformer(_get_text_data)),
        ('vectorizer', vectorizer)
    ])


def validate_inputs(
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    text_cols: Optional[List[str]] = None,
    func: Optional[BaseEstimator] = None
) -> None:
    """
    Validate input parameters for the setup function.
    
    Parameters
    ----------
    numeric_cols : Optional[List[str]], default=None
        List of numeric column names
    categorical_cols : Optional[List[str]], default=None
        List of categorical column names
    text_cols : Optional[List[str]], default=None
        List of text column names
    func : Optional[BaseEstimator], default=None
        Scikit-learn estimator
        
    Raises
    ------
    ValueError
        If validation fails
    """
    if func is None:
        raise ValueError("Estimator function 'func' must be provided")
    
    if not isinstance(func, BaseEstimator):
        raise ValueError("'func' must be a scikit-learn estimator")
    
    # Check for column overlap
    all_cols = []
    if numeric_cols:
        all_cols.extend(numeric_cols)
    if categorical_cols:
        all_cols.extend(categorical_cols)
    if text_cols:
        all_cols.extend(text_cols)
    
    if len(all_cols) != len(set(all_cols)):
        raise ValueError("Column names must be unique across all feature types")


def setup(
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    text_cols: Optional[List[str]] = None,
    func: Optional[BaseEstimator] = None,
    use_bert: bool = False,
    tfidf_params: Optional[dict] = None,
    **kwargs: Any
) -> Pipeline:
    """
    Setup a complete machine learning pipeline for NHL draft prediction.
    
    This function creates a scikit-learn pipeline that organizes preprocessing
    based on feature types (numeric, categorical, text). The pipeline handles
    missing values, feature scaling, encoding, and text vectorization.
    
    Parameters
    ----------
    numeric_cols : Optional[List[str]], default=None
        List of numeric column names. If None, uses DEFAULT_NUMERIC_COLS.
    categorical_cols : Optional[List[str]], default=None
        List of categorical column names. If None, uses DEFAULT_CATEGORICAL_COLS.
    text_cols : Optional[List[str]], default=None
        List of text column names. If None, uses DEFAULT_TEXT_COLS.
    func : Optional[BaseEstimator], default=None
        Scikit-learn estimator (classifier or regressor).
    use_bert : bool, default=False
        Whether to use BERT embeddings for text features instead of TF-IDF.
    tfidf_params : Optional[dict], default=None
        Parameters for TF-IDF vectorizer (only used if use_bert=False).
    **kwargs : Any
        Additional keyword arguments (currently unused).
        
    Returns
    -------
    Pipeline
        Complete scikit-learn pipeline ready for training.
        
    Raises
    ------
    ValueError
        If input validation fails.
        
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from setup_predictor import setup
    >>> 
    >>> # Create pipeline with default settings
    >>> pipeline = setup(func=RandomForestClassifier())
    >>> 
    >>> # Create pipeline with custom columns and BERT
    >>> pipeline = setup(
    ...     numeric_cols=['Height', 'Weight'],
    ...     categorical_cols=['Position'],
    ...     text_cols=['scouting_report'],
    ...     func=RandomForestClassifier(),
    ...     use_bert=True
    ... )
    """
    # Set default values
    numeric_cols = numeric_cols or DEFAULT_NUMERIC_COLS
    categorical_cols = categorical_cols or DEFAULT_CATEGORICAL_COLS
    text_cols = text_cols or DEFAULT_TEXT_COLS
    
    # Validate inputs
    validate_inputs(numeric_cols, categorical_cols, text_cols, func)
    
    # Create transformers
    transformers = []
    
    if numeric_cols:
        transformers.append(('numeric', create_numeric_transformer(), numeric_cols))
    
    if categorical_cols:
        transformers.append(('categorical', create_categorical_transformer(), categorical_cols))
    
    if text_cols:
        transformers.append(('text', create_text_transformer(use_bert, tfidf_params), text_cols))
    
    # Create feature transformer
    feature_transformer = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop any columns not explicitly handled
    )
    
    # Create final pipeline
    pipeline = Pipeline([
        ('features', feature_transformer),
        ('classifier', func)
    ])
    
    return pipeline


# Convenience functions for common use cases
def setup_bert_pipeline(
    func: BaseEstimator,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    text_cols: Optional[List[str]] = None,
    **kwargs: Any
) -> Pipeline:
    """
    Convenience function to create a pipeline with BERT embeddings.
    
    Parameters
    ----------
    func : BaseEstimator
        Scikit-learn estimator
    numeric_cols : Optional[List[str]], default=None
        List of numeric column names
    categorical_cols : Optional[List[str]], default=None
        List of categorical column names
    text_cols : Optional[List[str]], default=None
        List of text column names
    **kwargs : Any
        Additional arguments passed to setup()
        
    Returns
    -------
    Pipeline
        Pipeline configured with BERT embeddings
    """
    return setup(
        func=func,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        text_cols=text_cols,
        use_bert=True,
        **kwargs
    )


def setup_tfidf_pipeline(
    func: BaseEstimator,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    text_cols: Optional[List[str]] = None,
    tfidf_params: Optional[dict] = None,
    **kwargs: Any
) -> Pipeline:
    """
    Convenience function to create a pipeline with TF-IDF vectorization.
    
    Parameters
    ----------
    func : BaseEstimator
        Scikit-learn estimator
    numeric_cols : Optional[List[str]], default=None
        List of numeric column names
    categorical_cols : Optional[List[str]], default=None
        List of categorical column names
    text_cols : Optional[List[str]], default=None
        List of text column names
    tfidf_params : Optional[dict], default=None
        Parameters for TF-IDF vectorizer
    **kwargs : Any
        Additional arguments passed to setup()
        
    Returns
    -------
    Pipeline
        Pipeline configured with TF-IDF vectorization
    """
    return setup(
        func=func,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        text_cols=text_cols,
        use_bert=False,
        tfidf_params=tfidf_params,
        **kwargs
    )
