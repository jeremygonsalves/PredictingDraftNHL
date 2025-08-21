import json

import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score



def train_and_test(model, X, y, groups, param_grid, notes):
    r"""
    Parameters
    ----------
    model : sklearn.estimator
        An instance of an untrained scikit-learn estimator.
    X : pd.DataFrame
        The features to input into the model.
    y : pd.Series
        The labels to predict.
    param_grid : dict
        The parameter grid to input into GridSearchCV for
        hyperparameter tuning.
    notes : str
        Notes to include with the model.

    Returns
    -------
    metrics : dict
        The dictionary storing the testing metrics for each
        train/test split.
    """


    # store testing metrics for each train/test split
    metrics = defaultdict(list)

    gss = GroupShuffleSplit(n_splits=3, test_size=0.25, random_state=42)
    for i, (train_idx, test_idx) in enumerate(gss.split(X, y, groups)):
        # train/test splits
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Training config
        kfold = StratifiedKFold(n_splits=3)
        scoring = { 
            'f1': 'f1_macro', 
        }
        refit = 'f1'

        # Perform GridSearch for hyperparameter tuning
        gs_model = GridSearchCV(
            model, 
            param_grid=param_grid, 
            cv=kfold, 
            scoring=scoring, 
            refit=refit, 
            n_jobs=-1, 
            return_train_score=True, 
            verbose=1,
        )
        gs_model.fit(X_train, y_train)

        # obtain testing metrics
        y_test_pred = gs_model.predict(X_test)
        acc = gs_model.score(X_test, y_test)
        f1 = f1_score(y_test, y_test_pred, average='macro')
        pre = precision_score(y_test, y_test_pred, average='macro')
        rec = recall_score(y_test, y_test_pred, average='macro')

        metrics['accuracy'].append(acc)
        metrics['f1'].append(f1)
        metrics['precision'].append(pre)
        metrics['recall'].append(rec)

        # save the best parameters
        with open('artifacts/metrics.txt', 'a') as append_file:
            json_obj = json.dumps({
                'notes' : notes,
                **metrics,
                **gs_model.best_params_,
            })
            append_file.write(json_obj + '\n')
            
    return metrics
    