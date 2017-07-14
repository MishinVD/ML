import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone


class SimpleStacking:

    def __init__(self, estimator):
        self._estimator = clone(estimator)
        self._fitted = False
        
    def transform(self, X, use_proba=True):
        if not self._fitted:
            raise Exception("call 'fit_transform' first")
        
        if use_proba:
            pred = self._estimator.predict_proba(X)
        else:
            pred = self._estimator.predict(X)
        
        return pred
        
    def fit_transform(self, X, y, X_target=None, 
                      n_splits=5, shuffle=False, random_state=0, kfold=None,
                      use_proba=True, pred_col=-1, **fit_params):
        
        if type(y) == pd.Series:
            res = pd.Series(index=y.index)
        else:
            res = np.ndarray(y.shape)
            res.fill(0)
        
        if not kfold:
            kfold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        for train, test in kfold.split(X, y):
            if type(X) == pd.DataFrame:
                X_train = X.iloc[train]
                X_test = X.iloc[test]
            else:
                X_train = X[train]
                X_test = X[test]
                
            if type(y) == pd.Series:
                y_train = y.iloc[train]
                y_test = y.iloc[test]
            else:
                y_train = y[train]
                y_test = y[test]
                
            estimator = clone(self._estimator)
            estimator.fit(X_train, y_train, **fit_params)
            
            if use_proba:
                pred = estimator.predict_proba(X_test)[:, pred_col]
            else:
                pred = estimator.predict(X_test)
                
            if type(y) == pd.Series:
                res.iloc[test] = pred
            else:
                res[test] = pred
        
        pred_target = None
        if X_target is not None:
            self._estimator.fit(X, y, **fit_params)
            
            if use_proba:
                pred_target = self._estimator.predict_proba(X_target)[:, pred_col]
            else:
                pred_target = self._estimator.predict(X_target)
        self._fitted = True
        return res, pred_target
