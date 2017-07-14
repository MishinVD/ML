import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from hyperopt import hp
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


class SpaceSearchCV:

    def __init__(self, estimator, scoring, param_space,
                 cv=5, verbose=0, random_state=0, shuffle=False,
                 max_evals=10,
                 maximize_score=False,
                 fit_best_estimator=False):
        self._kfold = StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
        self._estimator = clone(estimator)
        self._scoring = scoring
        self._max_evals = max_evals
        self._param_space = param_space
        self._verbose = verbose
        self._maximize_score = maximize_score
        self._fit_best_estimator = fit_best_estimator
        self._fitted = False
    
    def _run(self, X, y, params, **fit_params):
        evals_results = []
        for train_index, test_index in self._kfold.split(X, y):
            if type(X) == pd.DataFrame:
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            if type(X) == np.ndarray:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
            estimator = self._estimator.__class__(**params)
            
            estimator.fit(X_train, y_train, **fit_params)
            
            evals_results.append(self._scoring(y_test, estimator.predict(X_test)))
            
        mean_score = np.mean(evals_results)
        std_score = np.std(evals_results)
        
        result = {
            'mean_score': mean_score,
            'loss': -mean_score if self._maximize_score else mean_score,
            'std_score': std_score,
            'params': params.copy(),
            'status': STATUS_OK
        }
        return result
    
    def fit(self, X, y, **fit_params):
        self._trials = Trials()
        fmin(
            rstate=np.random.RandomState(0),
            fn=lambda params: self._run(X=X, y=y, params=params),
            space=self._param_space,
            algo=tpe.suggest,
            max_evals=self._max_evals,
            trials=self._trials
        )
        self._estimator = self._estimator.__class__(**self.best_params_)
        self._fitted = True
        
        if self._fit_best_estimator:
            self._estimator.fit(X, y)
            
        return self.best_estimator_
    
    @property
    def trials_(self):
        return self._trials
    
    @property
    def space_scores_(self):
        res = []
        for r in self._trials.results:
            res.append({
                "mean": r['mean_score'],
                "std": r['std_score'],
                "params": r['params']
            })
        return res
    
    @property
    def best_score_(self):
        srtd = sorted(self._trials.results, key=lambda x:x['loss'])
        ind = -1 if self._maximize_score else 0
        return srtd[0]['mean_score']
        
    @property
    def best_params_(self):
        srtd = sorted(self._trials.results, key=lambda x:x['loss'])
        ind = -1 if self._maximize_score else 0
        return srtd[0]['params']
    
    @property
    def best_estimator_(self):
        if not self._fitted:
            raise Exception("call fit first")
        return self._estimator
