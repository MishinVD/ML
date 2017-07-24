import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


class CatTransformer:
    
    def __init__(self, func, cv=5, shuffle=False, random_state=0, kf=None):
        if kf:
            self._kf = kf
        else:
            self._kf = StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=random_state)
        self._func = func
        
    def fit_transform(self, X, y, column_by, verbose=False, split_by=None, **fit_params):
        r = X.loc[:, column_by].copy()
        self._fit_params = fit_params
        
        split_col = split_by if split_by else y
        
        for train_index, test_index in self._kf.split(X, split_col):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]
            
            unique_values = np.unique(X_test[[column_by]])
            
            mapped_values = {
                unique_value: self._func(X_train, y_train, column_by, unique_value, **self._fit_params) 
                for unique_value in unique_values
            }
            
            if verbose:
                print(mapped_values)

            r.iloc[test_index] = X_test.loc[:, column_by].map(mapped_values)
            
        return r
        
    def transform(self, X_train, y_train, X_target, column_by):      
        unique_values = np.unique(X_target[[column_by]])
        mapped_values = {
            unique_value: self._func(X_train, y_train, column_by, unique_value, **self._fit_params) 
            for unique_value in unique_values
        }
        r = X_target.loc[:, column_by].map(mapped_values).copy()
        return r


class BaseCategoryProcessor:

    def __init__(self, func=None, cv=5, shuffle=False, random_state=0, kf=None):
        self._func=self.get_func_by_name(func)
        self._transformer = CatTransformer(func=self._func, cv=cv, shuffle=shuffle, random_state=random_state, kf=kf)

    @staticmethod
    def get_func_by_name(func_name):
        if func_name == 'smoothed_likehood':
            return BaseCategoryProcessor._smoothed_likehood
        if func_name == 'weights_of_evidence':
            return BaseCategoryProcessor._weights_of_evidence
        if func_name == 'difference_in_counts':
            return BaseCategoryProcessor._difference_in_counts
        raise Exception('Unknown function!')

    @staticmethod
    def _smoothed_likehood(X_train, y_train, column_by, unique_value, alpha=1):
        mask = X_train.loc[:, column_by] == unique_value
        mn = y_train[mask].mean()
        nrows = mask.sum()
        global_mean = y_train.mean()
        smoothed_likehood = (mn*nrows + global_mean*alpha)/(nrows + alpha)
        return smoothed_likehood
    
    @staticmethod
    def _weights_of_evidence(X_train, y_train, column_by, unique_value, alpha=1, good_class=0):
        mask = X_train.loc[:, column_by] == unique_value
        good = (y_train[mask] == good_class).sum()
        bad = (y_train[mask] != good_class).sum()
        woe = np.log(np.float32(good + alpha)/np.float32(bad + alpha))
        return woe
    
    @staticmethod
    def _difference_in_counts(X_train, y_train, column_by, unique_value, alpha=1, good_class=0):
        mask = X_train.loc[:, column_by] == unique_value
        good = (y_train[mask] == good_class).sum()
        bad = (y_train[mask] != good_class).sum()
        dic = np.float32(good - bad)/alpha
        return dic      
    
    def fit_transform(self, X, y, column_by, verbose=False, split_by=None, **fit_params):
        return self._transformer.fit_transform(X=X, y=y, column_by=column_by,
                                               verbose=verbose, split_by=split_by, **fit_params)
        
    def transform(self, X_train, y_train, X_target, column_by):
        return self._transformer.transform(X_train, y_train, X_target, column_by=column_by)
    
    
class SmoothedLikehoodProcessor:
    def __init__(self, cv=5, shuffle=False, random_state=0, kf=None):
        self._preproc = BaseCategoryProcessor('smoothed_likehood', cv, shuffle, random_state, kf)

    def fit_transform(self, X, y, column_by, verbose=False, split_by=None, **fit_params):
        return self._preproc.fit_transform(X=X, y=y, column_by=column_by, verbose=verbose, split_by=split_by, **fit_params)

    def transform(self, X_train, y_train, X_target, column_by):
        return self._preproc.transform(X_train, y_train, X_target, column_by=column_by)
    
    
class WeightsOfEvidenceProcessor:
    def __init__(self, cv=5, shuffle=False, random_state=0, kf=None):
        self._preproc = BaseCategoryProcessor('weights_of_evidence', cv, shuffle, random_state, kf)

    def fit_transform(self, X, y, column_by, verbose=False, split_by=None, **fit_params):
        return self._preproc.fit_transform(X=X, y=y, column_by=column_by, verbose=verbose, split_by=split_by, **fit_params)

    def transform(self, X_train, y_train, X_target, column_by):
        return self._preproc.transform(X_train, y_train, X_target, column_by=column_by)
    
    
class DifferenceCountsProcessor:
    def __init__(self, cv=5, shuffle=False, random_state=0, kf=None):
        self._preproc = BaseCategoryProcessor('difference_in_counts', cv, shuffle, random_state, kf)

    def fit_transform(self, X, y, column_by, verbose=False, split_by=None, **fit_params):
        return self._preproc.fit_transform(X, y, column_by=column_by,
                                           verbose=verbose, split_by=split_by, **fit_params)

    def transform(self, X_train, y_train, X_target, column_by):
        return self._preproc.transform(X_train, y_train, X_target, column_by=column_by)
