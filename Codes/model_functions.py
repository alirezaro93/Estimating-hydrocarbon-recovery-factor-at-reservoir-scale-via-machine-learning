#%%

import xgboost as xgb

import numpy as np

import pandas as pd

from joblib import Parallel, delayed

from scipy.interpolate import interp1d

from scipy.special import erf, erfinv

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils.validation import FLOAT_DTYPES, check_array, check_is_fitted

import shap

import operator

from matplotlib import pyplot as plt

#%%

def ModelConstructor(dtrain ,dtest, c, params):
    
    num_boost_round = 999
    
    min_rmse = float("Inf")
    best_params = None
    
    for eta in [0.3, 0.2, 0.1, 0.05, 0.01, 0.005]:

        params['eta'] = eta
        
        cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                seed=42,
                nfold=10,
                metrics=['rmse'],
                early_stopping_rounds=10
              )
    
        mean_rmse = cv_results['test-rmse-mean'].min()

        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            best_params = eta
    
    params['eta'] = best_params
    
    print('')
    print("Best parameter: eta = {}, RMSE = {}".format(best_params, min_rmse))
    print('')
    
    gridsearch_params = [
        (max_depth, min_child_weight)
        for max_depth in range(1,12)
        for min_child_weight in range(1,11)
    ]
    
    min_rmse = float("Inf")
    best_params = None
    
    for max_depth, min_child_weight in gridsearch_params:
        
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=10,
            metrics={'rmse'},
            early_stopping_rounds=10
        )
    
        mean_rmse = cv_results['test-rmse-mean'].min()

        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            best_params = (max_depth,min_child_weight)
    
    params['max_depth'] = best_params[0]
    params['min_child_weight'] = best_params[1]
    
    print('')
    print("Best parameters: max_depth = {}, min_child_weight = {}, \
RMSE = {}".format(best_params[0], best_params[1], min_rmse))
    print('')
    
    gridsearch_params = [
        (subsample, colsample_bytree)
        for subsample in [i/10. for i in range(1,11)]
        for colsample_bytree in [i/10. for i in range(1,11)]
    ]
    
    min_rmse = float("Inf")
    best_params = None
    
    for subsample, colsample_bytree in reversed(gridsearch_params):

        params['subsample'] = subsample
        params['colsample_bytree'] = colsample_bytree
        
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=10,
            metrics={'rmse'},
            early_stopping_rounds=10
        )
    
        mean_rmse = cv_results['test-rmse-mean'].min()
        
        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            best_params = (subsample,colsample_bytree)
    
    params['subsample'] = best_params[0]
    params['colsample_bytree'] = best_params[1]
    
    print('')
    print("Best parameters: subsample = {}, colsample_bytree= {}, \
RMSE = {}".format(best_params[0], best_params[1], min_rmse))
    print('')
    
    gridsearch_params = [
        (max_delta_step, colsample_bylevel)
        for max_delta_step in [i/10. for i in range(1,11)]
        for colsample_bylevel in [i/10. for i in range(1,11)]
    ]
    
    min_rmse = float("Inf")
    best_params = None
    
    for max_delta_step, colsample_bylevel in reversed(gridsearch_params):

        params['max_delta_step'] = max_delta_step
        params['colsample_bylevel'] = colsample_bylevel
        
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=10,
            metrics={'rmse'},
            early_stopping_rounds=10
        )
    
        mean_rmse = cv_results['test-rmse-mean'].min()

        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            best_params = (max_delta_step,colsample_bylevel)
    
    params['max_delta_step'] = best_params[0]
    params['colsample_bylevel'] = best_params[1]
    
    print('')
    print("Best parameters: max_delta_step = {}, colsample_bylevel = {}, \
RMSE = {}".format(best_params[0], best_params[1], min_rmse))
    print('')   
    
    gridsearch_params = [
        (ralpha, rlambda)
        for ralpha in [i/10. for i in range(1,10)]
        for rlambda in [i/100. for i in range(1,10)]
    ]
    
    min_rmse = float("Inf")
    best_params = None
    
    for ralpha, rlambda in reversed(gridsearch_params):

        params['alpha'] = ralpha
        params['lambda'] = rlambda
        
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=10,
            metrics={'rmse'},
            early_stopping_rounds=10
        )
    
        mean_rmse = cv_results['test-rmse-mean'].min()
        
        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            best_params = (ralpha,rlambda)
    
    params['alpha'] = best_params[0]
    params['lambda'] = best_params[1]
    
    print('')
    print("Best parameters: alpha = {}, lambda = {}, \
RMSE = {}".format(best_params[0], best_params[1], min_rmse))
    print('') 
    
    gridsearch_params = [
        (gamma)
        for gamma in [i/100. for i in range(1,10)]
    ]
    
    min_rmse = float("Inf")
    best_params = None
    
    for gamma in reversed(gridsearch_params):

        params['gamma'] = gamma
        
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=10,
            metrics={'rmse'},
            early_stopping_rounds=10
        )
    
        mean_rmse = cv_results['test-rmse-mean'].min()
        
        if mean_rmse < min_rmse:
            min_rmse = mean_rmse
            best_params = (gamma)
    
    params['gamma'] = best_params
    
    print('')
    print("Best parameter: gama = {}, RMSE = {}".format(best_params, min_rmse))
    print('')
    
    best_model = xgb.train(
        params,
        dtrain,
        num_boost_round=999,
        evals=[(dtest, "Test")],
        verbose_eval = False
    )
    
    best_model.save_model('best_model_cluster_{}.model'.format(c))

    return print(''), print("Cluster {} was trained".format(c))

#%%

def ModeCalculator(df):
    mo = df.mode()
    if mo.shape[0] == 1:
        zmo=mo[0]
    else:
        zmo = mo.mean(axis=0)
    return zmo

#%%

def MissingValueHandler(i_start, i_end, df):
    
    t = 0.10
    
    temp = df[i_start:i_end]
    fr = temp.isnull().sum()/temp.shape[0]
    if fr > t:
        while fr > t:
            if df.shape[0] - i_end != 0:
                temp = df[i_start:i_end+1]
                i_end += 1
                fr = temp.isnull().sum()/temp.shape[0]
            zmo = ModeCalculator(temp)
            temp = temp.replace(np.nan, zmo)
            return temp
        else:
                zmo = ModeCalculator(temp)
                temp = temp.replace(np.nan, zmo)
                return temp
    elif fr <= t:
        zmo = ModeCalculator(temp)
        temp = temp.replace(np.nan, zmo)
        return temp
    elif fr == 0:
        return temp

#%%

def Imputer(dfnum, col):

    n = 6000
    
    my_col = dfnum[col]
    temp_df = pd.DataFrame()
    i_start = 0
    i_end = n
    
    while my_col.shape[0] - i_start >= n:
            curated_temp = MissingValueHandler(i_start, i_end, my_col)
            temp_df = temp_df.append(curated_temp.to_frame())
            i_start = i_start+curated_temp.shape[0]
            i_end = i_start + n
            
    last_piece = MissingValueHandler(i_start, my_col.shape[0], my_col)
    
    if last_piece is None:
        dfnum[col+"_curated1"] = temp_df
        dfnum=dfnum.drop([col], axis=1)


    else:
        temp_df = temp_df.append(last_piece.to_frame())
        dfnum[col+"_curated2"] = temp_df
        dfnum = dfnum.drop([col], axis=1)

    pass

#%%
    
class GaussRankScaler(BaseEstimator, TransformerMixin):
    """Transform features by scaling each feature to a normal distribution.
    Parameters
        ----------
        epsilon : float, optional, default 1e-4
            A small amount added to the lower bound or subtracted
            from the upper bound. This value prevents infinite number
            from occurring when applying the inverse error function.
        copy : boolean, optional, default True
            If False, try to avoid a copy and do inplace scaling instead.
            This is not guaranteed to always work inplace; e.g. if the data is
            not a NumPy array, a copy may still be returned.
        n_jobs : int or None, optional, default None
            Number of jobs to run in parallel.
            ``None`` means 1 and ``-1`` means using all processors.
        interp_kind : str or int, optional, default 'linear'
           Specifies the kind of interpolation as a string
            ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
            'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic'
            refer to a spline interpolation of zeroth, first, second or third
            order; 'previous' and 'next' simply return the previous or next value
            of the point) or as an integer specifying the order of the spline
            interpolator to use.
        interp_copy : bool, optional, default False
            If True, the interpolation function makes internal copies of x and y.
            If False, references to `x` and `y` are used.
        Attributes
        ----------
        interp_func_ : list
            The interpolation function for each feature in the training set.
        """

    def __init__(self, epsilon=1e-4, copy=True, n_jobs=None, interp_kind='linear', interp_copy=False):
        self.epsilon = epsilon
        self.copy = copy
        self.interp_kind = interp_kind
        self.interp_copy = interp_copy
        self.fill_value = 'extrapolate'
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit interpolation function to link rank with original data for future scaling
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data used to fit interpolation function for later scaling along the features axis.
        y
            Ignored
        """
        X = check_array(X, copy=self.copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite=True)

        self.interp_func_ = Parallel(n_jobs=self.n_jobs)(delayed(self._fit)(x) for x in X.T)
        return self

    def _fit(self, x):
        x = self.drop_duplicates(x)
        rank = np.argsort(np.argsort(x))
        bound = 1.0 - self.epsilon
        factor = np.max(rank) / 2.0 * bound
        scaled_rank = np.clip(rank / factor - bound, -bound, bound)
        return interp1d(
            x, scaled_rank, kind=self.interp_kind, copy=self.interp_copy, fill_value=self.fill_value)

    def transform(self, X, copy=None):
        """Scale the data with the Gauss Rank algorithm
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
        """
        check_is_fitted(self, 'interp_func_')

        copy = copy if copy is not None else self.copy
        X = check_array(X, copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite=True)

        X = np.array(Parallel(n_jobs=self.n_jobs)(delayed(self._transform)(i, x) for i, x in enumerate(X.T))).T
        return X

    def _transform(self, i, x):
        return erfinv(self.interp_func_[i](x))

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
        """
        check_is_fitted(self, 'interp_func_')

        copy = copy if copy is not None else self.copy
        X = check_array(X, copy=copy, estimator=self, dtype=FLOAT_DTYPES, force_all_finite=True)

        X = np.array(Parallel(n_jobs=self.n_jobs)(delayed(self._inverse_transform)(i, x) for i, x in enumerate(X.T))).T
        return X

    def _inverse_transform(self, i, x):
        inv_interp_func = interp1d(self.interp_func_[i].y, self.interp_func_[i].x, kind=self.interp_kind,
                                   copy=self.interp_copy, fill_value=self.fill_value)
        return inv_interp_func(erf(x))

    @staticmethod
    def drop_duplicates(x):
        is_unique = np.zeros_like(x, dtype=bool)
        is_unique[np.unique(x, return_index=True)[1]] = True
        return x[is_unique]

#%%

def SHAP_Feature_Importance(best_model, X, y, df_column_names, c,
                            dict_important_features):
    
    explainer = shap.TreeExplainer(best_model,
                                   feature_perturbation='tree_path_dependent')
    
    shap_values = explainer.shap_values(X, y)
    
    mean_abs_shap = np.absolute(shap_values).mean(axis=0)
    
    mean_abs_shap1 = mean_abs_shap.tolist()
    
    maxshap = max(mean_abs_shap1)
    
    mean_abs_shaprefined = mean_abs_shap[mean_abs_shap!=0]
    
    mean_abs_shap2 = dict(zip(df_column_names, mean_abs_shap1))
    
    mean_abs_shap2refined = {k: v for k, v in mean_abs_shap2.items() if v != 0}
    
    mean_abs_shap3 = sorted(mean_abs_shap2refined.items(),
                            key = operator.itemgetter(1))
    
    mean_abs_shap3 = pd.DataFrame(mean_abs_shap3,
                                  columns = ['Feature', 'Mean SHAP Value'])
    
    top10 = mean_abs_shap3.tail(10)
    
    plt.figure(figsize = (8, 4))
    plt.bar([x for x in range(len(mean_abs_shaprefined))],
            mean_abs_shaprefined)
    plt.ylim([0, maxshap])
    plt.autoscale('y')
    plt.xlabel('Features')
    plt.ylabel('Mean SHAP Value')
    plt.title('SHAP Feature Importance (Cluster {})'.format(c))
    plt.show()
    
    plt.figure(figsize = (9, 5))
    top10.plot.barh(x='Feature', y='Mean SHAP Value')
    plt.xlim([0, maxshap])
    plt.autoscale('y')
    plt.xlabel('Mean SHAP Value')
    plt.ylabel('Features')
    plt.title('Top 10 Most Important Features (Cluster {})'.format(c))
    plt.show()
    
    dict_important_features['cluster_{}_important_feature'.format(c)] = mean_abs_shap3
    
    return top10, print(''), print('Cluster {} Feature Importance is done'.format(c))
    