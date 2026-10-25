"""Shared scikit-learn preprocessing; fitted only on each training subset."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from .constants import MONTH_FEATURES

def regression_weights(y):
    """Equal-width target-range density weights fitted on training only."""
    values=np.asarray(y,dtype=float)
    edges=np.linspace(values.min(),values.max(),6)[1:-1] if values.max()>values.min() else np.array([])
    bins=np.searchsorted(edges,values,side='right')
    counts=pd.Series(bins).value_counts().sort_index()
    raw=np.array([len(values)/(len(counts)*counts.loc[b]) for b in bins])
    weights=np.clip(raw,.33,3.0);weights/=weights.mean()
    report=pd.DataFrame({'target_bin':bins,'weight':weights}).groupby('target_bin').agg(
        training_rows=('weight','size'),sample_weight=('weight','mean')).reset_index()
    return weights,edges,report

def monthly_preprocessor(scale=False):
    steps=[('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True))]
    if scale:steps.append(('scale',StandardScaler()))
    try:enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    except TypeError:enc=OneHotEncoder(handle_unknown='ignore',sparse=False)
    return ColumnTransformer([('numeric',Pipeline(steps),MONTH_FEATURES),('barangay',enc,['barangay'])])
