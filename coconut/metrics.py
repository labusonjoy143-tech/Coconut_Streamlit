"""Regression metrics and guardrails for almost-constant annual targets."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def safe_median(values):
    known=values.dropna()
    return float(known.median()) if len(known) else np.nan
def low_variation(values):
    x=np.asarray(values,dtype=float);x=x[np.isfinite(x)]
    return len(x)<2 or np.std(x)<max(1.0,abs(np.mean(x))*1e-4)
def scores(actual,predicted):
    a,p=np.asarray(actual,dtype=float),np.asarray(predicted,dtype=float)
    ok=np.isfinite(a)&np.isfinite(p);a,p=a[ok],p[ok]
    if not len(a):raise ValueError('No observed yield targets to score.')
    return dict(n=len(a),mae=float(mean_absolute_error(a,p)),rmse=float(np.sqrt(mean_squared_error(a,p))),
      wape_pct=float(100*np.abs(a-p).sum()/np.abs(a).sum()) if np.abs(a).sum() else np.nan,
      r2=np.nan if low_variation(a) else float(r2_score(a,p)),
      target_variation='Too small for informative R-squared' if low_variation(a) else 'Variable')
