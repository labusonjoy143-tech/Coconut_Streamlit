"""Lagged annual models with recursive, chronological forecast evaluation."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from .constants import FEATURES, ANNUAL_MODELS, FACTOR_GROUPS
from .metrics import safe_median, low_variation, scores
from .data import add_annual_lags

@dataclass
class AnnualModel:
    name:str
    model:object
    fallback:float=np.nan
    def predict(self,x):
        if self.name in ['Historical median','Last observed yield']:
            p=[self.model.get(str(b),self.fallback) for b in x.barangay]
        else:p=self.model.predict(x[['barangay']+FEATURES])
        p=np.asarray(p,dtype=float)
        if not np.isfinite(p).all():raise ValueError('A model returned unavailable predictions.')
        return np.maximum(0,p)

def fit_annual_model(name,train,trees=120):
    d=train[train.annual_yield_kg_ha.notna()].copy()
    if len(d)<20:raise ValueError('At least 20 complete annual targets are required in training.')
    if name=='Historical median':
        return AnnualModel(name,d.groupby('barangay').annual_yield_kg_ha.median().to_dict(),float(d.annual_yield_kg_ha.median()))
    if name=='Last observed yield':
        return AnnualModel(name,d.sort_values('year').groupby('barangay').annual_yield_kg_ha.last().to_dict(),float(d.annual_yield_kg_ha.median()))
    methods={
        'Ridge regression':Ridge(alpha=10.0),
        'Decision tree':DecisionTreeRegressor(max_depth=4,min_samples_leaf=6,random_state=42),
        'Random forest':RandomForestRegressor(n_estimators=trees,max_depth=6,min_samples_leaf=4,n_jobs=1,random_state=42),
        'Extra trees':ExtraTreesRegressor(n_estimators=trees,max_depth=6,min_samples_leaf=4,n_jobs=1,random_state=42),
        'Gradient boosting':HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=7,l2_regularization=1,
            early_stopping=False,random_state=42),
    }
    if name not in methods:raise ValueError('Unknown model: '+name)
    try:enc=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    except TypeError:enc=OneHotEncoder(handle_unknown='ignore',sparse=False)
    imp=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)
    numeric=Pipeline([('impute',imp),('scale',StandardScaler())]) if name=='Ridge regression' else imp
    pre=ColumnTransformer([('numbers',numeric,FEATURES),('barangay',enc,['barangay'])])
    model=Pipeline([('prepare',pre),('regressor',methods[name])]).fit(d[['barangay']+FEATURES],d.annual_yield_kg_ha)
    return AnnualModel(name,model)

def annual_projection(model,history,horizon,area_growth_pct=0):
    """Recursive forecast using only records at or before the issue year.

    Future climate/inflation inputs use a fixed median of the latest three available
    annual measurements. These are assumptions; no future observed drivers are used.
    """
    hist=history.copy();origin=int(hist.year.max());first_year=int(hist.year.min())
    places=sorted(hist.barangay.unique());output=[]
    climate={};area={};last_complete={}
    for b,g in hist.groupby('barangay'):
        climate[b]={c:safe_median(g.sort_values('year')[c].dropna().tail(3)) for c in ['rainfall_annual_mm','temperature_mean_c','inflation_mean_pct']}
        avail=g[g.annual_area_ha.notna()].sort_values('year')
        area[b]=float(avail.annual_area_ha.iloc[-1]) if len(avail) else np.nan
        complete=g[g.annual_yield_kg_ha.notna()]
        last_complete[b]=int(complete.year.max()) if len(complete) else None
    for step in range(1,int(horizon)+1):
        year=origin+step
        rows=[]
        for b in places:
            rows.append(dict(year=year,barangay=b,annual_yield_kg_ha=np.nan,
              annual_area_ha=area[b]*(1+area_growth_pct/100)**step,
              **climate[b]))
        future=pd.DataFrame(rows)
        combined=pd.concat([hist,future],ignore_index=True)
        combined=add_annual_lags(combined,first_year)
        x=combined[combined.year==year].copy()
        pred=model.predict(x)
        x['annual_yield_kg_ha']=pred
        for i,(_,row) in enumerate(x.iterrows()):
            output.append(dict(year=year,barangay=row.barangay,predicted_yield_kg_ha=pred[i],
              assumed_area_ha=row.annual_area_ha,predicted_production_kg=pred[i]*row.annual_area_ha,
              model=model.name,origin_year=origin,horizon=step,last_complete_yield_year=last_complete[row.barangay],
              area_growth_pct=area_growth_pct,missing_lag_inputs=int(row[FEATURES].isna().sum()),
              driver_basis='Future weather/inflation: fixed median of latest three available annual measurements; area: user assumption.'))
        # Add predicted targets for later horizons; original missing targets are preserved.
        hist=pd.concat([hist,x[hist.columns]],ignore_index=True)
    return pd.DataFrame(output)

class AnnualMixin:
    def reset_models(self):
        self.comparison=pd.DataFrame();self.test_predictions=pd.DataFrame();self.fitted={}
        self.winner=None;self.future=pd.DataFrame();self.config={};self.errors=[];self.reliance=pd.DataFrame()

    def compare_models(self,horizon=3,names=None,trees=120,progress=None):
        self.require_data();a=self.annual;h=int(horizon)
        names=list(ANNUAL_MODELS if names is None else names)
        if not names:raise ValueError('Select at least one model.')
        if not 1<=h<=5:raise ValueError('Use a horizon from one to five years.')
        years=sorted(a.year.unique());last=int(max(years))
        origins=[last-3*h,last-2*h,last-h]
        if origins[0]-int(min(years))+1<5:raise ValueError('Insufficient history for three backtest blocks. Reduce Years ahead.')
        rows=[];predictions=[];errors=[];done=0;total=len(names)*3
        for k,origin in enumerate(origins):
            hist=a[a.year<=origin].copy();actual=a[a.year.between(origin+1,origin+h)]
            for name in names:
                if progress:progress(done,total,f'{name}: origin {origin}, {h}-year horizon')
                try:
                    fitted=fit_annual_model(name,hist,trees)
                    p=annual_projection(fitted,hist,h)
                    p=p.merge(actual[['year','barangay','annual_yield_kg_ha']],on=['year','barangay'],how='left').rename(columns={'annual_yield_kg_ha':'actual_yield_kg_ha'})
                    p['split']='final test' if k==2 else 'validation'
                    rows.append(dict(model=name,origin_year=origin,split=p['split'].iloc[0],**scores(p.actual_yield_kg_ha,p.predicted_yield_kg_ha)))
                    predictions.append(p)
                except Exception as exc:errors.append(dict(model=name,origin_year=origin,error=str(exc)))
                done+=1
        r=pd.DataFrame(rows)
        if r.empty:raise ValueError('No model completed evaluation. Check annual target completeness.')
        v=r[r.split=='validation'].groupby('model').agg(mae=('mae','mean'),folds=('mae','size'))
        valid=v[(v.folds==2)&v.index.isin(r[r.split=='final test'].model)]
        if valid.empty:raise ValueError('No model completed all validation and final-test blocks.')
        winner=valid.mae.idxmin()
        for simple in ['Historical median','Last observed yield']:
            if simple in valid.index and valid.loc[simple,'mae']<=valid.loc[winner,'mae']+.1:
                winner=simple;break
        fitted={}
        for name in [winner]:
            try:fitted[name]=fit_annual_model(name,a,trees)
            except Exception as exc:errors.append(dict(model=name,origin_year=last,error='Full-history refit: '+str(exc)))
        if winner not in fitted:raise ValueError('Selected model could not be refitted for future predictions.')
        self.comparison=r;self.test_predictions=pd.concat(predictions,ignore_index=True);self.fitted=fitted;self.winner=winner;self.errors=errors
        self.config=dict(horizon=h,validation_origins=origins[:2],test_origin=origins[-1],forecast_origin=last,
            model_selection='Mean validation MAE. Simplicity tie tolerance: 0.1 kg/ha/year.',trees=trees,
            predictor_policy='Only lagged annual inputs. Missing predictors imputed inside each training pipeline.',
            forecast_policy='Recursive multi-year validation; future climate/inflation use fixed medians of the latest three available annual measurements.')
        self.future=annual_projection(fitted[winner],a,h)
        if progress:progress(total,total,'Annual comparison and future predictions complete.')
        return r

    def make_scenario(self,area_growth_pct=0,stress_pct=0):
        if not self.winner:raise ValueError('Run annual model comparison first.')
        if not -20<=area_growth_pct<=20 or not -30<=stress_pct<=30:raise ValueError('Scenario exceeds the supported planning control range.')
        p=annual_projection(self.fitted[self.winner],self.annual,self.config['horizon'],area_growth_pct)
        p['planning_production_kg']=p.predicted_production_kg*(1+stress_pct/100)
        p['planning_stress_pct']=stress_pct
        return p

    def model_reliance(self,model_name='Random forest',year=None,repeats=3):
        self.require_data();year=int(year or self.annual.year.max())
        past=self.annual[self.annual.year<year];origin=int(past.year.max())
        actual=self.annual[(self.annual.year==year)&self.annual.annual_yield_kg_ha.notna()]
        if not len(actual):raise ValueError('No complete annual targets in this selected year.')
        if low_variation(actual.annual_yield_kg_ha):
            self.reliance=pd.DataFrame([dict(year=year,feature_group='All factors',increase_mae_kg_ha=np.nan,
                interpretation='Annual yield varies too little within this year for informative factor ranking.')])
            return self.reliance
        model=fit_annual_model(model_name,past,self.config.get('trees',120))
        if model_name in ['Historical median','Last observed yield']:
            self.reliance=pd.DataFrame([dict(year=year,feature_group='Historical yield baseline',increase_mae_kg_ha=np.nan,
                interpretation='This baseline does not estimate separate weather or area effects.')])
            return self.reliance
        # Construct the same first-horizon inputs available at the previous-year origin.
        x=add_annual_lags(self.annual[self.annual.year<=year],int(self.annual.year.min()))
        x=x[(x.year==year)&x.barangay.isin(actual.barangay)].sort_values('barangay')
        y=actual.set_index('barangay').annual_yield_kg_ha.reindex(x.barangay).to_numpy()
        base=mean_absolute_error(y,model.predict(x));rng=np.random.default_rng(42);rows=[]
        for label,cols in FACTOR_GROUPS.items():
            effects=[]
            for _ in range(repeats):
                shuffled=x.copy();order=rng.permutation(len(x));shuffled[cols]=x[cols].iloc[order].to_numpy()
                effects.append(mean_absolute_error(y,model.predict(shuffled))-base)
            rows.append(dict(year=year,feature_group=label,increase_mae_kg_ha=float(np.mean(effects)),
                interpretation='Held-out-year model reliance; association, not causation.'))
        self.reliance=pd.DataFrame(rows)
        return self.reliance.sort_values('increase_mae_kg_ha',ascending=False)

