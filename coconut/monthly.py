"""Monthly yield prediction using calendar features and historical lags."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from .constants import MONTH_FEATURES, MONTH_METHODS
from .metrics import scores
from .preprocessing import monthly_preprocessor, regression_weights

def monthly_frame(d):
    z=d.sort_values(['barangay','date']).reset_index(drop=True).copy()
    z['year_index_m']=z.year-z.year.min()
    z['month_sin']=np.sin(2*np.pi*z.month/12);z['month_cos']=np.cos(2*np.pi*z.month/12)
    for source,lag,name in [('yield_kg_ha_month',12,'yield_lag12'),('yield_kg_ha_month',24,'yield_lag24'),
       ('rain_mm',12,'rain_lag12'),('temp_c',12,'temp_lag12')]:
        z[name]=z.groupby('barangay')[source].shift(lag)
    return z

@dataclass
class MonthlyModel:
    name:str
    fitted:object
    fallback:object=None
    def predict(self,x):
        if self.name=='Seasonal median':
            p=[self.fitted.get((str(b),int(m)),self.fallback.get(int(m),np.nan)) for b,m in zip(x.barangay,x.month)]
        elif self.name=='Harmonic regression':p=self.fitted.predict(x[['month_sin','month_cos']])
        else:p=self.fitted.predict(x[['barangay']+MONTH_FEATURES])
        p=np.asarray(p,dtype=float)
        if not np.isfinite(p).all():raise ValueError('Model returned missing predictions.')
        return np.maximum(0,p)

def fit_monthly(name,train,trees=100):
    y=train.yield_kg_ha_month
    if name=='Seasonal median':return MonthlyModel(name,train.groupby(['barangay','month']).yield_kg_ha_month.median().to_dict(),train.groupby('month').yield_kg_ha_month.median().to_dict())
    if name=='Harmonic regression':return MonthlyModel(name,LinearRegression().fit(train[['month_sin','month_cos']],y))
    methods={
      'Ridge regression':Ridge(alpha=1.0),
      'Decision tree':DecisionTreeRegressor(max_depth=6,min_samples_leaf=8,random_state=42),
      'Random forest':RandomForestRegressor(n_estimators=trees,max_depth=10,min_samples_leaf=4,n_jobs=1,random_state=42),
      'Random forest (yield-range weighted)':RandomForestRegressor(n_estimators=trees,max_depth=10,min_samples_leaf=4,n_jobs=1,random_state=42),
      'Extra trees':ExtraTreesRegressor(n_estimators=trees,min_samples_leaf=4,n_jobs=1,random_state=42),
      'Gradient boosting':HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,l2_regularization=1.0,early_stopping=False,random_state=42)}
    if name not in methods:raise ValueError('Unknown monthly method: '+name)
    model=Pipeline([('prepare',monthly_preprocessor(name=='Ridge regression')),('regressor',methods[name])])
    kwargs={}
    if name=='Random forest (yield-range weighted)':kwargs['regressor__sample_weight']=regression_weights(y)[0]
    model.fit(train[['barangay']+MONTH_FEATURES],y,**kwargs)
    return MonthlyModel(name,model)


class MonthlyMixin:
    def fit_monthly_models(self,names=None,trees=100,progress=None):
        self.require_data();d=self.monthly;last=int(d.year.max());names=list(MONTH_METHODS if names is None else names)
        if not names:raise ValueError('Select at least one monthly model.')
        folds=[last-3,last-2,last-1,last];rows=[];predictions=[];errors=[]
        total=len(names)*len(folds);done=0
        for year in folds:
            train=d[(d.year<year)&d.yield_kg_ha_month.notna()]
            test=d[(d.year==year)&d.yield_kg_ha_month.notna()]
            if train.date.nunique()<24 or test.empty:raise ValueError('Need at least six calendar years and observed targets in each recent year.')
            lo,hi=train.yield_kg_ha_month.quantile([.2,.8])
            for name in names:
                if progress:progress(done,total,f'Monthly {name}: year {year}')
                try:
                    model=fit_monthly(name,train,trees);p=model.predict(test)
                    metric=scores(test.yield_kg_ha_month,p)
                    tail=(test.yield_kg_ha_month<=lo)|(test.yield_kg_ha_month>=hi)
                    metric['tail_mae']=float(mean_absolute_error(test.loc[tail,'yield_kg_ha_month'],p[tail.to_numpy()])) if tail.any() else np.nan
                    rows.append(dict(model=name,year=year,split='final test' if year==last else 'validation',**metric))
                    q=test[['date','year','barangay','yield_kg_ha_month']].rename(columns={'yield_kg_ha_month':'actual_yield'}).copy()
                    q['predicted_yield']=p;q['model']=name;q['split']='final test' if year==last else 'validation';predictions.append(q)
                except Exception as exc:errors.append(dict(task='monthly prediction',model=name,year=year,error=str(exc)))
                done+=1
        r=pd.DataFrame(rows)
        if r.empty:raise ValueError('Monthly models failed.')
        validation=r[r.split=='validation'].groupby('model').agg(mae=('mae','mean'),folds=('mae','size'))
        valid=validation[(validation.folds==3)&validation.index.isin(r[r.split=='final test'].model)]
        if valid.empty:raise ValueError('No monthly model completed every fold.')
        selected=valid.mae.idxmin()
        if 'Seasonal median' in valid.index and valid.loc['Seasonal median','mae']<=valid.loc[selected,'mae']+.01:selected='Seasonal median'
        models={name:fit_monthly(name,d[d.yield_kg_ha_month.notna()],trees) for name in [selected]}
        # Save the actual final-test TRAIN weights for an auditable imbalance report.
        train=d[(d.year<last)&d.yield_kg_ha_month.notna()]
        weights,edges,report=regression_weights(train.yield_kg_ha_month)
        self.training_weights=train[['date','barangay','yield_kg_ha_month']].copy();self.training_weights['sample_weight']=weights
        self.weight_report=report;self.monthly_models=models;self.monthly_winner=selected
        self.monthly_comparison=r;self.monthly_predictions=pd.concat(predictions,ignore_index=True);self.errors.extend(errors)
        self.mining_settings.update(monthly_test_year=last,monthly_validation_years=folds[:-1],monthly_trees=trees,
            monthly_selected=selected,regression_weight_edges=edges.tolist(),
            regression_weights='Inverse training frequency across five equal-width target ranges, clipped to [0.33,3] then normalized to mean 1.')
        if progress:progress(total,total,'Monthly model comparison complete.')
        return r

    def predict_next_three_months(self,barangay=None):
        if not self.monthly_winner:raise ValueError('Run monthly prediction first.')
        places=[barangay] if barangay else self.audit['barangays'];rows=[];dates=pd.date_range(self.monthly.date.max()+pd.DateOffset(months=1),periods=3,freq='MS')
        for b in places:
            g=self.monthly[self.monthly.barangay==b].set_index('date')
            x=pd.DataFrame({'date':dates,'year':dates.year,'month':dates.month,'barangay':b})
            x['year_index_m']=x.year-self.monthly.year.min();x['month_sin']=np.sin(2*np.pi*x.month/12);x['month_cos']=np.cos(2*np.pi*x.month/12)
            for col,source,lag in [('yield_lag12','yield_kg_ha_month',12),('yield_lag24','yield_kg_ha_month',24),('rain_lag12','rain_mm',12),('temp_lag12','temp_c',12)]:
                x[col]=g[source].reindex(dates-pd.DateOffset(months=lag)).to_numpy()
            p=self.monthly_models[self.monthly_winner].predict(x)
            areas=g.area_ha.dropna();area=float(areas.iloc[-1]) if len(areas) else np.nan
            x['predicted_yield_kg_ha_month']=p;x['assumed_area_ha']=area;x['predicted_harvest_kg']=p*area
            x['missing_lag_inputs']=x[MONTH_FEATURES].isna().sum(axis=1);x['model']=self.monthly_winner
            rows.append(x[['date','barangay','predicted_yield_kg_ha_month','assumed_area_ha','predicted_harvest_kg','missing_lag_inputs','model']])
        self.three_month_predictions=pd.concat(rows,ignore_index=True)
        return self.three_month_predictions

    def monthly_model_reliance(self,model_name='Random forest',repeats=3):
        if self.monthly_comparison.empty:raise ValueError('Run monthly prediction first.')
        year=int(self.mining_settings['monthly_test_year']);d=self.monthly
        train=d[(d.year<year)&d.yield_kg_ha_month.notna()];test=d[(d.year==year)&d.yield_kg_ha_month.notna()].copy()
        model=fit_monthly(model_name,train,self.mining_settings['monthly_trees'])
        base=mean_absolute_error(test.yield_kg_ha_month,model.predict(test));rng=np.random.default_rng(42)
        groups={'Calendar month':['month','month_sin','month_cos'],'Year':['year_index_m'],'Barangay':['barangay'],
          'Past yield':['yield_lag12','yield_lag24'],'Past rainfall':['rain_lag12'],'Past temperature':['temp_lag12']}
        rows=[]
        for group,cols in groups.items():
            effect=[]
            for _ in range(repeats):
                q=test.copy();q[cols]=test[cols].iloc[rng.permutation(len(test))].to_numpy()
                effect.append(mean_absolute_error(test.yield_kg_ha_month,model.predict(q))-base)
            rows.append(dict(group=group,increase_mae_kg_ha=float(np.mean(effect))))
        self.monthly_reliance=pd.DataFrame(rows).sort_values('increase_mae_kg_ha',ascending=False)
        return self.monthly_reliance

