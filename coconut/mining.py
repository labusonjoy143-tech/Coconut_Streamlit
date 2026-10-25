"""Training-only class weighting, descriptive profiles, and record screening."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, balanced_accuracy_score, f1_score,
    average_precision_score, confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from .constants import MONTH_FEATURES
from .preprocessing import monthly_preprocessor

class MiningMixin:
    def fit_low_yield_classification(self,trees=100):
        self.require_data();d=self.monthly;last=int(d.year.max());rows=[];predictions=[];details={}
        methods=['Balanced logistic regression','Balanced random forest']
        for year in [last-2,last-1,last]:
            train=d[(d.year<year)&d.yield_kg_ha_month.notna()];test=d[(d.year==year)&d.yield_kg_ha_month.notna()]
            threshold=float(train.yield_kg_ha_month.quantile(.2))
            y=(train.yield_kg_ha_month<=threshold).astype(int);yt=(test.yield_kg_ha_month<=threshold).astype(int)
            if y.nunique()<2:raise ValueError('The training targets do not support two low-yield categories.')
            classes=np.array([0,1]);class_weights=compute_class_weight(class_weight='balanced',classes=classes,y=y)
            for name in methods:
                clf=LogisticRegression(class_weight='balanced',max_iter=1500,random_state=42) if name.startswith('Balanced logistic') else RandomForestClassifier(n_estimators=trees,max_depth=8,min_samples_leaf=4,class_weight='balanced',n_jobs=1,random_state=42)
                model=Pipeline([('prepare',monthly_preprocessor(name.startswith('Balanced logistic'))),('classifier',clf)])
                model.fit(train[['barangay']+MONTH_FEATURES],y)
                pred=model.predict(test[['barangay']+MONTH_FEATURES]);prob=model.predict_proba(test[['barangay']+MONTH_FEATURES])[:,1]
                split='final test' if year==last else 'validation'
                rows.append(dict(model=name,year=year,split=split,training_threshold_kg_ha=threshold,
                    balanced_accuracy=float(balanced_accuracy_score(yt,pred)) if yt.nunique()==2 else np.nan,
                    macro_f1=float(f1_score(yt,pred,average='macro',zero_division=0)),
                    average_precision=float(average_precision_score(yt,prob)) if yt.nunique()==2 else np.nan,
                    low_yield_recall=float(((pred==1)&(yt.to_numpy()==1)).sum()/(yt==1).sum()) if (yt==1).any() else np.nan))
                q=test[['date','year','barangay']].copy();q['observed_low_category']=yt.to_numpy();q['predicted_low_category']=pred;q['probability_low_category']=prob;q['model']=name;q['split']=split;predictions.append(q)
                if year==last:
                    details[name]={'confusion_matrix':confusion_matrix(yt,pred,labels=[0,1]).tolist(),
                      'report':classification_report(yt,pred,labels=[0,1],target_names=['Other yield range','At or below training 20th percentile'],zero_division=0,output_dict=True)}
            if year==last:
                self.class_counts=pd.DataFrame(dict(category=['Other yield range','At or below training 20th percentile'],
                  training_count=[int((y==c).sum()) for c in classes],class_weight=class_weights,
                  natural_test_count=[int((yt==c).sum()) for c in classes]))
        metrics=pd.DataFrame(rows);best=metrics[metrics.split=='validation'].groupby('model').macro_f1.mean().idxmax()
        self.classification_metrics=metrics;self.classification_predictions=pd.concat(predictions,ignore_index=True);self.class_details=details
        self.mining_settings.update(classification_selected=best,
            low_yield_definition='At or below the 20th percentile of the fold training yields; a seasonal category, not crop failure.',
            classification_weighting='Balanced class weights fitted on each training subset only. Natural test prevalence is retained.')
        return metrics

    def cluster_barangays(self):
        self.require_data();d=self.monthly
        profiles=d.groupby('barangay').agg(mean_yield_kg_ha_month=('yield_kg_ha_month','mean'),
          median_area_ha=('area_ha','median'),rainfall_mean_mm=('rain_mm','mean'),temperature_mean_c=('temp_c','mean'),
          observed_yield_pairs=('yield_kg_ha_month','count')).reset_index()
        # Capacity/climate profiles: yield is shown as context, not used as a
        # clustering input, because this source has near-identical normalized yields.
        cols=['median_area_ha','rainfall_mean_mm','temperature_mean_c']
        if len(profiles)<4:raise ValueError('At least four barangays are needed for the clustering comparison.')
        prep=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())])
        x=prep.fit_transform(profiles[cols]);candidates=[];labels={}
        for k in range(2,min(5,len(profiles)-1)+1):
            model=KMeans(n_clusters=k,n_init=10,random_state=42);lab=model.fit_predict(x)
            if len(np.unique(lab))<2:continue
            candidates.append(dict(k=k,silhouette=float(silhouette_score(x,lab))));labels[k]=lab
        if not candidates:raise ValueError('The profiles do not support distinct cluster labels.')
        self.cluster_scores=pd.DataFrame(candidates);k=int(self.cluster_scores.sort_values('silhouette',ascending=False).iloc[0].k)
        profiles['cluster']=labels[k]+1;self.clusters=profiles
        self.mining_settings.update(clustering_k=k,clustering_features=cols,
            clustering_scope='Descriptive capacity/climate profiles of all available records; yield is context only. Clusters are not fed to prediction models.')
        return profiles

    def detect_review_candidates(self,training_review_fraction=.03):
        self.require_data();d=self.monthly.copy();last=int(d.year.max());train=d[d.year<last]
        med=train.groupby('month').yield_kg_ha_month.median()
        d['yield_seasonal_residual']=d.yield_kg_ha_month-d.month.map(med)
        cols=['yield_seasonal_residual','area_ha','rain_mm','temp_c']
        prep=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',RobustScaler())])
        fit=d[d.year<last];x=prep.fit_transform(fit[cols])
        model=IsolationForest(n_estimators=150,contamination=float(training_review_fraction),random_state=42,n_jobs=1).fit(x)
        all_x=prep.transform(d[cols]);d['review_candidate']=model.predict(all_x)==-1;d['review_score']=-model.score_samples(all_x)
        d['inputs_missing']=d[cols].isna().sum(axis=1)
        self.anomalies=d[['date','year','barangay','harvest_kg','yield_kg_ha_month','review_candidate','review_score','inputs_missing']].copy()
        self.mining_settings.update(anomaly_training_cutoff=last-1,anomaly_review_fraction=float(training_review_fraction),
            anomaly_interpretation='Data-review candidates, not verified crop damage or pest diagnoses. Review fraction is a chosen training screening rate.')
        return self.anomalies

