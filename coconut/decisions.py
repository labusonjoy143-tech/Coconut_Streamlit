"""Observed factor associations and matched production-change accounting."""
import numpy as np
import pandas as pd
from .metrics import low_variation

class DecisionMixin:
    def factors_by_year(self):
        """Contemporaneous associations, never labeled causal effects."""
        self.require_data();rows=[]
        for year,g in self.annual.groupby('year'):
            for factor in ['rainfall_annual_mm','temperature_mean_c','annual_area_ha','inflation_mean_pct']:
                v=g[['annual_yield_kg_ha',factor]].dropna()
                reason='Descriptive association only'
                if len(v)<6:reason='Fewer than six complete barangay pairs'
                elif low_variation(v.annual_yield_kg_ha):reason='Annual yield has too little variation to assess this factor'
                elif v[factor].std()<1e-10:reason='Factor has no usable variation in this year'
                pearson=v.iloc[:,0].corr(v.iloc[:,1]) if reason=='Descriptive association only' else np.nan
                spearman=v.iloc[:,0].rank().corr(v.iloc[:,1].rank()) if reason=='Descriptive association only' else np.nan
                rows.append(dict(year=int(year),factor=factor,complete_pairs=len(v),pearson=pearson,spearman=spearman,
                    yield_sd_kg_ha=float(v.annual_yield_kg_ha.std()) if len(v)>1 else np.nan,interpretation=reason))
        return pd.DataFrame(rows)

    def production_change_factors(self):
        """Exact symmetric area/yield decomposition, using matched complete records.

        This is an accounting relationship, not a causal estimate.
        """
        self.require_data();a=self.annual;rows=[]
        for year in sorted(a.year.unique())[1:]:
            now=a[(a.year==year)&a.annual_target_complete]
            before=a[(a.year==year-1)&a.annual_target_complete]
            pair=now.merge(before,on='barangay',suffixes=('_now','_before'))
            if pair.empty:
                rows.append(dict(year=int(year),previous_year=int(year-1),matched_barangays=0,
                    area_component_kg=np.nan,yield_component_kg=np.nan,total_change_kg=np.nan,reconciliation_kg=np.nan))
                continue
            a1,a0=pair.annual_area_ha_now,pair.annual_area_ha_before
            y1,y0=pair.annual_yield_kg_ha_now,pair.annual_yield_kg_ha_before
            area=float(((a1-a0)*(y1+y0)/2).sum())
            productivity=float(((y1-y0)*(a1+a0)/2).sum())
            total=float((pair.annual_production_kg_now-pair.annual_production_kg_before).sum())
            rows.append(dict(year=int(year),previous_year=int(year-1),matched_barangays=len(pair),
                area_component_kg=area,yield_component_kg=productivity,total_change_kg=total,
                reconciliation_kg=area+productivity-total))
        return pd.DataFrame(rows)

