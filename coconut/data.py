"""Validate upload bytes, clean a monthly panel, and define observed annual targets."""
from pathlib import Path
import calendar
import hashlib
import io
import numpy as np
import pandas as pd
from .constants import NUMERIC, RENAME

def add_annual_lags(frame,first_year):
    z=frame.sort_values(['barangay','year']).copy()
    z['year_index']=z.year-first_year
    for source,lag,new in [('annual_yield_kg_ha',1,'yield_lag1'),('annual_yield_kg_ha',2,'yield_lag2'),
       ('rainfall_annual_mm',1,'rain_lag1'),('temperature_mean_c',1,'temp_lag1'),
       ('annual_area_ha',1,'area_lag1'),('inflation_mean_pct',1,'inflation_lag1')]:
        z[new]=z.groupby('barangay')[source].shift(lag)
    return z

class DataMixin:
    def _load_csv(self,content,name='coconut.csv'):
        if isinstance(content,(Path,str)):
            path=Path(content);name=path.name;content=path.read_bytes()
        content=bytes(content)
        for encoding in ['utf-8-sig','cp1252','latin1']:
            try:raw=pd.read_csv(io.BytesIO(content),encoding=encoding);break
            except UnicodeDecodeError:continue
        names={c:RENAME.get(str(c).strip().casefold(),str(c).strip()) for c in raw}
        names.update({c:'temp_c' for c in raw if str(c).strip().casefold().startswith('temperature')})
        d=raw.rename(columns=names).copy()
        if d.columns.duplicated().any():raise ValueError('Duplicate column names after header normalization; correct the source.')
        required=['year','month_name','barangay','area_ha','harvest_kg']
        missing=[c for c in required if c not in d]
        if missing:raise ValueError('Missing required columns: '+', '.join(missing))
        for c in NUMERIC+['weather_label']:
            if c not in d:d[c]=np.nan
        for c in ['barangay','month_name','weather_label']:
            d[c]=d[c].astype('string').str.strip().replace('',pd.NA)
        if d.barangay.isna().any():raise ValueError('Blank barangay; correct the source first.')
        month_map={calendar.month_name[m].casefold():m for m in range(1,13)}
        month_map.update({calendar.month_abbr[m].casefold():m for m in range(1,13)})
        # Normalize to NumPy floats: date validation must also work with pandas 3
        # Arrow-backed strings, whose integer modulo operation can be unavailable.
        d['month']=d.month_name.str.casefold().map(month_map).fillna(pd.to_numeric(d.month_name,errors='coerce')).astype('float64')
        d['year']=pd.to_numeric(d.year,errors='coerce').astype('float64')
        if d.year.isna().any() or d.month.isna().any() or d.year.mod(1).ne(0).any() or d.month.mod(1).ne(0).any():
            raise ValueError('Invalid year or month. Correct dates before analysis.')
        d['date']=pd.to_datetime(dict(year=d.year,month=d.month,day=1),errors='coerce')
        if d.date.isna().any():raise ValueError('Invalid calendar date in the source.')
        if d.duplicated(['barangay','date']).any():raise ValueError('Duplicate barangay-month keys; resolve them in the source.')
        invalid={}
        for c in NUMERIC:
            prior=d[c].copy();d[c]=pd.to_numeric(d[c],errors='coerce').astype('float64')
            bad=(prior.notna()&d[c].isna())|(~np.isfinite(d[c])&d[c].notna())
            if c=='area_ha':bad|=d[c].le(0)
            elif c not in ['temp_c','inflation_pct']:bad|=d[c].lt(0)
            d[c+'_invalid']=bad;invalid[c]=int(bad.sum());d.loc[bad,c]=np.nan
        source_rows=len(d);d['source_row_present']=True
        n_positions=len(d.barangay.unique())*len(pd.date_range(d.date.min(),d.date.max(),freq='MS'))
        if n_positions>2_000_000:raise ValueError('Calendar grid exceeds two million positions. Check dates or split the file.')
        grid=pd.MultiIndex.from_product([sorted(d.barangay.unique()),pd.date_range(d.date.min(),d.date.max(),freq='MS')],names=['barangay','date'])
        d=d.set_index(['barangay','date']).reindex(grid).reset_index()
        d['source_row_present']=d.source_row_present.eq(True)
        d['year']=d.date.dt.year;d['month']=d.date.dt.month
        d['yield_kg_ha_month']=d.harvest_kg/d.area_ha
        for c in NUMERIC:d[c+'_missing']=d[c].isna()
        rows=[]
        for (year,b),g in d.groupby(['year','barangay']):
            area_values=g.area_ha.dropna()
            consistent=bool(len(area_values) and np.isclose(area_values.max(),area_values.min(),rtol=1e-6,atol=.001))
            area=float(area_values.median()) if consistent else np.nan
            harvest_count=int(g.harvest_kg.notna().sum())
            reported=g.harvest_kg.sum(min_count=1)
            complete=len(g)==12 and harvest_count==12
            annual_yield=float(reported/area) if complete and np.isfinite(area) and area>0 else np.nan
            rows.append(dict(year=int(year),barangay=b,reported_production_kg=reported,
              harvest_months=harvest_count,calendar_months=len(g),missing_harvest_months=12-harvest_count,
              annual_area_ha=area,area_observations=int(len(area_values)),area_consistent=consistent,
              annual_production_kg=float(reported) if complete else np.nan,
              annual_yield_kg_ha=annual_yield,annual_target_complete=bool(np.isfinite(annual_yield)),
              rainfall_annual_mm=g.rain_mm.sum() if len(g)==12 and g.rain_mm.notna().sum()==12 else np.nan,
              temperature_mean_c=g.temp_c.mean() if len(g)==12 and g.temp_c.notna().sum()==12 else np.nan,
              inflation_mean_pct=g.inflation_pct.mean() if len(g)==12 and g.inflation_pct.notna().sum()==12 else np.nan,
              rain_observations=int(g.rain_mm.notna().sum()),temperature_observations=int(g.temp_c.notna().sum())))
        a=add_annual_lags(pd.DataFrame(rows),int(d.year.min()))
        paired=d.dropna(subset=['yield_kg_ha_month']);early=paired[paired.year<paired.year.max()]
        formula=850*(1+.25*np.sin(2*np.pi*early.month/12))
        formula_matches=int((early.yield_kg_ha_month-formula).abs().lt(.001).sum())
        self.source=content;self.source_name=name;self.monthly=d;self.annual=a;self.reset_models()
        self.audit=dict(source_name=name,sha256=hashlib.sha256(content).hexdigest(),encoding=encoding,
          source_rows=source_rows,panel_rows=len(d),inserted_months=len(d)-source_rows,
          first_month=str(d.date.min().date()),last_month=str(d.date.max().date()),
          barangays=sorted(d.barangay.unique().tolist()),missing={c:int(d[c].isna().sum()) for c in NUMERIC+['weather_label']},
          invalid_replaced=invalid,annual_rows=len(a),complete_annual_targets=int(a.annual_target_complete.sum()),
          formula_match_rows=formula_matches,formula_tested_rows=len(early),
          repeated_formula=bool(len(early)>24 and formula_matches/len(early)>.98),
          area_assumption='One consistent reported monthly area is used once as annual land area; confirm its meaning with the source.',
          target_definition='annual_yield_kg_ha = complete 12-month harvest kg / one consistent annual area ha')
        return self.audit

    def load_bytes(self, content, name="coconut.xlsx", sheet_name=0):
        """Load user bytes in memory; the uploaded workbook is never modified."""
        original = bytes(content)
        suffix = Path(name).suffix.lower()
        if suffix == ".xlsx":
            with pd.ExcelFile(io.BytesIO(original), engine="openpyxl") as workbook:
                sheets = list(workbook.sheet_names)
                if isinstance(sheet_name, int):
                    if not 0 <= sheet_name < len(sheets):
                        raise ValueError("Worksheet index is outside the workbook.")
                    sheet = sheets[sheet_name]
                else:
                    sheet = sheet_name
                if sheet not in sheets:
                    raise ValueError("Selected worksheet is not present.")
                frame = pd.read_excel(workbook, sheet_name=sheet)
            self._load_csv(frame.to_csv(index=False).encode("utf-8"), name)
            self.source = original
            self.audit.update(sha256=hashlib.sha256(original).hexdigest(), input_format="xlsx",
                              selected_sheet=sheet, worksheets=sheets, encoding="Excel Unicode")
        elif suffix == ".csv":
            self._load_csv(original, name)
            self.audit["input_format"] = "csv"
        else:
            raise ValueError("Choose an .xlsx or .csv file.")
        self.reset_mining()
        from .monthly import monthly_frame
        self.monthly = monthly_frame(self.monthly)
        return self.audit

    def load_file(self, path, sheet_name=0):
        path = Path(path)
        return self.load_bytes(path.read_bytes(), path.name, sheet_name)

    def require_data(self):
        if self.annual is None:raise ValueError('Load the coconut data first.')

    def annual_summary(self):
        self.require_data();rows=[]
        for year,g in self.annual.groupby('year'):
            complete=g[g.annual_target_complete]
            rows.append(dict(year=year,reported_production_kg=g.reported_production_kg.sum(min_count=1),
              recorded_harvest_months=int(g.harvest_months.sum()),expected_harvest_months=12*len(g),
              complete_barangays=len(complete),total_barangays=len(g),
              matching_area_yield_kg_ha=complete.annual_production_kg.sum()/complete.annual_area_ha.sum() if len(complete) else np.nan,
              recorded_total_complete=bool(g.harvest_months.eq(12).all())))
        return pd.DataFrame(rows)

