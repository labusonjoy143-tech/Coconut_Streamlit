"""Column definitions and selectable algorithms."""
NUMERIC=['area_ha','harvest_kg','rain_mm','temp_c','farmgate_php_kg','copra_php_kg','whole_nut_php_kg','inflation_pct']
RENAME={'year':'year','month':'month_name','barangay':'barangay','harvested area (ha)':'area_ha',
 'harvest (kg)':'harvest_kg','rainfall (mm)':'rain_mm','farmgate price (php/kg)':'farmgate_php_kg',
 'copra price (php/kg)':'copra_php_kg','whole nut price (php/kg)':'whole_nut_php_kg',
 'inflation rate (%)':'inflation_pct','weather':'weather_label'}
ANNUAL_MODELS=['Historical median','Last observed yield','Ridge regression','Decision tree',
               'Random forest','Extra trees','Gradient boosting']
FEATURES=['year_index','yield_lag1','yield_lag2','rain_lag1','temp_lag1','area_lag1','inflation_lag1']
FACTOR_GROUPS={'Yield history':['yield_lag1','yield_lag2'],'Previous-year rainfall':['rain_lag1'],
 'Previous-year temperature':['temp_lag1'],'Previous-year area':['area_lag1'],
 'Previous-year inflation':['inflation_lag1'],'Year':['year_index'],'Barangay':['barangay']}
MONTH_FEATURES=['year_index_m','month_sin','month_cos','yield_lag12','yield_lag24','rain_lag12','temp_lag12']
MONTH_METHODS=['Seasonal median','Harmonic regression','Ridge regression','Decision tree',
               'Random forest','Random forest (yield-range weighted)','Extra trees','Gradient boosting']

QUICK_MONTHLY = ["Seasonal median", "Harmonic regression", "Ridge regression", "Random forest", "Random forest (yield-range weighted)"]
QUICK_ANNUAL = ["Historical median", "Last observed yield", "Ridge regression", "Random forest"]
DEFAULT_TREES = 60
