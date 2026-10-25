# Coconut production — modular Streamlit application

A Streamlit conversion of the attached `coconut_excel_data_mining_colab.py`.
Upload your coconut Excel/CSV data and run only the modules you need. The project
includes all eight original monthly algorithms and seven annual algorithms,
imbalance handling, exploratory analysis, additional data mining, and a management
dashboard. No GPU is needed.

## Run the application

Install Python 3.11 or 3.12. Extract the ZIP, open a terminal inside the
`Coconut_Streamlit` folder, and run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed in the terminal, normally `http://localhost:8501`.
Keep the terminal running while using the application. On systems where Python
is named `python3`, use `python3` in these commands.

An isolated environment is optional: create it with `python -m venv .venv`, then
activate `.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on
macOS/Linux, before installing requirements. If a port is already occupied, run
`python -m streamlit run app.py --server.port 8502`.

## Suggested workflow

1. Upload `Coconut_Data.xlsx` or a CSV in the left panel. Select its worksheet,
   then click **Load and clean data**. The original dataset is not bundled.
2. Start with **Data quality** and **Explore production**.
3. Open **Monthly prediction** and click **Run monthly prediction** for model
   comparison, regression imbalance weights, and three monthly projections.
4. Open **Annual forecast** and click **Run annual forecast** for future annual
   yield and production. Change the horizon to one or two years if history is short.
5. In **Other data mining**, run classification, clustering, or record screening
   separately when needed.
6. Open **Management dashboard**. Select barangay and year; adjust harvested-area
   growth and planning volume, then click **Apply planning scenario**.
7. In **Download results**, click **Prepare results ZIP**, then download it.

You may use the dashboard before training to inspect recorded harvests and data
coverage. Future-volume planning becomes available after an annual model run.

## Modules and source files

| Screen | Streamlit code | Main analysis code |
| --- | --- | --- |
| Upload, cleaning, coverage | `app.py`, `views/data_quality.py` | `coconut/data.py`, `coconut/preprocessing.py` |
| Exploratory analysis | `views/explore.py` | Cleaned monthly/annual tables |
| Monthly yield and imbalance | `views/monthly.py` | `coconut/monthly.py`, `coconut/preprocessing.py` |
| Annual yield and forecasting | `views/annual.py` | `coconut/annual.py` |
| Classification, clustering, anomalies | `views/mining.py` | `coconut/mining.py` |
| Decision dashboard, yearly factors | `views/dashboard.py` | `coconut/decisions.py`, stored forecast models |
| Downloadable outputs | `views/downloads.py` | `coconut/exports.py` |
| Shared state, metrics, defaults | `views/common.py` | `coconut/workbench.py`, `metrics.py`, `constants.py` |

Each screen has a `render(wb)` function and shares a `CoconutWorkbench` object.
The workbench combines small analysis modules. Edit `coconut/constants.py` to
change default methods; edit one analysis module to change its logic. Run the
whole application from `app.py`; the screen files are not independent servers.

## Input data

Required headers:

```text
Year, Month, Barangay, Harvested Area (ha), Harvest (kg)
```

Optional headers:

```text
Rainfall (mm), Temperature (°C), Farmgate Price (PHP/kg),
Copra Price (PHP/kg), Whole Nut Price (PHP/kg), Inflation Rate (%), Weather
```

Use one row per barangay/month. Month can be an English name, abbreviation, or
integer 1–12. Headers are trimmed and matched without case sensitivity. Missing
optional columns remain missing. Only `.xlsx` and `.csv` are supported. The
configured upload limit is 50 MB. Wrong dates, blank barangays, and duplicate
barangay-month keys produce a correction message.

Monthly comparison needs at least six calendar years with observed targets in
the four latest years. Annual comparison needs three horizon-length backtest
blocks, at least five years before the first origin, and at least 20 complete
annual targets in each training set. Clustering requires at least four barangays
with distinguishable profiles. A failed analysis preserves the last successful
result and explains the issue.

## Lightweight execution

- Only the selected screen renders; model fitting starts from an explicit button.
- Results and fitted models stay in the current user's Streamlit session.
- Default monthly comparison: seasonal median, harmonic regression, Ridge,
  ordinary Random Forest, and yield-range-weighted Random Forest.
- Default annual comparison: historical median, last observed yield, Ridge,
  and Random Forest. Select additional original methods when needed.
- Default tree count is 60. Tree estimators use one worker; model runs also limit
  native numerical-library thread pools to one thread.
- Comparisons evaluate all selected algorithms; only the validation winner is
  refitted and stored for future projections.
- Explanations, classification, clustering, and anomaly screening are optional
  separate actions. Changing dashboard filters reuses fitted models.
- Streamlit/Altair charts replace Plotly, ipywidgets, IPython, and Colab helpers.
  There is no TensorFlow, CatBoost, XGBoost, SMOTE dependency, database, or external API.
- Uploads and output archives are handled in memory without shared result files.

Larger files and full algorithm comparisons still require more CPU and memory.
No app-specific cross-user model cache is used. A browser refresh, server restart,
or lost session may clear results; download them before closing the session.
Loading a new dataset clears old predictions and scenarios.

## Methods and interpretation

**Targets.** Monthly yield is observed harvest kg / matching positive area ha.
Annual yield requires all 12 observed harvest months and one consistent annual
area. Repeated monthly hectares are used once, not summed. Missing targets are
never replaced by synthetic harvests for evaluation. Verify the annual-area and
harvest-product meanings with the source.

**Preprocessing.** Standardize headers, trim blanks, parse dates, validate keys,
convert numeric values, flag invalid negative values/nonpositive area, and fill
absent calendar positions with missing values. Training pipelines fit median
predictor imputation, missing indicators, barangay encoding, and scaling for
linear models. Prices and outcome-bearing weather descriptions are excluded
from the predictive inputs.

**Imbalance.** Continuous yield uses five equal-width training target ranges and
inverse-frequency sample weights, clipped before normalization to mean one.
The weighted Random Forest is compared with ordinary fitting. Low-yield
classification uses balanced class weights and a training-only 20th-percentile
threshold. Validation/test targets keep their original distribution; SMOTE is
not applied to continuous yield or before splitting.

**Evaluation.** Monthly models use three earlier whole-year validation periods
and the latest source year as final test. Annual models use recursive multi-year
backtests with two earlier validation blocks and a final test block. The model
is chosen by validation MAE with a small simplicity preference for baseline
ties. R² and annual correlations are withheld when yield variation is too small.
Final-test scores are saved before full-history refitting. All barangays share
the date split.

**Forecasts.** The monthly method projects the next three months after the source
cutoff, with area held at its latest observed value. Annual forecasts project
one to five years; future weather and inflation use fixed medians of the latest
three available annual observations. Forecast dates follow the data cutoff,
not today's date. Monthly and annual models are selected independently and are
not mathematically reconciled. These are point projections without calibrated
prediction intervals.

**Management.** The dashboard reports harvest coverage, annual yield on matching
complete records, projected tonnes, and assumed-area/planning scenarios. Scenario
tables remain separate from baseline projections. Area growth is an assumption
about harvested land, not an estimate of new-tree bearing. Planning adjustments
are not probability intervals. Missing area produces explicit projection
subtotals. No profit is inferred from unverified product/price units.

**Other mining.** Classification predicts a relative seasonal category, not crop
failure. K-Means groups standardized area and climate profiles; observed yield
is context only. Isolation Forest flags records for review, not verified damage.
Grouped permutation results describe model reliance, not causal effects.
Annual production-change decomposition is an accounting identity on matched
complete barangays, not causal inference.

The supplied dataset has an unusually repeated seasonal yield pattern and almost
constant annual normalized yield. High model accuracy on this file does not
establish field accuracy or prove that rainfall/temperature cause yield changes.
Verify source measurements and collect bearing-tree counts, tree age, variety,
soil, inputs, pests, and independent climate observations before making claims
about yield improvement.

## Use the analysis without Streamlit

Run this from the project folder in a Python script or notebook:

```python
from coconut import CoconutWorkbench
from coconut.exports import build_results_zip

wb = CoconutWorkbench()
wb.load_file("Coconut_Data.xlsx", sheet_name=0)
wb.fit_monthly_models(names=["Seasonal median", "Ridge regression"], trees=60)
monthly = wb.predict_next_three_months()
wb.compare_models(horizon=3, names=["Historical median", "Ridge regression"], trees=60)
scenario = wb.make_scenario(area_growth_pct=2.5, stress_pct=-5)
with open("Coconut_Analysis_Results.zip", "wb") as output:
    output.write(build_results_zip(wb, scenario))
```

## Validation and method documentation

Run the included data/evaluation checks:

```bash
python -m unittest discover -s tests -v
```

`VALIDATION.md` describes the workbook and Streamlit interaction checks used for
this delivery. `SOURCE.md` records the conversion source and execution changes.

Primary documentation:

- [Streamlit Session State](https://docs.streamlit.io/develop/concepts/architecture/session-state)
- [Streamlit forms](https://docs.streamlit.io/develop/concepts/architecture/forms)
- [Streamlit application testing](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
- [scikit-learn preprocessing pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
- [Balanced class weights](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)
- [Random Forest sample weights](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
