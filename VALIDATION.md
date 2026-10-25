# Validation

Delivery date: 24 October 2026.

The uploaded Python attachment was read without executing its Colab setup,
uploads, or demonstration cells. Its analysis definitions were split into the
provided modules, then executed against `Coconut_Data.xlsx` from the same task.

## Original workbook

- Worksheet: `Sheet1 (2)`.
- 3,432 source rows, 11 barangays, monthly observations from 2000 through 2025.
- 197 complete barangay-year annual yield targets.
- Original workbook bytes preserved; no source file overwrite.
- Monthly evaluation: all eight original methods completed, 123 observed targets
  per method in the 2025 final test. Validation selected the seasonal median.
- Annual evaluation: all seven original methods completed for the three-year
  horizon, 19 complete targets per method in the 2023–2025 final test. Validation
  selected the historical median.
- Full-method checks used 40 trees; the Streamlit default workflow used 60.
- Only the selected monthly and annual models were retained/refitted for future
  projections. Three-month and three-year projections each produced 33 rows.

## Streamlit interaction checks

Streamlit's real `AppTest` runner exercised:

1. Startup before upload, Excel upload, worksheet selection, and load/clean.
2. All seven screens before training; navigation did not trigger model fitting.
3. Default monthly comparison, regression imbalance table, three-month
   projections, and grouped Random Forest reliance.
4. Default annual comparison, projections, and factor-variation safeguards.
5. Class-weighted classification, confusion matrix, K-Means profiles,
   and Isolation Forest screening.
6. Management barangay filters and a +5% annual area / −10% planning scenario.
   Training functions were patched to fail if called during those controls;
   no retraining occurred. Baseline projections and test metrics stayed unchanged.
7. ZIP preparation, archive integrity, separate baseline/scenario exports,
   dataset reload resetting results, and duplicate-key errors retaining the
   last valid dataset.

These checks execute real Streamlit code, data transformations, scikit-learn
models, chart serialization, session state, and widget events. They do not
constitute visual inspection in a separate browser or public deployment testing.

## Included automated checks

The five tests in `tests/test_workflow.py` cover missing targets/partial annual
records, annual area handling, duplicate keys, imbalance weights, chronological
test counts, projection dates, independent scenarios, and export integrity.
They passed on both numerical environments below.

## Environments checked

- Python 3.12.14, scikit-learn 1.8.0.
- pandas 2.2.3 / NumPy 2.3.5 for the full algorithm comparison and unit checks.
- Streamlit 1.66.1, Altair 6.3.0, pandas 3.1.0 / NumPy 2.5.4 for AppTest and
  the second unit-check run.

The source's strongly repeated seasonal values were flagged by the app.
These are software/data-consistency checks, not independent agronomic validation
or a claim that the same accuracy will hold on future field measurements.
