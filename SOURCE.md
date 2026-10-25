# Conversion source

Converted from the user attachment `coconut_excel_data_mining_colab.py`.
SHA-256: `3e14a85ff32460312643eb37f3bbef67a2f6284bd65d49ac1cc7fdcc77594a3c`.

The original cleaning, lag definitions, evaluation folds, weighting, classifiers,
clustering, anomaly screening, and production accounting are preserved in modules.
Colab installation cells, duplicated uploads/dashboard calls, and hard-coded demo
filter changes are replaced with Streamlit inputs.

Differences for responsiveness:
- Each screen executes only when selected.
- Analysis runs on an explicit button; results remain in the current session.
- The default comparison uses a smaller selectable method set and 60 trees.
- All original eight monthly and seven annual methods remain available.
- Only the validation-selected method is refitted/stored for future forecasts.
- Tree methods use one worker; optional mining tasks run separately.
- Streamlit's native charts replace Plotly/ipywidgets/IPython.
- User scenarios are separate from zero-growth annual baseline projections.
- Input/output bytes stay in memory; no shared result filenames are written.
