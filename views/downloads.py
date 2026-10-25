import pandas as pd
import streamlit as st
from coconut.exports import build_results_zip
from .common import table


def render(wb):
    st.title("Download analysis results")
    status = pd.DataFrame([
        {"Module": "Cleaning and annual targets", "Ready": True},
        {"Module": "Monthly prediction", "Ready": bool(wb.monthly_winner)},
        {"Module": "Annual forecast", "Ready": bool(wb.winner)},
        {"Module": "Low-yield classification", "Ready": not wb.classification_metrics.empty},
        {"Module": "Barangay profiles", "Ready": not wb.clusters.empty},
        {"Module": "Record screening", "Ready": not wb.anomalies.empty},
        {"Module": "Management scenario", "Ready": "decision_scenario" in st.session_state},
    ])
    table(status)
    st.write("The archive contains cleaned data, available model results, projections, factor checks, "
             "and an audit of the source and settings. Baseline projections and the latest applied scenario are separate tables.")
    if st.button("Prepare results ZIP", type="primary", key="prepare_export"):
        st.session_state["export_bytes"] = build_results_zip(
            wb, st.session_state.get("decision_scenario"), st.session_state.get("task_runs"))
    if "export_bytes" in st.session_state:
        st.download_button("Download results ZIP", st.session_state["export_bytes"],
                           file_name="Coconut_Analysis_Results.zip", mime="application/zip", key="results_zip")
    st.caption("Unrun modules are listed as unavailable in the audit. Reloading data clears earlier results. "
               "The original uploaded workbook is unchanged.")
