"""Small presentation helpers shared by the module screens."""
import copy
import math
from datetime import datetime
from zoneinfo import ZoneInfo
import altair as alt
import pandas as pd
import streamlit as st
from threadpoolctl import threadpool_limits
from coconut.exports import csv_bytes

LABELS = {
    "yield_kg_ha_month": "Monthly yield (kg/ha)",
    "rain_mm": "Monthly rainfall (mm)",
    "temp_c": "Mean temperature (°C)",
    "area_ha": "Reported area (ha)",
    "harvest_kg": "Harvest (kg)",
    "farmgate_php_kg": "Farmgate price (PHP/kg)",
    "copra_php_kg": "Copra price (PHP/kg)",
    "whole_nut_php_kg": "Whole-nut price (PHP/kg)",
    "inflation_pct": "Inflation (%)",
    "weather_label": "Weather description",
}


def filtered(frame, barangay):
    return frame if barangay == "All barangays" else frame[frame.barangay == barangay]


def place_filter(wb, key):
    return st.selectbox("Barangay", ["All barangays"] + wb.audit["barangays"], key=key)


def fmt(value, digits=1):
    return f"{value:,.{digits}f}" if value is not None and math.isfinite(value) else "Unavailable"


def table(frame, limit=500):
    if frame is None or frame.empty:
        st.info("No results are available for this selection.")
    else:
        st.dataframe(frame.head(limit), width="stretch", hide_index=True)
        if len(frame) > limit:
            st.caption(f"Showing {limit:,} of {len(frame):,} rows. Download the full table for all records.")


def download_table(frame, filename, label="Download table (CSV)"):
    st.download_button(label, csv_bytes(frame), file_name=filename, mime="text/csv", key=filename)


def chart(chart):
    st.altair_chart(chart)


def bar(frame, x, y, x_title, y_title, color=None):
    enc = {"x": alt.X(x, title=x_title), "y": alt.Y(y, title=y_title),
           "tooltip": list(frame.columns)}
    if color:
        enc["color"] = color
    chart(alt.Chart(frame).mark_bar().encode(**enc).properties(height=300))


def run_task(wb, label, action, has_progress=False):
    """Commit a task atomically. A failed run keeps the last successful result."""
    candidate = copy.deepcopy(wb)
    progress = st.progress(0, text=label)
    def callback(done, total, message):
        progress.progress(min(1.0, done / max(total, 1)), text=message)
    try:
        with threadpool_limits(limits=1):
            action(candidate, callback) if has_progress else action(candidate)
    except Exception as exc:
        progress.empty()
        st.error(f"{label} could not finish: {exc}")
        st.caption("The previous successful results are still available. Check data coverage and settings.")
        return wb
    progress.empty()
    st.session_state["workbench"] = candidate
    st.session_state.pop("export_bytes", None)
    if label == "Annual forecast":
        st.session_state.pop("decision_scenario", None)
        st.session_state.pop("scenario_signature", None)
    runs = dict(st.session_state.get("task_runs", {}))
    runs[label] = datetime.now(ZoneInfo("Asia/Manila")).isoformat(timespec="seconds")
    st.session_state["task_runs"] = runs
    st.success(f"{label} completed.")
    return candidate


def source_note(wb):
    st.caption(f"Source: {wb.audit['source_name']} · {wb.audit['first_month']} to "
               f"{wb.audit['last_month']} · {len(wb.audit['barangays'])} barangays")


def final_test_table(metrics, selected, monthly=True):
    units = "kg/ha/month" if monthly else "kg/ha/year"
    st.success(f"Selected using earlier validation periods: {selected}")
    final = metrics[metrics.split == "final test"].copy()
    final["selected"] = final.model.eq(selected)
    columns = [c for c in ["model", "selected", "n", "mae", "rmse", "wape_pct", "tail_mae", "r2"] if c in final]
    table(final[columns].sort_values("mae"))
    st.caption(f"MAE and RMSE: {units}; WAPE: %. Final-test results do not choose the winner. "
               "A blank R² means observed yield varies too little for an informative score.")
    with st.expander("Validation and test details"):
        table(metrics)
