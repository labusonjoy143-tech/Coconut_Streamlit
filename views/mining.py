import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from coconut.constants import DEFAULT_TREES
from .common import run_task, table, chart, place_filter, filtered, download_table


def classification(wb):
    st.subheader("Class-weighted low-yield classification")
    st.info("The low category is at or below each training set's 20th percentile of monthly yield. "
            "It describes relative seasonal output; it is not a crop-failure category.")
    with st.form("classification_settings"):
        trees = st.slider("Trees for balanced Random Forest", 40, 200, DEFAULT_TREES, 20)
        run = st.form_submit_button("Run low-yield classification", type="primary")
    if run:
        wb = run_task(wb, "Low-yield classification", lambda x: x.fit_low_yield_classification(trees=trees))
    if wb.classification_metrics.empty:
        return
    selected = wb.mining_settings["classification_selected"]
    st.success(f"Classifier selected on validation macro F1: {selected}")
    table(wb.class_counts)
    table(wb.classification_metrics[wb.classification_metrics.split == "final test"])
    st.caption("Class weights are computed only from training labels. The final test keeps its natural class counts. "
               "Balanced accuracy and macro F1 account for uneven category sizes.")
    matrix = np.array(wb.class_details[selected]["confusion_matrix"])
    labels = ["Other yield range", "Low yield category"]
    cells = pd.DataFrame([{"Actual": labels[i], "Predicted": labels[j], "Records": int(matrix[i, j])}
                          for i in range(2) for j in range(2)])
    base = alt.Chart(cells).encode(x="Predicted:N", y="Actual:N", tooltip=list(cells))
    chart((base.mark_rect().encode(color=alt.Color("Records:Q", scale=alt.Scale(scheme="blues"))) +
           base.mark_text(fontSize=18).encode(text="Records:Q")).properties(height=240))
    with st.expander("Classification report and validation"):
        report = wb.class_details[selected]["report"]
        table(pd.DataFrame({k: v for k, v in report.items() if isinstance(v, dict)}).T.reset_index(names="Category"))
        table(wb.classification_metrics)


def clustering(wb):
    st.subheader("Barangay capacity and climate profiles")
    st.write("Compare K-Means groups using standardized area, rainfall, and temperature profiles. "
             "Silhouette scores compare two to five groups. Yield is displayed as context and is excluded from fitting.")
    if st.button("Group barangay profiles", type="primary", key="run_clusters"):
        wb = run_task(wb, "Barangay profiles", lambda x: x.cluster_barangays())
    if wb.clusters.empty:
        return
    table(wb.clusters)
    profiles = wb.clusters.copy()
    profiles["Group"] = profiles.cluster.astype(str)
    chart(alt.Chart(profiles).mark_circle(size=160).encode(
        x=alt.X("median_area_ha:Q", title="Median reported area (ha)"),
        y=alt.Y("rainfall_mean_mm:Q", title="Mean monthly rainfall (mm)", scale=alt.Scale(zero=False)),
        color="Group:N", tooltip=["barangay:N", "Group:N", "median_area_ha:Q", "rainfall_mean_mm:Q", "temperature_mean_c:Q"]
    ).properties(height=300).interactive())
    table(wb.cluster_scores)
    st.caption("Descriptive profiles use all available records. Group numbers are labels, not productivity rankings, "
               "and are not inputs to the prediction backtests.")


def anomalies(wb):
    st.subheader("Records for data review")
    st.write("Isolation Forest compares season-adjusted yield, area, rainfall, and temperature. "
             "It is fitted before the latest source year.")
    with st.form("anomaly_settings"):
        percent = st.slider("Training screening fraction (%)", 1, 10, 3,
                            help="A chosen review workload, not an observed disease or loss rate.")
        run = st.form_submit_button("Screen unusual records", type="primary")
    if run:
        wb = run_task(wb, "Record screening", lambda x: x.detect_review_candidates(percent/100))
    if wb.anomalies.empty:
        return
    place = place_filter(wb, "review_place")
    years = sorted(map(int, wb.anomalies.year.unique()))
    year = st.selectbox("Recorded year", years, index=len(years)-1, key="review_year")
    scope = filtered(wb.anomalies[wb.anomalies.year == year], place)
    candidates = scope[scope.review_candidate].sort_values("review_score", ascending=False)
    st.metric("Records to review", f"{len(candidates)}/{len(scope)}")
    table(candidates)
    download_table(scope, "anomaly_review_selection.csv")
    st.warning("Flags are review candidates, not confirmed crop damage. Check the original records, "
               "missing inputs, recording methods, and current farm reports.")


def render(wb):
    st.title("Other data mining")
    task = st.selectbox("Choose a task", ["Low-yield classification", "Barangay profiles", "Review unusual records"], key="mining_task")
    {"Low-yield classification": classification,
     "Barangay profiles": clustering,
     "Review unusual records": anomalies}[task](wb)
