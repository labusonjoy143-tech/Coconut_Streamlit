import json
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from .common import table, chart, fmt, filtered, place_filter, download_table


def render(wb):
    st.title("Coconut production management dashboard")
    st.write("Review recorded production, check missing records, and plan collection volumes under stated assumptions.")
    place = place_filter(wb, "decision_place")
    years = sorted(map(int, wb.annual.year.unique()))
    year = st.selectbox("Recorded year", years, index=len(years)-1, key="decision_year")
    actual = filtered(wb.annual[wb.annual.year == year], place)
    complete = actual[actual.annual_target_complete]
    harvest_months = int(actual.harvest_months.sum())
    expected = len(actual) * 12
    missing = expected - harvest_months
    reported = actual.reported_production_kg.sum(min_count=1) / 1000
    yield_value = (complete.annual_production_kg.sum() / complete.annual_area_ha.sum()) if len(complete) else np.nan
    cols = st.columns(3)
    cols[0].metric(f"{year} reported harvest", f"{fmt(reported)} t")
    cols[0].caption(f"{harvest_months}/{expected} harvest months recorded")
    cols[1].metric("Observed annual yield", f"{fmt(yield_value)} kg/ha")
    cols[1].caption(f"{len(complete)}/{len(actual)} complete barangay-years; matching area only")
    cols[2].metric("Harvest records to recover", missing)
    if missing:
        st.warning("This year's reported production is incomplete. Recover missing harvest records before comparing annual growth.")
    historical = filtered(wb.annual, place).groupby("year").agg(
        reported_kg=("reported_production_kg", lambda s: s.sum(min_count=1)),
        harvest_months=("harvest_months", "sum"), barangays=("barangay", "size")).reset_index()
    historical["Reported tonnes"] = historical.reported_kg / 1000
    historical["Coverage"] = np.where(historical.harvest_months.eq(historical.barangays * 12), "Complete", "Incomplete")
    chart(alt.Chart(historical).mark_bar().encode(
        x=alt.X("year:O", title="Recorded year"), y="Reported tonnes:Q", color="Coverage:N",
        tooltip=["year:O", "Reported tonnes:Q", "harvest_months:Q", "Coverage:N"]).properties(height=260))
    st.subheader("Plan future collection volumes")
    if wb.winner:
        future_years = sorted(map(int, wb.future.year.unique()))
        future_year = st.selectbox("Projected year", future_years, index=len(future_years)-1, key="decision_future_year")
        with st.form("planning_controls"):
            growth = st.slider("Assumed harvested-area growth (% per year)", -10.0, 10.0, 0.0, 0.5, key="decision_growth")
            stress = st.slider("Planning volume adjustment (%)", -20, 20, 0, key="decision_stress")
            st.form_submit_button("Apply planning scenario")
        signature = (wb.audit["sha256"], json.dumps(wb.config, sort_keys=True), growth, stress)
        if st.session_state.get("scenario_signature") != signature:
            scenario = wb.make_scenario(growth, stress)
            st.session_state["decision_scenario"] = scenario
            st.session_state["scenario_signature"] = signature
            st.session_state.pop("export_bytes", None)
        scenario = filtered(st.session_state["decision_scenario"], place)
        selected = scenario[scenario.year == future_year]
        valid = selected.dropna(subset=["assumed_area_ha", "predicted_production_kg"])
        base = valid.predicted_production_kg.sum(min_count=1) / 1000
        plan = valid.planning_production_kg.sum(min_count=1) / 1000
        cols = st.columns(3)
        cols[0].metric(f"{future_year} model volume", f"{fmt(base)} t")
        cols[1].metric("Adjusted planning volume", f"{fmt(plan)} t")
        cols[2].metric("Forecast origin year", wb.config["forecast_origin"])
        st.caption(f"Annual method: {wb.winner}. Area growth {growth:+.1f}%/year; planning adjustment {stress:+d}%. "
                   f"Volume available for {len(valid)}/{len(selected)} barangays in this selection.")
        if len(valid) < len(selected):
            st.warning("Some projected production is unavailable because assumed area is missing. Do not interpret the subtotal as a full-area forecast.")
        trend = scenario.groupby("year")[["predicted_production_kg", "planning_production_kg"]].sum(min_count=1).reset_index()
        trend = trend.rename(columns={"predicted_production_kg": "Model volume", "planning_production_kg": "Adjusted planning volume"})
        trend[["Model volume", "Adjusted planning volume"]] /= 1000
        long = trend.melt(id_vars="year", var_name="Volume", value_name="Tonnes")
        chart(alt.Chart(long).mark_bar().encode(
            x=alt.X("year:O", title="Projected year"), xOffset="Volume:N", y="Tonnes:Q", color="Volume:N",
            tooltip=["year:O", "Volume:N", "Tonnes:Q"]).properties(height=280))
        report = selected[["year", "barangay", "predicted_yield_kg_ha", "assumed_area_ha", "predicted_production_kg", "planning_production_kg"]].copy()
        report["Projected tonnes"] = report.predicted_production_kg / 1000
        report["Planning tonnes"] = report.planning_production_kg / 1000
        report = report.rename(columns={"year": "Year", "barangay": "Barangay", "predicted_yield_kg_ha": "Predicted yield (kg/ha/year)", "assumed_area_ha": "Assumed area (ha)"})
        table(report.drop(columns=["predicted_production_kg", "planning_production_kg"]))
        download_table(selected, "selected_annual_planning_scenario.csv")
        st.info("Area growth is an assumption about harvested land, not a prediction of young-tree bearing. "
                "Planning adjustments are user assumptions, not confidence intervals. "
                "Update elapsed projection years with actual harvest records when available.")
    else:
        st.info("Open Annual forecast and run a comparison to enable future-volume planning.")
    review_count = None
    if not wb.anomalies.empty:
        review = filtered(wb.anomalies[(wb.anomalies.year == year) & wb.anomalies.review_candidate], place)
        review_count = len(review)
        with st.expander(f"Records to review in {year}: {review_count}"):
            table(review.sort_values("review_score", ascending=False))
    if not wb.clusters.empty:
        with st.expander("Descriptive barangay profiles"):
            table(filtered(wb.clusters, place))
    st.subheader("Recommended management follow-up")
    actions = [
        ["Data completeness", f"{missing} harvest records absent", "Recover records before judging annual growth or ranking barangays."],
        ["Collection and logistics", "Use projected tonnes with stated assumptions" if wb.winner else "Annual models have not been run", "Confirm farm schedules before reserving transport, storage, and staff."],
        ["Record verification", f"{review_count} flagged records in this selection" if review_count is not None else "Optional record screening has not been run", "Check source documents and field reports; screening flags are not crop-damage diagnoses."],
        ["Yield improvement", "Repeated seasonal source pattern" if wb.audit["repeated_formula"] else "Review factor diagnostics", "Verify source measurements; add bearing-tree counts, age, variety, soil, inputs, pests, and independent climate observations."],
        ["Forecast refresh", f"Latest source month: {wb.audit['last_month']}", "Upload newer actual observations and rerun the required models before commitments."],
    ]
    table(pd.DataFrame(actions, columns=["Management check", "Finding", "Action"]))
    with st.expander(f"Yearly factor checks across all barangays: {year}"):
        factors = wb.factors_by_year()
        table(factors[factors.year == year])
        st.caption("These associations use all barangays with complete matching records in the selected year. "
                   "They are unavailable when the data do not support interpretation; they do not establish causal effects.")
    with st.expander("Production-change accounting across matched barangays"):
        changes = wb.production_change_factors()
        table(changes[changes.year == year])
        st.caption("Municipality-wide matched complete records; separate area and yield arithmetic, not a causal estimate.")
