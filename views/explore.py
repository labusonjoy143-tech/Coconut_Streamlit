import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from .common import LABELS, place_filter, filtered, table, chart, bar


def render(wb):
    st.title("Explore coconut production")
    place = place_filter(wb, "explore_place")
    data = filtered(wb.monthly, place)
    observed = data[data.yield_kg_ha_month.notna()]
    season = data.groupby("month").yield_kg_ha_month.agg(["mean", "median", "count"]).reset_index()
    st.subheader("Typical yield by month")
    chart(alt.Chart(season).mark_line(point=True).encode(
        x=alt.X("month:O", title="Calendar month"),
        y=alt.Y("mean:Q", title="Mean yield (kg/ha/month)", scale=alt.Scale(zero=False)),
        tooltip=["month", "mean", "median", "count"]).properties(height=300))
    annual = filtered(wb.annual, place).groupby("year").agg(
        reported_kg=("reported_production_kg", lambda s: s.sum(min_count=1)),
        recorded_months=("harvest_months", "sum"),
        expected_months=("calendar_months", "size")).reset_index()
    annual["expected_months"] *= 12
    annual["Reported tonnes"] = annual.reported_kg / 1000
    annual["Coverage"] = np.where(annual.recorded_months.eq(annual.expected_months), "Complete harvest records", "Incomplete harvest records")
    st.subheader("Reported production and coverage")
    bar(annual, "year:O", "Reported tonnes:Q", "Year", "Reported tonnes", "Coverage:N")
    st.caption("Bars sum available harvest records. A partial annual total is unsuitable for an annual growth comparison.")
    table(annual)
    left, right = st.columns(2)
    with left:
        st.subheader("Observed yield distribution")
        if len(observed):
            counts, edges = np.histogram(observed.yield_kg_ha_month, bins=25)
            hist = pd.DataFrame({"From": edges[:-1], "To": edges[1:], "Records": counts})
            chart(alt.Chart(hist).mark_bar().encode(
                x=alt.X("From:Q", bin="binned", title="kg/ha/month"), x2="To:Q", y="Records:Q",
                tooltip=["From", "To", "Records"]).properties(height=280))
    with right:
        st.subheader("Climate and yield")
        factor = st.selectbox("Compare yield with", ["rain_mm", "temp_c"], format_func=lambda x: LABELS[x])
        complete = observed.dropna(subset=[factor])
        sample = complete.sample(min(len(complete), 3000), random_state=42)
        chart(alt.Chart(sample).mark_circle(opacity=0.45).encode(
            x=alt.X(f"{factor}:Q", title=LABELS[factor]),
            y=alt.Y("yield_kg_ha_month:Q", title="Yield (kg/ha/month)", scale=alt.Scale(zero=False)),
            tooltip=["date:T", "barangay:N", "yield_kg_ha_month:Q", f"{factor}:Q"]
        ).properties(height=280).interactive())
        st.caption(f"Showing {len(sample):,} complete records; chart sample only. Correlation does not establish cause.")
    with st.expander("Descriptive statistics and correlations"):
        columns = ["yield_kg_ha_month", "area_ha", "rain_mm", "temp_c", "farmgate_php_kg", "inflation_pct"]
        table(data[columns].describe().T.reset_index(names="Variable"))
        corr = data[columns].corr()
        grid = corr.rename_axis("Factor").reset_index().melt(id_vars="Factor", var_name="Other factor", value_name="Correlation")
        chart(alt.Chart(grid).mark_rect().encode(
            x="Factor:N", y="Other factor:N", color=alt.Color("Correlation:Q", scale=alt.Scale(domain=[-1, 1], scheme="redblue")),
            tooltip=["Factor", "Other factor", "Correlation"]).properties(height=320))
        st.caption("Calendar seasonality and repeated source values can explain apparent climate relationships.")
