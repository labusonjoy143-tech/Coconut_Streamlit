import pandas as pd
import streamlit as st
from coconut.constants import NUMERIC
from .common import LABELS, bar, table, download_table


def render(wb):
    st.title("Data quality and preprocessing")
    audit = wb.audit
    cols = st.columns(4)
    cols[0].metric("Source records", f"{audit['source_rows']:,}")
    cols[1].metric("Barangays", len(audit["barangays"]))
    cols[2].metric("Inserted missing months", audit["inserted_months"])
    cols[3].metric("Complete annual targets", f"{audit['complete_annual_targets']}/{audit['annual_rows']}")
    if audit["repeated_formula"]:
        st.warning(f"Source verification needed: {audit['formula_match_rows']:,} of "
                   f"{audit['formula_tested_rows']:,} observed yields before the latest year match a repeated seasonal formula. "
                   "Confirm whether the source values were measured, estimated, or generated before relying on model accuracy.")
    quality = pd.DataFrame([{
        "Variable": LABELS.get(c, c), "Missing values": count,
        "Invalid values changed to missing": audit["invalid_replaced"].get(c, 0),
    } for c, count in audit["missing"].items()])
    st.subheader("Records to recover or verify")
    bar(quality, "Variable:N", "Missing values:Q", "", "Missing records")
    table(quality)
    coverage = wb.monthly.groupby("barangay").agg(
        calendar_positions=("date", "size"), observed_yield_pairs=("yield_kg_ha_month", "count"),
        missing_harvest=("harvest_kg", lambda s: int(s.isna().sum())),
        missing_area=("area_ha", lambda s: int(s.isna().sum()))).reset_index()
    st.subheader("Barangay coverage")
    table(coverage)
    st.subheader("Cleaned data preview")
    table(wb.monthly[["date", "barangay"] + NUMERIC + ["yield_kg_ha_month"]], limit=100)
    download_table(wb.monthly, "cleaned_coconut_monthly.csv")
    with st.expander("Cleaning and preprocessing rules", expanded=False):
        st.markdown("""
- Standardize known column headers and trim blank text.
- Validate dates and reject duplicate barangay-month keys.
- Convert numeric fields; flag invalid values and set them to missing.
- Treat negative harvest, rainfall, and prices, and nonpositive area as invalid.
- Insert absent calendar months with missing values; preserve unobserved targets.
- Monthly yield equals matching harvest kg divided by positive reported hectares.
- Annual yield requires 12 observed harvest months and one consistent annual area.
- Fit predictor imputation, missing indicators, encoding, scaling, and imbalance weights inside each training subset.
""")
        st.info(audit["area_assumption"] + " Confirm whether harvest kg refers to whole nuts, copra, or another product.")
    with st.expander("Source audit"):
        st.json(audit)
