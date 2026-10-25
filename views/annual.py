import pandas as pd
import streamlit as st
from coconut.constants import ANNUAL_MODELS, QUICK_ANNUAL, DEFAULT_TREES
from .common import run_task, table, filtered, place_filter, final_test_table, download_table


def render(wb):
    st.title("Annual yield and production forecast")
    st.write("Annual yield uses all 12 harvest months and one consistent annual reported area. "
             "Earlier multi-year blocks select the method; a separate final block measures it.")
    with st.form("annual_settings"):
        methods = st.multiselect("Methods to compare", ANNUAL_MODELS, default=QUICK_ANNUAL)
        horizon = st.slider("Years to project", 1, 5, 3)
        trees = st.slider("Trees per ensemble", 40, 200, DEFAULT_TREES, 20)
        submitted = st.form_submit_button("Run annual forecast", type="primary")
    if submitted:
        def action(candidate, progress):
            candidate.reliance = pd.DataFrame()
            candidate.compare_models(horizon=horizon, names=methods, trees=trees, progress=progress)
        wb = run_task(wb, "Annual forecast", action, has_progress=True)
    if not wb.comparison.empty:
        st.subheader("Model evaluation")
        final_test_table(wb.comparison, wb.winner, monthly=False)
        st.caption(f"Stored run: {wb.config['horizon']}-year horizon; validation origins "
                   f"{wb.config['validation_origins']}; final-test origin {wb.config['test_origin']}; "
                   f"forecast origin {wb.config['forecast_origin']}.")
        st.subheader("Annual baseline projections")
        place = place_filter(wb, "annual_place")
        table(filtered(wb.future, place))
        download_table(wb.future, "annual_baseline_projections.csv")
        st.info("Baseline area stays at each barangay's latest valid area. Future weather and inflation "
                "use fixed medians of the latest three available annual measurements. "
                "Use the management dashboard to change area and planning assumptions.")
    else:
        st.info("Run annual forecast when you need future annual volumes. Factor diagnostics below need no training.")
    st.subheader("Yearly factor diagnostics")
    years = sorted(map(int, wb.annual.year.unique()))
    year = st.selectbox("Recorded year for factors", years, index=len(years)-1, key="annual_factor_year")
    factors = wb.factors_by_year()
    table(factors[factors.year == year])
    st.caption("Associations use same-year complete barangay records. They are withheld when yield varies too little, "
               "a factor is constant, or fewer than six pairs are available. They do not establish causes.")
    st.subheader("What contributed arithmetically to production change?")
    decomposition = wb.production_change_factors()
    table(decomposition[decomposition.year == year])
    st.caption("Only barangays with complete targets in both consecutive years are compared. "
               "The area and yield components add to the observed production change for those matched barangays.")
    with st.expander("Optional: held-out-year annual model reliance"):
        st.write("This fits a separate Random Forest before the chosen recorded year. "
                 "Near-constant annual yields cannot support a useful factor ranking.")
        if st.button("Calculate annual model reliance", key="annual_reliance"):
            wb = run_task(wb, "Annual model explanations", lambda x: x.model_reliance("Random forest", year=year))
        if not wb.reliance.empty:
            table(wb.reliance)
