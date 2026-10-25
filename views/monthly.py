import altair as alt
import pandas as pd
import streamlit as st
from coconut.constants import MONTH_METHODS, QUICK_MONTHLY, DEFAULT_TREES
from .common import run_task, table, chart, bar, filtered, place_filter, final_test_table, download_table


def render(wb):
    st.title("Monthly yield prediction")
    st.write("Compare methods on earlier years, evaluate the latest source year separately, "
             "then predict the three months after the data cutoff.")
    with st.form("monthly_settings"):
        methods = st.multiselect("Methods to compare", MONTH_METHODS, default=QUICK_MONTHLY)
        trees = st.slider("Trees per ensemble", 40, 200, DEFAULT_TREES, 20,
                          help="More trees cost additional CPU time and memory.")
        submitted = st.form_submit_button("Run monthly prediction", type="primary")
    if submitted:
        def action(candidate, progress):
            candidate.monthly_reliance = pd.DataFrame()
            candidate.fit_monthly_models(names=methods, trees=trees, progress=progress)
            candidate.predict_next_three_months()
        wb = run_task(wb, "Monthly prediction", action, has_progress=True)
    if wb.monthly_comparison.empty:
        st.info("Choose methods and click Run monthly prediction. Data exploration works before training.")
        return
    st.subheader("Model evaluation")
    final_test_table(wb.monthly_comparison, wb.monthly_winner)
    st.caption(f"Validation years: {wb.mining_settings['monthly_validation_years']}; "
               f"final test: {wb.mining_settings['monthly_test_year']}. All barangays share each chronological split.")
    st.subheader("Continuous-target imbalance")
    table(wb.weight_report)
    bar(wb.weight_report, "target_bin:O", "training_rows:Q", "Training yield range (0 = lowest)", "Training records")
    st.info("Five equal-width ranges and inverse-frequency weights use training targets only. "
            "Compare the ordinary and weighted Random Forest; weighting may help or hurt. "
            "Test records retain their natural distribution. SMOTE is not used on continuous yield.")
    st.subheader("Three monthly projections")
    place = place_filter(wb, "monthly_place")
    predictions = filtered(wb.three_month_predictions, place)
    table(predictions)
    download_table(predictions, "three_month_predictions.csv")
    st.caption("Forecast dates follow the last source month. Production = predicted kg/ha × last observed area. "
               "Missing input lags use the training-fitted imputer. Replace elapsed forecast months with actual observations when available.")
    scored = wb.monthly_predictions
    scored = filtered(scored[(scored.model == wb.monthly_winner) & (scored.split == "final test")], place)
    if not scored.empty:
        curve = scored.groupby("date")[["actual_yield", "predicted_yield"]].mean().reset_index().melt(
            id_vars="date", var_name="Series", value_name="Yield")
        st.subheader("Final-test monthly yield")
        chart(alt.Chart(curve).mark_line(point=True).encode(
            x="date:T", y=alt.Y("Yield:Q", title="Mean of scored records (kg/ha/month)", scale=alt.Scale(zero=False)),
            color="Series:N", tooltip=["date:T", "Series:N", "Yield:Q"]).properties(height=300))
    with st.expander("Optional: factors used by a Random Forest"):
        st.write("This comparison model is fitted before the final-test year. Shuffling grouped inputs "
                 "measures its reliance on those inputs; it does not estimate agricultural causes.")
        if st.button("Calculate monthly model reliance", key="monthly_reliance"):
            wb = run_task(wb, "Monthly model explanations", lambda x: x.monthly_model_reliance("Random forest"))
        if not wb.monthly_reliance.empty:
            table(wb.monthly_reliance)
            bar(wb.monthly_reliance, "increase_mae_kg_ha:Q", "group:N", "Increase in MAE (kg/ha/month)", "Input group")
    if wb.errors:
        with st.expander("Model messages"):
            table(pd.DataFrame(wb.errors))
