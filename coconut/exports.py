"""Create downloadable result archives in memory, including task status."""
import io
import json
import zipfile
import pandas as pd


def csv_bytes(frame):
    return frame.to_csv(index=False).encode("utf-8-sig")


def build_results_zip(wb, scenario=None, task_runs=None):
    wb.require_data()
    tables = {
        "cleaned_monthly.csv": wb.monthly,
        "annual_targets.csv": wb.annual,
        "annual_summary.csv": wb.annual_summary(),
        "regression_training_weights.csv": wb.training_weights,
        "yield_range_weights.csv": wb.weight_report,
        "monthly_model_comparison.csv": wb.monthly_comparison,
        "monthly_scored_predictions.csv": wb.monthly_predictions,
        "three_month_predictions.csv": wb.three_month_predictions,
        "annual_model_comparison.csv": wb.comparison,
        "annual_scored_predictions.csv": wb.test_predictions,
        "annual_baseline_projections.csv": wb.future,
        "low_yield_class_distribution.csv": wb.class_counts,
        "classification_comparison.csv": wb.classification_metrics,
        "classification_scored_predictions.csv": wb.classification_predictions,
        "barangay_profiles_and_clusters.csv": wb.clusters,
        "cluster_comparison.csv": wb.cluster_scores,
        "anomaly_review_records.csv": wb.anomalies,
        "annual_factor_associations.csv": wb.factors_by_year(),
        "production_change_accounting.csv": wb.production_change_factors(),
        "monthly_model_reliance.csv": wb.monthly_reliance,
        "annual_model_reliance.csv": wb.reliance,
        "model_errors.csv": pd.DataFrame(wb.errors),
    }
    if scenario is not None and not scenario.empty:
        tables["annual_user_scenario.csv"] = scenario
    status = {name: {"rows": len(df), "available": not df.empty}
              for name, df in tables.items()}
    metadata = {
        "audit": wb.audit,
        "annual_settings": wb.config,
        "mining_settings": wb.mining_settings,
        "classifier_details": wb.class_details,
        "tables": status,
        "task_runs": task_runs or {},
        "monthly_selected": wb.monthly_winner,
        "annual_selected": wb.winner,
        "scenario": None if scenario is None or scenario.empty else {
            "area_growth_pct": float(scenario.area_growth_pct.iloc[0]),
            "planning_adjustment_pct": float(scenario.planning_stress_pct.iloc[0]),
        },
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, frame in tables.items():
            # Unrun modules are recorded in metadata instead of blank result files.
            if len(frame.columns):
                archive.writestr(name, csv_bytes(frame))
        archive.writestr("audit_and_settings.json", json.dumps(metadata, indent=2, default=str))
        archive.writestr("README.txt", (
            "Coconut production analysis\n"
            "See audit_and_settings.json for completed tasks, settings, and available tables.\n"
            "Missing target harvests remain missing and are excluded from scoring.\n"
            "Annual yield requires twelve harvest months and one consistent annual area.\n"
            "Regression/class weights are training-only; tests retain natural prevalence.\n"
            "Monthly and annual models are selected separately by validation performance.\n"
            "Projections follow the source cutoff, not the date the app is opened.\n"
            "Baseline and user scenarios are separate. Adjustments are assumptions, not intervals.\n"
            "Low-category labels are seasonal categories, not crop-failure diagnoses.\n"
            "Clusters are descriptive profiles; anomaly flags are records to review.\n"
            "Factor associations and model reliance do not establish agricultural causes.\n"
            "Confirm area/product units before using production or price calculations.\n"
        ))
    return buf.getvalue()
