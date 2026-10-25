"""Run from the project folder: python -m unittest discover -s tests -v"""
import io
import json
import unittest
import zipfile
import numpy as np
import pandas as pd
from coconut import CoconutWorkbench
from coconut.preprocessing import regression_weights
from coconut.exports import build_results_zip


def fixture():
    rows = []
    for year in range(2018, 2026):
        for month in range(1, 13):
            for i in range(5):
                area = 10 + i
                yield_value = 600 + 100*np.sin(2*np.pi*month/12) + i*4 + (year-2018)*2
                rows.append({"Year": year, "Month": month, "Barangay": f"Place {i}",
                             "Harvested Area (ha)": area, "Harvest (kg)": area*yield_value,
                             "Rainfall (mm)": 80+month*3+i*15, "Temperature (°C)": 26+month*.03+i*.1,
                             "Inflation Rate (%)": 3.2})
    return pd.DataFrame(rows)


def loaded(frame):
    wb = CoconutWorkbench()
    wb.load_bytes(frame.to_csv(index=False).encode(), "fixture.csv")
    return wb


class WorkflowTests(unittest.TestCase):
    def test_missing_targets_remain_missing_and_partial_year_is_not_a_target(self):
        raw = fixture()
        raw.loc[(raw.Year == 2025) & (raw.Month == 2) & (raw.Barangay == "Place 0"), "Harvest (kg)"] = np.nan
        wb = loaded(raw)
        row = wb.annual[(wb.annual.year == 2025) & (wb.annual.barangay == "Place 0")].iloc[0]
        self.assertEqual(row.harvest_months, 11)
        self.assertTrue(pd.isna(row.annual_yield_kg_ha))
        self.assertTrue(pd.isna(row.annual_production_kg))
        self.assertEqual(int(wb.monthly.yield_kg_ha_month.isna().sum()), 1)

    def test_repeated_monthly_hectares_are_used_once_and_inconsistent_area_is_rejected(self):
        raw = fixture()
        wb = loaded(raw)
        first = wb.annual[(wb.annual.year == 2018) & (wb.annual.barangay == "Place 0")].iloc[0]
        self.assertAlmostEqual(first.annual_yield_kg_ha, 7200)
        raw.loc[(raw.Year == 2018) & (raw.Month == 12) & (raw.Barangay == "Place 0"), "Harvested Area (ha)"] = 99
        changed = loaded(raw)
        target = changed.annual[(changed.annual.year == 2018) & (changed.annual.barangay == "Place 0")].iloc[0]
        self.assertFalse(target.annual_target_complete)

    def test_duplicate_keys_are_rejected(self):
        raw = fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate barangay-month"):
            loaded(pd.concat([raw, raw.iloc[[0]]], ignore_index=True))

    def test_imbalance_weights_are_finite_and_do_not_create_records(self):
        values = np.array([10.0]*90 + [50.0]*8 + [100.0]*2)
        weights, edges, report = regression_weights(values)
        self.assertEqual(len(weights), len(values))
        self.assertAlmostEqual(weights.mean(), 1)
        self.assertTrue(np.isfinite(weights).all())
        self.assertGreater(weights[-1], weights[0])
        constant_weights, _, _ = regression_weights(np.ones(20))
        np.testing.assert_allclose(constant_weights, 1)

    def test_chronological_scoring_forecast_dates_and_separate_scenarios(self):
        wb = loaded(fixture())
        observed_before = wb.monthly.yield_kg_ha_month.copy()
        metrics = wb.fit_monthly_models(names=["Seasonal median", "Harmonic regression"], trees=40)
        self.assertEqual(set(metrics[metrics.split == "validation"].year), {2022, 2023, 2024})
        self.assertTrue(metrics[metrics.split == "final test"].n.eq(60).all())
        future = wb.predict_next_three_months()
        self.assertEqual(future.date.min(), pd.Timestamp("2026-01-01"))
        self.assertEqual(len(future), 15)
        wb.compare_models(horizon=1, names=["Historical median", "Last observed yield"], trees=40)
        baseline = wb.future.copy(deep=True)
        scenario = wb.make_scenario(5, -10)
        pd.testing.assert_frame_equal(wb.future, baseline)
        np.testing.assert_allclose(scenario.planning_production_kg, scenario.predicted_production_kg*.9)
        pd.testing.assert_series_equal(wb.monthly.yield_kg_ha_month, observed_before)
        archive = build_results_zip(wb, scenario)
        with zipfile.ZipFile(io.BytesIO(archive)) as output:
            self.assertIsNone(output.testzip())
            self.assertIn("annual_baseline_projections.csv", output.namelist())
            self.assertIn("annual_user_scenario.csv", output.namelist())
            meta = json.loads(output.read("audit_and_settings.json"))
            self.assertEqual(meta["scenario"]["area_growth_pct"], 5)


if __name__ == "__main__":
    unittest.main()
