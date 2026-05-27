import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drift_demo import add_deployment_features, evaluate_scenario, make_modified_current


class EvidentlyDriftDemoTests(unittest.TestCase):
    def test_add_deployment_features_matches_api_features(self):
        raw = pd.DataFrame(
            [
                {
                    "FL_DATE": "2023-12-23",
                    "AIRLINE_CODE": "ua",
                    "ORIGIN": "fll",
                    "DEST": "ewr",
                    "CRS_DEP_TIME": "55",
                    "CANCELLED": "0",
                }
            ]
        )

        result = add_deployment_features(raw).iloc[0]

        self.assertEqual(result["AIRLINE_CODE"], "UA")
        self.assertEqual(result["ORIGIN"], "FLL")
        self.assertEqual(result["DEST"], "EWR")
        self.assertEqual(result["ROUTE"], "FLL_EWR")
        self.assertEqual(result["DEP_HOUR"], 0)
        self.assertEqual(result["DEP_MIN"], 55)
        self.assertEqual(result["DEP_PERIOD"], "red_eye")
        self.assertEqual(result["IS_WEEKEND"], 1)
        self.assertEqual(result["IS_HOLIDAY_SEASON"], 1)

    def test_make_modified_current_changes_monitoring_distributions(self):
        reference = pd.DataFrame(
            {
                "AIRLINE_CODE": ["UA"] * 20,
                "ORIGIN": ["FLL"] * 20,
                "DEST": ["EWR"] * 20,
                "ROUTE": ["FLL_EWR"] * 20,
                "DEP_HOUR": [11] * 20,
                "DEP_MIN": [30] * 20,
                "DEP_PERIOD": ["morning"] * 20,
                "IS_HOLIDAY_SEASON": [0] * 20,
            }
        )

        current = make_modified_current(reference)

        self.assertEqual(len(current), len(reference))
        self.assertGreater((current["AIRLINE_CODE"] == "WN").sum(), 0)
        self.assertGreater((current["ORIGIN"] == "LAX").sum(), 0)
        self.assertGreater((current["DEP_PERIOD"] == "red_eye").sum(), 0)
        self.assertGreater((current["ROUTE"].str.startswith("LAX_")).sum(), 0)

    def test_evaluate_scenario_outputs_metrics_for_both_frames(self):
        reference = pd.DataFrame(
            {
                "CANCELLED": [0, 1],
                "cancellation_probability": [0.1, 0.2],
                "will_cancel": [0, 1],
            }
        )
        current = pd.DataFrame(
            {
                "CANCELLED": [0, 1],
                "cancellation_probability": [0.3, 0.5],
                "will_cancel": [1, 1],
            }
        )

        metrics = evaluate_scenario(reference, current)

        self.assertEqual(metrics["Scenario"].tolist(), ["Original_2023_Test", "Modified_Current"])
        self.assertEqual(metrics["Rows"].tolist(), [2, 2])
        self.assertAlmostEqual(metrics.loc[0, "Cancellation Rate"], 0.5)
        self.assertAlmostEqual(metrics.loc[1, "Predicted Cancellation Rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
