import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drift_demo import (
    build_monitoring_row,
    categorical_psi,
    create_modified_current_rows,
)


class DriftDemoTests(unittest.TestCase):
    def test_build_monitoring_row_creates_deployment_features(self):
        raw = {
            "FL_DATE": "2023-12-23",
            "AIRLINE_CODE": "ua",
            "ORIGIN": "fll",
            "DEST": "ewr",
            "CRS_DEP_TIME": "55",
            "CANCELLED": "0",
        }

        row = build_monitoring_row(raw)

        self.assertEqual(row["YEAR"], 2023)
        self.assertEqual(row["AIRLINE_CODE"], "UA")
        self.assertEqual(row["ORIGIN"], "FLL")
        self.assertEqual(row["DEST"], "EWR")
        self.assertEqual(row["ROUTE"], "FLL_EWR")
        self.assertEqual(row["DEP_HOUR"], 0)
        self.assertEqual(row["DEP_MIN"], 55)
        self.assertEqual(row["DEP_PERIOD"], "red_eye")
        self.assertEqual(row["IS_WEEKEND"], 1)
        self.assertEqual(row["IS_HOLIDAY_SEASON"], 1)

    def test_create_modified_current_rows_changes_selected_features(self):
        rows = [
            {
                "AIRLINE_CODE": "UA",
                "ORIGIN": "FLL",
                "DEST": "EWR",
                "ROUTE": "FLL_EWR",
                "DEP_HOUR": 11,
                "DEP_MIN": 30,
                "DEP_PERIOD": "morning",
                "IS_WEEKEND": 0,
                "IS_HOLIDAY_SEASON": 0,
                "IS_SUMMER_PEAK": 0,
                "IS_COVID_YEAR": 0,
            }
            for _ in range(10)
        ]

        modified = create_modified_current_rows(rows)

        self.assertEqual(len(modified), len(rows))
        self.assertGreater(sum(row["AIRLINE_CODE"] == "WN" for row in modified), 0)
        self.assertGreater(sum(row["ORIGIN"] == "LAX" for row in modified), 0)
        self.assertGreater(sum(row["DEP_PERIOD"] == "red_eye" for row in modified), 0)

    def test_categorical_psi_detects_distribution_shift(self):
        reference = ["UA"] * 90 + ["WN"] * 10
        current = ["UA"] * 50 + ["WN"] * 50

        score = categorical_psi(reference, current)

        self.assertGreater(score, 0.25)


if __name__ == "__main__":
    unittest.main()
