import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EMPIRICAL_DIR = DATA_DIR / "empirical"
SCENARIOS_DIR = EMPIRICAL_DIR / "scenarios"

from src.prototype_dp import (
    load_calibration,
    load_empirical_calibration,
    PrototypeDP,
    validate_csv_width
)


class EmpiricalPackageTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ops_mid = pd.read_csv(EMPIRICAL_DIR / "duty_cycle_operations.csv")
        cls.ops_low = pd.read_csv(SCENARIOS_DIR / "duty_cycle_operations_theta_low.csv")
        cls.ops_high = pd.read_csv(SCENARIOS_DIR / "duty_cycle_operations_theta_high.csv")
        cls.depot = pd.read_csv(EMPIRICAL_DIR / "depot_energy.csv")
        cls.assets = pd.read_csv(EMPIRICAL_DIR / "assets.csv")
        cls.mon = pd.read_csv(EMPIRICAL_DIR / "monitoring_protocols.csv")
        cls.policy = pd.read_csv(EMPIRICAL_DIR / "carbon_and_policy_inputs.csv")

    def test_rfc4180_column_alignment(self):
        """All empirical CSV tables must have perfectly aligned row widths."""
        for p in EMPIRICAL_DIR.glob("*.csv"):
            width, count = validate_csv_width(p)
            self.assertGreater(width, 0)
            self.assertGreater(count, 0)
        for p in SCENARIOS_DIR.glob("*.csv"):
            width, count = validate_csv_width(p)
            self.assertGreater(width, 0)
            self.assertGreater(count, 0)

    def test_vehicle_scheduling_non_duplication(self):
        """Zero overlapping assignments: exactly 10 distinct physical vehicles per operating day."""
        dups = self.ops_mid.groupby(["date", "vehicle_id"]).size()
        self.assertEqual(len(dups[dups > 1]), 0, "Found duplicate vehicle assignments on the same day")
        daily_vehicle_counts = self.ops_mid.groupby("date")["vehicle_id"].nunique()
        self.assertTrue((daily_vehicle_counts == 10).all(), "Each day must have exactly 10 distinct vehicles")

    def test_pairwise_matched_counterfactuals(self):
        """The three regime datasets must be strictly matched on all operational background conditions."""
        self.assertEqual(len(self.ops_mid), 2600)
        self.assertEqual(len(self.ops_low), 2600)
        self.assertEqual(len(self.ops_high), 2600)

        for col in ["date", "vehicle_id", "duty_id", "duty_class", "distance_km",
                    "duration_hr", "ambient_temp_c", "weather_cond", "service_status",
                    "is_outsourced", "telemetry_status", "departure_time", "return_time"]:
            diff_low = (self.ops_mid[col] != self.ops_low[col]).sum()
            diff_high = (self.ops_mid[col] != self.ops_high[col]).sum()
            self.assertEqual(diff_low, 0, f"Mismatch in {col} between mid and low regime datasets")
            self.assertEqual(diff_high, 0, f"Mismatch in {col} between mid and high regime datasets")

        # Verify strict monotonic energy scaling across non-outsourced electric duties
        ev_comp = (self.ops_mid["technology"] == "Electric") & (self.ops_mid["is_outsourced"] == 0)
        trac_mid = self.ops_mid.loc[ev_comp, "sim_true_traction_kwh"].astype(float)
        trac_low = self.ops_low.loc[ev_comp, "sim_true_traction_kwh"].astype(float)
        trac_high = self.ops_high.loc[ev_comp, "sim_true_traction_kwh"].astype(float)

        self.assertTrue((trac_low < trac_mid).all(), "trac_low must be strictly less than trac_mid")
        self.assertTrue((trac_mid < trac_high).all(), "trac_mid must be strictly less than trac_high")

    def test_operational_counts_reconciliation(self):
        """Counts across assigned, completed, late, and outsourced duties must reconcile exactly."""
        tech_counts = self.ops_mid["technology"].value_counts().to_dict()
        self.assertEqual(tech_counts["Electric"], 585)
        self.assertEqual(tech_counts["Diesel"], 2015)

        comp = self.ops_mid[self.ops_mid["service_status"] == "Completed"]["technology"].value_counts().to_dict()
        self.assertEqual(comp["Electric"], 579)
        self.assertEqual(comp["Diesel"], 1988)

        late = self.ops_mid[self.ops_mid["service_status"] == "Late"]["technology"].value_counts().to_dict()
        self.assertEqual(late["Electric"], 4)
        self.assertEqual(late["Diesel"], 18)

        out = self.ops_mid[self.ops_mid["service_status"] == "Outsourced"]["technology"].value_counts().to_dict()
        self.assertEqual(out["Electric"], 2)
        self.assertEqual(out["Diesel"], 9)

    def test_information_chronology_and_telemetry_masking(self):
        """Baseline EV rows must hide telemetry; enhanced records must observe missingness."""
        base_ev = self.ops_mid[self.ops_mid["monitoring_tier"] == "Baseline_Meter"]
        self.assertEqual(len(base_ev), 130)
        self.assertEqual(base_ev["obs_traction_kwh"].notna().sum(), 0, "Baseline EV exposed traction telemetry")

        # Verify baseline reporting availability is post-quarter + 45 days
        expected_recon_dates = {
            "2025-05-15T00:00:00", "2025-08-15T00:00:00",
            "2025-11-15T00:00:00", "2026-02-15T00:00:00"
        }
        self.assertTrue(set(base_ev["report_available_at"]).issubset(expected_recon_dates))

        # Enhanced telemetry missingness rate check
        enh_ev = self.ops_mid[self.ops_mid["monitoring_tier"] == "Enhanced_Telemetry"]
        self.assertEqual(len(enh_ev), 453)
        dropped = enh_ev[enh_ev["telemetry_status"] == "Dropped_Cellular_Deadzone"]
        self.assertEqual(len(dropped), 4)
        self.assertTrue(dropped["obs_traction_kwh"].isna().all())

    def test_asset_linkage_and_commissioning_dates(self):
        """All vehicles must link to assets and operations must respect commissioning dates."""
        asset_ids = set(self.assets["asset_id"])
        ops_veh_ids = set(self.ops_mid["vehicle_id"])
        self.assertTrue(ops_veh_ids.issubset(asset_ids))

        # Check commissioning date respect
        ev02_first = self.ops_mid[self.ops_mid["vehicle_id"] == "EV-02"]["date"].min()
        self.assertEqual(ev02_first, "2025-04-03", "EV-02 operated prior to Q2 commissioning")

        ev03_first = self.ops_mid[self.ops_mid["vehicle_id"] == "EV-03"]["date"].min()
        self.assertEqual(ev03_first, "2025-07-03", "EV-03 operated prior to Q3 commissioning")

    def test_unified_economic_loader(self):
        """Loader must import asset costs, monitoring subscriptions, and updated policy factors."""
        cal = load_empirical_calibration()
        p = cal.parameters

        self.assertEqual(p["c_E"], 185000.0)  # Average net capital cost of BEVs
        self.assertEqual(p["c_C"], 39000.0)   # Average net capital cost of 50kW DCFCs
        self.assertEqual(p["c_D"], 96200.0)   # Average diesel chassis cost
        self.assertEqual(p["c_M"], 112.50)    # Enhanced quarterly monitoring fee
        self.assertEqual(p["c_M_hardware"], 850.0)
        self.assertAlmostEqual(p["g_E_CAMX"], 0.000195, places=6)
        self.assertAlmostEqual(p["g_E_RFCE"], 0.000272, places=6)
        self.assertEqual(p["C_cap"], 300.0)
        self.assertEqual(p["b_1"], 836.0)

    def test_empirical_dp_solve(self):
        """PrototypeDP must compute a finite, feasible policy under empirical parameters."""
        cal = load_empirical_calibration()
        model = PrototypeDP(cal, "active")
        expected_cost = model.solve()
        self.assertTrue(expected_cost < float("inf"), "Empirical active scenario should be feasible")
        path = model.most_likely_path()
        self.assertEqual(len(path), 12, "Should evaluate a full 12-quarter horizon")


if __name__ == "__main__":
    unittest.main()
