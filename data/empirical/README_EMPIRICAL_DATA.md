# Calibrated Computational Benchmark Suite: Documentation & Verification

**Study:** Pilot, Scale, or Wait? Learning and Staged Fleet Electrification Under Carbon Budgets
**Directory:** data/empirical/ and data/empirical/scenarios/
**Scientific Status:** Calibrated Computational Simulation Package (Synthetic trajectories with source-attributed inputs and assumptions; not measured carrier operational logs; not all attributions independently verified).
**Field-Level Provenance:** Detailed in research/PROVENANCE_AND_FIELD_MAPPING.md.
**Temporal Horizon:** 260 Operating Business Days (Calendar Year 2025: 2025-01-02 to 2025-12-31).
**Fleet Architecture:** 10 Dedicated Physical Vehicles per operating day (zero vehicle scheduling overlaps).

## Availability and verification

The row-level operational CSV files and matched scenario files described below are included in the replication package. The critical source and input files match the SHA-256 digests recorded in `results/revised_study_metadata.json`, and the three automated test modules pass in the verified release. These files remain synthetic computational records rather than measured carrier data.

---

## 1. Operational Counts & Reconciliation

| Metric | Battery-Electric (BEV) | Diesel Internal Combustion | Total Fleet |
| :--- | :---: | :---: | :---: |
| Assigned Duties | 585 | 2,015 | 2,600 |
| Completed Duties (On-Time) | 579 | 1,988 | 2,567 |
| Late Duties (Completed without Recourse) | 4 | 18 | 22 |
| Outsourced Duties (3PL Recourse Contractor) | 2 | 9 | 11 |
| Duplicate Vehicle-Days | 0 | 0 | 0 |

- Electric Fleet Expansion Progression:
  - Quarter 1 (Days 1-65): EV-01 active on DUTY_URB_01 (65 BEV duty-days).
  - Quarter 2 (Days 66-130): EV-02 commissioned; joins EV-01 (130 BEV duty-days).
  - Quarters 3-4 (Days 131-260): EV-03 commissioned; all 3 urban duties electrified (390 BEV duty-days).
  - Total BEV assignments: 65 + 130 + 390 = 585 duty-days.

---

## 2. Telemetry and Information Nonanticipativity

- Enhanced Telemetry (EV-01, EV-02):
  - High-frequency CAN-bus direct traction monitoring.
  - Specified packet-loss parameter: 1.32%; realized in this frozen package as 4 dropped records out of 453 enhanced observations (0.88%) flagged as Dropped_Cellular_Deadzone with empty obs_traction_kwh.
  - Available at telemetry_reported_at (5 minutes post-shift).
- Baseline Metering (EV-03, Diesel):
  - obs_traction_kwh is strictly hidden across all 130 baseline EV observations.
  - Reporting timestamp (report_available_at) is set to the post-quarter accounting reconciliation date (45 days after quarter end: 2025-05-15, 2025-08-15, 2025-11-15, 2026-02-15), enforcing decision nonanticipativity.

---

## 3. Matched Counterfactual Scenario Datasets

Matched trajectories are included for the parameter support Theta = {1.15, 1.40, 1.75} kWh/km:
- duty_cycle_operations.csv: Primary baseline (theta* = 1.40 kWh/km, Mid-consumption).
- scenarios/duty_cycle_operations_theta_low.csv: Counterfactual favorable regime (theta* = 1.15 kWh/km).
- scenarios/duty_cycle_operations_theta_high.csv: Counterfactual adverse regime (theta* = 1.75 kWh/km).

---

## 4. Key Calibration Parameters and Public Sources

1. EPA eGRID2023 (January 15, 2025 release; June 12, 2025 revision 2). Published rates below were checked October 3, 2026; parentheses give rounded solver assumptions:
   - RFC East (RFCE): 599.170 lb/MWh = 0.271779 kg CO2e/kWh (0.000272 t/kWh)
   - WECC California (CAMX): 429.983 lb/MWh = 0.195037 kg CO2e/kWh (0.000195 t/kWh)
   - US National Average: 770.884 lb/MWh = 0.349667 kg CO2e/kWh (0.000350 t/kWh)
2. EPA GHG Emission Factors Hub (2025 Mobile Combustion):
   - Diesel Combustion: 10.21 kg CO2/gallon = 0.002697 tonnes CO2/liter
3. Depot Commercial Tariff:
   - Commonwealth Edison (ComEd) Rate BES / OpenEI URDB Rate ID: 5da63f0b5457a3253b2184bb
   - Energy: 0.14 USD/kWh, Monthly Peak Demand: 12.50 USD/kW.
4. Charging Efficiency:
   - eta = 0.88 is an assumed scenario value. The earlier EPRI/NREL attribution has not been tied to an identifiable report passage and must not be presented as a verified extraction.

## Source-verification boundary

See `research/SOURCE_VERIFICATION.md` for independently inspected evidence and unresolved attributions. Credit-adjusted asset values are hypothetical horizon-wide offsets, not a statement of current tax-credit eligibility. See `research/REPLICATION.md` for the intended synthetic-path evaluation workflow and the current missing-file warning; saved outputs do not convert this package into empirical data.
