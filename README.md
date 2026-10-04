# Computational Replication Package: Learning and Staged Fleet Electrification Under Carbon Budgets

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-30%20passed-brightgreen.svg)](tests/)

This repository contains the computational replication package, source code, test suites, simulation inputs, and experimental results for the study investigating the economics and carbon dynamics of staged fleet electrification under operational learning.

---

## 1. Overview and Key Findings

Fleet electrification involves irreversible commitments in vehicles and charging infrastructure before operating performance is fully understood. An operational pilot generates real-world information while serving freight, but delaying commitment also consumes depleting carbon headroom.

This study formulates a finite-horizon deployment-and-learning stochastic dynamic programming problem with delivery obligations, charging lead times, annual emission caps, and a cumulative carbon budget. 

### Principal Findings

1. **Global-Support Hard-Compliance Control (Null Finding):**  
   Under exact global-support hard compliance, enhanced telemetry does not alter the investment policy or expected cost ($924,568 finite-model objective). Because all initially possible energy regimes remain in the compliance envelope, learning cannot manufacture regulatory headroom by lowering posterior belief probabilities.
2. **Prespecified Risk-Screened Instance Family:**  
   Across six predefined instances evaluated under an explicit 1% whole-horizon risk screen:
   - **4 instances** exhibit **zero information value** (Base budget, Loose budget, Initial EV & charger, Changed duty mix).
   - **1 instance** (Diesel factor +3%) is **structurally infeasible** under the restricted action space.
   - **1 instance** (Tight budget, 832 tonnes) exhibits **positive telemetry value** ($100,407 and $101,925 savings across held-out seeds).
3. **The Cost–Carbon Trade-Off:**  
   The positive cost savings in the tight-budget instance arise because the informed policy avoids precautionary purchase of a second electric vehicle on approximately two-thirds of sample paths. Consequently, mean physical emissions under telemetry **increase by 8.68 to 8.84 tonnes CO2** relative to the baseline-only policy, demonstrating that operational information reduces precautionary investment rather than automatically deepening decarbonization.

---

## 2. Repository Structure

```text
.
├── src/                        # Core dynamic programming and simulation code
│   ├── prototype_dp.py         # Stochastic dynamic program and state transitions
│   ├── validate_study.py       # Monte Carlo path generator and policy replay harness
│   ├── run_revised_study.py    # Predefined instance runner and merge workflow
│   ├── generate_empirical_data.py # Calibrated simulation dataset generator
│   ├── run_experiments.py      # Benchmark experiment runner
│   └── audit_posterior.py      # Posterior support and belief audit
├── tests/                      # Automated unit and regression test suite (30 checks)
│   ├── test_prototype_dp.py    # DP state transitions, beliefs, and accounting checks
│   ├── test_validation_study.py# Policy replay, nonanticipativity, and evaluation checks
│   └── test_empirical_package.py # Schema and parameter integrity checks
├── data/                       # Input data and scenario definitions
│   ├── empirical/              # Row-level CSV inputs (assets, energy, duty cycles)
│   │   ├── assets.csv
│   │   ├── carbon_and_policy_inputs.csv
│   │   ├── depot_energy.csv
│   │   ├── duty_cycle_operations.csv
│   │   ├── monitoring_protocols.csv
│   │   └── scenarios/
│   ├── fleet_asset_cost_parameters.csv
│   ├── depot_daily_energy.csv
│   └── duty_cycle_operations_sample.csv
├── results/                    # Generated experimental results and metadata
│   ├── revised_parts/          # 36 isolated candidate JSON result payloads
│   ├── revised_candidate_screen.csv
│   ├── revised_evaluation.csv
│   ├── revised_instance_family.csv
│   └── revised_study_metadata.json # Input SHA-256 hashes, seeds, and configurations
├── figures/                    # High-resolution result figures (JPEG format)
│   └── revised_cost_carbon_tradeoff.jpg
├── LICENSE                     # MIT Open Source License
└── README.md                   # This replication guide
```

---

## 3. Replication Instructions

### Prerequisites

- Python 3.10 or higher
- Standard scientific Python stack:
  ```bash
  pip install numpy pandas scipy matplotlib pytest
  ```

### Step 1: Run Automated Integrity Tests

Run the full automated test suite (30 unit and regression tests):
```bash
python -m pytest tests/ -q
```
*Expected output:* `30 passed in ~11s`

### Step 2: Regenerate All Experimental Results

To re-execute all 36 candidate configurations across the six predefined instances and merge outputs:

```bash
python -c "
import subprocess, sys

instances = [
    'Tight budget', 'Base budget', 'Loose budget',
    'Initial EV and charger', 'One regional duty becomes suburban',
    'Diesel factor plus 3 percent'
]
infos = ['baseline', 'telemetry']
candidates = ['Global hard', 'Quantile 99%', 'Quantile 66%']

for inst in instances:
    for inf in infos:
        for cand in candidates:
            cmd = [sys.executable, 'src/run_revised_study.py', '--instance', inst, '--information', inf, '--candidate', cand]
            subprocess.run(cmd, check=True)

subprocess.run([sys.executable, 'src/run_revised_study.py', '--merge'], check=True)
print('Replication workflow completed successfully!')
"
```

To quickly merge existing candidate payloads and regenerate CSV outputs and figures:
```bash
python src/run_revised_study.py --merge
```

---

## 4. Input Verification and Provenance

Every input file used in the revised study is tracked by its SHA-256 hash in [`results/revised_study_metadata.json`](results/revised_study_metadata.json):

| File Path | SHA-256 Digest | Status |
|---|---|---|
| `src/prototype_dp.py` | `8310ffa44bbb3bd022e580bdd7a05bc1ca5433a42f0ef5a8e00097579cf8bd71` | Verified |
| `src/validate_study.py` | `dfe3faf82c00c2eaaedf77852aa057fc4750cf03f9069be49f9889599d5806d8` | Verified |
| `src/run_revised_study.py` | `d05ef6f9e94a7b1f362a5f11875b44304d1df8b8254cf9cb89cc2439c3875e87` | Verified |
| `data/empirical/assets.csv` | `7f03c31d09412ca17b2fa42842a08c10657f1dd80dd32f950656bd90174d1ef8` | Verified |
| `data/empirical/carbon_and_policy_inputs.csv` | `aa5036f6154f46bc7fe9155ee8fc6673f0b16f399532371942c2f21edb0f2a3a` | Verified |
| `data/empirical/depot_energy.csv` | `262ccfd510afb9288b5b50533edc87c1d6fc95497193989eeb4dfcd84ac20df2` | Verified |
| `data/empirical/duty_cycle_operations.csv` | `d9a5335b8b5f7cfd812748c03c232818ce298863d91aa5134bb037a67bf161ca` | Verified |
| `data/empirical/monitoring_protocols.csv` | `db51197a76dc36be1a4c3a7518eca89c7afb6000d10ca9c9c0d151b266d6e44b` | Verified |

---

## 5. License and Citation

This computational replication package is licensed under the [MIT License](LICENSE).

```bibtex
@misc{pilotscaleorwait2026,
  title={Replication Package: Learning and Staged Fleet Electrification Under Carbon Budgets},
  author={Anonymous Authors},
  year={2026},
  howpublished={\url{https://github.com/DeckardShaw31/Pilot-Scale-or-Wait-}}
}
```
