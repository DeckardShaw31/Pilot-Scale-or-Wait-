# Data Dictionary, Calibration Protocol, and Prototype Conventions

**Manuscript Reference:** `main.tex` Section 4 (Model formulation), Section 5 (Properties), Section 6 (Solution approach), Section 7 (Calibration), and Appendix A (`main.tex:558-592`)  
**Status:** Synthetic diagnostic test fixtures for schema validation and dynamic programming prototype testing (not yet validated empirical public calibration).

**Archived companion files (not present in the current workspace snapshot):**
- `data/duty_cycle_operations_sample.csv` — previously described as 70 synthetic vehicle-day records across 5 vehicles and 14 days.
- `data/depot_daily_energy.csv` — previously described as a synthetic shared depot-facility baseload table.
- `data/fleet_asset_cost_parameters.csv` — previously described as a synthetic financial, charging, lead-time, noise, and carbon-configuration table.

The earlier documentation linked to a private Windows path and asserted RFC 4180 validation without retaining the files. Those links and validation claims are not usable evidence. Restore the exact hashed inputs before treating this dictionary as a reproducible package description.

---

## 1. Energy Accounting and Upper-Envelope Analysis

The data schema partitions vehicle energy into four distinct quantities to reconcile battery depletion, auxiliary loads, and grid utility purchases:

1. **Traction Energy ($e^{\text{trac}}_{jvtd}$, kWh):**
   $$e^{\text{trac}}_{jvtd} = \ell_{jt}\theta_j + \epsilon^{\text{trac}}_{jvtd}$$
   Governed by duty distance $\ell_{jt}$, persistent efficiency regime $\theta_j$, and zero-mean residual road/driver variation $\epsilon^{\text{trac}}_{jvtd} \sim \mathcal{N}(0, \sigma_{\text{trac}}^2)$ truncated to $[-10.0, +10.0]\text{ kWh}$ with $\sigma_{\text{trac}} = 3.5\text{ kWh}$.
2. **Onboard Vehicle Auxiliary Energy ($e^{\text{aux,veh}}_{jvtd}$, kWh):**
   $$e^{\text{aux,veh}}_{jvtd} = a_0 + a_1 \cdot (15 - T_{\text{amb}}) + \epsilon^{\text{aux}}_{jvtd}$$
   Power draw from cabin HVAC, battery thermal conditioning, and 12V/24V chassis electronics. Bounded strictly in $[5.0, 20.0]\text{ kWh}$ (mean $\approx 7.5\text{ kWh}$ at $15^\circ\text{C}$, $\sigma_{\text{aux}} = 1.5\text{ kWh}$).
3. **Total Battery Withdrawal ($e^{\text{batt}}_{jvtd}$, kWh):**
   $$e^{\text{batt}}_{jvtd} = e^{\text{trac}}_{jvtd} + e^{\text{aux,veh}}_{jvtd}$$
   **Urban Worst-Case Battery Feasibility Envelope:**
   For urban duty D1 ($\ell_{\text{D1}} = 85\text{ km}$) under the high-consumption regime ($\theta_{\text{high}} = 1.75\text{ kWh/km}$):
   $$\bar{e}^{\text{batt}}_{\text{D1}} = 85 \times 1.75 + 10.0 + 20.0 = 148.75 + 10.0 + 20.0 = 178.75\text{ kWh} \le Q = 200.0\text{ kWh}$$
   This confirms that under simultaneous upper-envelope traction and auxiliary outcomes, the urban duty remains strictly feasible without mid-day opportunity charging.
4. **Grid-Purchased Vehicle Energy ($e^{\text{grid}}_{jvtd}$, kWh):**
   $$e^{\text{grid}}_{jvtd} = \frac{e^{\text{batt}}_{jvtd}}{\eta}$$
   Includes AC-DC conversion and battery electrochemical round-trip charging resistance ($\eta = 0.88$). Worst-case urban grid draw is $178.75 / 0.88 = 203.125\text{ kWh/cycle}$.

---

## 2. Vehicle–Duty Compatibility Matrix ($\Xi_{jq}$)

| Duty Class | Distance | Traction ($\theta_{\text{high}}=1.75$) | Max Aux Load | Worst-Case Battery Draw | Compatibility $\Xi_{j,E}$ | Compatibility $\Xi_{j,D}$ |
|---|---|---|---|---|---|---|
| **D1 (Urban)** | 85 km | 148.75 kWh | 20.0 kWh | **178.75 kWh $\le$ 200** | **1 (Electrifiable)** | **1 (Compatible)** |
| **D2 (Suburban)**| 135 km | 236.25 kWh | 20.0 kWh | **266.25 kWh > 200** | **0 (Incompatible)** | **1 (Compatible)** |
| **D3 (Regional)**| 210 km | 367.50 kWh | 20.0 kWh | **397.50 kWh > 200** | **0 (Incompatible)** | **1 (Compatible)** |

- **Urban Duty (D1):** Candidate for electrification, pilots, and Bayesian learning ($n_{\text{D1}} = 3$ bundles/cycle).
- **Suburban & Regional Duties (D2, D3):** Retained as diesel-only duties ($n_{\text{D2}} = 4, n_{\text{D3}} = 3$ bundles/cycle). Their emissions form a structural baseline floor.

---

## 3. Carbon Constraints & Experimental Configurations

### Analytical Floor of Baseline Diesel Operations:
- Operating schedule: $w_t = 65$ cycles/quarter $\times 4 = 260$ operating days/year.
- Diesel combustion factor: $g^D = 0.002697\text{ tonnes CO}_2/\text{liter}$ (EPA 2025 Mobile Combustion Table).
- Average diesel consumption from sample: $f_{\text{D2}} \approx 37.82\text{ L/cycle}$, $f_{\text{D3}} \approx 57.25\text{ L/cycle}$.
- **Diesel-Only Duties Emissions (D2 + D3):**
  $$260 \times (4 \times 37.82 + 3 \times 57.25) \times 0.002697 = 260 \times 323.03 \times 0.002697 \approx 226.52\text{ tonnes CO}_2/\text{year}$$
- **All-Diesel Urban Duties (D1):**
  $$260 \times (3 \times 25.6\text{ L}) \times 0.002697 \approx 53.86\text{ tonnes CO}_2/\text{year}$$
- **Depot Facility Baseline ($A_{td} \approx 15\text{ kWh/day}$ at $g^E = 0.000272\text{ t/kWh}$):**
  $$260 \times 15.0 \times 0.000272 \approx 1.06\text{ tonnes CO}_2\text{e}/\text{year}$$
- **Total All-Diesel Fleet Emissions:** $\approx 281.44\text{ tonnes CO}_2/\text{year}$ ($\approx 844.32\text{ tonnes}$ over 3 years).

### Why $C^{\text{cap}} = 220\text{ t/year}$ is Infeasible:
The diesel-only duties alone generate $226.52\text{ t/year} > 220\text{ t/year}$. Furthermore, third-party spot outsourcing factors ($g^O_{\text{D2}} = 0.105, g^O_{\text{D3}} = 0.160\text{ t/bundle}$) exceed internal diesel emissions ($0.102$ and $0.154\text{ t/bundle}$), meaning outsourcing cannot relieve this floor.

### Experimental Configurations in `data/fleet_asset_cost_parameters.csv`:
1. **`Infeasible_Diagnostic` ($C^{\text{cap}} = 220\text{ t/yr}, b_1 = 800\text{ t}$):**
   Used to confirm that the prototype dynamic program correctly detects structural infeasibility without masking violations via ad-hoc penalties (`main.tex:263`). Notice that under a 220 t cap, the cumulative budget of 800 t is mathematically redundant since $3 \times 220 = 660 < 800$.
2. **`Transition_Active` ($C^{\text{cap}} = 300\text{ t/yr}, b_1 = 836\text{ t}$):**
   The adaptive learning diagnostic setting.
   - One charger is ordered in Quarter 2 and one electric truck in Quarter 3, making the pilot operational in Quarter 5 under the prototype timing convention.
   - A coarse baseline aggregate signal is observed after the first pilot quarter; enhanced traction telemetry is selected in Quarter 6.
   - Low- and mid-energy telemetry signals retain one electric truck. The high-energy signal triggers a second charger in Quarter 7 and a second truck in Quarter 8, with both available in Quarter 10.
   - The 836 t budget is a synthetic mechanism-test value, not an observed regulatory limit. It is deliberately positioned so that the posterior regime changes the required fleet scale.
3. **`Unconstrained_Control` ($C^{\text{cap}} = 400\text{ t/yr}, b_1 = 1200\text{ t}$):**
   Non-binding carbon constraints; isolates purely financial TCO and learning dynamics.

---

## 4. State Representation and Two Annual Ledgers

To model annual caps and investment limits within quarterly decision epochs ($T = 12$ quarters, 3 years), the state vector is:
$$s_t = (N_t^D, N_t^E, k_t, \Gamma_t, b_t, b_t^{\text{yr}}, W_t, W_t^{\text{yr}}, \beta_t, h_t^Z, h_t^Y)$$
where $\tau(t) = ((t-1) \pmod 4) + 1 \in \{1, 2, 3, 4\}$ indexes the quarter within the calendar year.
The indicator $h_t^Z$ records whether the one-time coarse baseline calibration signal has already been observed. This prevents repeated quarterly bills affected by the same persistent allocation confounder from being counted as independent evidence.
The indicator $h_t^Y$ records whether the one-time enhanced pilot-evaluation signal has been collected. The minimal prototype therefore evaluates whether and when to conduct one enhanced review, rather than allowing repeated telemetry purchases to create an unlimited sequence of independent experiments.

### Transitions:
- **Cumulative Carbon Budget ($b_t$):**
  $$b_{t+1} = b_t - C_t$$
- **Annual Carbon Headroom Ledger ($b_t^{\text{yr}}$):**
  $$\text{Within year } (\tau(t) \in \{1, 2, 3\}): \quad b_{t+1}^{\text{yr}} = b_t^{\text{yr}} - C_t$$
  $$\text{At year boundary } (\tau(t) = 4): \quad b_{t+1}^{\text{yr}} = C^{\text{cap}}$$
- **Cumulative Capital Budget ($W_t$):**
  $$W_{t+1} = W_t - K_t$$
- **Annual Capital Allowance Ledger ($W_t^{\text{yr}}$):**
  $$\text{Within year } (\tau(t) \in \{1, 2, 3\}): \quad W_{t+1}^{\text{yr}} = W_t^{\text{yr}} - K_t$$
  $$\text{At year boundary } (\tau(t) = 4): \quad W_{t+1}^{\text{yr}} = B_{\text{yr}} = \$500,000$$
- **Admissibility Envelopes at Period $t$:**
  $$\overline{C}_t(a_t) \le b_t^{\text{yr}}, \qquad \overline{C}_t(a_t) \le b_t, \qquad K_t \le W_t^{\text{yr}}, \qquad K_t \le W_t$$

---

## 5. Bayesian Observation Model and Likelihood Functions

Let $\Theta = \{\theta_{\text{low}}, \theta_{\text{mid}}, \theta_{\text{high}}\} = \{1.15, 1.40, 1.75\}\text{ kWh/km}$, with initial prior $\beta_1 = \left(\frac{1}{3}, \frac{1}{3}, \frac{1}{3}\right)$.

### 1. Enhanced Telemetry Likelihood ($\mathcal{L}_t(Y_t \mid \theta)$):
When $z_{jvt} = 1$, vehicle CAN-bus logs report average daily traction energy over $w_t = 65$ cycles:
$$Y_{jvt} = \frac{1}{w_t} \sum_{d=1}^{w_t} e^{\text{trac}}_{jvtd} + \nu_{jvt}$$
where sensor measurement noise is $\nu_{jvt} \sim \mathcal{N}(0, \sigma_{\nu}^2)$ with $\sigma_{\nu} = 0.5\text{ kWh}$.
By independence of daily cycles, the sample mean traction residual has variance $\frac{\sigma_{\text{trac}}^2}{w_t} = \frac{3.5^2}{65} \approx 0.1885\text{ kWh}^2$.
Thus:
$$Y_{jvt} \mid \theta \sim \mathcal{N}\left(85\theta, \frac{\sigma_{\text{trac}}^2}{65} + \sigma_{\nu}^2\right) = \mathcal{N}(85\theta, 0.4385)$$
Standard deviation is $\sqrt{0.4385} \approx 0.662\text{ kWh}$, providing sharp discrimination between:
- $\E[Y \mid \theta_{\text{low}}] = 85 \times 1.15 = 97.75\text{ kWh}$
- $\E[Y \mid \theta_{\text{mid}}] = 85 \times 1.40 = 119.00\text{ kWh}$
- $\E[Y \mid \theta_{\text{high}}] = 85 \times 1.75 = 148.75\text{ kWh}$

### 2. Baseline Aggregate Metering Likelihood ($\mathcal{L}_t(Z_t^E \mid \theta)$):
The baseline policy observes total depot grid electricity at epoch end:
$$Z_t^E = \sum_{d=1}^{w_t} \left(\sum_{v=1}^{N_t^E} \frac{e^{\text{batt}}_{\text{D1},v,t,d}}{\eta} + A_{td}\right) + \nu_Z$$
where daily facility variation contributes $\sigma_Z = 5.0\text{ kWh/cycle}$ and unresolved quarterly allocation and reconciliation error contributes $\sigma_{Z,\mathrm{qtr}}=1600\text{ kWh/quarter}$. The baseline observation pools vehicle charging with depot electricity and gives one coarse calibration signal. Subsequent bills are not treated as independent replications of the same persistent confounding error. Enhanced telemetry isolates traction energy and can therefore change the posterior before a later scale commitment.

### 3. Posterior-Supported Carbon Envelope

Battery and charging feasibility retain the global regime support $\Theta$. Carbon accounting in the adaptive diagnostic uses
$$\Theta_t^{\varepsilon}=\{\theta\in\Theta:\beta_t(\theta)>\varepsilon\}$$
and debits the annual and cumulative ledgers using the worst regime remaining in this numerical support. The prototype tolerance is $10^{-7}$ after categorical likelihood updating. This convention creates an information-sensitive compliance margin without allowing learning to change physical battery capacity. It is a diagnostic approximation; the calibrated study must compare it with global-support and realized-emissions evaluations.

---

## 6. Prototype Sizing and Grid Upgrade Scoping

- **Fleet Demand:** $n_{\text{D1}} = 3$ (Urban), $n_{\text{D2}} = 4$ (Suburban), $n_{\text{D3}} = 3$ (Regional). Initial fleet: $N_0^D = 10, N_0^E = 0, k_0 = 0$.
- **Grid Upgrade Scoping:**
  With at most 3 electrifiable urban duties ($N_t^E \le 3$), maximum depot charging power with three 50 kW plugs is:
  $$3 \times 50\text{ kW} = 150\text{ kW} < G_0 = 250\text{ kW}$$
  Total overnight energy draw for 3 EVs plus depot baseload is:
  $$3 \times 136.4\text{ kWh} + 16.0\text{ kWh} = 425.2\text{ kWh} \ll G_0 \cdot h_t = 2,500\text{ kWh}$$
  Therefore, grid substation upgrades ($c_G, L_G$) are **strictly non-binding and omitted** from the minimal prototype, keeping the state and action spaces compact.
