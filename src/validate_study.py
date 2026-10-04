"""Independent synthetic-path evaluation and numerical-refinement experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm, t

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prototype_dp import (
    Calibration, HORIZON_QUARTERS, MAX_ELECTRIC_DUTIES, PrototypeDP,
    THETA, URBAN_DISTANCE_KM, load_empirical_calibration,
)


@dataclass(frozen=True)
class PotentialPaths:
    regimes: np.ndarray
    traction: np.ndarray
    battery: np.ndarray
    facility: np.ndarray
    telemetry_noise: np.ndarray
    missingness_noise: np.ndarray
    meter_noise: np.ndarray
    contamination: np.ndarray
    replacement_signal: np.ndarray


def generate_paths(calibration: Calibration, count: int, seed: int) -> PotentialPaths:
    if count < 2:
        raise ValueError("At least two independent paths are required")
    parameters = calibration.parameters
    generator = np.random.default_rng(seed)
    regimes = generator.choice(len(THETA), count, p=(1 / 3, 1 / 3, 1 / 3))
    cycles = int(parameters["w_t"])
    traction = np.empty((count, HORIZON_QUARTERS, MAX_ELECTRIC_DUTIES))
    battery = np.empty_like(traction)
    facility = np.empty((count, HORIZON_QUARTERS))
    depot = pd.read_csv("data/empirical/depot_energy.csv").facility_aux_kwh.to_numpy()
    mean_energy = URBAN_DISTANCE_KM * np.asarray(THETA)[regimes, None, None]
    for quarter in range(HORIZON_QUARTERS):
        residual = generator.normal(
            0, parameters["sigma_trac"], (count, cycles, MAX_ELECTRIC_DUTIES)
        ).clip(parameters["eps_trac_min"], parameters["eps_trac_max"])
        auxiliary = generator.normal(
            (parameters["aux_min"] + parameters["aux_max"]) / 2,
            (parameters["aux_max"] - parameters["aux_min"]) / 6,
            residual.shape,
        ).clip(parameters["aux_min"], parameters["aux_max"])
        daily_traction = mean_energy + residual
        traction[:, quarter] = daily_traction.sum(axis=1)
        battery[:, quarter] = (daily_traction + auxiliary).sum(axis=1)
        facility[:, quarter] = generator.choice(depot, (count, cycles)).sum(axis=1)
    shape = (count, HORIZON_QUARTERS)
    result = PotentialPaths(
        regimes, traction, battery, facility,
        generator.normal(size=shape), generator.normal(size=shape),
        generator.normal(size=shape), generator.random(shape), generator.random(shape),
    )
    for array in vars(result).values():
        array.setflags(write=False)
    return result


def thresholds(means: tuple[float, ...], categories: int) -> np.ndarray:
    midpoints = np.asarray([(left + right) / 2 for left, right in zip(means, means[1:])])
    if categories == 3:
        return midpoints
    return np.linspace(midpoints[0], midpoints[-1], categories - 1)


def mean_interval(values: np.ndarray) -> tuple[float, float, float]:
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Intervals require at least two finite independent paths")
    mean = float(np.mean(values))
    half_width = float(t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))
    return mean, mean - half_width, mean + half_width


def rate_interval(events: np.ndarray) -> tuple[float, float, float]:
    count = len(events)
    rate = float(np.mean(events))
    critical = norm.ppf(0.975)
    denominator = 1 + critical ** 2 / count
    center = (rate + critical ** 2 / (2 * count)) / denominator
    half_width = critical * math.sqrt(rate * (1 - rate) / count + critical ** 2 / (4 * count ** 2)) / denominator
    return rate, center - half_width, center + half_width


def replay(
    model: PrototypeDP | tuple[PrototypeDP, ...], paths: PotentialPaths, *, contamination_rate: float = 0.0
) -> pd.DataFrame:
    oracle_models = model if isinstance(model, tuple) else None
    if oracle_models is not None:
        model = oracle_models[0]
    parameters = model.parameters
    cycles = parameters["w_t"]
    posterior_cache = {}
    results = []
    for path_index in range(len(paths.regimes)):
        if oracle_models is not None:
            model = oracle_models[int(paths.regimes[path_index])]
        state = model.initial_state()
        cost = 0.0
        emissions = np.zeros(HORIZON_QUARTERS)
        monitored_quarters = 0
        maximum_electric = 0
        failure = False
        for quarter in range(HORIZON_QUARTERS):
            action = model.policy.get(state)
            if action is None:
                failure = True
                break
            electric = model.electric_duties(state)
            maximum_electric = max(maximum_electric, electric)
            monitored_quarters += int(action.monitored_vehicles > 0)
            regime = int(paths.regimes[path_index])
            truth_belief = tuple(float(index == regime) for index in range(len(THETA)))
            truth_state = replace(state, belief=truth_belief)
            expected_grid = cycles * (
                electric * (URBAN_DISTANCE_KM * THETA[regime] + (parameters["aux_min"] + parameters["aux_max"]) / 2) / parameters["eta"]
                + model.calibration.facility_mean_kwh
            )
            purchased_grid = paths.battery[path_index, quarter, :electric].sum() / parameters["eta"] + paths.facility[path_index, quarter]
            investment = model.investment_cost(action)
            actual_operating = model.operating_cost(truth_state, action.monitored_vehicles) + parameters["p_E"] * (purchased_grid - expected_grid)
            cost += parameters["delta"] ** quarter * (model.from_money_units(investment) + actual_operating)
            diesel = cycles * (
                (MAX_ELECTRIC_DUTIES - electric) * model.calibration.diesel_litres["D1_Urban"]
                + parameters["n_D2"] * model.calibration.diesel_litres["D2_Suburban"]
                + parameters["n_D3"] * model.calibration.diesel_litres["D3_Regional"]
            )
            emissions[quarter] = diesel * parameters["g_D"] + purchased_grid * parameters[model.grid_factor_symbol]
            baseline_parameters = model.channel_signal_parameters(state, "baseline")
            enhanced_parameters = None
            if action.monitored_vehicles > 0 and not model.suppress_enhanced_signal:
                enhanced_parameters = model.channel_signal_parameters(
                    state, "enhanced", action.monitored_vehicles
                )
            baseline_category = None
            if baseline_parameters is not None:
                means, _ = baseline_parameters
                generating_quarter = quarter - 1 if state.pending_baseline > 0 else quarter
                exposure = state.pending_baseline or electric
                observed = paths.battery[path_index, generating_quarter, :exposure].sum() / parameters["eta"] + paths.facility[path_index, generating_quarter]
                measurement_std = math.sqrt(cycles * parameters["sigma_Z"] ** 2 + parameters["sigma_Z_qtr"] ** 2)
                observed += measurement_std * paths.meter_noise[path_index, generating_quarter]
                baseline_category = int(
                    np.searchsorted(thresholds(means, model.signal_categories), observed)
                )
            enhanced_category = None
            if enhanced_parameters is not None:
                means, _ = enhanced_parameters
                monitored = action.monitored_vehicles
                effective_count = max(1.0, cycles * monitored * (1 - parameters.get("missingness_rate", 0.0)))
                additional_variance = parameters["sigma_trac"] ** 2 * max(0.0, 1 / effective_count - 1 / (cycles * monitored))
                observed = paths.traction[path_index, quarter, :monitored].sum() / (cycles * monitored)
                observed += parameters["sigma_nu"] * paths.telemetry_noise[path_index, quarter]
                observed += math.sqrt(additional_variance) * paths.missingness_noise[path_index, quarter]
                enhanced_category = int(
                    np.searchsorted(thresholds(means, model.signal_categories), observed)
                )
                if paths.contamination[path_index, quarter] < contamination_rate:
                    enhanced_category = min(
                        model.signal_categories - 1,
                        int(paths.replacement_signal[path_index, quarter] * model.signal_categories),
                    )
            if baseline_category is not None and enhanced_category is not None:
                category = baseline_category * model.signal_categories + enhanced_category
            elif baseline_category is not None:
                category = baseline_category
            else:
                category = enhanced_category
            key = (state, action.monitored_vehicles)
            if key not in posterior_cache:
                conditional = model.signal_probabilities(state, action.monitored_vehicles)
                posterior_cache[key] = {
                    branch_category: posterior
                    for _, posterior, branch_category in model.posterior_branches(state.belief, conditional)
                }
            if category not in posterior_cache[key]:
                failure = True
                break
            state = model.transition(
                state, action, posterior_cache[key][category],
                model.robust_carbon(state), investment,
            )
        if not failure:
            cost += parameters["delta"] ** HORIZON_QUARTERS * model.terminal_value(state)
        results.append({
            "cost_usd": cost if not failure else math.nan,
            "carbon_tonnes": float(emissions.sum()) if not failure else math.nan,
            "carbon_violation": bool(emissions.sum() > model.from_carbon_units(model.cumulative_carbon_limit) + 1e-9 or np.any(emissions.reshape(3, 4).sum(axis=1) > model.from_carbon_units(model.annual_carbon_limit) + 1e-9)) if not failure else False,
            "policy_failure": failure,
            "maximum_electric_vehicles": maximum_electric,
            "monitoring_selected": monitored_quarters > 0,
        })
    return pd.DataFrame(results)


def build_policies(calibration: Calibration) -> dict[str, PrototypeDP | tuple[PrototypeDP, ...]]:
    models = {}
    for name, strategy, monitoring, suppression in (
        ("Immediate scale", "immediate_scale", False, False),
        ("Flexible timing", "flexible", False, False),
        ("Fixed pilot", "fixed_pilot", False, False),
        ("Joint policy", "joint", True, False),
        ("Matched ablation", "matched_ablation", True, True),
    ):
        model = PrototypeDP(calibration, "active", strategy=strategy, allow_monitoring=monitoring, suppress_enhanced_signal=suppression)
        if not math.isfinite(model.solve()):
            raise ValueError(f"Cannot evaluate infeasible policy: {name}")
        models[name] = model
    oracle_models = []
    for regime in range(len(THETA)):
        prior = tuple(float(index == regime) for index in range(len(THETA)))
        oracle = PrototypeDP(calibration, "active", strategy="oracle", allow_monitoring=False, initial_belief=prior)
        if not math.isfinite(oracle.solve()):
            raise ValueError("Cannot evaluate an infeasible oracle")
        oracle_models.append(oracle)
    models["Parameter oracle"] = tuple(oracle_models)
    return models


def refinement(calibration: Calibration) -> pd.DataFrame:
    cases = (
        ("Base", {}),
        ("Belief precision 4", {"belief_digits": 4}),
        ("Belief precision 5", {"belief_digits": 5}),
        ("Five signal categories", {"signal_categories": 5}),
        ("Seven signal categories", {"signal_categories": 7}),
        ("Literal positive support", {"support_tolerance": 0.0}),
        ("Support tolerance 0.001", {"support_tolerance": 0.001}),
        ("Denser budget 834", {"b_1_active": 834}),
        ("Denser budget 838", {"b_1_active": 838}),
        ("No incentive offsets", {"c_E": calibration.parameters["c_E"] + 40000, "c_C": calibration.parameters["c_C"] + 7500}),
        ("Half charger demand", {"charger_kw": 25}),
        ("Diesel factor plus 3 percent", {"g_D": 1.03 * calibration.parameters["g_D"]}),
        ("One regional duty becomes suburban", {"n_D2": 5, "n_D3": 2}),
        ("No EV terminal credit", {"s_E": 0}),
        ("Initial EV and charger", {"N_0_E": 1, "k_0": 1}),
    )
    rows = []
    for case, updates in cases:
        parameters = dict(calibration.parameters)
        parameters.update(updates)
        revised = replace(calibration, parameters=parameters)
        for strategy, monitoring in (("joint", True), ("flexible", False)):
            model = PrototypeDP(revised, "active", strategy=strategy, allow_monitoring=monitoring)
            started = time.perf_counter()
            objective = model.solve()
            elapsed = time.perf_counter() - started
            path = model.most_likely_path() if math.isfinite(objective) else []
            rows.append({"case": case, "strategy": strategy, "cost_usd": objective, "feasible": math.isfinite(objective), "states": model.states_evaluated, "runtime_seconds": elapsed, "monitoring_quarters": sum(row["monitored_vehicles"] > 0 for row in path), "maximum_electric_vehicles": max((row["electric_vehicles"] for row in path), default=0)})
            print(case, strategy, round(objective, 2), model.states_evaluated, flush=True)
    return pd.DataFrame(rows)


def write_artifacts() -> None:
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(Path("../../tmp/prism-matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path("results/tables")
    directory.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv("results/independent_paired_comparisons.csv")
    refinement_data = pd.read_csv("results/numerical_refinement.csv")
    policies = pd.read_csv("results/independent_policy_evaluation.csv")
    lines = [r"\begin{table}[H]", r"\centering\small",
             r"\caption{Paired joint-policy savings on independent simulated paths; 2,000 paths per seed. Intervals describe simulation sampling uncertainty.}",
             r"\label{tab:independentpairs}",
             r"\begin{tabularx}{\linewidth}{@{}p{0.17\linewidth}p{0.16\linewidth}Yrr@{}}",
             r"\toprule", r"Evaluation & Seed & Comparator & Savings (USD) & 95\% CI (USD) \\", r"\midrule"]
    for row in pairs.itertuples():
        case = "Bounded" if row.case == "Independent bounded paths" else "Contaminated"
        lines.append(f"{case} & {row.seed} & {row.comparator} & {row.mean_savings_usd:,.0f} & [{row.ci_low:,.0f}, {row.ci_high:,.0f}] " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    (directory / "independent_comparisons.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{table}[H]", r"\centering\small",
             r"\caption{Numerical refinements and structural robustness. Savings compare the joint policy with flexible timing; monitoring counts describe the representative joint path.}",
             r"\label{tab:refinement}",
             r"\begin{tabularx}{\linewidth}{@{}Yrrr@{}}", r"\toprule",
             r"Case & Joint cost (USD) & Savings (USD) & Monitor periods \\", r"\midrule"]
    for case in refinement_data.case.drop_duplicates():
        display_case = "Positive rounded support" if case == "Literal positive support" else case
        subset = refinement_data.loc[refinement_data.case.eq(case)].set_index("strategy")
        joint, baseline = subset.loc["joint"], subset.loc["flexible"]
        if math.isfinite(joint.cost_usd) and math.isfinite(baseline.cost_usd):
            lines.append(f"{display_case} & {joint.cost_usd:,.2f} & {baseline.cost_usd-joint.cost_usd:,.2f} & {int(joint.monitoring_quarters)} " + r"\\")
        else:
            lines.append(f"{display_case} & Infeasible & --- & --- " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    (directory / "numerical_refinement.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{table}[H]", r"\centering\small",
             r"\caption{Independent bounded-path outcomes for seed 20261003. Emissions are simulated physical totals, not planning-envelope debits.}",
             r"\label{tab:independentphysical}",
             r"\begin{tabularx}{\linewidth}{@{}Yrrrr@{}}", r"\toprule",
             r"Policy & Mean cost (USD) & Mean carbon (t) & Two EVs (\%) & Violations \\", r"\midrule"]
    subset = policies.loc[policies.case.eq("Independent bounded paths") & policies.seed.eq(20261003)]
    for row in subset.itertuples():
        lines.append(f"{row.policy} & {row.mean_cost_usd:,.0f} & {row.mean_carbon_tonnes:.3f} & {100*row.two_ev_frequency:.2f} & {int(round(row.carbon_violation_rate*row.paths))}/{row.paths} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    (directory / "independent_physical.tex").write_text("\n".join(lines) + "\n")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    comparisons = pairs.loc[pairs.comparator.eq("Flexible timing")]
    for index, row in enumerate(comparisons.itertuples()):
        axes[0].errorbar(row.mean_savings_usd/1000, index,
                        xerr=(row.ci_high-row.mean_savings_usd)/1000,
                        fmt="o", color="#2166ac", capsize=4)
    axes[0].set_yticks(range(len(comparisons)))
    axes[0].set_yticklabels([f"{'Bounded' if row.case == 'Independent bounded paths' else 'Contaminated'}\nseed {row.seed}" for row in comparisons.itertuples()])
    axes[0].axvline(89982.30/1000, color="#777777", linestyle="--", label="Finite-model comparison")
    axes[0].set_xlabel("Joint savings versus flexible timing (USD thousands)")
    axes[0].set_title("Paired means and 95% intervals")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="x", alpha=0.25)
    for index, row in enumerate(subset.itertuples()):
        axes[1].errorbar(row.mean_carbon_tonnes, index,
                        xerr=(row.carbon_ci_high-row.mean_carbon_tonnes),
                        fmt="o", color="#b35806", capsize=4)
    axes[1].set_yticks(range(len(subset)))
    axes[1].set_yticklabels(subset.policy.tolist())
    axes[1].set_xlabel("Mean cumulative physical emissions (tonnes)")
    axes[1].set_title("Independent bounded paths, seed 20261003")
    axes[1].grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig("figures/independent_validation.jpg", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-per-seed", type=int, default=2000)
    parser.add_argument("--skip-refinement", action="store_true")
    parser.add_argument("--render-saved", action="store_true")
    arguments = parser.parse_args()
    if arguments.render_saved:
        write_artifacts()
        return
    calibration = load_empirical_calibration()
    models = build_policies(calibration)
    policy_rows, comparison_rows = [], []
    for seed in (20261003, 20261004):
        paths = generate_paths(calibration, arguments.paths_per_seed, seed)
        for case, contamination in (("Independent bounded paths", 0.0), ("Ten percent telemetry contamination", 0.1)):
            samples = {name: replay(model, paths, contamination_rate=contamination) for name, model in models.items()}
            for name, sample in samples.items():
                failures = int(sample.policy_failure.sum())
                cost_stats = mean_interval(sample.cost_usd.to_numpy()) if not failures else (math.nan,) * 3
                carbon_stats = mean_interval(sample.carbon_tonnes.to_numpy()) if not failures else (math.nan,) * 3
                violation_stats = rate_interval(sample.carbon_violation.to_numpy()) if not failures else (math.nan,) * 3
                policy_rows.append({"case": case, "seed": seed, "policy": name, "paths": len(sample), "mean_cost_usd": cost_stats[0], "cost_ci_low": cost_stats[1], "cost_ci_high": cost_stats[2], "mean_carbon_tonnes": carbon_stats[0], "carbon_ci_low": carbon_stats[1], "carbon_ci_high": carbon_stats[2], "carbon_violation_rate": violation_stats[0], "carbon_violation_ci_low": violation_stats[1], "carbon_violation_ci_high": violation_stats[2], "policy_failures": failures, "monitoring_frequency": float(sample.monitoring_selected.mean()), "two_ev_frequency": float(sample.maximum_electric_vehicles.ge(2).mean())})
            for comparator in ("Flexible timing", "Matched ablation", "Immediate scale"):
                differences = samples[comparator].cost_usd.to_numpy() - samples["Joint policy"].cost_usd.to_numpy()
                statistics = mean_interval(differences) if np.isfinite(differences).all() else (math.nan,) * 3
                comparison_rows.append({"case": case, "seed": seed, "comparator": comparator, "paths": len(differences), "mean_savings_usd": statistics[0], "ci_low": statistics[1], "ci_high": statistics[2], "precision_target_usd": 5000, "precision_target_met": statistics[2] - statistics[0] <= 5000})
            print(case, seed, "completed", flush=True)
    output = Path("results")
    pd.DataFrame(policy_rows).to_csv(output / "independent_policy_evaluation.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(output / "independent_paired_comparisons.csv", index=False)
    if not arguments.skip_refinement:
        refinement(calibration).to_csv(output / "numerical_refinement.csv", index=False)
    files = sorted(Path("data/empirical").glob("*.csv")) + [Path("data/fleet_asset_cost_parameters.csv"), Path("src/prototype_dp.py"), Path("src/validate_study.py")]
    metadata = {"seeds": [20261003, 20261004], "paths_per_seed": arguments.paths_per_seed, "precision_target_usd": 5000, "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}, "physical_process": "Clipped normal traction and auxiliary draws; resampled simulated facility loads; fixed diesel fuel means; shared potential electric duties and observation errors; no stochastic service failures.", "evaluation_boundary": "Independent simulated paths; not empirical validation. Telemetry contamination is an unretrained distribution-shift stress test. No failed paths are silently excluded."}
    (output / "validation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not arguments.skip_refinement:
        write_artifacts()


if __name__ == "__main__":
    main()