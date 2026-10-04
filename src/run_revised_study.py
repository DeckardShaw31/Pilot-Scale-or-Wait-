#!/usr/bin/env python3
"""Evaluate hard-compliance controls and explicitly risk-screened policies."""

from __future__ import annotations

import argparse
import hashlib
import gc
import json
import math
import platform
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prototype_dp import Calibration, PrototypeDP, load_empirical_calibration
from src.validate_study import generate_paths, mean_interval, rate_interval, replay


RISK_LIMIT = 0.01
DESIGN_SEED = 20261001
EVALUATION_SEEDS = (20261002, 20261003)
DESIGN_PATHS = 2000
EVALUATION_PATHS = 2000
CANDIDATE_RULES = (
    ("Global hard", "global_hard", 0.0),
    ("Quantile 99%", "posterior_quantile", 0.01),
    ("Quantile 66%", "posterior_quantile", 0.34),
)
INSTANCE_UPDATES = (
    ("Tight budget", {"b_1_active": 832.0}),
    ("Base budget", {"b_1_active": 836.0}),
    ("Loose budget", {"b_1_active": 840.0}),
    ("Initial EV and charger", {"N_0_E": 1.0, "k_0": 1.0}),
    ("One regional duty becomes suburban", {"n_D2": 5.0, "n_D3": 2.0}),
    ("Diesel factor plus 3 percent", {"g_D_multiplier": 1.03}),
)


def revised_calibration(base: Calibration, updates: dict[str, float]) -> Calibration:
    parameters = dict(base.parameters)
    multiplier = updates.get("g_D_multiplier")
    parameters.update({key: value for key, value in updates.items() if key != "g_D_multiplier"})
    if multiplier is not None:
        parameters["g_D"] *= multiplier
    return replace(base, parameters=parameters)


def solve_candidate(
    calibration: Calibration,
    *,
    monitoring: bool,
    accounting: str,
    tolerance: float,
) -> tuple[PrototypeDP, float]:
    model = PrototypeDP(
        calibration,
        "active",
        strategy="joint" if monitoring else "flexible",
        allow_monitoring=monitoring,
        carbon_accounting=accounting,
        risk_tolerance=tolerance,
    )
    return model, model.solve()


def screen_candidates(
    calibration: Calibration,
    instance: str,
    monitoring: bool,
) -> tuple[list[dict[str, object]], PrototypeDP | None, str | None]:
    paths = generate_paths(calibration, DESIGN_PATHS, DESIGN_SEED)
    rows = []
    selected: tuple[float, str, PrototypeDP] | None = None
    for label, accounting, tolerance in CANDIDATE_RULES:
        model, objective = solve_candidate(
            calibration,
            monitoring=monitoring,
            accounting=accounting,
            tolerance=tolerance,
        )
        if math.isfinite(objective):
            samples = replay(model, paths)
            failures = int(samples.policy_failure.sum())
            if failures:
                mean_cost = math.nan
                violation = (math.nan, math.nan, math.nan)
            else:
                mean_cost = float(samples.cost_usd.mean())
                violation = rate_interval(samples.carbon_violation.to_numpy())
            passes = not failures and violation[2] <= RISK_LIMIT
        else:
            failures = DESIGN_PATHS
            mean_cost = math.nan
            violation = (math.nan, math.nan, math.nan)
            passes = False
        row = {
            "instance": instance,
            "information_policy": "Baseline plus telemetry" if monitoring else "Baseline only",
            "candidate": label,
            "accounting": accounting,
            "local_quantile_tolerance": tolerance,
            "finite_model_cost_usd": objective,
            "design_mean_cost_usd": mean_cost,
            "design_violation_rate": violation[0],
            "design_violation_upper_95": violation[2],
            "risk_limit": RISK_LIMIT,
            "passes_joint_risk_screen": passes,
            "policy_failures": failures,
            "states_evaluated": model.states_evaluated,
        }
        rows.append(row)
        if passes and (selected is None or (mean_cost, label) < selected[:2]):
            if selected is not None:
                selected[2]._value.cache_clear()
            selected = (mean_cost, label, model)
        else:
            model._value.cache_clear()
        gc.collect()
        print(instance, row["information_policy"], label, passes, flush=True)
    if selected is None:
        return rows, None, None
    _, selected_label, selected_model = selected
    selected_model._value.cache_clear()
    return rows, selected_model, selected_label


def evaluate_selected(
    calibration: Calibration,
    instance: str,
    policy_name: str,
    model: PrototypeDP | None,
    selected_candidate: str | None,
) -> list[dict[str, object]]:
    rows = []
    for seed in EVALUATION_SEEDS:
        if model is None:
            rows.append({
                "instance": instance,
                "information_policy": policy_name,
                "selected_candidate": selected_candidate,
                "seed": seed,
                "paths": EVALUATION_PATHS,
                "feasible_candidate": False,
            })
            continue
        samples = replay(model, generate_paths(calibration, EVALUATION_PATHS, seed))
        failures = int(samples.policy_failure.sum())
        cost = mean_interval(samples.cost_usd.to_numpy()) if not failures else (math.nan,) * 3
        carbon = mean_interval(samples.carbon_tonnes.to_numpy()) if not failures else (math.nan,) * 3
        violation = rate_interval(samples.carbon_violation.to_numpy()) if not failures else (math.nan,) * 3
        rows.append({
            "instance": instance,
            "information_policy": policy_name,
            "selected_candidate": selected_candidate,
            "seed": seed,
            "paths": len(samples),
            "feasible_candidate": True,
            "mean_cost_usd": cost[0],
            "cost_ci_low": cost[1],
            "cost_ci_high": cost[2],
            "mean_carbon_tonnes": carbon[0],
            "carbon_ci_low": carbon[1],
            "carbon_ci_high": carbon[2],
            "violation_rate": violation[0],
            "violation_upper_95": violation[2],
            "risk_limit": RISK_LIMIT,
            "policy_failures": failures,
            "monitoring_frequency": float(samples.monitoring_selected.mean()),
            "two_ev_frequency": float(samples.maximum_electric_vehicles.ge(2).mean()),
        })
    return rows


def write_outputs(candidate_rows: list[dict[str, object]], evaluation_rows: list[dict[str, object]]) -> None:
    output = Path("results")
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    candidates = pd.DataFrame(candidate_rows)
    evaluations = pd.DataFrame(evaluation_rows)
    candidates.to_csv(output / "revised_candidate_screen.csv", index=False)
    evaluations.to_csv(output / "revised_evaluation.csv", index=False)

    summary_rows = []
    for instance in evaluations.instance.drop_duplicates():
        subset = evaluations.loc[
            evaluations.instance.eq(instance) & evaluations.seed.eq(EVALUATION_SEEDS[0])
        ].set_index("information_policy")
        telemetry = subset.loc["Baseline plus telemetry"]
        baseline = subset.loc["Baseline only"]
        if bool(telemetry.feasible_candidate) and bool(baseline.feasible_candidate):
            information_value = baseline.mean_cost_usd - telemetry.mean_cost_usd
            summary_rows.append({
                "instance": instance,
                "telemetry_candidate": telemetry.selected_candidate,
                "baseline_candidate": baseline.selected_candidate,
                "telemetry_mean_cost_usd": telemetry.mean_cost_usd,
                "baseline_mean_cost_usd": baseline.mean_cost_usd,
                "information_value_usd": information_value,
                "telemetry_mean_carbon_tonnes": telemetry.mean_carbon_tonnes,
                "baseline_mean_carbon_tonnes": baseline.mean_carbon_tonnes,
                "telemetry_violation_upper_95": telemetry.violation_upper_95,
                "baseline_violation_upper_95": baseline.violation_upper_95,
                "monitoring_frequency": telemetry.monitoring_frequency,
                "classification": "Positive information" if information_value > 1.0 else "Zero information",
            })
        else:
            summary_rows.append({
                "instance": instance,
                "telemetry_candidate": telemetry.get("selected_candidate"),
                "baseline_candidate": baseline.get("selected_candidate"),
                "telemetry_mean_cost_usd": telemetry.get("mean_cost_usd"),
                "baseline_mean_cost_usd": baseline.get("mean_cost_usd"),
                "information_value_usd": math.nan,
                "telemetry_mean_carbon_tonnes": telemetry.get("mean_carbon_tonnes"),
                "baseline_mean_carbon_tonnes": baseline.get("mean_carbon_tonnes"),
                "telemetry_violation_upper_95": telemetry.get("violation_upper_95"),
                "baseline_violation_upper_95": baseline.get("violation_upper_95"),
                "monitoring_frequency": telemetry.get("monitoring_frequency"),
                "classification": "No admissible candidate",
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "revised_instance_family.csv", index=False)

    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\caption{Predefined instance family under a 1\% whole-horizon carbon-violation limit. Candidate policies are selected on an independent design sample; this table reports held-out seed 20261002.}",
        r"\label{tab:revisedfamily}",
        r"\textit{Panel A: Expected discounted cost}\par\smallskip",
        r"\begin{tabularx}{\linewidth}{@{}Yrr@{}}",
        r"\toprule",
        r"Instance & Baseline cost (USD) & Telemetry cost (USD) \\",
        r"\midrule",
    ]
    for row in summary.itertuples():
        baseline_cost = (
            "N/A (infeasible)"
            if not math.isfinite(row.baseline_mean_cost_usd)
            else rf"\${row.baseline_mean_cost_usd:,.0f}"
        )
        telemetry_cost = (
            "N/A (infeasible)"
            if not math.isfinite(row.telemetry_mean_cost_usd)
            else rf"\${row.telemetry_mean_cost_usd:,.0f}"
        )
        lines.append(
            f"{row.instance} & {baseline_cost} & {telemetry_cost} "
            + r"\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        r"\par\vspace{1.1em}",
        r"\textit{Panel B: Mean physical carbon}\par\smallskip",
        r"\begin{tabularx}{\linewidth}{@{}Yrr@{}}",
        r"\toprule",
        r"Instance & Baseline (t) & Telemetry (t) \\",
        r"\midrule",
    ])
    for row in summary.itertuples():
        if math.isfinite(row.baseline_mean_carbon_tonnes):
            baseline_carbon = f"{row.baseline_mean_carbon_tonnes:.3f}"
            telemetry_carbon = f"{row.telemetry_mean_carbon_tonnes:.3f}"
        else:
            baseline_carbon = telemetry_carbon = "N/A"
        lines.append(
            f"{row.instance} & {baseline_carbon} & {telemetry_carbon} "
            + r"\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        r"\par\vspace{0.25em}\footnotesize Notes: Values are held-out means for seed 20261002. Equal baseline and telemetry entries mean that telemetry does not change the selected policy or outcome. ``N/A (infeasible)'' means that neither information design has an admissible candidate; it is not missing data.",
        r"\end{table}",
    ])
    (tables / "revised_instance_family.tex").write_text("\n".join(lines) + "\n")

    plot_data = summary.loc[summary.classification.ne("No admissible candidate")].copy()
    labels = [
        value.replace("One regional duty becomes suburban", "Regional to suburban")
        .replace("Initial EV and charger", "Initial EV + charger")
        for value in plot_data.instance
    ]
    positions = np.arange(len(plot_data))
    figure, axes = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)
    width = 0.38
    axes[0].bar(
        positions - width / 2,
        plot_data.baseline_mean_cost_usd / 1000.0,
        width,
        label="Baseline information",
        color="#777777",
    )
    axes[0].bar(
        positions + width / 2,
        plot_data.telemetry_mean_cost_usd / 1000.0,
        width,
        label="Baseline + telemetry",
        color="#2b6f9f",
    )
    axes[0].set_ylabel("Expected discounted cost\n(thousand USD)")
    axes[0].legend(frameon=False, ncol=2, loc="upper center")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(
        positions - width / 2,
        plot_data.baseline_mean_carbon_tonnes,
        width,
        color="#777777",
    )
    axes[1].bar(
        positions + width / 2,
        plot_data.telemetry_mean_carbon_tonnes,
        width,
        color="#b44b4b",
    )
    carbon_min = min(
        plot_data.baseline_mean_carbon_tonnes.min(),
        plot_data.telemetry_mean_carbon_tonnes.min(),
    )
    carbon_max = max(
        plot_data.baseline_mean_carbon_tonnes.max(),
        plot_data.telemetry_mean_carbon_tonnes.max(),
    )
    axes[1].set_ylim(carbon_min - 4.0, carbon_max + 4.0)
    axes[1].set_ylabel("Mean physical carbon (t)")
    axes[1].set_xticks(positions, labels, rotation=22, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    figure.savefig("figures/revised_cost_carbon_tradeoff.jpg", dpi=300, bbox_inches="tight")
    plt.close(figure)

    files = [
        Path("src/prototype_dp.py"),
        Path("src/validate_study.py"),
        Path("src/run_revised_study.py"),
        *sorted(Path("data/empirical").glob("*.csv")),
    ]
    metadata = {
        "risk_limit": RISK_LIMIT,
        "risk_event": "Any annual or cumulative physical-carbon limit violation over the 12-quarter horizon.",
        "selection_rule": "Minimum design-sample mean cost among predefined candidates whose two-sided Wilson 95% upper violation bound is at most the risk limit.",
        "design_seed": DESIGN_SEED,
        "design_paths": DESIGN_PATHS,
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "evaluation_paths_per_seed": EVALUATION_PATHS,
        "candidate_rules": [list(candidate) for candidate in CANDIDATE_RULES],
        "instance_updates": [[name, updates] for name, updates in INSTANCE_UPDATES],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        },
    }
    (output / "revised_study_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def run_instances(instance_names: set[str] | None = None) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    base = load_empirical_calibration()
    candidate_rows: list[dict[str, object]] = []
    evaluation_rows: list[dict[str, object]] = []
    for instance, updates in INSTANCE_UPDATES:
        if instance_names is not None and instance not in instance_names:
            continue
        calibration = revised_calibration(base, updates)
        for monitoring, policy_name in (
            (False, "Baseline only"),
            (True, "Baseline plus telemetry"),
        ):
            rows, model, selected = screen_candidates(calibration, instance, monitoring)
            candidate_rows.extend(rows)
            evaluation_rows.extend(
                evaluate_selected(calibration, instance, policy_name, model, selected)
            )
    return candidate_rows, evaluation_rows


def slug(value: str) -> str:
    return "_".join(value.lower().replace("%", "percent").split())


def json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def evaluate_one_candidate(
    instance: str,
    monitoring: bool,
    candidate_label: str,
) -> dict[str, object]:
    base = load_empirical_calibration()
    updates = dict(INSTANCE_UPDATES)[instance]
    calibration = revised_calibration(base, updates)
    candidate = next(rule for rule in CANDIDATE_RULES if rule[0] == candidate_label)
    _, accounting, tolerance = candidate
    model, objective = solve_candidate(
        calibration,
        monitoring=monitoring,
        accounting=accounting,
        tolerance=tolerance,
    )
    result: dict[str, object] = {
        "instance": instance,
        "information_policy": "Baseline plus telemetry" if monitoring else "Baseline only",
        "candidate": candidate_label,
        "accounting": accounting,
        "local_quantile_tolerance": tolerance,
        "finite_model_cost_usd": objective,
        "states_evaluated": model.states_evaluated,
        "design": None,
        "evaluation": [],
    }
    if not math.isfinite(objective):
        return result
    design_samples = replay(model, generate_paths(calibration, DESIGN_PATHS, DESIGN_SEED))
    design_failures = int(design_samples.policy_failure.sum())
    if design_failures:
        design = {
            "paths": DESIGN_PATHS,
            "mean_cost_usd": math.nan,
            "violation_rate": math.nan,
            "violation_upper_95": math.nan,
            "policy_failures": design_failures,
            "passes_joint_risk_screen": False,
        }
    else:
        violation = rate_interval(design_samples.carbon_violation.to_numpy())
        design = {
            "paths": DESIGN_PATHS,
            "mean_cost_usd": float(design_samples.cost_usd.mean()),
            "violation_rate": violation[0],
            "violation_upper_95": violation[2],
            "policy_failures": 0,
            "passes_joint_risk_screen": violation[2] <= RISK_LIMIT,
        }
    result["design"] = design
    evaluation = []
    for seed in EVALUATION_SEEDS:
        samples = replay(model, generate_paths(calibration, EVALUATION_PATHS, seed))
        failures = int(samples.policy_failure.sum())
        if failures:
            evaluation.append({"seed": seed, "paths": EVALUATION_PATHS, "policy_failures": failures})
            continue
        cost = mean_interval(samples.cost_usd.to_numpy())
        carbon = mean_interval(samples.carbon_tonnes.to_numpy())
        violation = rate_interval(samples.carbon_violation.to_numpy())
        evaluation.append({
            "seed": seed,
            "paths": EVALUATION_PATHS,
            "policy_failures": 0,
            "mean_cost_usd": cost[0],
            "cost_ci_low": cost[1],
            "cost_ci_high": cost[2],
            "mean_carbon_tonnes": carbon[0],
            "carbon_ci_low": carbon[1],
            "carbon_ci_high": carbon[2],
            "violation_rate": violation[0],
            "violation_upper_95": violation[2],
            "monitoring_frequency": float(samples.monitoring_selected.mean()),
            "two_ev_frequency": float(samples.maximum_electric_vehicles.ge(2).mean()),
        })
    result["evaluation"] = evaluation
    return result


def merge_candidate_parts(parts: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    payloads = [json.loads(path.read_text()) for path in sorted(parts.glob("*.json"))]
    expected = len(INSTANCE_UPDATES) * 2 * len(CANDIDATE_RULES)
    if len(payloads) != expected:
        raise ValueError(f"Expected {expected} candidate parts, found {len(payloads)}")
    candidate_rows = []
    evaluation_rows = []
    for payload in payloads:
        design = payload.get("design") or {}
        candidate_rows.append({
            "instance": payload["instance"],
            "information_policy": payload["information_policy"],
            "candidate": payload["candidate"],
            "accounting": payload["accounting"],
            "local_quantile_tolerance": payload["local_quantile_tolerance"],
            "finite_model_cost_usd": payload["finite_model_cost_usd"],
            "design_mean_cost_usd": design.get("mean_cost_usd"),
            "design_violation_rate": design.get("violation_rate"),
            "design_violation_upper_95": design.get("violation_upper_95"),
            "risk_limit": RISK_LIMIT,
            "passes_joint_risk_screen": bool(design.get("passes_joint_risk_screen", False)),
            "policy_failures": design.get("policy_failures", DESIGN_PATHS),
            "states_evaluated": payload["states_evaluated"],
        })
    candidate_frame = pd.DataFrame(candidate_rows)
    for (instance, information_policy), group in candidate_frame.groupby(
        ["instance", "information_policy"], sort=False
    ):
        eligible = group.loc[group.passes_joint_risk_screen]
        if eligible.empty:
            evaluation_rows.extend([
                {
                    "instance": instance,
                    "information_policy": information_policy,
                    "selected_candidate": None,
                    "seed": seed,
                    "paths": EVALUATION_PATHS,
                    "feasible_candidate": False,
                }
                for seed in EVALUATION_SEEDS
            ])
            continue
        selected = eligible.sort_values(["design_mean_cost_usd", "candidate"]).iloc[0]
        payload = next(
            item for item in payloads
            if item["instance"] == instance
            and item["information_policy"] == information_policy
            and item["candidate"] == selected.candidate
        )
        for row in payload["evaluation"]:
            evaluation_rows.append({
                "instance": instance,
                "information_policy": information_policy,
                "selected_candidate": selected.candidate,
                "feasible_candidate": True,
                "risk_limit": RISK_LIMIT,
                **row,
            })
    return candidate_rows, evaluation_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", choices=[name for name, _ in INSTANCE_UPDATES])
    parser.add_argument("--information", choices=("baseline", "telemetry"))
    parser.add_argument("--candidate", choices=[name for name, _, _ in CANDIDATE_RULES])
    parser.add_argument("--merge", action="store_true")
    arguments = parser.parse_args()
    parts = Path("results/revised_parts")
    parts.mkdir(parents=True, exist_ok=True)
    if arguments.merge:
        candidate_rows, evaluation_rows = merge_candidate_parts(parts)
        write_outputs(candidate_rows, evaluation_rows)
        return
    if arguments.instance and arguments.information and arguments.candidate:
        payload = evaluate_one_candidate(
            arguments.instance,
            arguments.information == "telemetry",
            arguments.candidate,
        )
        filename = "__".join((
            slug(arguments.instance),
            arguments.information,
            slug(arguments.candidate),
        )) + ".json"
        (parts / filename).write_text(
            json.dumps(payload, indent=2, default=json_scalar) + "\n"
        )
        return
    if any((arguments.instance, arguments.information, arguments.candidate)):
        parser.error("--instance, --information, and --candidate must be supplied together")
    selected = {arguments.instance} if arguments.instance else None
    candidate_rows, evaluation_rows = run_instances(selected)
    if arguments.instance:
        stem = slug(arguments.instance)
        pd.DataFrame(candidate_rows).to_csv(parts / f"{stem}_candidates.csv", index=False)
        pd.DataFrame(evaluation_rows).to_csv(parts / f"{stem}_evaluation.csv", index=False)
    else:
        write_outputs(candidate_rows, evaluation_rows)


if __name__ == "__main__":
    main()