#!/usr/bin/env python3
"""Run benchmark policies and diagnostic sensitivity experiments."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from src.prototype_dp import Calibration, PrototypeDP, THETA, load_calibration, load_empirical_calibration


RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def calibration_with(calibration: Calibration, **updates: float) -> Calibration:
    parameters = dict(calibration.parameters)
    parameters.update(updates)
    return replace(calibration, parameters=parameters)


def path_metrics(model: PrototypeDP) -> dict[str, object]:
    path = model.most_likely_path()
    return {
        "representative_carbon_tonnes": round(
            sum(float(row["carbon_tonnes"]) for row in path), 3
        ),
        "maximum_electric_vehicles": max(
            (int(row["electric_vehicles"]) for row in path), default=0
        ),
        "monitoring_quarters": sum(
            int(row["monitored_vehicles"] > 0) for row in path
        ),
        "investment_sequence": "; ".join(
            f"Q{row['quarter']}:E{row['order_electric']}/C{row['order_chargers']}"
            for row in path
            if row["order_electric"] or row["order_chargers"]
        )
        or "No investment",
    }


def solve_model(
    calibration: Calibration,
    strategy: str,
    *,
    allow_monitoring: bool,
    initial_belief: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    carbon_accounting: str = "global_hard",
    risk_tolerance: float = 0.01,
    suppress_enhanced_signal: bool = False,
) -> tuple[PrototypeDP, float, dict[str, object]]:
    model = PrototypeDP(
        calibration,
        "active",
        allow_monitoring=allow_monitoring,
        strategy=strategy,
        initial_belief=initial_belief,
        carbon_accounting=carbon_accounting,
        risk_tolerance=risk_tolerance,
        suppress_enhanced_signal=suppress_enhanced_signal,
    )
    objective = model.solve()
    metrics = path_metrics(model) if math.isfinite(objective) else {
        "representative_carbon_tonnes": None,
        "maximum_electric_vehicles": None,
        "monitoring_quarters": None,
        "investment_sequence": "Infeasible",
    }
    return model, objective, metrics


def benchmark_results(calibration: Calibration) -> pd.DataFrame:
    specifications = [
        ("Immediate scale", "immediate_scale", False, False),
        ("Flexible timing", "flexible", False, False),
        ("Fixed pilot", "fixed_pilot", False, False),
        ("Joint policy", "joint", True, False),
        ("Matched ablation", "matched_ablation", True, True),
    ]
    rows = []
    for label, strategy, allow_monitoring, suppress_signal in specifications:
        model, objective, metrics = solve_model(
            calibration,
            strategy,
            allow_monitoring=allow_monitoring,
            suppress_enhanced_signal=suppress_signal,
        )
        rows.append(
            {
                "policy": label,
                "feasible": math.isfinite(objective),
                "expected_discounted_cost_usd": None
                if math.isinf(objective)
                else round(objective, 2),
                "states_evaluated": model.states_evaluated,
                **metrics,
            }
        )

    oracle_rows = []
    for index, theta in enumerate(THETA):
        belief = tuple(1.0 if position == index else 0.0 for position in range(3))
        model, objective, metrics = solve_model(
            calibration,
            "oracle",
            allow_monitoring=False,
            initial_belief=belief,
        )
        oracle_rows.append((theta, model, objective, metrics))
    oracle_feasible = all(math.isfinite(row[2]) for row in oracle_rows)
    rows.append(
        {
            "policy": "Parameter oracle",
            "feasible": oracle_feasible,
            "expected_discounted_cost_usd": round(
                sum(row[2] for row in oracle_rows) / len(oracle_rows), 2
            )
            if oracle_feasible
            else None,
            "states_evaluated": sum(row[1].states_evaluated for row in oracle_rows),
            "representative_carbon_tonnes": round(
                sum(float(row[3]["representative_carbon_tonnes"]) for row in oracle_rows)
                / len(oracle_rows),
                3,
            )
            if oracle_feasible
            else None,
            "maximum_electric_vehicles": "/".join(
                str(row[3]["maximum_electric_vehicles"]) for row in oracle_rows
            ),
            "monitoring_quarters": 0,
            "investment_sequence": "; ".join(
                f"theta={theta:.2f}: {metrics['investment_sequence']}"
                for theta, _, _, metrics in oracle_rows
            ),
        }
    )
    return pd.DataFrame(rows)


def sensitivity_results(calibration: Calibration) -> pd.DataFrame:
    rows = []

    for budget in (832.0, 836.0, 840.0, 844.0):
        case = calibration_with(calibration, b_1_active=budget)
        joint, joint_value, joint_metrics = solve_model(
            case, "joint", allow_monitoring=True
        )
        _, base_value, base_metrics = solve_model(
            case, "flexible", allow_monitoring=False
        )
        rows.append(
            {
                "factor": "Cumulative carbon budget",
                "level": budget,
                "unit": "tonnes",
                "joint_cost_usd": None if math.isinf(joint_value) else round(joint_value, 2),
                "baseline_cost_usd": None if math.isinf(base_value) else round(base_value, 2),
                "information_value_usd": None
                if math.isinf(joint_value) or math.isinf(base_value)
                else round(base_value - joint_value, 2),
                "joint_monitoring_quarters": joint_metrics["monitoring_quarters"],
                "joint_maximum_electric_vehicles": joint_metrics["maximum_electric_vehicles"],
                "baseline_maximum_electric_vehicles": base_metrics["maximum_electric_vehicles"],
                "joint_feasible": math.isfinite(joint_value),
                "baseline_feasible": math.isfinite(base_value),
                "states_evaluated": joint.states_evaluated,
            }
        )

    for monitoring_cost in (0.0, 10_000.0, 50_000.0, 100_000.0):
        case = calibration_with(calibration, c_M=monitoring_cost)
        joint, joint_value, metrics = solve_model(
            case, "joint", allow_monitoring=True
        )
        _, base_value, _ = solve_model(case, "flexible", allow_monitoring=False)
        rows.append(
            {
                "factor": "Monitoring cost",
                "level": monitoring_cost,
                "unit": "USD per monitored quarter",
                "joint_cost_usd": None if math.isinf(joint_value) else round(joint_value, 2),
                "baseline_cost_usd": None if math.isinf(base_value) else round(base_value, 2),
                "information_value_usd": None
                if math.isinf(joint_value) or math.isinf(base_value)
                else round(base_value - joint_value, 2),
                "joint_monitoring_quarters": metrics["monitoring_quarters"],
                "joint_maximum_electric_vehicles": metrics["maximum_electric_vehicles"],
                "baseline_maximum_electric_vehicles": None,
                "joint_feasible": math.isfinite(joint_value),
                "baseline_feasible": math.isfinite(base_value),
                "states_evaluated": joint.states_evaluated,
            }
        )

    for telemetry_noise in (0.5, 10.0, 25.0):
        case = calibration_with(calibration, sigma_nu=telemetry_noise)
        joint, joint_value, metrics = solve_model(
            case, "joint", allow_monitoring=True
        )
        _, base_value, _ = solve_model(case, "flexible", allow_monitoring=False)
        rows.append(
            {
                "factor": "Telemetry noise",
                "level": telemetry_noise,
                "unit": "kWh standard deviation",
                "joint_cost_usd": None if math.isinf(joint_value) else round(joint_value, 2),
                "baseline_cost_usd": None if math.isinf(base_value) else round(base_value, 2),
                "information_value_usd": None
                if math.isinf(joint_value) or math.isinf(base_value)
                else round(base_value - joint_value, 2),
                "joint_monitoring_quarters": metrics["monitoring_quarters"],
                "joint_maximum_electric_vehicles": metrics["maximum_electric_vehicles"],
                "baseline_maximum_electric_vehicles": None,
                "joint_feasible": math.isfinite(joint_value),
                "baseline_feasible": math.isfinite(base_value),
                "states_evaluated": joint.states_evaluated,
            }
        )

    for label, electric_lead, charger_lead in (
        ("Faster", 1.0, 2.0),
        ("Base", 2.0, 3.0),
        ("Slower", 3.0, 4.0),
    ):
        case = calibration_with(calibration, L_E=electric_lead, L_C=charger_lead)
        joint, joint_value, metrics = solve_model(
            case, "joint", allow_monitoring=True
        )
        rows.append(
            {
                "factor": "Asset lead times",
                "level": label,
                "unit": f"EV={int(electric_lead)}, charger={int(charger_lead)} quarters",
                "joint_cost_usd": None if math.isinf(joint_value) else round(joint_value, 2),
                "baseline_cost_usd": None,
                "information_value_usd": None,
                "joint_monitoring_quarters": metrics["monitoring_quarters"],
                "joint_maximum_electric_vehicles": metrics["maximum_electric_vehicles"],
                "baseline_maximum_electric_vehicles": None,
                "joint_feasible": math.isfinite(joint_value),
                "baseline_feasible": None,
                "states_evaluated": joint.states_evaluated,
            }
        )

    for label, belief in (
        ("Favorable-heavy", (0.6, 0.3, 0.1)),
        ("Balanced", (1 / 3, 1 / 3, 1 / 3)),
        ("Adverse-heavy", (0.1, 0.3, 0.6)),
    ):
        joint, joint_value, metrics = solve_model(
            calibration,
            "joint",
            allow_monitoring=True,
            initial_belief=belief,
        )
        rows.append(
            {
                "factor": "Prior distribution",
                "level": label,
                "unit": str(belief),
                "joint_cost_usd": None if math.isinf(joint_value) else round(joint_value, 2),
                "baseline_cost_usd": None,
                "information_value_usd": None,
                "joint_monitoring_quarters": metrics["monitoring_quarters"],
                "joint_maximum_electric_vehicles": metrics["maximum_electric_vehicles"],
                "baseline_maximum_electric_vehicles": None,
                "joint_feasible": math.isfinite(joint_value),
                "baseline_feasible": None,
                "states_evaluated": joint.states_evaluated,
            }
        )

    for label, accounting, tolerance in (
        ("Posterior quantile 99%", "posterior_quantile", 0.01),
        ("Global hard support", "global_hard", 0.0),
    ):
        joint, joint_value, metrics = solve_model(
            calibration,
            "joint",
            allow_monitoring=True,
            carbon_accounting=accounting,
            risk_tolerance=tolerance,
        )
        rows.append(
            {
                "factor": "Carbon accounting support",
                "level": label,
                "unit": "rule",
                "joint_cost_usd": None if math.isinf(joint_value) else round(joint_value, 2),
                "baseline_cost_usd": None,
                "information_value_usd": None,
                "joint_monitoring_quarters": metrics["monitoring_quarters"],
                "joint_maximum_electric_vehicles": metrics["maximum_electric_vehicles"],
                "baseline_maximum_electric_vehicles": None,
                "joint_feasible": math.isfinite(joint_value),
                "baseline_feasible": None,
                "states_evaluated": joint.states_evaluated,
            }
        )
    return pd.DataFrame(rows)


def save_figures(benchmarks: pd.DataFrame, sensitivities: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})

    feasible = benchmarks[benchmarks["feasible"]].copy()
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    bars = axis.barh(
        feasible["policy"],
        feasible["expected_discounted_cost_usd"] / 1000.0,
        color=["#7f8c8d", "#5b8ff9", "#f6bd16", "#2ca02c", "#e8684a", "#9270ca"],
    )
    axis.invert_yaxis()
    axis.set_xlabel("Expected discounted cost (thousand USD)")
    axis.set_title("Synthetic benchmark-policy comparison")
    axis.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, feasible["expected_discounted_cost_usd"]):
        axis.text(
            bar.get_width() + 4,
            bar.get_y() + bar.get_height() / 2,
            f"{value / 1000:.1f}",
            va="center",
        )
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "benchmark_policy_costs.jpg",
        dpi=220,
        bbox_inches="tight",
        pil_kwargs={"quality": 95},
    )
    plt.close(figure)

    carbon = sensitivities[sensitivities["factor"].eq("Cumulative carbon budget")]
    monitoring = sensitivities[sensitivities["factor"].eq("Monitoring cost")]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].plot(
        carbon["level"].astype(float),
        carbon["information_value_usd"] / 1000.0,
        marker="o",
        color="#2ca02c",
    )
    axes[0].set_xlabel("Cumulative carbon budget (t)")
    axes[0].set_ylabel("Net information value (thousand USD)")
    axes[0].set_title("Carbon headroom")
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        monitoring["level"].astype(float) / 1000.0,
        monitoring["information_value_usd"] / 1000.0,
        marker="o",
        color="#5b8ff9",
    )
    axes[1].set_xlabel("Monitoring cost (thousand USD/quarter)")
    axes[1].set_title("Monitoring cost")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "information_value_sensitivity.jpg",
        dpi=220,
        bbox_inches="tight",
        pil_kwargs={"quality": 95},
    )
    plt.close(figure)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    calibration = load_empirical_calibration()
    benchmarks = benchmark_results(calibration)
    sensitivities = sensitivity_results(calibration)
    benchmarks.to_csv(RESULTS_DIR / "benchmark_results.csv", index=False)
    sensitivities.to_csv(RESULTS_DIR / "sensitivity_results.csv", index=False)
    save_figures(benchmarks, sensitivities)
    summary = {
        "benchmarks": benchmarks.to_dict(orient="records"),
        "sensitivities": sensitivities.to_dict(orient="records"),
    }
    (RESULTS_DIR / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(benchmarks.to_string(index=False))
    print()
    print(sensitivities.to_string(index=False))


if __name__ == "__main__":
    main()