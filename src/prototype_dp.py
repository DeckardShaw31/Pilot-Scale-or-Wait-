#!/usr/bin/env python3
"""Finite-state diagnostic prototype for staged fleet electrification.

The prototype intentionally stays smaller than the manuscript formulation:

* the incumbent diesel fleet is fixed;
* only three urban duties can be electrified;
* grid upgrades and outsourcing are omitted because they are non-binding in
  the supplied synthetic fixture;
* exact hard-compliance runs debit carbon using the global regime support;
  posterior quantiles are available only as explicitly parameterized candidate-
  policy generators for separate whole-horizon risk screening;
* continuous meter readings are represented by three likelihood-matched
  signal categories so the belief-state dynamic program remains finite;
* when a delayed baseline record is due at a monitoring epoch, the implementation
  retains it and combines its likelihood with current telemetry. The records come
  from different quarter blocks and are conditionally independent given the
  persistent regime; their marginal dependence through that regime is retained.

These conventions make the code a model-integrity and scenario-feasibility
prototype, not an empirical result generator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, NamedTuple

import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEFAULT_PARAMETER_FILE = DATA_DIR / "fleet_asset_cost_parameters.csv"
DEFAULT_OPERATIONS_FILE = DATA_DIR / "duty_cycle_operations_sample.csv"
DEFAULT_DEPOT_FILE = DATA_DIR / "depot_daily_energy.csv"

THETA = (1.15, 1.40, 1.75)
INITIAL_BELIEF = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
URBAN_DISTANCE_KM = 85.0
SUBURBAN_DISTANCE_KM = 135.0
REGIONAL_DISTANCE_KM = 210.0
NOMINAL_DUTY_DISTANCES = {
    "D1_Urban": URBAN_DISTANCE_KM,
    "D2_Suburban": SUBURBAN_DISTANCE_KM,
    "D3_Regional": REGIONAL_DISTANCE_KM,
}


@dataclass(frozen=True)
class InformationChannel:
    channel_id: int
    name: str
    reporting_delay_days: int
    reporting_latency_quarters: int
    missingness_rate: float
    description: str


INFORMATION_CHANNELS = {
    1: InformationChannel(1, "Operational_Dispatch", 0, 0, 0.0, "Daily dispatch logs (same-day, 0-day delay)"),
    2: InformationChannel(2, "Enhanced_Telemetry", 0, 0, 0.0132, "Real-time CAN logger (< 5 min latency, 1.32% packet loss)"),
    3: InformationChannel(3, "Depot_Utility_Meter", 30, 1, 0.0, "Monthly utility billing invoice (30-day settlement lag)"),
    4: InformationChannel(4, "Corporate_Accounting_Reconciliation", 45, 1, 0.0, "Quarterly financial/fuel reconciliation (45-day post-quarter lag)"),
}

HORIZON_QUARTERS = 12
MAX_ELECTRIC_DUTIES = 3
BELIEF_DIGITS = 3
SUPPORT_TOLERANCE = 1e-7
TONNES_TO_MILLITONNES = 1_000
USD_TO_CENTS = 100
INFEASIBLE_COST = math.inf


SCENARIOS = {
    "infeasible": ("C_cap_infeasible", "b_1_infeasible"),
    "active": ("C_cap_active", "b_1_active"),
    "control": ("C_cap_control", "b_1_control"),
}

STRATEGIES = {
    "joint",
    "flexible",
    "immediate_scale",
    "fixed_pilot",
    "matched_ablation",
    "oracle",
}


class Action(NamedTuple):
    order_electric: int
    order_chargers: int
    monitored_vehicles: int


@dataclass(frozen=True)
class State:
    quarter: int
    electric_vehicles: int
    chargers: int
    electric_pipeline: tuple[int, ...]
    charger_pipeline: tuple[int, ...]
    cumulative_carbon: int
    annual_carbon: int
    cumulative_capital: int
    annual_capital: int
    belief: tuple[float, float, float]
    baseline_observed: bool = False
    enhanced_observed: bool = False
    pending_baseline: int = 0


@dataclass(frozen=True)
class Calibration:
    parameters: dict[str, float]
    diesel_litres: dict[str, float]
    facility_mean_kwh: float
    facility_max_kwh: float
    diesel_maintenance: dict[str, float] = field(default_factory=dict)

    @property
    def electric_lead_time(self) -> int:
        return int(self.parameters["L_E"])

    @property
    def charger_lead_time(self) -> int:
        return int(self.parameters["L_C"])


def validate_csv_width(path: Path) -> tuple[int, int]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream, strict=True))
    if not rows:
        raise ValueError(f"{path} is empty")
    width = len(rows[0])
    bad_rows = [number for number, row in enumerate(rows[1:], 2) if len(row) != width]
    if bad_rows:
        raise ValueError(f"{path} has non-RFC-aligned rows: {bad_rows}")
    return width, len(rows) - 1


def load_calibration(
    parameter_file: Path = DEFAULT_PARAMETER_FILE,
    operations_file: Path = DEFAULT_OPERATIONS_FILE,
    depot_file: Path = DEFAULT_DEPOT_FILE,
    assets_file: Path | None = None,
    monitoring_file: Path | None = None,
    policy_file: Path | None = None,
) -> Calibration:
    for path in (parameter_file, operations_file, depot_file):
        validate_csv_width(path)

    parameter_frame = pd.read_csv(parameter_file)
    parameters = {
        row.symbol: float(row.value)
        for row in parameter_frame.itertuples(index=False)
    }

    if assets_file is not None and Path(assets_file).exists():
        validate_csv_width(Path(assets_file))
        assets_df = pd.read_csv(assets_file)
        bev_rows = assets_df[assets_df["technology"] == "Battery_Electric"]
        if not bev_rows.empty:
            parameters["c_E"] = float(bev_rows["net_capital_cost_usd"].mean())
            parameters["m_E"] = round(float(bev_rows["maintenance_rate_usd_per_km"].mean()) * URBAN_DISTANCE_KM, 2)
            parameters["L_E"] = float(bev_rows["order_lead_time_quarters"].iloc[0])
            parameters["Q"] = float(bev_rows["usable_energy_capacity_kwh"].iloc[0])

        d_rows = assets_df[assets_df["technology"] == "Diesel"]
        if not d_rows.empty:
            parameters["c_D"] = float(d_rows["net_capital_cost_usd"].mean())
            parameters["m_D"] = round(float(d_rows["maintenance_rate_usd_per_km"].mean()) * URBAN_DISTANCE_KM, 2)
            parameters["L_D"] = float(d_rows["order_lead_time_quarters"].iloc[0])

        chg_rows = assets_df[assets_df["asset_type"] == "Charger"]
        if not chg_rows.empty:
            parameters["c_C"] = float(chg_rows["net_capital_cost_usd"].mean())
            parameters["L_C"] = float(chg_rows["order_lead_time_quarters"].iloc[0])

    if monitoring_file is not None and Path(monitoring_file).exists():
        validate_csv_width(Path(monitoring_file))
        mon_df = pd.read_csv(monitoring_file)
        enh_row = mon_df[mon_df["monitoring_tier"] == "Enhanced_Telemetry"]
        if not enh_row.empty:
            parameters["c_M"] = float(enh_row["quarterly_subscription_usd"].iloc[0])
            parameters["c_M_hardware"] = float(enh_row["hardware_capex_per_vehicle_usd"].iloc[0])
            parameters["sigma_nu"] = float(enh_row["sensor_measurement_noise_std_kwh"].iloc[0])
            parameters["missingness_rate"] = float(enh_row["missingness_rate_pct"].iloc[0]) / 100.0
            parameters["enhanced_delay_days"] = float(enh_row["reporting_delay_days"].iloc[0])
        base_row = mon_df[mon_df["monitoring_tier"] == "Baseline_Meter"]
        if not base_row.empty:
            parameters["sigma_Z"] = float(base_row["sensor_measurement_noise_std_kwh"].iloc[0])
            parameters["baseline_delay_days"] = float(base_row["reporting_delay_days"].iloc[0])

    parameters.setdefault("c_M_hardware", 0.0)
    parameters.setdefault("missingness_rate", 0.0)
    parameters.setdefault("enhanced_delay_days", 0.0)
    parameters.setdefault("baseline_delay_days", 0.0)

    if policy_file is not None and Path(policy_file).exists():
        validate_csv_width(Path(policy_file))
        pol_df = pd.read_csv(policy_file)
        for row in pol_df.itertuples(index=False):
            if hasattr(row, "symbol") and hasattr(row, "value"):
                try:
                    parameters[str(row.symbol)] = float(row.value)
                except (ValueError, TypeError):
                    pass

    operations = pd.read_csv(operations_file, na_values=["NA"])
    completed_diesel = operations[
        operations["assigned_tech"].eq("Diesel")
        & operations["service_status"].eq("Completed")
    ]
    diesel_litres = (
        completed_diesel.groupby("duty_class")["diesel_liters"].mean().to_dict()
    )
    required_duties = {"D1_Urban", "D2_Suburban", "D3_Regional"}
    if set(diesel_litres) != required_duties:
        raise ValueError("Completed diesel fixture does not cover all three duty classes")

    electric = operations[operations["assigned_tech"].eq("Electric")].copy()
    partition_error = (
        electric["sim_true_traction_kwh"]
        + electric["sim_true_onboard_aux_kwh"]
        - electric["sim_true_battery_withdrawal_kwh"]
    ).abs().max()
    if partition_error > 0.11:
        raise ValueError(f"Battery-energy partition error is {partition_error:.3f} kWh")
    if electric["sim_true_battery_withdrawal_kwh"].max() > parameters["Q"]:
        raise ValueError("Synthetic EV observation exceeds usable battery capacity")
    baseline_observations = electric.loc[
        electric["monitoring_tier"].eq("Baseline_Meter"), "obs_traction_kwh"
    ]
    if baseline_observations.notna().any():
        raise ValueError("Baseline EV rows expose traction telemetry")

    depot = pd.read_csv(depot_file)
    if "demand_charge_usd_per_kw" in depot.columns:
        parameters["p_demand"] = float(depot["demand_charge_usd_per_kw"].iloc[0])
    else:
        parameters.setdefault("p_demand", 0.0)
    if "peak_demand_kw" in depot.columns:
        zero_chg = depot[depot.get("charger_electricity_kwh", 0) == 0]
        parameters["facility_peak_kw"] = float(zero_chg["peak_demand_kw"].mean()) if not zero_chg.empty else float(depot["peak_demand_kw"].min())
    else:
        parameters.setdefault("facility_peak_kw", 0.0)
    parameters.setdefault("charger_kw", 50.0)

    diesel_maintenance: dict[str, float] = {}
    if assets_file is not None and Path(assets_file).exists():
        d_assets = assets_df[assets_df["technology"] == "Diesel"]
        m_rate_urban = float(d_assets[d_assets["asset_id"].isin(["D-02", "D-03"])]["maintenance_rate_usd_per_km"].mean()) if not d_assets[d_assets["asset_id"].isin(["D-02", "D-03"])].empty else float(d_assets["maintenance_rate_usd_per_km"].mean())
        m_rate_suburban = float(d_assets[d_assets["asset_id"].isin(["D-04", "D-05", "D-06", "D-07"])]["maintenance_rate_usd_per_km"].mean()) if not d_assets[d_assets["asset_id"].isin(["D-04", "D-05", "D-06", "D-07"])].empty else float(d_assets["maintenance_rate_usd_per_km"].mean())
        m_rate_regional = float(d_assets[d_assets["asset_id"].isin(["D-08", "D-09", "D-10"])]["maintenance_rate_usd_per_km"].mean()) if not d_assets[d_assets["asset_id"].isin(["D-08", "D-09", "D-10"])].empty else float(d_assets["maintenance_rate_usd_per_km"].mean())

        dist_urban = float(completed_diesel[completed_diesel["duty_class"] == "D1_Urban"]["distance_km"].mean()) if (completed_diesel["duty_class"] == "D1_Urban").any() else URBAN_DISTANCE_KM
        dist_suburban = float(completed_diesel[completed_diesel["duty_class"] == "D2_Suburban"]["distance_km"].mean()) if (completed_diesel["duty_class"] == "D2_Suburban").any() else SUBURBAN_DISTANCE_KM
        dist_regional = float(completed_diesel[completed_diesel["duty_class"] == "D3_Regional"]["distance_km"].mean()) if (completed_diesel["duty_class"] == "D3_Regional").any() else REGIONAL_DISTANCE_KM

        diesel_maintenance = {
            "D1_Urban": round(m_rate_urban * dist_urban, 2),
            "D2_Suburban": round(m_rate_suburban * dist_suburban, 2),
            "D3_Regional": round(m_rate_regional * dist_regional, 2),
        }
    else:
        m_D_urban = parameters.get("m_D", 18.5)
        m_D_rate = m_D_urban / URBAN_DISTANCE_KM
        diesel_maintenance = {
            "D1_Urban": m_D_urban,
            "D2_Suburban": round(m_D_rate * SUBURBAN_DISTANCE_KM, 2),
            "D3_Regional": round(m_D_rate * REGIONAL_DISTANCE_KM, 2),
        }

    return Calibration(
        parameters=parameters,
        diesel_litres=diesel_litres,
        facility_mean_kwh=float(depot["facility_aux_kwh"].mean()),
        facility_max_kwh=float(depot["facility_aux_kwh"].max()),
        diesel_maintenance=diesel_maintenance,
    )



def load_empirical_calibration(
    operations_file: Path = DATA_DIR / "empirical" / "duty_cycle_operations.csv",
    depot_file: Path = DATA_DIR / "empirical" / "depot_energy.csv",
    assets_file: Path = DATA_DIR / "empirical" / "assets.csv",
    monitoring_file: Path = DATA_DIR / "empirical" / "monitoring_protocols.csv",
    policy_file: Path = DATA_DIR / "empirical" / "carbon_and_policy_inputs.csv",
    parameter_file: Path = DEFAULT_PARAMETER_FILE,
) -> Calibration:
    """Unified loader importing the full calibrated computational empirical package."""
    return load_calibration(
        parameter_file=parameter_file,
        operations_file=operations_file,
        depot_file=depot_file,
        assets_file=assets_file,
        monitoring_file=monitoring_file,
        policy_file=policy_file,
    )

def normalized_belief(
    values: Iterable[float], digits: int = BELIEF_DIGITS
) -> tuple[float, float, float]:
    values = tuple(max(0.0, float(value)) for value in values)
    total = sum(values)
    if total <= 0:
        raise ValueError("Belief has zero mass")
    normalized = tuple(value / total for value in values)
    maximum = max(normalized)
    if maximum >= 1.0 - 1e-7:
        index = normalized.index(maximum)
        return tuple(1.0 if position == index else 0.0 for position in range(3))
    rounded = tuple(round(value, digits) for value in normalized)
    correction = round(1.0 - sum(rounded), digits)
    result = (rounded[0], rounded[1], round(rounded[2] + correction, digits))
    return result


def categorical_normal_probabilities(
    means: tuple[float, ...], std: float, categories: int = 3
) -> tuple[tuple[float, ...], ...]:
    """Return P(category | theta) using midpoint thresholds between means."""
    if std <= 0:
        raise ValueError("Signal standard deviation must be positive")
    if categories < 3:
        raise ValueError("At least three signal categories are required")
    thresholds = tuple((left + right) / 2.0 for left, right in zip(means, means[1:]))
    if categories != 3:
        lower, upper = thresholds[0], thresholds[-1]
        thresholds = tuple(
            lower + (upper - lower) * index / (categories - 2)
            for index in range(categories - 1)
        )
    probabilities = []
    for mean in means:
        bounds = (-math.inf, *thresholds, math.inf)
        row = []
        for lower, upper in zip(bounds, bounds[1:]):
            lower_cdf = 0.0 if lower == -math.inf else norm.cdf((lower - mean) / std)
            upper_cdf = 1.0 if upper == math.inf else norm.cdf((upper - mean) / std)
            row.append(max(0.0, upper_cdf - lower_cdf))
        row_total = sum(row)
        probabilities.append(tuple(value / row_total for value in row))
    return tuple(probabilities)


class PrototypeDP:
    def __init__(
        self,
        calibration: Calibration,
        scenario: str,
        allow_monitoring: bool = True,
        grid_factor_symbol: str = "g_E_RFCE",
        strategy: str = "joint",
        initial_belief: tuple[float, float, float] = INITIAL_BELIEF,
        carbon_accounting: str = "global_hard",
        risk_tolerance: float = 0.01,
        suppress_enhanced_signal: bool = False,
    ) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {scenario!r}")
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}")
        self.calibration = calibration
        self.parameters = calibration.parameters
        self.scenario = scenario
        self.allow_monitoring = allow_monitoring
        self.grid_factor_symbol = grid_factor_symbol
        self.strategy = strategy
        self.belief_digits = int(self.parameters.get("belief_digits", BELIEF_DIGITS))
        self.support_tolerance = self.parameters.get("support_tolerance", SUPPORT_TOLERANCE)
        self.signal_categories = int(self.parameters.get("signal_categories", 3))
        if not 2 <= self.belief_digits <= 8:
            raise ValueError("Belief precision must be between two and eight digits")
        if not 0.0 <= self.support_tolerance < 1.0 / 3.0:
            raise ValueError("Support tolerance must lie in [0, 1/3)")
        if self.signal_categories < 3:
            raise ValueError("At least three signal categories are required")
        self.initial_belief = normalized_belief(initial_belief, self.belief_digits)
        if carbon_accounting not in {"global_hard", "posterior_quantile"}:
            raise ValueError("Carbon accounting must be global_hard or posterior_quantile")
        if not 0.0 <= risk_tolerance < 1.0:
            raise ValueError("Risk tolerance must lie in [0, 1)")
        self.carbon_accounting = carbon_accounting
        self.risk_tolerance = risk_tolerance
        self.suppress_enhanced_signal = suppress_enhanced_signal
        cap_symbol, budget_symbol = SCENARIOS[scenario]
        self.annual_carbon_limit = self.to_carbon_units(self.parameters[cap_symbol])
        self.cumulative_carbon_limit = self.to_carbon_units(self.parameters[budget_symbol])
        self.annual_capital_limit = self.to_money_units(self.parameters["B_yr"])
        self.cumulative_capital_limit = self.to_money_units(self.parameters["W_1"])
        self.policy: dict[State, Action] = {}
        self.states_evaluated = 0

    @staticmethod
    def to_carbon_units(tonnes: float) -> int:
        return int(round(tonnes * TONNES_TO_MILLITONNES))

    @staticmethod
    def to_money_units(usd: float) -> int:
        return int(round(usd * USD_TO_CENTS))

    @staticmethod
    def from_carbon_units(value: int) -> float:
        return value / TONNES_TO_MILLITONNES

    @staticmethod
    def from_money_units(value: int) -> float:
        return value / USD_TO_CENTS

    def initial_state(self) -> State:
        return State(
            quarter=1,
            electric_vehicles=int(self.parameters["N_0_E"]),
            chargers=int(self.parameters["k_0"]),
            electric_pipeline=(0,) * (int(self.parameters["L_E"]) - 1),
            charger_pipeline=(0,) * (int(self.parameters["L_C"]) - 1),
            cumulative_carbon=self.cumulative_carbon_limit,
            annual_carbon=self.annual_carbon_limit,
            cumulative_capital=self.cumulative_capital_limit,
            annual_capital=self.annual_capital_limit,
            belief=self.initial_belief,
            baseline_observed=False,
            enhanced_observed=False,
            pending_baseline=0,
        )

    @staticmethod
    def advance_pipeline(
        installed: int, pipeline: tuple[int, ...], order: int
    ) -> tuple[int, tuple[int, ...]]:
        if not pipeline:
            return installed + order, ()
        arrivals = pipeline[0]
        return installed + arrivals, (*pipeline[1:], order)

    def electric_duties(self, state: State) -> int:
        return min(state.electric_vehicles, state.chargers, MAX_ELECTRIC_DUTIES)

    def expected_theta(self, belief: tuple[float, float, float]) -> float:
        return sum(probability * theta for probability, theta in zip(belief, THETA))

    def operating_cost(self, state: State, monitored_vehicles: int) -> float:
        p = self.parameters
        cycles = p["w_t"]
        electric_duties = self.electric_duties(state)
        urban_diesel = MAX_ELECTRIC_DUTIES - electric_duties
        fixed_diesel_litres = (
            p["n_D2"] * self.calibration.diesel_litres["D2_Suburban"]
            + p["n_D3"] * self.calibration.diesel_litres["D3_Regional"]
        )
        diesel_litres_per_cycle = (
            fixed_diesel_litres
            + urban_diesel * self.calibration.diesel_litres["D1_Urban"]
        )
        m_dict = self.calibration.diesel_maintenance or {
            "D1_Urban": p.get("m_D", 18.5),
            "D2_Suburban": round(p.get("m_D", 18.5) * SUBURBAN_DISTANCE_KM / URBAN_DISTANCE_KM, 2),
            "D3_Regional": round(p.get("m_D", 18.5) * REGIONAL_DISTANCE_KM / URBAN_DISTANCE_KM, 2),
        }
        fixed_diesel_maint = (
            p["n_D2"] * m_dict["D2_Suburban"]
            + p["n_D3"] * m_dict["D3_Regional"]
        )
        diesel_maintenance_per_cycle = (
            fixed_diesel_maint
            + urban_diesel * m_dict["D1_Urban"]
        )

        expected_battery = (
            URBAN_DISTANCE_KM * self.expected_theta(state.belief)
            + (p["aux_min"] + p["aux_max"]) / 2.0
        )
        vehicle_grid_kwh = electric_duties * expected_battery / p["eta"]
        total_grid_kwh = vehicle_grid_kwh + self.calibration.facility_mean_kwh
        daily_cost = (
            p["p_D"] * diesel_litres_per_cycle
            + diesel_maintenance_per_cycle
            + p["p_E"] * total_grid_kwh
            + p["m_E"] * electric_duties
        )
        monitoring_cost = monitored_vehicles * p["c_M"]

        quarterly_demand_charge = 0.0
        if p.get("p_demand", 0.0) > 0.0:
            peak_kw = p.get("facility_peak_kw", 9.8) + electric_duties * p.get("charger_kw", 50.0)
            quarterly_demand_charge = 3.0 * p["p_demand"] * peak_kw

        return cycles * daily_cost + monitoring_cost + quarterly_demand_charge

    def carbon_theta(self, belief: tuple[float, float, float]) -> float:
        if self.carbon_accounting == "global_hard":
            return max(THETA)
        cumulative = 0.0
        target = 1.0 - self.risk_tolerance
        for probability, theta in zip(belief, THETA):
            cumulative += probability
            if cumulative + 1e-12 >= target:
                return theta
        return max(THETA)

    def robust_carbon(self, state: State) -> int:
        p = self.parameters
        cycles = p["w_t"]
        electric_duties = self.electric_duties(state)
        urban_diesel = MAX_ELECTRIC_DUTIES - electric_duties
        diesel_litres_per_cycle = (
            p["n_D2"] * self.calibration.diesel_litres["D2_Suburban"]
            + p["n_D3"] * self.calibration.diesel_litres["D3_Regional"]
            + urban_diesel * self.calibration.diesel_litres["D1_Urban"]
        )
        supported_theta = self.carbon_theta(state.belief)
        worst_battery = (
            URBAN_DISTANCE_KM * supported_theta
            + p["eps_trac_max"]
            + p["aux_max"]
        )
        grid_kwh_per_cycle = (
            electric_duties * worst_battery / p["eta"]
            + self.calibration.facility_max_kwh
        )
        tonnes = cycles * (
            diesel_litres_per_cycle * p["g_D"]
            + grid_kwh_per_cycle * p[self.grid_factor_symbol]
        )
        return self.to_carbon_units(tonnes)

    def investment_cost(self, action: Action) -> int:
        p = self.parameters
        usd = (
            action.order_electric * p["c_E"]
            + action.order_chargers * p["c_C"]
            + action.monitored_vehicles * p.get("c_M_hardware", 0.0)
        )
        return self.to_money_units(usd)

    def channel_signal_parameters(
        self, state: State, channel: str, monitored_vehicles: int = 0
    ) -> tuple[tuple[float, ...], float] | None:
        electric_duties = self.electric_duties(state)
        pending_duties = int(getattr(state, "pending_baseline", 0))
        p = self.parameters
        if channel == "enhanced":
            if monitored_vehicles <= 0 or electric_duties == 0:
                return None
            r_miss = p.get("missingness_rate", 0.0)
            effective_n = max(1.0, p["w_t"] * monitored_vehicles * (1.0 - r_miss))
            means = tuple(URBAN_DISTANCE_KM * theta for theta in THETA)
            variance = (
                p["sigma_trac"] ** 2 / effective_n
                + p["sigma_nu"] ** 2
            )
            return means, math.sqrt(variance)
        if channel == "baseline":
            if state.baseline_observed:
                return None
            baseline_delay_days = p.get("baseline_delay_days", 0.0)
            if pending_duties > 0:
                effective_duties = pending_duties
            elif baseline_delay_days > 0:
                return None
            else:
                if electric_duties == 0:
                    return None
                effective_duties = electric_duties
            means = tuple(
                p["w_t"]
                * (
                    effective_duties
                    * (URBAN_DISTANCE_KM * theta + (p["aux_min"] + p["aux_max"]) / 2.0)
                    / p["eta"]
                    + self.calibration.facility_mean_kwh
                )
                for theta in THETA
            )
            variance = p["w_t"] * (
                effective_duties
                * (p["sigma_trac"] ** 2 + ((p["aux_max"] - p["aux_min"]) / 6.0) ** 2)
                / p["eta"] ** 2
                + p["sigma_Z"] ** 2
            ) + p["sigma_Z_qtr"] ** 2
            return means, math.sqrt(variance)
        raise ValueError(f"Unknown information channel {channel!r}")

    def signal_parameters(
        self, state: State, monitored_vehicles: int
    ) -> tuple[tuple[float, ...], float] | None:
        """Return a single-channel summary for diagnostics and path replay.

        Joint baseline-plus-telemetry transitions are represented by
        :meth:`signal_probabilities`; callers needing realized categories should
        inspect the two channel-specific parameter sets.
        """
        if monitored_vehicles > 0 and not self.suppress_enhanced_signal:
            return self.channel_signal_parameters(state, "enhanced", monitored_vehicles)
        return self.channel_signal_parameters(state, "baseline")

    def signal_probabilities(
        self, state: State, monitored_vehicles: int
    ) -> tuple[tuple[float, ...], ...] | None:
        baseline_parameters = self.channel_signal_parameters(state, "baseline")
        enhanced_parameters = None
        if monitored_vehicles > 0 and not self.suppress_enhanced_signal:
            enhanced_parameters = self.channel_signal_parameters(
                state, "enhanced", monitored_vehicles
            )
        channels = []
        for parameters in (baseline_parameters, enhanced_parameters):
            if parameters is None:
                continue
            means, standard_deviation = parameters
            channels.append(
                categorical_normal_probabilities(
                    means, standard_deviation, self.signal_categories
                )
            )
        if not channels:
            return None
        if len(channels) == 1:
            return channels[0]
        baseline, enhanced = channels
        return tuple(
            tuple(
                baseline[regime][baseline_category]
                * enhanced[regime][enhanced_category]
                for baseline_category in range(self.signal_categories)
                for enhanced_category in range(self.signal_categories)
            )
            for regime in range(len(THETA))
        )

    def posterior_branches(
        self, belief: tuple[float, float, float],
        conditional: tuple[tuple[float, ...], ...] | None,
    ) -> tuple[tuple[float, tuple[float, float, float], int | None], ...]:
        if conditional is None:
            return ((1.0, belief, None),)
        branches = []
        category_count = len(conditional[0])
        for category in range(category_count):
            weighted = tuple(
                belief[index] * conditional[index][category]
                for index in range(len(belief))
            )
            predictive = sum(weighted)
            if predictive > 1e-8:
                branches.append((predictive, normalized_belief(weighted, self.belief_digits), category))
        return tuple(branches)

    def unrestricted_actions(self, state: State) -> tuple[Action, ...]:
        actions = []
        installed_or_committed_e = state.electric_vehicles + sum(state.electric_pipeline)
        installed_or_committed_c = state.chargers + sum(state.charger_pipeline)
        max_order_e = MAX_ELECTRIC_DUTIES - installed_or_committed_e
        max_order_c = MAX_ELECTRIC_DUTIES - installed_or_committed_c
        if state.quarter + self.calibration.electric_lead_time > HORIZON_QUARTERS:
            max_order_e = 0
        if state.quarter + self.calibration.charger_lead_time > HORIZON_QUARTERS:
            max_order_c = 0
        electric_duties = self.electric_duties(state)
        belief_is_uncertain = max(state.belief) < 1.0 - 1e-7
        monitoring_choices = (0,)
        if (
            self.allow_monitoring
            and electric_duties > 0
            and belief_is_uncertain
            and not state.enhanced_observed
        ):
            if self.suppress_enhanced_signal:
                monitoring_choices = (0, 1)
            else:
                enhanced = self.channel_signal_parameters(state, "enhanced", 1)
                if enhanced is not None:
                    monitoring_choices = (0, 1)
        for order_electric in range(max_order_e + 1):
            for order_chargers in range(max_order_c + 1):
                committed_electric = installed_or_committed_e + order_electric
                committed_chargers = installed_or_committed_c + order_chargers
                if committed_electric > committed_chargers:
                    continue
                for monitored_vehicles in monitoring_choices:
                    actions.append(Action(order_electric, order_chargers, monitored_vehicles))
        return tuple(actions)

    def actions(self, state: State) -> Iterable[Action]:
        unrestricted = self.unrestricted_actions(state)
        if self.strategy == "joint":
            return unrestricted
        if self.strategy in {"flexible", "oracle"}:
            return tuple(action for action in unrestricted if action.monitored_vehicles == 0)

        forced_action = None
        if self.strategy == "immediate_scale":
            forced_action = {
                1: Action(0, 2, 0),
                2: Action(2, 0, 0),
            }.get(state.quarter, Action(0, 0, 0))
        elif self.strategy == "fixed_pilot":
            if state.quarter <= 6:
                forced_action = {
                    2: Action(0, 1, 0),
                    3: Action(1, 0, 0),
                }.get(state.quarter, Action(0, 0, 0))
            else:
                return tuple(action for action in unrestricted if action.monitored_vehicles == 0)
        elif self.strategy == "matched_ablation":
            if state.quarter <= 6:
                forced_action = {
                    2: Action(0, 1, 0),
                    3: Action(1, 0, 0),
                    6: Action(0, 0, 1),
                }.get(state.quarter, Action(0, 0, 0))
            else:
                return tuple(action for action in unrestricted if action.monitored_vehicles == 0)

        if forced_action is None:
            return unrestricted
        return (forced_action,) if forced_action in unrestricted else ()

    def transition(
        self,
        state: State,
        action: Action,
        posterior: tuple[float, float, float],
        carbon_debit: int,
        capital_debit: int,
    ) -> State:
        next_electric, electric_pipeline = self.advance_pipeline(
            state.electric_vehicles, state.electric_pipeline, action.order_electric
        )
        next_chargers, charger_pipeline = self.advance_pipeline(
            state.chargers, state.charger_pipeline, action.order_chargers
        )
        cumulative_carbon = state.cumulative_carbon - carbon_debit
        cumulative_capital = state.cumulative_capital - capital_debit
        if state.quarter % 4 == 0:
            annual_carbon = self.annual_carbon_limit
            annual_capital = self.annual_capital_limit
        else:
            annual_carbon = state.annual_carbon - carbon_debit
            annual_capital = state.annual_capital - capital_debit

        pending_duties = int(getattr(state, "pending_baseline", 0))
        baseline_delay_days = self.parameters.get("baseline_delay_days", 0.0)

        if action.monitored_vehicles > 0:
            next_enhanced_observed = True
        else:
            next_enhanced_observed = state.enhanced_observed

        if baseline_delay_days > 0:
            if pending_duties > 0:
                next_baseline_observed = True
                next_pending_baseline = 0
            elif self.electric_duties(state) > 0 and not state.baseline_observed:
                next_baseline_observed = False
                next_pending_baseline = self.electric_duties(state)
            else:
                next_baseline_observed = state.baseline_observed
                next_pending_baseline = 0
        else:
            next_baseline_observed = (
                state.baseline_observed
                or self.electric_duties(state) > 0
                or (action.monitored_vehicles > 0 and not self.suppress_enhanced_signal)
            )
            next_pending_baseline = 0

        return State(
            quarter=state.quarter + 1,
            electric_vehicles=next_electric,
            chargers=next_chargers,
            electric_pipeline=electric_pipeline,
            charger_pipeline=charger_pipeline,
            cumulative_carbon=cumulative_carbon,
            annual_carbon=annual_carbon,
            cumulative_capital=cumulative_capital,
            annual_capital=annual_capital,
            belief=posterior,
            baseline_observed=next_baseline_observed,
            enhanced_observed=next_enhanced_observed,
            pending_baseline=next_pending_baseline,
        )

    def terminal_value(self, state: State) -> float:
        return -self.parameters["s_E"] * state.electric_vehicles

    def solve(self) -> float:
        self.policy.clear()
        self.states_evaluated = 0
        self._value.cache_clear()
        return self._value(self.initial_state())

    @lru_cache(maxsize=None)
    def _value(self, state: State) -> float:
        self.states_evaluated += 1
        if state.quarter > HORIZON_QUARTERS:
            return self.terminal_value(state)
        carbon_debit = self.robust_carbon(state)
        if carbon_debit > state.cumulative_carbon or carbon_debit > state.annual_carbon:
            return INFEASIBLE_COST

        best_cost = INFEASIBLE_COST
        best_action = None
        discount = self.parameters["delta"]
        for action in self.actions(state):
            capital_debit = self.investment_cost(action)
            if capital_debit > state.cumulative_capital or capital_debit > state.annual_capital:
                continue
            operating_cost = self.operating_cost(state, action.monitored_vehicles)
            immediate_cost = self.from_money_units(capital_debit) + operating_cost
            conditional = self.signal_probabilities(state, action.monitored_vehicles)
            continuation = 0.0
            feasible_action = True
            for probability, posterior, _ in self.posterior_branches(state.belief, conditional):
                next_state = self.transition(
                    state,
                    action,
                    posterior,
                    carbon_debit,
                    capital_debit,
                )
                next_value = self._value(next_state)
                if math.isinf(next_value):
                    feasible_action = False
                    break
                continuation += probability * next_value
            if not feasible_action:
                continue
            total_cost = immediate_cost + discount * continuation
            if total_cost < best_cost - 1e-6:
                best_cost = total_cost
                best_action = action
        if best_action is not None:
            self.policy[state] = best_action
        return best_cost

    def most_likely_path(self) -> list[dict[str, object]]:
        state = self.initial_state()
        path = []
        while state.quarter <= HORIZON_QUARTERS and state in self.policy:
            action = self.policy[state]
            carbon_debit = self.robust_carbon(state)
            capital_debit = self.investment_cost(action)
            electric_duties = self.electric_duties(state)
            path.append(
                {
                    "quarter": state.quarter,
                    "electric_vehicles": state.electric_vehicles,
                    "chargers": state.chargers,
                    "electric_duties": electric_duties,
                    "order_electric": action.order_electric,
                    "order_chargers": action.order_chargers,
                    "monitored_vehicles": action.monitored_vehicles,
                    "carbon_tonnes": self.from_carbon_units(carbon_debit),
                    "remaining_cumulative_carbon": self.from_carbon_units(state.cumulative_carbon),
                    "remaining_annual_carbon": self.from_carbon_units(state.annual_carbon),
                    "remaining_cumulative_capital": self.from_money_units(state.cumulative_capital),
                    "remaining_annual_capital": self.from_money_units(state.annual_capital),
                    "belief_low": state.belief[0],
                    "belief_mid": state.belief[1],
                    "belief_high": state.belief[2],
                }
            )
            conditional = self.signal_probabilities(state, action.monitored_vehicles)
            branches = self.posterior_branches(state.belief, conditional)
            _, posterior, _ = max(branches, key=lambda branch: branch[0])
            state = self.transition(
                state,
                action,
                posterior,
                carbon_debit,
                capital_debit,
            )
        return path

    def summary(self, objective: float) -> dict[str, object]:
        path = self.most_likely_path() if not math.isinf(objective) else []
        return {
            "scenario": self.scenario,
            "strategy": self.strategy,
            "carbon_accounting": self.carbon_accounting,
            "risk_tolerance": self.risk_tolerance,
            "feasible": not math.isinf(objective),
            "objective_usd": None if math.isinf(objective) else round(objective, 2),
            "states_evaluated": self.states_evaluated,
            "initial_action": self.policy.get(self.initial_state())._asdict()
            if self.initial_state() in self.policy
            else None,
            "most_likely_path": path,
            "first_monitored_signal_branches": self.first_monitored_signal_branches()
            if not math.isinf(objective)
            else [],
        }

    def first_monitored_signal_branches(self) -> list[dict[str, object]]:
        state = self.initial_state()
        while state.quarter <= HORIZON_QUARTERS and state in self.policy:
            action = self.policy[state]
            conditional = self.signal_probabilities(state, action.monitored_vehicles)
            branches = self.posterior_branches(state.belief, conditional)
            carbon_debit = self.robust_carbon(state)
            capital_debit = self.investment_cost(action)
            if action.monitored_vehicles > 0:
                result = []
                for probability, posterior, category in branches:
                    next_state = self.transition(
                        state,
                        action,
                        posterior,
                        carbon_debit,
                        capital_debit,
                    )
                    next_action = self.policy.get(next_state)
                    lookahead_state = next_state
                    electric_orders = 0
                    maximum_electric_vehicles = next_state.electric_vehicles
                    for _ in range(4):
                        lookahead_action = self.policy.get(lookahead_state)
                        if lookahead_action is None:
                            break
                        electric_orders += lookahead_action.order_electric
                        lookahead_carbon = self.robust_carbon(lookahead_state)
                        lookahead_capital = self.investment_cost(lookahead_action)
                        lookahead_conditional = self.signal_probabilities(
                            lookahead_state,
                            lookahead_action.monitored_vehicles,
                        )
                        lookahead_branches = self.posterior_branches(
                            lookahead_state.belief,
                            lookahead_conditional,
                        )
                        _, lookahead_posterior, _ = max(
                            lookahead_branches,
                            key=lambda branch: branch[0],
                        )
                        lookahead_state = self.transition(
                            lookahead_state,
                            lookahead_action,
                            lookahead_posterior,
                            lookahead_carbon,
                            lookahead_capital,
                        )
                        maximum_electric_vehicles = max(
                            maximum_electric_vehicles,
                            lookahead_state.electric_vehicles,
                        )
                    result.append(
                        {
                            "signal_category": category,
                            "probability": round(float(probability), 6),
                            "posterior_low": posterior[0],
                            "posterior_mid": posterior[1],
                            "posterior_high": posterior[2],
                            "next_order_electric": None
                            if next_action is None
                            else next_action.order_electric,
                            "next_order_chargers": None
                            if next_action is None
                            else next_action.order_chargers,
                            "electric_orders_next_four_quarters": electric_orders,
                            "maximum_electric_vehicles_next_four_quarters": maximum_electric_vehicles,
                        }
                    )
                return result
            _, posterior, _ = max(branches, key=lambda branch: branch[0])
            state = self.transition(
                state,
                action,
                posterior,
                carbon_debit,
                capital_debit,
            )
        return []


def run_scenarios(
    scenarios: Iterable[str], allow_monitoring: bool = True
) -> list[dict[str, object]]:
    calibration = load_calibration()
    summaries = []
    for scenario in scenarios:
        model = PrototypeDP(calibration, scenario, allow_monitoring=allow_monitoring)
        objective = model.solve()
        summaries.append(model.summary(objective))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=(*SCENARIOS, "all"),
        default="all",
        help="Carbon configuration to solve",
    )
    parser.add_argument(
        "--disable-monitoring",
        action="store_true",
        help="Remove enhanced telemetry from the action space",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path",
    )
    args = parser.parse_args()
    scenarios = SCENARIOS if args.scenario == "all" else (args.scenario,)
    summaries = run_scenarios(scenarios, allow_monitoring=not args.disable_monitoring)
    output = json.dumps(summaries, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()