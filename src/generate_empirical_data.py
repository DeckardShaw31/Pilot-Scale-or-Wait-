import csv
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EMPIRICAL_DIR = DATA_DIR / "empirical"
SCENARIOS_DIR = EMPIRICAL_DIR / "scenarios"
EMPIRICAL_DIR.mkdir(parents=True, exist_ok=True)
SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)

# 1. 260 Operating Dates (Calendar Year 2025: Mon-Fri)
start_date = date(2025, 1, 2)
operating_dates = []
curr = start_date
while len(operating_dates) < 260:
    if curr.weekday() < 5:
        operating_dates.append(curr)
    curr += timedelta(days=1)

def get_ambient_temperature(day):
    doy = day.timetuple().tm_yday
    base_temp = 14.5 - 14.0 * math.cos(2.0 * math.pi * (doy - 20) / 365.25)
    noise = random.gauss(0.0, 3.2)
    temp = round(base_temp + noise, 1)
    precip_prob = 0.22 if (3 <= day.month <= 8) else 0.28
    precip_mm = round(random.expovariate(0.35), 1) if random.random() < precip_prob else 0.0
    if precip_mm > 0.0 and temp <= 1.0:
        weather = "Snow"
    elif precip_mm > 8.0:
        weather = "Heavy_Rain"
    elif precip_mm > 0.0:
        weather = "Rain"
    elif random.random() < 0.12:
        weather = "Cold_Wind" if temp < 10.0 else "Windy"
    else:
        weather = "Dry"
    return temp, weather, precip_mm

BATTERY_CAPACITY_Q = 200.0  # Usable kWh
CHARGING_EFFICIENCY_ETA = 0.88  # EPRI demonstration benchmark

# 2. Pre-generate master physical realization (FROZEN across all counterfactual scenarios)
master_days = []
rng = random.Random(42)

for day_idx, op_date in enumerate(operating_dates, 1):
    iso_date = op_date.isoformat()
    temp_c, weather, precip_mm = get_ambient_temperature(op_date)
    quarter = (op_date.month - 1) // 3 + 1

    duties_config = [
        ("DUTY_URB_01", "D1_Urban", 85.0, 5.8, 30, 140, "EV-01", "Electric"),
        ("DUTY_URB_02", "D1_Urban", 84.5, 5.7, 28, 135, "EV-02" if day_idx >= 66 else "D-02", "Electric" if day_idx >= 66 else "Diesel"),
        ("DUTY_URB_03", "D1_Urban", 85.5, 5.9, 32, 145, "EV-03" if day_idx >= 131 else "D-03", "Electric" if day_idx >= 131 else "Diesel"),
        ("DUTY_SUB_01", "D2_Suburban", 134.0, 6.4, 16, 280, "D-04", "Diesel"),
        ("DUTY_SUB_02", "D2_Suburban", 135.0, 6.5, 15, 290, "D-05", "Diesel"),
        ("DUTY_SUB_03", "D2_Suburban", 136.0, 6.6, 17, 310, "D-06", "Diesel"),
        ("DUTY_SUB_04", "D2_Suburban", 135.5, 6.5, 14, 275, "D-07", "Diesel"),
        ("DUTY_REG_01", "D3_Regional", 210.0, 7.8, 6, 520, "D-08", "Diesel"),
        ("DUTY_REG_02", "D3_Regional", 208.5, 7.7, 5, 490, "D-09", "Diesel"),
        ("DUTY_REG_03", "D3_Regional", 212.0, 7.9, 7, 550, "D-10", "Diesel")
    ]

    day_duties = []
    for duty_id, duty_class, base_dist, base_dur, base_stops, base_elev, veh_id, tech in duties_config:
        r = rng.random()
        if r < 0.98846:
            service_status = "Completed"
            is_outsourced = 0
            outsourced_cost = 0.0
        elif r < 0.99577:
            service_status = "Late"
            is_outsourced = 0
            outsourced_cost = 0.0
        else:
            service_status = "Outsourced"
            is_outsourced = 1
            outsourced_cost = 180.0 if duty_class == "D1_Urban" else (270.0 if duty_class == "D2_Suburban" else 420.0)

        dist_km = round(max(50.0, base_dist + rng.gauss(0.0, 1.8)), 1)
        dur_hr = round(max(3.0, base_dur + rng.gauss(0.0, 0.25) + (0.6 if service_status == "Late" else 0.0)), 1)
        stops = max(2, int(base_stops + rng.randint(-3, 3)))
        elev = max(50, int(base_elev + rng.randint(-20, 20)))
        payload_ratio = round(min(0.98, max(0.55, 0.72 + rng.gauss(0.0, 0.06))), 2)
        payload_kg = int(payload_ratio * 4000)

        dep_hour = 7
        dep_min = rng.choice([0, 15, 30])
        departure_time = f"{dep_hour:02d}:{dep_min:02d}"
        ret_total_min = dep_hour * 60 + dep_min + int(dur_hr * 60)
        ret_h = ret_total_min // 60
        ret_m = ret_total_min % 60
        return_time = f"{ret_h:02d}:{ret_m:02d}"
        charging_window_hr = round(24.0 - (ret_total_min / 60.0) + 7.0, 1)

        eps_trac = max(-10.0, min(10.0, rng.gauss(0.0, 3.5)))
        eps_aux = max(-2.5, min(2.5, rng.gauss(0.0, 1.0)))
        sensor_noise = max(-1.5, min(1.5, rng.gauss(0.0, 0.5)))
        is_packet_dropped = rng.random() < 0.0132  # 1.32% packet loss rate

        noise_l = rng.gauss(0.0, 0.5)
        if weather in ("Snow", "Heavy_Rain"):
            noise_l += 1.2

        day_duties.append({
            "duty_id": duty_id, "duty_class": duty_class, "veh_id": veh_id, "tech": tech,
            "dist_km": dist_km, "dur_hr": dur_hr, "stops": stops, "elev": elev,
            "payload_kg": payload_kg, "payload_ratio": payload_ratio,
            "departure_time": departure_time, "return_time": return_time,
            "ret_h": ret_h, "ret_m": ret_m, "charging_window_hr": charging_window_hr,
            "service_status": service_status, "is_outsourced": is_outsourced,
            "outsourced_cost": outsourced_cost, "eps_trac": eps_trac, "eps_aux": eps_aux,
            "sensor_noise": sensor_noise, "is_packet_dropped": is_packet_dropped,
            "noise_l": noise_l
        })

    master_days.append({
        "day_idx": day_idx, "op_date": op_date, "iso_date": iso_date,
        "temp_c": temp_c, "weather": weather, "precip_mm": precip_mm,
        "quarter": quarter, "duties": day_duties
    })

# Channel 4 Corporate Quarterly Fuel & Accounting Reconciliation Dates (End of Quarter + 45 Days)
quarter_recon_dates = {
    1: "2025-05-15T00:00:00",
    2: "2025-08-15T00:00:00",
    3: "2025-11-15T00:00:00",
    4: "2026-02-15T00:00:00"
}

ops_headers = [
    "date","vehicle_id","assigned_tech","technology","vehicle_class","duty_id",
    "duty_class","distance_km","duration_hr","payload_kg","payload_ratio",
    "stop_count","elevation_gain_m","departure_time","return_time",
    "charging_window_hr","ambient_temp_c","weather_cond","precipitation_mm",
    "service_status","is_outsourced","outsourced_cost_usd","diesel_liters",
    "sim_true_traction_kwh","sim_true_onboard_aux_kwh","sim_true_battery_withdrawal_kwh",
    "sim_true_grid_charging_kwh","state_of_charge_start_pct","state_of_charge_end_pct",
    "charger_id","charger_power_kw","monitoring_tier","obs_traction_kwh",
    "telemetry_status","obs_availability","dispatch_reported_at","telemetry_reported_at",
    "report_available_at"
]

def build_scenario_operations(latent_theta, target_path):
    rows = []
    daily_bevs = {}

    for day in master_days:
        iso_date = day["iso_date"]
        temp_c = day["temp_c"]
        weather = day["weather"]
        precip_mm = day["precip_mm"]
        quarter = day["quarter"]
        recon_date = quarter_recon_dates[quarter]

        daily_bevs[iso_date] = []

        for d in day["duties"]:
            veh_id = d["veh_id"]
            tech = d["tech"]
            duty_id = d["duty_id"]
            duty_class = d["duty_class"]
            dist_km = d["dist_km"]
            dur_hr = d["dur_hr"]
            payload_kg = d["payload_kg"]
            payload_ratio = d["payload_ratio"]
            stops = d["stops"]
            elev = d["elev"]
            departure_time = d["departure_time"]
            return_time = d["return_time"]
            charging_window_hr = d["charging_window_hr"]
            service_status = d["service_status"]
            is_outsourced = d["is_outsourced"]
            outsourced_cost = d["outsourced_cost"]
            ret_h = d["ret_h"]
            ret_m = d["ret_m"]

            # Channel 1: Operational Dispatch Reporting (shift conclusion + 30 min)
            dispatch_avail = f"{iso_date}T{ret_h:02d}:{min(59, ret_m + 30):02d}:00"

            if is_outsourced:
                diesel_liters_val = ""
                sim_trac_val = ""
                sim_aux_val = ""
                sim_batt_val = ""
                sim_grid_val = ""
                soc_start_val = ""
                soc_end_val = ""
                charger_id_val = ""
                charger_kw_val = ""
                tier_val = "None"
                obs_trac_val = ""
                telemetry_status = "Not_Applicable_Outsourced"
                obs_avail_val = "Quarter_End"
                telemetry_avail = ""
                report_avail = dispatch_avail
            elif tech == "Diesel":
                base_l = 25.6 if duty_class == "D1_Urban" else (37.82 if duty_class == "D2_Suburban" else 56.85)
                diesel_liters_val = f"{round(max(10.0, base_l + d['noise_l']), 1):.1f}"
                sim_trac_val = ""
                sim_aux_val = ""
                sim_batt_val = ""
                sim_grid_val = ""
                soc_start_val = ""
                soc_end_val = ""
                charger_id_val = ""
                charger_kw_val = ""
                tier_val = "None"
                obs_trac_val = ""
                telemetry_status = "Not_Installed_Diesel"
                obs_avail_val = "Quarter_End"
                telemetry_avail = ""
                report_avail = recon_date
            else:
                # Electric duty
                eps_trac = d["eps_trac"]
                eps_aux = d["eps_aux"]
                sensor_noise = d["sensor_noise"]
                is_packet_dropped = d["is_packet_dropped"]

                sim_trac = round(dist_km * latent_theta + eps_trac, 1)
                raw_aux = 7.5 + 0.35 * abs(15.0 - temp_c) + eps_aux
                if weather in ("Snow", "Heavy_Rain"):
                    raw_aux += 2.0
                sim_aux = round(max(5.0, min(20.0, raw_aux)), 1)
                sim_batt = round(sim_trac + sim_aux, 1)
                sim_grid = round(sim_batt / CHARGING_EFFICIENCY_ETA, 1)
                soc_start = 100.0
                soc_end = round(100.0 - (sim_batt / BATTERY_CAPACITY_Q * 100.0), 1)

                chg_id = "CHG-01" if veh_id in ("EV-01", "EV-03") else "CHG-02"
                charger_id_val = chg_id
                charger_kw_val = "50.0"

                if veh_id in ("EV-01", "EV-02"):
                    tier_val = "Enhanced_Telemetry"
                    # Channel 2: Real-time CAN Telemetry (shift conclusion + 5 min)
                    telemetry_avail = f"{iso_date}T{ret_h:02d}:{min(59, ret_m + 5):02d}:00"
                    if is_packet_dropped:
                        obs_trac_val = ""
                        telemetry_status = "Dropped_Cellular_Deadzone"
                    else:
                        obs_trac_val = f"{round(sim_trac + sensor_noise, 1):.1f}"
                        telemetry_status = "Received_Valid"
                    obs_avail_val = "Immediate_CAN_Bus"
                    report_avail = telemetry_avail if not is_packet_dropped else dispatch_avail
                else:
                    # EV-03: Baseline_Meter
                    tier_val = "Baseline_Meter"
                    obs_trac_val = ""
                    telemetry_status = "Not_Subscribed_Baseline"
                    obs_avail_val = "Quarter_End"
                    telemetry_avail = ""
                    report_avail = recon_date

                diesel_liters_val = ""
                sim_trac_val = f"{sim_trac:.1f}"
                sim_aux_val = f"{sim_aux:.1f}"
                sim_batt_val = f"{sim_batt:.1f}"
                sim_grid_val = f"{sim_grid:.1f}"
                soc_start_val = f"{soc_start:.1f}"
                soc_end_val = f"{soc_end:.1f}"
                daily_bevs[iso_date].append(sim_grid)

            rows.append([
                iso_date, veh_id, tech, tech, "Class6_Box", duty_id, duty_class,
                f"{dist_km:.1f}", f"{dur_hr:.1f}", str(payload_kg), f"{payload_ratio:.2f}",
                str(stops), str(elev), departure_time, return_time, f"{charging_window_hr:.1f}",
                f"{temp_c:.1f}", weather, f"{precip_mm:.1f}", service_status, str(is_outsourced),
                f"{outsourced_cost:.1f}", diesel_liters_val, sim_trac_val, sim_aux_val,
                sim_batt_val, sim_grid_val, soc_start_val, soc_end_val, charger_id_val,
                charger_kw_val, tier_val, obs_trac_val, telemetry_status, obs_avail_val,
                dispatch_avail, telemetry_avail, report_avail
            ])

    with target_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(ops_headers)
        writer.writerows(rows)
    return daily_bevs, len(rows)

# Generate Primary Benchmark (theta = 1.40)
daily_bevs, n_rows = build_scenario_operations(1.40, EMPIRICAL_DIR / "duty_cycle_operations.csv")
print(f"Wrote primary operations ({n_rows} rows)")

# Generate Matched Counterfactuals (theta = 1.15 and theta = 1.75)
build_scenario_operations(1.15, SCENARIOS_DIR / "duty_cycle_operations_theta_low.csv")
build_scenario_operations(1.75, SCENARIOS_DIR / "duty_cycle_operations_theta_high.csv")
print("Wrote truly matched counterfactual scenarios (theta=1.15 and theta=1.75)")

# 3. Depot Energy Table (Channel 3: Monthly Billing with 30-day reporting delay)
depot_file = EMPIRICAL_DIR / "depot_energy.csv"
depot_headers = [
    "date","depot_id","ambient_temp_c","weather_cond","charger_electricity_kwh",
    "facility_aux_kwh","total_grid_kwh","peak_demand_kw","tariff_id",
    "energy_rate_usd_per_kwh","demand_charge_usd_per_kw","meter_reporting_delay_days","notes"
]
depot_rows = []
for day in master_days:
    iso_date = day["iso_date"]
    temp_c = day["temp_c"]
    weather = day["weather"]
    base_fac = 14.2 + 0.25 * abs(15.0 - temp_c) + random.gauss(0.0, 0.8)
    if weather in ("Snow", "Heavy_Rain"):
        base_fac += 2.5
    facility_kwh = round(max(11.0, min(24.5, base_fac)), 1)
    bev_kwh_list = daily_bevs.get(iso_date, [])
    charger_kwh = round(sum(bev_kwh_list), 1)
    total_grid_kwh = round(charger_kwh + facility_kwh, 1)
    num_chargers = min(2, len(bev_kwh_list))
    peak_kw = round(num_chargers * 48.5 + 6.2 + random.uniform(0.5, 2.5), 1) if num_chargers > 0 else round(8.5 + random.uniform(0.5, 2.0), 1)
    if num_chargers == 0:
        notes = "Baseline facility load only; no active fleet charging"
    elif num_chargers == 1:
        notes = "Single Class 6 BEV overnight dwell charging and facility baseload"
    else:
        notes = "Dual Class 6 BEV concurrent overnight depot charging"
    depot_rows.append([
        iso_date, "Depot_Central", f"{temp_c:.1f}", weather, f"{charger_kwh:.1f}",
        f"{facility_kwh:.1f}", f"{total_grid_kwh:.1f}", f"{peak_kw:.1f}",
        "TOU_COMM_COMED_BES_2025", "0.14", "12.50", "30", notes
    ])

with depot_file.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(depot_headers)
    writer.writerows(depot_rows)
print(f"Wrote {len(depot_rows)} depot records")

# 4. Assets Table
assets_file = EMPIRICAL_DIR / "assets.csv"
assets_headers = [
    "asset_id","asset_type","technology","make_model_specification","depot_id",
    "rated_power_or_engine","usable_energy_capacity_kwh","charger_plugs",
    "order_lead_time_quarters","order_date","delivery_date","commission_date",
    "gross_purchase_cost_usd","incentive_ira_section_usd","net_capital_cost_usd",
    "maintenance_rate_usd_per_km","expected_useful_life_years","salvage_value_fraction",
    "charging_efficiency_eta","operational_status"
]
assets_rows = [
    ["EV-01", "Vehicle", "Battery_Electric", "Freightliner eM2 (200 kWh LFP, 250 kW eAxle)", "Depot_Central", "250 kW Peak", "200.0", "N/A", "2", "2024-05-01", "2024-11-15", "2024-12-15", "220000", "40000", "180000", "0.18", "10", "0.25", "0.88", "Active_From_Day_1"],
    ["EV-02", "Vehicle", "Battery_Electric", "Lion6 Class 6 Electric Box Truck (210 kWh)", "Depot_Central", "250 kW Peak", "200.0", "N/A", "2", "2024-09-10", "2025-03-15", "2025-04-01", "225000", "40000", "185000", "0.18", "10", "0.25", "0.88", "Commissioned_Q2_Day_66"],
    ["EV-03", "Vehicle", "Battery_Electric", "Kenworth K270E (212 kWh High-Density Pack)", "Depot_Central", "265 kW Peak", "200.0", "N/A", "2", "2024-12-05", "2025-06-15", "2025-07-01", "230000", "40000", "190000", "0.18", "10", "0.25", "0.88", "Commissioned_Q3_Day_131"],
    ["D-01", "Vehicle", "Diesel", "Freightliner M2 106 (6.7L Cummins ISB)", "Depot_Central", "240 HP", "N/A", "N/A", "1", "2022-05-01", "2022-08-15", "2022-09-01", "92000", "0", "92000", "0.34", "10", "0.18", "N/A", "Depot_Reserve_Standby"],
    ["D-02", "Vehicle", "Diesel", "Freightliner M2 106 (6.7L Cummins ISB)", "Depot_Central", "240 HP", "N/A", "N/A", "1", "2023-08-15", "2023-11-20", "2023-12-01", "95000", "0", "95000", "0.32", "10", "0.20", "N/A", "Active_Q1_Then_Reserve"],
    ["D-03", "Vehicle", "Diesel", "Freightliner M2 106 (6.7L Cummins ISB)", "Depot_Central", "240 HP", "N/A", "N/A", "1", "2023-08-15", "2023-11-20", "2023-12-01", "95000", "0", "95000", "0.32", "10", "0.20", "N/A", "Active_Q1_Q2_Then_Reserve"],
    ["D-04", "Vehicle", "Diesel", "International MV607 (6.7L Cummins)", "Depot_Central", "250 HP", "N/A", "N/A", "1", "2023-09-01", "2023-12-10", "2024-01-02", "98000", "0", "98000", "0.33", "10", "0.20", "N/A", "Active_Suburban_DUTY_SUB_01"],
    ["D-05", "Vehicle", "Diesel", "International MV607 (6.7L Cummins)", "Depot_Central", "250 HP", "N/A", "N/A", "1", "2023-09-01", "2023-12-10", "2024-01-02", "98000", "0", "98000", "0.33", "10", "0.20", "N/A", "Active_Suburban_DUTY_SUB_02"],
    ["D-06", "Vehicle", "Diesel", "Kenworth T280 (PACCAR PX-7)", "Depot_Central", "240 HP", "N/A", "N/A", "1", "2024-01-10", "2024-04-15", "2024-05-01", "102000", "0", "102000", "0.31", "10", "0.22", "N/A", "Active_Suburban_DUTY_SUB_03"],
    ["D-07", "Vehicle", "Diesel", "Kenworth T280 (PACCAR PX-7)", "Depot_Central", "240 HP", "N/A", "N/A", "1", "2024-01-10", "2024-04-15", "2024-05-01", "102000", "0", "102000", "0.31", "10", "0.22", "N/A", "Active_Suburban_DUTY_SUB_04"],
    ["D-08", "Vehicle", "Diesel", "Freightliner M2 106 (6.7L Cummins ISB)", "Depot_Central", "240 HP", "N/A", "N/A", "1", "2022-05-01", "2022-08-15", "2022-09-01", "92000", "0", "92000", "0.34", "10", "0.18", "N/A", "Active_Regional_DUTY_REG_01"],
    ["D-09", "Vehicle", "Diesel", "International MV607 (6.7L Cummins)", "Depot_Central", "250 HP", "N/A", "N/A", "1", "2022-06-15", "2022-09-20", "2022-10-01", "94000", "0", "94000", "0.34", "10", "0.18", "N/A", "Active_Regional_DUTY_REG_02"],
    ["D-10", "Vehicle", "Diesel", "International MV607 (6.7L Cummins)", "Depot_Central", "250 HP", "N/A", "N/A", "1", "2022-06-15", "2022-09-20", "2022-10-01", "94000", "0", "94000", "0.34", "10", "0.18", "N/A", "Active_Regional_DUTY_REG_03"],
    ["CHG-01", "Charger", "DCFC_CCS1", "ChargePoint Express 250 (50 kW DC)", "Depot_Central", "50 kW", "N/A", "1", "3", "2024-02-15", "2024-10-20", "2024-11-30", "45000", "7500", "37500", "N/A", "10", "0.10", "0.92", "Active"],
    ["CHG-02", "Charger", "DCFC_CCS1", "ABB Terra 54 Dual-Outlet (50 kW DC)", "Depot_Central", "50 kW", "N/A", "2", "3", "2024-06-10", "2025-02-15", "2025-03-25", "48000", "7500", "40500", "N/A", "10", "0.10", "0.92", "Active"],
    ["GRID-01", "Grid_Infrastructure", "Electric_Service", "Medium General Service (250 kW Capacity)", "Depot_Central", "250 kW", "N/A", "N/A", "4", "2023-10-01", "2024-09-01", "2024-10-15", "60000", "0", "60000", "N/A", "25", "0.00", "N/A", "Active"]
]
with assets_file.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(assets_headers)
    writer.writerows(assets_rows)
print(f"Wrote {len(assets_rows)} assets records")

# 5. Monitoring Protocols Table
mon_file = EMPIRICAL_DIR / "monitoring_protocols.csv"
mon_headers = [
    "monitoring_tier","hardware_description","hardware_capex_per_vehicle_usd",
    "quarterly_subscription_usd","sampling_frequency","data_latency",
    "telemetry_channels","sensor_measurement_noise_std_kwh","missingness_rate_pct",
    "reporting_delay_days","bayesian_learning_capability","notes"
]
mon_rows = [
    ["Baseline_Meter", "Utility depot meter and quarterly fuel delivery reconciliation", "0.0", "0.0", "Monthly Billing Interval", "30 to 45 Days Post-Billing", "Total depot gross electricity (kWh) and bulk diesel receipts (L)", "5.0", "0.0", "45", "Confounded; uninformative for duty-level traction regime identification", "Channel 4 corporate accounting reconciliation; cannot isolate individual route efficiency from facility baseline"],
    ["Enhanced_Telemetry", "High-frequency cellular CAN-bus logger and high-precision current shunt", "850.0", "112.50", "1 Hz raw sampling; 1-minute averaged packet transmission", "Real-time cellular transmission (< 5 minutes latency)", "Separated traction kWh, auxiliary HVAC kWh, battery SoC, pack temperature, speed, elevation", "0.5", "1.32", "0", "Direct Bayesian updating of posterior regime distribution p(theta)", "Channel 2 high-frequency telemetry; unlocks empirical value-of-information within 1 quarter"]
]
with mon_file.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(mon_headers)
    writer.writerows(mon_rows)
print(f"Wrote {len(mon_rows)} monitoring protocols")

# 6. Carbon & Policy Inputs
carbon_json = EMPIRICAL_DIR / "carbon_and_policy_inputs.json"
carbon_dict = {
    "provenance": {
        "title": "Calibrated Carbon Policy, Accounting Boundaries, and Agency Emission Factors",
        "scientific_status": "Calibrated Simulation Package (not raw unmanipulated carrier records)",
        "dataset_releases": {
            "diesel_combustion": "EPA GHG Emission Factors Hub (2025 Mobile Combustion Table)",
            "electric_grid": "EPA eGRID2023 (released January 2025)",
            "depot_tariffs": "ComEd Rate BES / OpenEI URDB Rate ID: 5da63f0b5457a3253b2184bb"
        },
        "geography": "Depot Central located in RFC East (RFCE) subregion, United States",
        "temporal_horizon": "12 Quarters (3 Calendar Years: 2025-2027), 260 operating days/year (65 cycles/quarter)"
    },
    "emissions_factors": {
        "diesel_combustion_tonnes_co2_per_liter": 0.002697,
        "diesel_combustion_notes": "10.21 kg CO2/gallon pure combustion (EPA GHG Hub 2025)",
        "electricity_grid_rfce_tonnes_co2e_per_kwh": 0.000271994,
        "electricity_grid_rfce_lb_per_mwh": 599.645,
        "electricity_grid_rfce_notes": "EPA eGRID2023 RFC East subregion output rate",
        "electricity_grid_camx_california_tonnes_co2e_per_kwh": 0.000195037,
        "electricity_grid_camx_lb_per_mwh": 429.983,
        "electricity_grid_camx_notes": "EPA eGRID2023 WECC California subregion output rate (429.983 lb/MWh)",
        "electricity_grid_us_average_tonnes_co2e_per_kwh": 0.000350000,
        "electricity_grid_us_average_lb_per_mwh": 771.618,
        "outsourced_spot_contractor_tonnes_co2_per_bundle": {
            "D1_Urban": 0.065, "D2_Suburban": 0.105, "D3_Regional": 0.160
        }
    },
    "policy_scenarios": {
        "Active_Transition": {
            "annual_emissions_cap_tonnes_co2_per_year": 300.0,
            "cumulative_carbon_budget_3year_tonnes_co2": 836.0,
            "status": "Binding policy target; induces adaptive staged electrification and learning margin"
        },
        "Infeasible_Diagnostic": {
            "annual_emissions_cap_tonnes_co2_per_year": 220.0,
            "cumulative_carbon_budget_3year_tonnes_co2": 800.0,
            "status": "Strictly infeasible benchmark; diesel-only duties alone generate 226.52 t/yr"
        },
        "Unconstrained_Control": {
            "annual_emissions_cap_tonnes_co2_per_year": 400.0,
            "cumulative_carbon_budget_3year_tonnes_co2": 1200.0,
            "status": "Non-binding control benchmark"
        }
    },
    "tax_incentive_assumptions": {
        "ira_section_45w": "Assumes eligibility for 40,000 USD credit for medium-duty commercial clean vehicles acquired during initial transition. Policy sensitivity evaluates scenario without 45W credits."
    }
}
with carbon_json.open("w", encoding="utf-8") as f:
    json.dump(carbon_dict, f, indent=2)

carbon_csv = EMPIRICAL_DIR / "carbon_and_policy_inputs.csv"
carbon_rows = [
    ["parameter_name", "symbol", "value", "unit", "source_reference", "notes"],
    ["diesel_combustion_factor", "g_D", "0.002697", "tonnes_CO2/liter", "EPA GHG Hub 2025 Mobile Combustion", "Direct Scope 1 combustion (10.21 kg/gal)"],
    ["electricity_grid_factor_RFCE", "g_E_RFCE", "0.000272", "tonnes_CO2e/kWh", "EPA eGRID2023 (released 2025)", "RFC East: 599.645 lb/MWh"],
    ["electricity_grid_factor_CAMX", "g_E_CAMX", "0.000195", "tonnes_CO2e/kWh", "EPA eGRID2023 (released 2025)", "WECC California: 429.983 lb/MWh"],
    ["electricity_grid_factor_USavg", "g_E_USavg", "0.000350", "tonnes_CO2e/kWh", "EPA eGRID2023 (released 2025)", "US National Average: 771.618 lb/MWh"],
    ["annual_emissions_cap_active", "C_cap", "300.0", "tonnes_CO2/year", "Active Transition Policy Scenario", "Binding annual threshold"],
    ["cumulative_carbon_budget_active", "b_1", "836.0", "tonnes_CO2", "Active Transition Policy Scenario", "3-year cumulative budget ceiling"],
    ["annual_emissions_cap_infeasible", "C_cap_inf", "220.0", "tonnes_CO2/year", "Infeasible Diagnostic Benchmark", "Below 226.52 t/yr diesel floor"],
    ["cumulative_carbon_budget_infeasible", "b_1_inf", "800.0", "tonnes_CO2", "Infeasible Diagnostic Benchmark", "Diagnostic check"],
    ["annual_emissions_cap_control", "C_cap_ctrl", "400.0", "tonnes_CO2/year", "Unconstrained Control Scenario", "Non-binding upper bound"],
    ["cumulative_carbon_budget_control", "b_1_ctrl", "1200.0", "tonnes_CO2", "Unconstrained Control Scenario", "Non-binding cumulative upper bound"],
    ["outsourced_emissions_factor_D1", "g_O_D1", "0.065", "tonnes_CO2/bundle", "3PL Contractor Model", "Urban spot contractor emissions"],
    ["outsourced_emissions_factor_D2", "g_O_D2", "0.105", "tonnes_CO2/bundle", "3PL Contractor Model", "Suburban spot contractor emissions"],
    ["outsourced_emissions_factor_D3", "g_O_D3", "0.160", "tonnes_CO2/bundle", "3PL Contractor Model", "Regional spot contractor emissions"]
]
with carbon_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(carbon_rows[0])
    writer.writerows(carbon_rows[1:])
print("Successfully generated all tables.")
