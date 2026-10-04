import math
import unittest
from dataclasses import replace

from src.prototype_dp import (
    Action,
    INITIAL_BELIEF,
    PrototypeDP,
    load_calibration,
)


class PrototypeDPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = load_calibration()

    def test_fixture_energy_and_masking_validation(self):
        self.assertEqual(self.calibration.parameters["Q"], 200.0)
        self.assertGreater(self.calibration.facility_max_kwh, self.calibration.facility_mean_kwh)

    def test_pipeline_lead_times(self):
        model = PrototypeDP(self.calibration, "control")
        state = model.initial_state()
        self.assertEqual(state.electric_pipeline, (0,))
        self.assertEqual(state.charger_pipeline, (0, 0))

        installed, pipeline = model.advance_pipeline(0, state.electric_pipeline, 1)
        self.assertEqual((installed, pipeline), (0, (1,)))
        installed, pipeline = model.advance_pipeline(installed, pipeline, 0)
        self.assertEqual((installed, pipeline), (1, (0,)))

        installed, pipeline = model.advance_pipeline(0, state.charger_pipeline, 1)
        self.assertEqual((installed, pipeline), (0, (0, 1)))
        installed, pipeline = model.advance_pipeline(installed, pipeline, 0)
        self.assertEqual((installed, pipeline), (0, (1, 0)))
        installed, pipeline = model.advance_pipeline(installed, pipeline, 0)
        self.assertEqual((installed, pipeline), (1, (0, 0)))

    def test_annual_ledgers_reset_after_fourth_quarter(self):
        model = PrototypeDP(self.calibration, "control")
        state = model.initial_state()
        state = state.__class__(
            quarter=4,
            electric_vehicles=state.electric_vehicles,
            chargers=state.chargers,
            electric_pipeline=state.electric_pipeline,
            charger_pipeline=state.charger_pipeline,
            cumulative_carbon=state.cumulative_carbon,
            annual_carbon=state.annual_carbon,
            cumulative_capital=state.cumulative_capital,
            annual_capital=state.annual_capital,
            belief=INITIAL_BELIEF,
            baseline_observed=state.baseline_observed,
            enhanced_observed=state.enhanced_observed,
        )
        next_state = model.transition(
            state,
            action=Action(0, 0, 0),
            posterior=INITIAL_BELIEF,
            carbon_debit=10_000,
            capital_debit=25_000,
        )
        self.assertEqual(next_state.annual_carbon, model.annual_carbon_limit)
        self.assertEqual(next_state.annual_capital, model.annual_capital_limit)
        self.assertEqual(next_state.cumulative_carbon, state.cumulative_carbon - 10_000)
        self.assertEqual(next_state.cumulative_capital, state.cumulative_capital - 25_000)

    def test_infeasible_scenario_is_detected(self):
        model = PrototypeDP(self.calibration, "infeasible")
        self.assertTrue(math.isinf(model.solve()))

    def test_monitoring_option_cannot_worsen_optimum(self):
        with_monitoring = PrototypeDP(self.calibration, "active", allow_monitoring=True)
        without_monitoring = PrototypeDP(self.calibration, "active", allow_monitoring=False)
        value_with = with_monitoring.solve()
        value_without = without_monitoring.solve()
        self.assertLessEqual(value_with, value_without + 1e-6)

    def test_global_hard_compliance_has_no_information_value(self):
        with_monitoring = PrototypeDP(self.calibration, "active", allow_monitoring=True)
        without_monitoring = PrototypeDP(self.calibration, "active", allow_monitoring=False)
        value_with = with_monitoring.solve()
        value_without = without_monitoring.solve()
        self.assertTrue(math.isfinite(value_with))
        self.assertAlmostEqual(value_without, value_with)
        self.assertFalse(any(row["monitored_vehicles"] for row in with_monitoring.most_likely_path()))

    def test_joint_baseline_and_telemetry_create_nine_branches(self):
        model = PrototypeDP(
            self.calibration,
            "active",
            allow_monitoring=True,
            carbon_accounting="posterior_quantile",
            risk_tolerance=0.01,
        )
        state = replace(
            model.initial_state(),
            quarter=6,
            electric_vehicles=1,
            chargers=1,
            pending_baseline=1,
        )
        branches = model.posterior_branches(
            state.belief,
            model.signal_probabilities(state, 1),
        )
        self.assertEqual(len(branches), 9)
        self.assertAlmostEqual(sum(branch[0] for branch in branches), 1.0, places=5)

    def test_global_hard_carbon_is_belief_invariant(self):
        model = PrototypeDP(self.calibration, "active")
        state = model.initial_state()
        state = state.__class__(
            quarter=6,
            electric_vehicles=1,
            chargers=1,
            electric_pipeline=state.electric_pipeline,
            charger_pipeline=state.charger_pipeline,
            cumulative_carbon=state.cumulative_carbon,
            annual_carbon=state.annual_carbon,
            cumulative_capital=state.cumulative_capital,
            annual_capital=state.annual_capital,
            belief=(1.0, 0.0, 0.0),
            baseline_observed=True,
            enhanced_observed=True,
        )
        low_carbon = model.robust_carbon(state)
        high_state = state.__class__(**{**state.__dict__, "belief": (0.0, 0.0, 1.0)})
        high_carbon = model.robust_carbon(high_state)
        self.assertEqual(low_carbon, high_carbon)

    def test_fixed_posterior_quantile_changes_carbon_debit(self):
        model = PrototypeDP(
            self.calibration,
            "active",
            carbon_accounting="posterior_quantile",
            risk_tolerance=0.1,
        )
        state = replace(
            model.initial_state(),
            quarter=6,
            electric_vehicles=1,
            chargers=1,
            belief=(0.95, 0.04, 0.01),
        )
        low_carbon = model.robust_carbon(state)
        high_carbon = model.robust_carbon(replace(state, belief=(0.01, 0.04, 0.95)))
        self.assertLess(low_carbon, high_carbon)

    def test_immediate_scale_schedule_is_enforced(self):
        model = PrototypeDP(
            self.calibration,
            "active",
            allow_monitoring=False,
            strategy="immediate_scale",
        )
        state = model.initial_state()
        self.assertEqual(tuple(model.actions(state)), (Action(0, 2, 0),))
        next_state = model.transition(
            state,
            Action(0, 2, 0),
            state.belief,
            model.robust_carbon(state),
            model.investment_cost(Action(0, 2, 0)),
        )
        self.assertEqual(tuple(model.actions(next_state)), (Action(2, 0, 0),))

    def test_matched_ablation_pays_for_but_suppresses_signal(self):
        model = PrototypeDP(
            self.calibration,
            "active",
            allow_monitoring=True,
            strategy="matched_ablation",
            suppress_enhanced_signal=True,
        )
        state = replace(
            model.initial_state(),
            quarter=6,
            electric_vehicles=1,
            chargers=1,
            baseline_observed=True,
        )
        self.assertEqual(tuple(model.actions(state)), (Action(0, 0, 1),))
        self.assertIsNone(model.signal_probabilities(state, 1))

    def test_pending_baseline_delivered_when_enhanced_suppressed(self):
        model_ablation = PrototypeDP(
            self.calibration,
            "active",
            allow_monitoring=True,
            strategy="matched_ablation",
            suppress_enhanced_signal=True,
        )
        model_base = PrototypeDP(
            self.calibration,
            "active",
            allow_monitoring=False,
            strategy="flexible",
        )
        state_with_pending = replace(
            model_ablation.initial_state(),
            quarter=6,
            electric_vehicles=1,
            chargers=1,
            baseline_observed=False,
            pending_baseline=1,
        )
        base_signal = model_base.signal_probabilities(state_with_pending, 0)
        self.assertIsNotNone(base_signal)
        # Matched ablation suppresses enhanced signal but must deliver pending baseline
        ablation_signal = model_ablation.signal_probabilities(state_with_pending, 1)
        self.assertIsNotNone(ablation_signal)
        self.assertEqual(base_signal, ablation_signal)

    def test_pending_baseline_marked_delivered_on_transition(self):
        model_ablation = PrototypeDP(
            self.calibration,
            "active",
            allow_monitoring=True,
            strategy="matched_ablation",
            suppress_enhanced_signal=True,
        )
        state_with_pending = replace(
            model_ablation.initial_state(),
            quarter=6,
            electric_vehicles=1,
            chargers=1,
            baseline_observed=False,
            pending_baseline=1,
        )
        # Monitoring action with enhanced signal suppressed
        next_monitored = model_ablation.transition(
            state_with_pending,
            Action(0, 0, 1),
            state_with_pending.belief,
            model_ablation.robust_carbon(state_with_pending),
            model_ablation.investment_cost(Action(0, 0, 1)),
        )
        self.assertTrue(next_monitored.baseline_observed)
        self.assertEqual(next_monitored.pending_baseline, 0)
        self.assertTrue(next_monitored.enhanced_observed)

        # Unmonitored action
        next_unmonitored = model_ablation.transition(
            state_with_pending,
            Action(0, 0, 0),
            state_with_pending.belief,
            model_ablation.robust_carbon(state_with_pending),
            0,
        )
        self.assertTrue(next_unmonitored.baseline_observed)
        self.assertEqual(next_unmonitored.pending_baseline, 0)

    def test_monitoring_retains_due_baseline_likelihood(self):
        model = PrototypeDP(
            self.calibration,
            "active",
            allow_monitoring=True,
            carbon_accounting="posterior_quantile",
            risk_tolerance=0.01,
        )
        state = replace(
            model.initial_state(),
            quarter=6,
            electric_vehicles=1,
            chargers=1,
            baseline_observed=False,
            pending_baseline=1,
        )
        baseline = model.signal_probabilities(state, 0)
        enhanced = model.channel_signal_parameters(state, "enhanced", 1)
        joint = model.signal_probabilities(state, 1)
        self.assertIsNotNone(baseline)
        self.assertIsNotNone(enhanced)
        self.assertEqual(len(joint[0]), 9)
        enhanced_probabilities = model.signal_categories
        self.assertAlmostEqual(
            sum(joint[0][:enhanced_probabilities]),
            baseline[0][0],
        )

    def test_delayed_baseline_retains_exposure_from_generating_quarter(self):
        model = PrototypeDP(self.calibration, "active", allow_monitoring=False)
        # 2 electric duties running currently, but only 1 duty generated the pending baseline
        state_changed_duties = replace(
            model.initial_state(),
            quarter=6,
            electric_vehicles=2,
            chargers=2,
            baseline_observed=False,
            pending_baseline=1,
        )
        # Reference state where 1 duty is currently running with pending_baseline=1
        state_reference_1duty = replace(
            model.initial_state(),
            quarter=6,
            electric_vehicles=1,
            chargers=1,
            baseline_observed=False,
            pending_baseline=1,
        )
        # State where 2 duties generated the pending baseline
        state_pending_2duties = replace(
            model.initial_state(),
            quarter=6,
            electric_vehicles=2,
            chargers=2,
            baseline_observed=False,
            pending_baseline=2,
        )
        sig_changed = model.signal_probabilities(state_changed_duties, 0)
        sig_ref1 = model.signal_probabilities(state_reference_1duty, 0)
        sig_pend2 = model.signal_probabilities(state_pending_2duties, 0)

        self.assertEqual(sig_changed, sig_ref1)
        self.assertNotEqual(sig_changed, sig_pend2)

    def test_enhanced_monitoring_is_a_one_time_review(self):
        model = PrototypeDP(self.calibration, "active", allow_monitoring=True)
        state = replace(
            model.initial_state(),
            quarter=7,
            electric_vehicles=1,
            chargers=1,
            baseline_observed=True,
            enhanced_observed=True,
        )
        self.assertTrue(all(action.monitored_vehicles == 0 for action in model.actions(state)))


if __name__ == "__main__":
    unittest.main()