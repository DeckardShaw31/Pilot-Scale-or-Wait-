import unittest
from dataclasses import replace

import numpy as np

from src.audit_posterior import audit_configuration
from src.prototype_dp import Action, PrototypeDP, categorical_normal_probabilities, load_empirical_calibration, normalized_belief
from src.validate_study import generate_paths, mean_interval, rate_interval, replay, thresholds


class ValidationStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = load_empirical_calibration()

    def test_refined_signal_normalization(self):
        means = (97.75, 119.0, 148.75)
        for categories in (3, 5, 7):
            probabilities = categorical_normal_probabilities(means, 10.0, categories)
            self.assertEqual(len(probabilities), 3)
            self.assertTrue(all(len(row) == categories for row in probabilities))
            np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0)
            self.assertEqual(len(thresholds(means, categories)), categories - 1)

    def test_belief_precision_is_model_specific(self):
        parameters = dict(self.calibration.parameters, belief_digits=5, signal_categories=5)
        model = PrototypeDP(replace(self.calibration, parameters=parameters), "active")
        self.assertEqual(model.belief_digits, 5)
        self.assertEqual(model.initial_belief, normalized_belief((1 / 3, 1 / 3, 1 / 3), 5))
        state = replace(model.initial_state(), electric_vehicles=1, chargers=1)
        self.assertEqual(len(model.signal_probabilities(state, 1)[0]), 5)

    def test_common_paths_are_reproducible_and_immutable(self):
        first = generate_paths(self.calibration, 5, 7)
        second = generate_paths(self.calibration, 5, 7)
        for name, values in vars(first).items():
            np.testing.assert_array_equal(values, getattr(second, name))
            self.assertFalse(values.flags.writeable)
        self.assertFalse(np.array_equal(first.traction, generate_paths(self.calibration, 5, 8).traction))

    def test_intervals_include_sampling_uncertainty(self):
        mean, lower, upper = mean_interval(np.asarray([10.0, 20.0, 30.0]))
        self.assertEqual(mean, 20.0)
        self.assertLess(lower, mean)
        self.assertGreater(upper, mean)
        rate, lower, upper = rate_interval(np.zeros(100, dtype=bool))
        self.assertEqual(rate, 0.0)
        self.assertGreater(upper, 0.0)
        self.assertAlmostEqual(lower, 0.0)

    def test_probability_audit_deduplicates_likelihood_pairs(self):
        model = PrototypeDP(self.calibration, "active")
        first = replace(model.initial_state(), quarter=3, electric_vehicles=1, chargers=1, pending_baseline=1)
        second = replace(first, quarter=4)
        model.policy[first] = Action(0, 0, 0)
        model.policy[second] = Action(0, 0, 0)
        result = audit_configuration(model)
        self.assertEqual(result["cached_policy_states"], 2)
        self.assertEqual(result["distinct_belief_likelihood_pairs"], 1)
        self.assertGreaterEqual(result["max_posterior_component_error"], 0.0)
        self.assertLess(result["max_omitted_predictive_category_mass"], 1e-8)

    def test_fixed_policy_replay_is_complete_and_reproducible(self):
        model = PrototypeDP(self.calibration, "active", strategy="immediate_scale", allow_monitoring=False)
        model.solve()
        paths = generate_paths(self.calibration, 5, 10)
        first = replay(model, paths)
        second = replay(model, paths)
        self.assertTrue(first.equals(second))
        self.assertFalse(first.policy_failure.any())
        self.assertTrue(np.isfinite(first.cost_usd).all())
        self.assertFalse(first.monitoring_selected.any())
        self.assertTrue(first.maximum_electric_vehicles.eq(2).all())


if __name__ == "__main__":
    unittest.main()