"""Audit probability truncation for solved finite-belief configurations."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prototype_dp import PrototypeDP, load_empirical_calibration


def audit_configuration(model: PrototypeDP) -> dict:
    conditional_cache = {}
    largest_omitted_branch_mass = 0.0
    largest_removed_regime_mass = 0.0
    largest_posterior_error = 0.0
    support_changes = 0
    for state, action in model.policy.items():
        signal = model.signal_parameters(state, action.monitored_vehicles)
        if signal is None:
            continue
        key = (state.belief, signal)
        if key in conditional_cache:
            continue
        conditional = model.signal_probabilities(state, action.monitored_vehicles)
        branches = model.posterior_branches(state.belief, conditional)
        conditional_cache[key] = True
        retained = {category: posterior for _, posterior, category in branches}
        omitted_mass = 0.0
        for category in range(len(conditional[0])):
            weights = [
                state.belief[index] * conditional[index][category]
                for index in range(len(state.belief))
            ]
            predictive = sum(weights)
            if predictive <= 0:
                continue
            if category not in retained:
                omitted_mass += predictive
                continue
            exact = [weight / predictive for weight in weights]
            rounded = retained[category]
            removed = sum(
                probability for index, probability in enumerate(exact)
                if rounded[index] <= model.support_tolerance
                and probability > model.support_tolerance
            )
            support_changes += int(removed > 0)
            largest_removed_regime_mass = max(largest_removed_regime_mass, removed)
            largest_posterior_error = max(
                largest_posterior_error,
                max(abs(probability - rounded[index]) for index, probability in enumerate(exact)),
            )
        largest_omitted_branch_mass = max(largest_omitted_branch_mass, omitted_mass)
    return {
        'cached_policy_states': len(model.policy),
        'distinct_belief_likelihood_pairs': len(conditional_cache),
        'max_omitted_predictive_category_mass': largest_omitted_branch_mass,
        'max_regime_mass_removed_by_quantization_or_tolerance': largest_removed_regime_mass,
        'max_posterior_component_error': largest_posterior_error,
        'posterior_categories_with_support_changes': support_changes,
    }


def main() -> None:
    calibration = load_empirical_calibration()
    configurations = (
        ('Base', {}),
        ('Belief precision 5', {'belief_digits': 5}),
        ('Seven signal categories', {'signal_categories': 7}),
        ('Support tolerance 0.001', {'support_tolerance': 0.001}),
    )
    findings = []
    for name, updates in configurations:
        parameters = dict(calibration.parameters)
        parameters.update(updates)
        model = PrototypeDP(replace(calibration, parameters=parameters), 'active')
        objective = model.solve()
        if not math.isfinite(objective):
            raise RuntimeError(f'Audit configuration is infeasible: {name}')
        result = {'case': name, 'objective_usd': objective, **audit_configuration(model)}
        findings.append(result)
        print(name, result, flush=True)
    sources = ('src/audit_posterior.py', 'src/prototype_dp.py')
    payload = {
        'scope': 'Maxima over distinct belief/likelihood pairs in cached subproblem policies, not path-weighted error bounds. Computed Gaussian category probabilities are subject to floating-point underflow. Positive unrounded Gaussian likelihoods do not eliminate latent regimes; this audit does not certify exact-support hard-carbon compliance.',
        'source_sha256': {
            source: hashlib.sha256(Path(source).read_bytes()).hexdigest() for source in sources
        },
        'configurations': findings,
    }
    Path('results/posterior_truncation_audit.json').write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()