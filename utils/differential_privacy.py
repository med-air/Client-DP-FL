# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Library for computing privacy values for DP-SGD."""

import math
import dp_accounting
from opacus.accountants.rdp import RDPAccountant
from opacus.accountants.prv import PRVAccountant
from opacus.accountants.gdp import GaussianAccountant
from opacus.accountants.utils import get_noise_multiplier


r"""
Below are TensorFlow style, only support RDP
"""


def compute_dp_sgd_privacy(q, noise_multiplier, steps, delta=1e-9):
    """Compute epsilon based on the given hyperparameters.

    Args:
      q: probability of each client is selected.
      noise_multiplier: Noise multiplier used in training.
      steps: Local epochs * training rounds.
      delta: Value of delta for which to compute epsilon.

    Returns:
      Value of epsilon corresponding to input hyperparameters.
    """
    orders = (
        [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
        + list(range(5, 64))
        + [128, 256, 512]
    )

    accountant = dp_accounting.rdp.RdpAccountant(orders)

    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(q, dp_accounting.GaussianDpEvent(noise_multiplier)),
        steps,
    )

    accountant.compose(event)

    eps, opt_order = accountant.get_epsilon_and_optimal_order(delta)

    return eps, opt_order


def compute_noise(q, target_epsilon, steps, delta, noise_lbd=1e-5):
    """Compute noise based on the given hyperparameters."""
    orders = (
        [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
        + list(range(5, 64))
        + [128, 256, 512]
    )

    def make_event_from_noise(sigma):
        return dp_accounting.SelfComposedDpEvent(
            dp_accounting.PoissonSampledDpEvent(q, dp_accounting.GaussianDpEvent(sigma)), steps
        )

    def make_accountant():
        return dp_accounting.rdp.RdpAccountant(orders)

    accountant = make_accountant()
    accountant.compose(make_event_from_noise(noise_lbd))
    init_epsilon = accountant.get_epsilon(delta)

    if init_epsilon < target_epsilon:  # noise_lbd was an overestimate
        print("noise_lbd too large for target epsilon.")
        return 0

    target_noise = dp_accounting.calibrate_dp_mechanism(
        make_accountant,
        make_event_from_noise,
        target_epsilon,
        delta,
        dp_accounting.LowerEndpointAndGuess(noise_lbd, noise_lbd * 2),
    )

    return target_noise


r"""
Below are PyTorch opacus stype, support accountant:
rdp: Renyi DP Accountant
gdp: Gaussian DP Accountant
prv: Privacy loss Random Variables (PRVs) Accountant [1]

[1] https://arxiv.org/abs/2106.02848
"""


def create_accountant(mechanism: str):
    if mechanism == "rdp":
        return RDPAccountant()
    elif mechanism == "gdp":
        return GaussianAccountant()
    elif mechanism == "prv":
        return PRVAccountant()

    raise ValueError(f"Unexpected accounting mechanism: {mechanism}")


def get_epsilon(steps, noise_multiplier, sample_rate, delta, mechanism: str = "prv", **kwargs):
    accountant = create_accountant(mechanism=mechanism)
    accountant.history = [(noise_multiplier, sample_rate, steps)]

    return accountant.get_epsilon(delta=delta, **kwargs)


def get_noise_multiplier_from_epsilon(
    epsilon, steps, sample_rate, delta, mechanism: str = "prv", **kwargs
):
    return get_noise_multiplier(
        target_epsilon=epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        steps=steps,
        accountant=mechanism,
        **kwargs,
    )

