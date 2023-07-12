"""
FedAvg Trainer with DP
"""
import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import copy
import random
import math
from .base_trainer import BaseFederatedTrainer
from utils.differential_privacy import get_epsilon, get_noise_multiplier_from_epsilon


class PrivateFederatedTrainer(BaseFederatedTrainer):
    def __init__(
        self,
        args,
        logging,
        device,
        server_model,
        train_sites,
        val_sites,
        client_weights=None,
        **kwargs,
    ):
        super().__init__(
            args, logging, device, server_model, train_sites, val_sites, client_weights, **kwargs
        )

        self.client_grad_norms = []
        self.client_grad_norms_sum = []
        self.aggregated_grad_norm = []
        self.noise_norm = []

        """DP setting"""
        self.delta = self.args.delta  # DP budget
        self.S = self.args.S

        if self.args.dp_mode == "overhead":
            if args.epsilon is not None:
                self.epsilon = self.args.epsilon  # DP budget
                self.noise_multiplier = get_noise_multiplier_from_epsilon(
                    epsilon=self.epsilon,
                    steps=self.args.rounds,
                    sample_rate=self.sample_rate,
                    delta=self.delta,
                    mechanism=self.args.accountant,
                )
            else:
                self.noise_multiplier = self.args.noise_multiplier
                self.epsilon = get_epsilon(
                    steps=self.args.rounds,
                    noise_multiplier=self.noise_multiplier,
                    sample_rate=self.sample_rate,
                    delta=self.delta,
                    mechanism=self.args.accountant,
                )

            self.epsilon_cur = self.epsilon
            self.sigma_averaged = (
                self.noise_multiplier * self.S / (1 * self.aggregation_client_num)
            )

            self.logging.info(
                f"Using noise multiplier={self.noise_multiplier} and sigma on aggregation={self.sigma_averaged} to satisfy ({self.epsilon},{self.delta})-DP."
            )
        elif self.args.dp_mode == "bounded":
            assert (
                args.epsilon is not None and args.noise_multiplier is not None
            ), "To use bounded dp mode, noise_multiplier and target epsilon must both be specified."
            self.sigma_averaged = 0
            self.epsilon_cur = 0
            self.logging.info(
                f"Bounding ({self.epsilon},{self.delta})-DP with noise multiplier={self.noise_multiplier}."
            )
        else:
            raise NotImplementedError

        if self.args.adaclip:
            self.logging.info(f"Using adaptive clipping.")
        self.logging.info(f"Using {self.args.accountant} accountant.")

    # Override: communication with private communication
    def communication_grad(self, server_model, models, client_weights):
        with torch.no_grad():
            aggregated_grads = self._private_grad_aggregation(client_weights)

            assert len(server_model.state_dict().keys()) == len(aggregated_grads)
            for idx, key in enumerate(server_model.state_dict().keys()):
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if "num_batches_tracked" in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    server_model.state_dict()[key].data.add_(aggregated_grads[idx])

                    # distribute back to clients
                    if self.args.ema > 0.0:
                        for client_idx in range(len(client_weights)):
                            models[client_idx].state_dict()[key].data.mul_(self.args.ema).add_(
                                (1 - self.args.ema) * server_model.state_dict()[key]
                            )
            """
            This optimize effiency of communication, but assign aggregated num_batches_tracked to
            every local model. However, as momentum is set to 0.1 in BN layers, num_batches_tracked
            does not actually work during both training and inference.
            """
            if not self.args.ema > 0.0:
                for client_idx in range(len(client_weights)):
                    models[client_idx].load_state_dict(server_model.state_dict())

        return server_model, models

    def _private_grad_aggregation(self, client_weights, noise_scale=1.0):
        # clipping gradients
        self.client_grad_norms = []
        for i in self.aggregation_idxs:
            client_grad_norm, _, _ = self._compute_grad_norm(
                copy.deepcopy(self.client_grads[i]), self.server_model.state_dict().keys()
            )
            self.client_grad_norms.append(client_grad_norm)

        self.client_grad_norms_sum.append(
            torch.asarray(self.client_grad_norms).sum().item() * noise_scale
        )

        if self.args.adaclip:
            max_clip_norm = float(torch.median(torch.asarray(self.client_grad_norms)))
            max_clip_norm = min(max_clip_norm, 25.0)  # this is important for dp2
            self.sigma_averaged = (
                self.noise_multiplier
                * max_clip_norm
                * noise_scale
                / (1 * self.aggregation_client_num)
            )
            self.logging.info(
                f"Update sigma on aggregation={self.sigma_averaged}; clip bound update={max_clip_norm * noise_scale}"
            )
        else:
            if self.args.noclip:
                max_clip_norm = 1e8
            else:
                max_clip_norm = self.S

        if self.args.dp_mode == "bounded":
            self.epsilon_cur = get_epsilon(
                steps=self.cur_iter + 1,
                noise_multiplier=self.noise_multiplier,
                sample_rate=self.sample_rate,
                delta=self.delta,
                mechanism=self.args.accountant,
            )
            self.logging.info(f"Current epsilon on epoch {self.cur_iter} is {self.epsilon_cur}.")

            if self.epsilon_cur > self.epsilon:
                raise Exception(f"DP budget exceed upper bound {self.epsilon}.")

        aggregated_grads = [torch.zeros_like(grad_term) for grad_term in self.client_grads[0]]

        for i, client_idx in enumerate(self.aggregation_idxs):
            """clip client updates (gradients)"""
            self.client_grads[client_idx] = self._global_l2_clip_with_norm(
                max_clip_norm,
                self.client_grads[client_idx],
                keys=self.server_model.state_dict().keys(),
                grad_norm=self.client_grad_norms[i],
            )

            for idx in range(len(aggregated_grads)):
                aggregated_grads[idx] = (
                    aggregated_grads[idx]
                    + self.client_grads[client_idx][idx] * client_weights[client_idx]
                )

        agg_grad_norm, _, _ = self._compute_grad_norm(
            aggregated_grads, self.server_model.state_dict().keys()
        )
        self.aggregated_grad_norm.append(agg_grad_norm.numpy())

        noise_list = []
        for idx, key in enumerate(self.server_model.state_dict().keys()):
            if (
                "running_mean" not in key
                and "running_var" not in key
                and "num_batches_tracked" not in key
            ):
                gaussian_noise = self._generate_noise(
                    std=self.sigma_averaged,
                    reference=self.server_model.state_dict()[key].data,
                )
                noise_list.append(gaussian_noise)
                """ dp-sgd add noise"""
                aggregated_grads[idx] = aggregated_grads[idx] + gaussian_noise
            else:
                noise_list.append(torch.zeros_like(self.server_model.state_dict()[key].data))

        noise_norm_, _, _ = self._compute_grad_norm(
            noise_list, self.server_model.state_dict().keys()
        )
        self.noise_norm.append(noise_norm_.numpy())

        return aggregated_grads

    def select_split_num(self):
        assert self.sample_rate == 1, "Only support sample rate 1 at this moment"
        xi = self.noise_norm[-1] / self.aggregated_grad_norm[-1]
        phi = self.client_grad_norms_sum[-1] / self.aggregated_grad_norm[-1]

        ratio = xi / phi
        vn = pow(ratio * self.args.clients, 1.0 / 2)
        if vn < 1:
            vn = math.floor(self.virtual_clients * vn)
        else:
            if self.args.ada_stable:
                vn = math.floor(vn) * self.virtual_clients
            elif self.args.ada_prog:
                vn = (
                    math.floor((vn * self.virtual_clients - self.virtual_clients) / 2.0)
                    + self.virtual_clients
                )
            else:
                vn = math.floor(vn * self.virtual_clients)

        # note that this should return an value of N / real_client_num
        return vn

    def _compute_grad_norm(self, gradients, keys):
        grad_norm = []
        # skip the num_batches_tracked/running_mean/var parameter
        grad_mins = []
        grad_maxs = []
        for idx, key in enumerate(keys):
            if "num_batches_tracked" in key or "running_mean" in key or "running_var" in key:
                continue
            else:
                grad_norm.append(torch.norm(gradients[idx].reshape(-1), p=2))
                grad_mins.append(torch.min(gradients[idx]))
                grad_maxs.append(torch.max(gradients[idx]))

        grad_norm = torch.asarray(grad_norm)
        global_norm = torch.norm(grad_norm, p=2)

        return global_norm, min(grad_mins), max(grad_maxs)

    def _global_l2_clip(self, clip_bound, gradients, keys):
        grad_norm = []
        # skip the num_batches_tracked/running_mean/var parameter
        for idx, key in enumerate(keys):
            if "num_batches_tracked" in key or "running_mean" in key or "running_var" in key:
                continue
            else:
                grad_norm.append(torch.norm(gradients[idx].reshape(-1), p=2))

        grad_norm = torch.asarray(grad_norm)
        global_norm = torch.norm(grad_norm, p=2)
        print("global norm: ", global_norm)
        norm_factor = torch.minimum(clip_bound / (global_norm + 1e-15), torch.tensor(1.0))
        for idx, key in enumerate(keys):
            if "num_batches_tracked" in key or "running_mean" in key or "running_var" in key:
                continue
            else:
                gradients[idx] = gradients[idx] * norm_factor

        return gradients

    def _global_l2_clip_with_norm(self, clip_bound, gradients, keys, grad_norm):
        norm_factor = torch.minimum(clip_bound / (grad_norm + 1e-15), torch.tensor(1.0))
        for idx, key in enumerate(keys):
            if "num_batches_tracked" in key or "running_mean" in key or "running_var" in key:
                continue
            else:
                gradients[idx] = gradients[idx] * norm_factor

        return gradients

    def _generate_noise(
        self,
        std: float,
        reference: torch.Tensor,
    ) -> torch.Tensor:

        zeros = torch.zeros(reference.shape, device=reference.device)
        if std == 0:
            return zeros

        return torch.normal(
            mean=0,
            std=std,
            size=reference.shape,
            device=reference.device,
        )
