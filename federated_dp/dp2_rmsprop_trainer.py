"""
DP2-RMSProp algorithm (ICLR 2023)
"""
import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import copy
import random
import math
import logging
import pandas as pd
from federated_dp.private_trainer import PrivateFederatedTrainer


class DP2RMSPropTrainer(PrivateFederatedTrainer):
    def __init__(
        self,
        args,
        logging,
        device,
        server_model,
        train_sites,
        val_sites,
        client_weights=None,
        **kwargs
    ) -> None:
        super().__init__(
            args, logging, device, server_model, train_sites, val_sites, client_weights, **kwargs
        )

        assert self.sample_rate == 1, "assume all clients join the training"

        self.interval = args.dp2_interval

        self.Gt = None
        self.At = None

        self.init_At = False

        self.rmsprop_gamma = 0.9
        self.rmsprop_epsilon = 1e-7
        self.rmsprop_lr = self.args.rmsprop_lr
        self.count = 0

    def _private_grad_aggregation(self, client_weights, noise_scale=1):
        branch = self.branch_func(self.interval)

        if branch == 2:
            for i in range(len(self.client_grads[0])):
                denom = torch.sqrt(self.At[i]).add(self.rmsprop_epsilon)
                for idx_client in range(self.client_num):
                    self.client_grads[idx_client][i].div_(denom)

        if self.Gt is None:
            self.Gt = [torch.zeros_like(grad_term) for grad_term in self.client_grads[0]]

        aggregated_grads = super()._private_grad_aggregation(client_weights, noise_scale)

        if branch <= 1:
            for i in range(len(aggregated_grads)):
                self.Gt[i].add_(aggregated_grads[i])
            self.count += 1

        if branch == 1:
            Gt_avg_sq = copy.deepcopy(self.Gt)
            for i in range(len(aggregated_grads)):
                Gt_avg_sq[i] = torch.pow(Gt_avg_sq[i].div(self.count), 2)
                if self.init_At:
                    self.At[i].mul_(self.rmsprop_gamma).add_(
                        Gt_avg_sq[i], alpha=1 - self.rmsprop_gamma
                    )

            if not self.init_At:
                self.At = copy.deepcopy(Gt_avg_sq)
                self.init_At = True

            self.Gt = [torch.zeros_like(grad_term) for grad_term in self.client_grads[0]]
            self.count = 0

        if branch == 2:
            for i in range(len(aggregated_grads)):
                aggregated_grads[i].mul_(self.rmsprop_lr)

            self.client_grad_norms = []
            for i in range(self.client_num):
                client_grad_norm, _, _ = self._compute_grad_norm(
                    copy.deepcopy(self.client_grads[i]), self.server_model.state_dict().keys()
                )
                self.client_grad_norms.append(client_grad_norm)

            self.client_grad_norms_sum[-1] = (
                torch.asarray(self.client_grad_norms).sum().item() * noise_scale
            )

        return aggregated_grads

    def branch_func(self, interval):
        i = self.cur_iter

        s1 = interval
        s2 = interval
        on_interval = (i + 1 + s2) % (s1 + s2) == 0
        use_adaptive = (i // interval) % (1 + 1) > 0
        return 2 if use_adaptive else (1 if on_interval else 0)
