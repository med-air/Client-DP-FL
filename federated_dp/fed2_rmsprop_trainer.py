"""
Federated RMSProp trainer for federated learning with DP
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
from utils.loss import DiceLoss
from utils.util import _connectivity_region_analysis, _eval_haus
from federated_dp.base_trainer import BaseFederatedTrainer


class Fed2RMSPropTrainer(BaseFederatedTrainer):
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

    def communication_grad(self, server_model, models, client_weights):
        branch = self.branch_func(self.interval)

        with torch.no_grad():
            if branch == 2:
                for i in range(len(self.client_grads[0])):
                    denom = torch.sqrt(self.At[i]).add(self.rmsprop_epsilon)
                    for idx_client in range(self.client_num):
                        self.client_grads[idx_client][i].div_(denom)
                lr = self.rmsprop_lr

            aggregated_grads = [torch.zeros_like(grad_term) for grad_term in self.client_grads[0]]

            for i in range(self.client_num):
                for idx in range(len(aggregated_grads)):
                    aggregated_grads[idx] = (
                        aggregated_grads[idx] + self.client_grads[i][idx] * client_weights[i]
                    )

            if self.Gt is None:
                self.Gt = [torch.zeros_like(grad_term) for grad_term in self.client_grads[0]]

            if branch <= 1:
                for i in range(len(aggregated_grads)):
                    self.Gt[i].add_(aggregated_grads[i])
                self.count += 1
                lr = 1

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

            assert len(server_model.state_dict().keys()) == len(aggregated_grads)
            for idx, key in enumerate(server_model.state_dict().keys()):
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if "num_batches_tracked" in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    server_model.state_dict()[key].data.add_(aggregated_grads[idx], alpha=lr)
                    # distribute back to clients
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(
                            server_model.state_dict()[key]
                        )

        return server_model, models

    def branch_func(self, interval):
        i = self.cur_iter
        # if ("dp" in self.args.mode) and (self.args.ada_vn or self.args.init_vn):
        #     i -= 1

        s1 = interval
        s2 = interval
        on_interval = (i + 1 + s2) % (s1 + s2) == 0
        use_adaptive = (i // interval) % (1 + 1) > 0
        return 2 if use_adaptive else (1 if on_interval else 0)
