"""
FedAdam 
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
from federated_dp.base_trainer import BaseFederatedTrainer


class FedAdamTrainer(BaseFederatedTrainer):
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

        assert self.sample_rate == 1, "Only support sample rate 1 at this moment"

        self.mt = None
        self.vt = None
        # momentum param for Adam optim
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.adam_tau = 1e-9

        self.adam_lr = args.adam_lr

    def communication_grad(self, server_model, models, client_weights):
        with torch.no_grad():
            aggregated_grads = [torch.zeros_like(grad_term) for grad_term in self.client_grads[0]]
            for i in range(self.client_num):
                for idx in range(len(aggregated_grads)):
                    aggregated_grads[idx] = (
                        aggregated_grads[idx] + self.client_grads[i][idx] * client_weights[i]
                    )

            if self.mt is None:
                self.mt = [torch.zeros_like(grad_term) for grad_term in self.client_grads[0]]

            if self.vt is None:
                self.vt = [torch.zeros_like(grad_term) for grad_term in self.client_grads[0]]

            for idx, key in enumerate(server_model.state_dict().keys()):
                assert self.mt[idx].shape == aggregated_grads[idx].shape
                if "num_batches_tracked" in key or "running_mean" in key or "running_var" in key:
                    continue

                self.mt[idx].mul_(self.beta1).add_(aggregated_grads[idx], alpha=1 - self.beta1)
                self.vt[idx].mul_(self.beta2).add_(
                    torch.pow(aggregated_grads[idx], 2), alpha=1 - self.beta2
                )

                mt_h = self.mt[idx]
                denom = torch.sqrt(self.vt[idx]) + self.adam_tau

                aggregated_grads[idx].copy_(mt_h.mul(self.adam_lr).div(denom))

            assert len(server_model.state_dict().keys()) == len(aggregated_grads)
            for idx, key in enumerate(server_model.state_dict().keys()):
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

            if not self.args.ema > 0.0:
                for client_idx in range(len(client_weights)):
                    models[client_idx].load_state_dict(server_model.state_dict())

        return server_model, models
