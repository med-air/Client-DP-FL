"""
FedNova
"""
import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import torch.nn as nn

from utils.nova_utils import FedNova, SimpleFedNova4Adam
from federated_dp.fedavg_trainer import FedAvgTrainer

class FedNovaTrainer(FedAvgTrainer):
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

        self.tau_effs = [0 for _ in range(self.client_num)]

    def communication_grad(self, server_model, models, client_weights):
        self._update_norm_grad()
        tau_eff = sum(self.tau_effs)
        nova_client_weights = [tau_eff for w in client_weights]
        # real_client_weights = [tau_eff for _ in range(len(client_weights))]
        return super().communication_grad(server_model, models, nova_client_weights)

    # in-place update of client grads and weights
    def _update_norm_grad(self, weight=0):
        for i in range(self.client_num):
            if weight == 0:
                weight = self.optimizers[i].ratio
            for idx in range(len(self.client_grads[i])):
                # skip the track param in BN layers
                scale = 1.0 / self.optimizers[i].local_normalizing_vec
                self.client_grads[i][idx].mul_(weight * scale)

            self.tau_effs[i] = self.get_local_tau_eff(self.optimizers[i])

            self.reset_nova_optimizer(self.optimizers[i])

    def get_local_tau_eff(self, opt):
        if opt.mu != 0:
            return opt.local_steps * opt.ratio
        else:
            return opt.local_normalizing_vec * opt.ratio

    def init_optims(self):
        self.optimizers = []
        self.schedulers = []

        for idx in range(self.client_num):
            optimizer = SimpleFedNova4Adam(
                self.client_models[idx].parameters(),
                lr=self.args.lr,
                ratio=torch.FloatTensor([self.client_weights[idx]]),
                amsgrad=True,
            )
            self.optimizers.append(optimizer)
            if self.lr_decay:
                # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = self.args.lr_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.args.rounds
                )
                self.schedulers.append(scheduler)

    def reset_nova_optimizer(self, opt):
        opt.local_normalizing_vec = 0
        opt.local_steps = 0
