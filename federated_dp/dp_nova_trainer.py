"""
Trainer with FedNova optimizer
"""
import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import pandas as pd
from utils.nova_utils import FedNova, SimpleFedNova4Adam
from federated_dp.private_trainer import PrivateFederatedTrainer


class DPNovaTrainer(PrivateFederatedTrainer):
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

        self.tau_effs = []

    def _private_grad_aggregation(self, client_weights):
        self._update_norm_grad()
        tau_eff = sum(self.tau_effs)
        nova_client_weights = [tau_eff * w for w in client_weights]

        self.tau_effs = []

        return super()._private_grad_aggregation(nova_client_weights, noise_scale=float(tau_eff))

    # in-place update of client grads and weights
    def _update_norm_grad(self):
        for i in range(self.client_num):
            for idx in range(len(self.client_grads[i])):
                # skip the track param in BN layers
                scale = 1.0 / self.optimizers[i].local_normalizing_vec
                self.client_grads[i][idx].mul_(scale)

            self.tau_effs.append(self.get_local_tau_eff(self.optimizers[i]))

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
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.args.rounds
                )
                self.schedulers.append(scheduler)

    def reset_nova_optimizer(self, opt):
        opt.local_normalizing_vec = 0
        opt.local_steps = 0

    def split_virtual_client(self, vn, real_train_loaders):
        super_return = super().split_virtual_client(vn, real_train_loaders)

        for idx, opt in enumerate(self.optimizers):
            opt.ratio = torch.FloatTensor([self.client_weights[idx]])

        return super_return
