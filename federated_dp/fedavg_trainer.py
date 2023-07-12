"""
FedAvg Trainer
"""
import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import torch.nn as nn
from federated_dp.base_trainer import BaseFederatedTrainer


class FedAvgTrainer(BaseFederatedTrainer):
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

    def communication_grad(self, server_model, models, client_weights):
        with torch.no_grad():
            aggregated_grads = [torch.zeros_like(grad_term) for grad_term in self.client_grads[0]]
            # print(len(aggregated_grads))
            for i in range(self.client_num):
                for idx in range(len(aggregated_grads)):
                    aggregated_grads[idx] = (
                        aggregated_grads[idx] + self.client_grads[i][idx] * client_weights[i]
                    )

            assert len(server_model.state_dict().keys()) == len(aggregated_grads)
            for idx, key in enumerate(server_model.state_dict().keys()):
                if "num_batches_tracked" in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    server_model.state_dict()[key].data.copy_(
                        server_model.state_dict()[key].data + aggregated_grads[idx]
                    )
                    # distribute back to clients
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(
                            server_model.state_dict()[key]
                        )

        return server_model, models
