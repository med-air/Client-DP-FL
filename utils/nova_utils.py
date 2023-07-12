# Utils for FedNova
from typing import cast, List, Optional, Dict, Tuple

import torch
import torch.distributed as dist
from torch.optim import Adam
from torch.optim.optimizer import Optimizer, required, _use_grad_for_differentiable
import math


class SimpleFedNova4Adam(Adam):
    def __init__(
        self,
        params,
        ratio,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.ratio = ratio
        self.gmf = 0
        self.mu = 0

        self.local_normalizing_vec = 0
        self.local_steps = 0

    @_use_grad_for_differentiable
    def step(self, closure=None):
        self.local_normalizing_vec += 1
        self.local_steps += 1

        return super().step(closure)


class FedNova4Adam(Optimizer):
    def __init__(
        self,
        params,
        ratio,
        gmf,
        mu=0,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        variance=0,
    ):

        self.gmf = gmf
        self.ratio = ratio

        self.momentum = betas[0]
        self.mu = mu
        self.local_normalizing_vec = 0
        self.local_counter = 0
        self.local_steps = 0

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            variance=variance,
        )
        super(FedNova4Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FedNova4Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # scale = 1**self.itr
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]
            eps = group["eps"]
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                ## Nova logic
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                if "old_init" not in state:
                    state["old_init"] = torch.clone(p.data).detach()

                state["step"] += 1

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                ## Nova logic

                if self.mu != 0:
                    grad.add_(self.mu, p.data - state["old_init"])

                if "cum_grad" not in state:
                    state["cum_grad"] = torch.clone(grad).detach()
                    state["cum_grad"].mul_(lr)
                else:
                    state["cum_grad"].add_(grad, alpha=lr)

                ##

                state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    torch.maximum(
                        state["max_exp_avg_sq"], state["exp_avg_sq"], out=state["max_exp_avg_sq"]
                    )
                    denom = state["max_exp_avg_sq"].sqrt() / math.sqrt(bias_correction2).add_(eps)
                else:
                    denom = (state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)).add_(eps)

                step_size = lr / bias_correction1

                p.data.addcdiv_(state["exp_avg"], denom, value=-step_size)

        # compute local normalizing vector a_i
        if self.momentum != 0:
            self.local_counter = self.local_counter * self.momentum + 1
            self.local_normalizing_vec += self.local_counter

        self.etamu = lr * self.mu
        if self.etamu != 0:
            self.local_normalizing_vec *= 1 - self.etamu
            self.local_normalizing_vec += 1

        if self.momentum == 0 and self.etamu == 0:
            self.local_normalizing_vec += 1

        self.local_steps += 1

        return loss


"""
FedNova Optimizer implementation cited from https://github.com/JYWa/FedNova/tree/master
This version is modified from https://github.com/FedML-AI/FedML/blob/master/fedml_api/standalone/fednova/fednova.py
"""


class FedNova(Optimizer):
    r"""Implements federated normalized averaging (FedNova).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        ratio (float): relative sample size of client
        gmf (float): global/server/slow momentum factor
        mu (float): parameter for proximal local SGD
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v
        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
             v = \rho * v + lr * g \\
             p = p - v
        The Nesterov version is analogously modified.
    """

    def __init__(
        self,
        params,
        ratio,
        gmf,
        mu=0,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        variance=0,
    ):

        self.gmf = gmf
        self.ratio = ratio
        self.momentum = momentum
        self.mu = mu
        self.local_normalizing_vec = 0
        self.local_counter = 0
        self.local_steps = 0

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            variance=variance,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(FedNova, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FedNova, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        # scale = 1**self.itr
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(alpha=weight_decay, other=p.data)

                param_state = self.state[p]
                if "old_init" not in param_state:
                    param_state["old_init"] = torch.clone(p.data).detach()

                local_lr = group["lr"]

                # apply momentum updates
                if momentum != 0:
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(alpha=1 - dampening, other=d_p)
                    if nesterov:
                        d_p = d_p.add(alpha=momentum, other=buf)
                    else:
                        d_p = buf

                # apply proximal updates
                if self.mu != 0:
                    d_p.add_(alpha=self.mu, other=p.data - param_state["old_init"])

                # update accumalated local updates
                if "cum_grad" not in param_state:
                    param_state["cum_grad"] = torch.clone(d_p).detach()
                    param_state["cum_grad"].mul_(local_lr)

                else:
                    param_state["cum_grad"].add_(alpha=local_lr, other=d_p)

                p.data.add_(alpha=-local_lr, other=d_p)

        # compute local normalizing vector a_i
        if self.momentum != 0:
            self.local_counter = self.local_counter * self.momentum + 1
            self.local_normalizing_vec += self.local_counter

        self.etamu = local_lr * self.mu
        if self.etamu != 0:
            self.local_normalizing_vec *= 1 - self.etamu
            self.local_normalizing_vec += 1

        if self.momentum == 0 and self.etamu == 0:
            self.local_normalizing_vec += 1

        self.local_steps += 1

        return loss


def get_local_norm_grad(opt, cur_params, init_params, weight=0):
    if weight == 0:
        weight = opt.ratio
    grad_dict = {}
    for k in cur_params.keys():
        # skip the track param in BN layers
        if "num_batches_tracked" in k:
            continue
        scale = 1.0 / opt.local_normalizing_vec
        cum_grad = init_params[k] - cur_params[k]
        cum_grad.mul_(weight * scale)
        grad_dict[k] = cum_grad
    return grad_dict


def get_local_tau_eff(opt):
    if opt.mu != 0:
        return opt.local_steps * opt.ratio
    else:
        return opt.local_normalizing_vec * opt.ratio


def reset_fednova_optimizer(opt):
    opt.local_counter = 0
    opt.local_normalizing_vec = 0
    opt.local_steps = 0
    for group in opt.param_groups:
        for p in group["params"]:
            param_state = opt.state[p]
            param_state["cum_grad"].zero_()
            # Reinitialize momentum buffer
            if "momentum_buffer" in param_state:
                param_state["momentum_buffer"].zero_()
