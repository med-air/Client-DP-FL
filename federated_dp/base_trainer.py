"""
Base FedAvg Trainer
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
from sklearn import metrics
from utils.loss import DiceLoss
from utils.util import _eval_haus, _eval_iou
from dataset.dataset import DatasetSplit
from utils.nova_utils import SimpleFedNova4Adam


def dict_append(key, value, dict_):
    """
    dict_[key] = list()
    """
    if key not in dict_:
        dict_[key] = [value]
    else:
        dict_[key].append(value)
    return dict_


def cvt_np(lst):
    res_np = np.array([each.numpy() for each in lst])
    return res_np


def cvt_dict(lst):
    res_dct = {i: cvt_np(lst[i]) for i in range(len(lst))}
    return res_dct


def metric_calc(gt, pred, score):
    tn, fp, fn, tp = metrics.confusion_matrix(gt, pred).ravel()
    acc = metrics.accuracy_score(gt, pred)
    try:
        auc = metrics.roc_auc_score(gt, score)
    except ValueError:
        auc = 0
    sen = metrics.recall_score(gt, pred)  # recall = sensitivity = TP/TP+FN
    spe = tn / (tn + fp)  # specificity = TN / (TN+FP)
    f1 = metrics.f1_score(gt, pred)
    return [tn, fp, fn, tp], auc, acc, sen, spe, f1


def metric_log_print(metric_dict, cur_metric):
    if "AUC" in list(cur_metric.keys()):
        clients_accs_avg = np.mean(
            [v[-1] for k, v in metric_dict.items() if "mean" not in k and "Acc" in k]
        )
        metric_dict = dict_append("mean_Acc", clients_accs_avg, metric_dict)

        clients_aucs_avg = np.mean(
            [v[-1] for k, v in metric_dict.items() if "mean" not in k and "AUC" in k]
        )
        metric_dict = dict_append("mean_AUC", clients_aucs_avg, metric_dict)

        clients_sens_avg = np.mean(
            [v[-1] for k, v in metric_dict.items() if "mean" not in k and "Sen" in k]
        )
        metric_dict = dict_append("mean_Sen", clients_sens_avg, metric_dict)

        clients_spes_avg = np.mean(
            [v[-1] for k, v in metric_dict.items() if "mean" not in k and "Spe" in k]
        )
        metric_dict = dict_append("mean_Spe", clients_spes_avg, metric_dict)

        clients_f1_avg = np.mean(
            [v[-1] for k, v in metric_dict.items() if "mean" not in k and "F1" in k]
        )
        metric_dict = dict_append("mean_F1", clients_f1_avg, metric_dict)

        out_str = f" | {'AUC'}: {clients_aucs_avg:.4f} | {'Acc'}: {clients_accs_avg:.4f} | {'Sen'}: {clients_sens_avg:.4f} | {'Spe'}: {clients_spes_avg:.4f} | {'F1'}: {clients_f1_avg:.4f}"
    elif "Dice" in list(cur_metric.keys()):
        clients_dice_avg = np.mean(
            [v[-1] for k, v in metric_dict.items() if "mean" not in k and "Dice" in k]
        )
        metric_dict = dict_append("mean_Dice", clients_dice_avg, metric_dict)

        clients_hd_avg = np.mean(
            [v[-1] for k, v in metric_dict.items() if "mean" not in k and "HD" in k]
        )
        metric_dict = dict_append("mean_HD", clients_hd_avg, metric_dict)

        clients_iou_avg = np.mean(
            [v[-1] for k, v in metric_dict.items() if "mean" not in k and "IoU" in k]
        )
        metric_dict = dict_append("mean_IoU", clients_iou_avg, metric_dict)

        out_str = f" | {'Dice'}: {clients_dice_avg:.4f} | {'HD'}: {clients_hd_avg:.4f} | {'IoU'}: {clients_iou_avg:.4f}"
    else:
        raise NotImplementedError

    return metric_dict, out_str


class BaseFederatedTrainer(object):
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
    ) -> None:
        self.args = args
        self.logging = logging
        self.device = device
        self.lr_decay = args.lr_decay > 0
        self.server_model = server_model
        self.train_sites = train_sites
        self.val_sites = val_sites
        self.client_num = len(train_sites)
        self.client_num_val = len(val_sites)
        self.sample_rate = args.sample_rate
        assert self.sample_rate > 0 and self.sample_rate <= 1
        self.aggregation_idxs = None
        self.aggregation_client_num = int(self.client_num * self.sample_rate)
        self.client_weights = (
            [1 / self.aggregation_client_num for i in range(self.client_num)]
            if client_weights is None
            else client_weights
        )
        self.client_models = [copy.deepcopy(server_model) for idx in range(self.client_num)]
        self.client_grads = [None for i in range(self.client_num)]
        (
            self.train_loss,
            self.train_acc,
            self.val_loss,
            self.val_acc,
            self.test_loss,
            self.test_acc,
        ) = ({}, {}, {}, {}, {}, {})

        self.generalize_sites = (
            kwargs["generalize_sites"] if "generalize_sites" in kwargs.keys() else None
        )

        self.train_loss["mean"] = []
        self.val_loss["mean"] = []
        self.test_loss["mean"] = []

        self.virtual_clients = args.virtual_clients if not args.ada_vn else 1

    def train(self, model, data_loader, optimizer, loss_fun):
        model.to(self.device)
        # optimizer.load_state_dict(optimizer.state_dict())
        # print(optimizer.state.values())
        for optimizer_metrics in optimizer.state.values():
            for metric_name, metric in optimizer_metrics.items():
                if torch.is_tensor(metric):
                    optimizer_metrics[metric_name] = metric.to(self.device)

        model.train()
        loss_all = 0
        # segmentation = model.__class__.__name__ == 'UNet'
        segmentation = "UNet" in model.__class__.__name__
        train_acc = 0.0 if not segmentation else {}
        model_pred, label_gt, pred_prob = [], [], []
        num_sample_test = 0

        for step, data in enumerate(data_loader):
            if self.args.data.startswith("prostate"):
                inp = data["Image"]
                target = data["Mask"]
                target = target.to(self.device)
            else:
                inp = data["Image"]
                target = data["Label"]
                target = target.to(self.device)

            optimizer.zero_grad()
            inp = inp.to(self.device)
            output = model(inp)

            if self.args.data.startswith("prostate"):
                loss = loss_fun(output[:, 0, :, :], target)
            else:
                loss = loss_fun(output, target)

            loss_all += loss.item()

            if segmentation:
                if self.args.data.startswith("prostate"):
                    if len(train_acc.keys()) == 0:
                        train_acc["Dice"] = DiceLoss().dice_coef(output[:, 0, :, :], target).item()
                        train_acc["IoU"] = _eval_iou(output[:, 0, :, :], target).item()
                        train_acc["HD"] = 0.0
                    else:
                        train_acc["Dice"] += (
                            DiceLoss().dice_coef(output[:, 0, :, :], target).item()
                        )
                        train_acc["IoU"] += _eval_iou(output[:, 0, :, :], target).item()

                    for i_b in range(output.shape[0]):
                        hd = _eval_haus(output[i_b, 0, :, :], target[i_b]).item()
                        if hd > 0:
                            train_acc["HD"] += hd
                            num_sample_test += 1

                else:
                    if len(train_acc.keys()) == 0:
                        train_acc["Dice"] = DiceLoss().dice_coef(output, target).item()
                    else:
                        train_acc["Dice"] += DiceLoss().dice_coef(output, target).item()
            else:
                out_prob = torch.nn.functional.softmax(output, dim=1)
                model_pred.extend(out_prob.data.max(1)[1].view(-1).detach().cpu().numpy())
                pred_prob.extend(out_prob.data[:, 1].view(-1).detach().cpu().numpy())
                label_gt.extend(target.view(-1).detach().cpu().numpy())

            loss.backward()
            optimizer.step()

        loss = loss_all / len(data_loader)
        if segmentation:
            acc = {
                "Dice": train_acc["Dice"] / len(data_loader),
                "IoU": train_acc["IoU"] / len(data_loader),
                "HD": train_acc["HD"] / num_sample_test,
            }
        else:
            model_pred = np.asarray(model_pred)
            pred_prob = np.asarray(pred_prob)
            label_gt = np.asarray(label_gt)
            metric_res = metric_calc(label_gt, model_pred, pred_prob)
            acc = {
                "AUC": metric_res[1],
                "Acc": metric_res[2],
                "Sen": metric_res[3],
                "Spe": metric_res[4],
                "F1": metric_res[5],
            }

        model.to("cpu")
        for optimizer_metrics in optimizer.state.values():
            for metric_name, metric in optimizer_metrics.items():
                if torch.is_tensor(metric):
                    optimizer_metrics[metric_name] = metric.cpu()
        return loss, acc

    def test(self, model, data_loader, loss_fun, process=False):
        model.to(self.device)
        model.eval()
        loss_all = 0
        total = 0
        correct = 0
        num_sample_test = 0

        segmentation = "UNet" in model.__class__.__name__
        test_acc = 0.0 if not segmentation else {}
        model_pred, label_gt, pred_prob = [], [], []
        for step, data in enumerate(data_loader):
            if self.args.data.startswith("prostate"):
                inp = data["Image"]
                target = data["Mask"]
                target = target.to(self.device)
            else:
                inp = data["Image"]
                target = data["Label"]
                target = target.to(self.device)

            inp = inp.to(self.device)
            output = model(inp)

            if self.args.data.startswith("prostate"):
                loss = loss_fun(output[:, 0, :, :], target)
            else:
                loss = loss_fun(output, target)

            loss_all += loss.item()

            if segmentation:
                if self.args.data.startswith("prostate"):
                    if len(test_acc.keys()) == 0:
                        test_acc["Dice"] = DiceLoss().dice_coef(output[:, 0, :, :], target).item()
                        test_acc["IoU"] = _eval_iou(output[:, 0, :, :], target).item()
                        test_acc["HD"] = 0.0
                    else:
                        test_acc["Dice"] += DiceLoss().dice_coef(output[:, 0, :, :], target).item()
                        test_acc["IoU"] += _eval_iou(output[:, 0, :, :], target).item()

                    for i_b in range(output.shape[0]):
                        hd = _eval_haus(output[i_b, 0, :, :], target[i_b]).item()
                        if hd > 0:
                            test_acc["HD"] += hd
                            num_sample_test += 1
                else:
                    if len(test_acc.keys()) == 0:
                        test_acc["Dice"] = DiceLoss().dice_coef(output, target).item()
                    else:
                        test_acc["Dice"] += DiceLoss().dice_coef(output, target).item()
            else:
                out_prob = torch.nn.functional.softmax(output, dim=1)
                model_pred.extend(out_prob.data.max(1)[1].view(-1).detach().cpu().numpy())
                pred_prob.extend(out_prob.data[:, 1].view(-1).detach().cpu().numpy())
                label_gt.extend(target.view(-1).detach().cpu().numpy())

        loss = loss_all / len(data_loader)
        # acc = test_acc/ len(data_loader) if segmentation else correct/total
        if segmentation:
            acc = {
                "Dice": test_acc["Dice"] / len(data_loader),
                "IoU": test_acc["IoU"] / len(data_loader),
                "HD": test_acc["HD"] / num_sample_test,
            }
        else:
            model_pred = np.asarray(model_pred)
            pred_prob = np.asarray(pred_prob)
            label_gt = np.asarray(label_gt)
            metric_res = metric_calc(label_gt, model_pred, pred_prob)
            acc = {
                "AUC": metric_res[1],
                "Acc": metric_res[2],
                "Sen": metric_res[3],
                "Spe": metric_res[4],
                "F1": metric_res[5],
            }
        model.to("cpu")
        return loss, acc

    def inference_test(self, save_path, model, data_loader, loss_fun, process=False):
        def _culmode(anno, img):
            # Find external contours
            img_cp = img.copy()
            contours, _ = cv2.findContours(anno, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in contours:
                cv2.drawContours(img_cp, [c], -1, [0, 255, 0], 3)

            return img_cp

        """inference and save results"""
        model.to(self.device)
        model.eval()
        loss_all = 0
        total = 0
        correct = 0
        num_sample_test = 0
        segmentation = model.__class__.__name__ == "UNet"
        test_acc = 0.0 if not segmentation else {}
        test_his = {}
        save_res_path = save_path + "_inference_results"
        os.makedirs(save_res_path, exist_ok=True)
        for step, data in enumerate(data_loader):
            if self.args.data.startswith("prostate"):
                inp = data["Image"]
                target = data["Mask"]
                target = target.to(self.device)
            else:
                inp = data["Image"]
                target = data["Label"]
                # inp, target = data[0], data[1]
                target = target.to(self.device)

            inp = inp.to(self.device)
            output = model(inp)
            pred = torch.sigmoid(output)

            if self.args.data.startswith("prostate"):
                loss = loss_fun(output[:, 0, :, :], target)
            else:
                loss = loss_fun(output, target)

            loss_all += loss.item()
            batch_dice = 0.0
            if segmentation:
                if self.args.data.startswith("prostate"):
                    if len(test_acc.keys()) == 0:
                        test_acc["Dice"] = DiceLoss().dice_coef(output[:, 0, :, :], target).item()
                        test_acc["IoU"] = _eval_iou(output[:, 0, :, :], target).item()
                        test_acc["HD"] = 0.0

                        test_his["Dice"] = [
                            DiceLoss().dice_coef(output[:, 0, :, :], target).item()
                        ]
                    else:
                        test_acc["Dice"] += DiceLoss().dice_coef(output[:, 0, :, :], target).item()
                        test_acc["IoU"] += _eval_iou(output[:, 0, :, :], target).item()

                        test_his["Dice"].append(
                            DiceLoss().dice_coef(output[:, 0, :, :], target).item()
                        )

                    for i_b in range(output.shape[0]):
                        hd = _eval_haus(output[i_b, 0, :, :], target[i_b]).item()
                        if hd > 0:
                            test_acc["HD"] += hd
                            num_sample_test += 1

                    batch_dice = DiceLoss().dice_coef(output[:, 0, :, :], target).item()
                else:
                    if len(test_acc.keys()) == 0:
                        test_acc["Dice"] = DiceLoss().dice_coef(output, target).item()
                    else:
                        test_acc["Dice"] += DiceLoss().dice_coef(output, target).item()
            else:
                total += target.size(0)
                pred = output.data.max(1)[1]
                batch_correct = pred.eq(target.view(-1)).sum().item()
                correct += batch_correct
                if self.args.data == "camelyon17":
                    if step % math.ceil(len(data_loader) * 0.05) == 0:
                        print(
                            " [Step-{}|{}]| Test Acc: {:.4f}".format(
                                step, len(data_loader), batch_correct / target.size(0)
                            ),
                            end="\r",
                        )

            import cv2

        loss = loss_all / len(data_loader)

        if segmentation:
            acc = {
                "Dice": test_acc["Dice"] / len(data_loader),
                "IoU": test_acc["IoU"] / len(data_loader),
                "HD": test_acc["HD"] / num_sample_test,
            }
        else:
            acc = correct / total
        model.to("cpu")
        return loss, acc

    def inference(self, ckpt_path, data_loaders, loss_fun, datasites, process=False):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        global_model = copy.deepcopy(self.server_model).to(self.device)
        # mean_loss, mean_acc = [], []
        global_model.load_state_dict(checkpoint["server_model"])
        assert len(datasites) == len(data_loaders)
        for client_idx in range(len(data_loaders)):
            with torch.no_grad():
                test_loss, test_acc = self.inference_test(
                    ckpt_path, global_model, data_loaders[client_idx], loss_fun, process
                )

            self.test_loss = dict_append(
                "client_{}".format(str(datasites[client_idx])), test_loss, self.test_loss
            )

            if isinstance(test_acc, dict):
                out_str = ""
                for k, v in test_acc.items():
                    out_str += " | Test {}: {:.6f}".format(k, v)
                    self.test_acc = dict_append(
                        f"client{datasites[client_idx]}_" + k, v, self.test_acc
                    )
                self.logging.info(
                    " Site-{:<10s}| Test Loss: {:.6f}{}".format(
                        str(datasites[client_idx]), test_loss, out_str
                    )
                )

            else:
                self.test_acc["client_{}".format(datasites[client_idx])].append(round(test_acc, 4))
                self.logging.info(
                    " Site-{:<10s}| Test Loss: {:.4f} | Test Acc: {:.4f}".format(
                        str(datasites[client_idx]), test_loss, test_acc
                    )
                )

            if client_idx == len(data_loaders) - 1:
                clients_loss_avg = np.mean(
                    [v[-1] for k, v in self.test_loss.items() if "mean" not in k]
                )
                self.test_loss["mean"].append(clients_loss_avg)

                self.test_acc, out_str = metric_log_print(self.test_acc, test_acc)
                self.logging.info(
                    " Site-Average | Test Loss: {:.4f}{}".format(clients_loss_avg, out_str)
                )

        metrics_pd = pd.DataFrame.from_dict(self.test_acc)
        metrics_pd.to_csv(os.path.join(self.args.log_path, "test_acc.csv"))

        del global_model

    def test_ckpt(self, ckpt_path, data_loaders, loss_fun, datasites, process=False):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        global_model = copy.deepcopy(self.server_model).to(self.device)
        # mean_loss, mean_acc = [], []
        global_model.load_state_dict(checkpoint["server_model"])
        self.test_acc = dict_append("round", self.cur_iter, self.test_acc)

        if self.args.merge:
            test_loss, test_acc = self.test(global_model, data_loaders, loss_fun, process)
            self.test_loss["mean"].append(test_loss)
            if isinstance(test_acc, dict):
                out_str = ""
                for k, v in test_acc.items():
                    out_str += " | Test {}: {:.4f} ".format(k, v)
                    self.test_acc = dict_append("mean_" + k, v, self.test_acc)
                self.logging.info(" Site-Average | Test Loss: {:.4f}{}".format(test_loss, out_str))
            else:
                self.logging.info(
                    " Site-Average | Test Loss: {:.4f} | Test Acc: {:.4f}".format(
                        test_loss, test_acc
                    )
                )
                self.test_acc = dict_append("mean", test_acc, self.test_acc)
        else:
            assert len(datasites) == len(data_loaders)
            for client_idx in range(len(data_loaders)):
                test_loss, test_acc = self.test(
                    global_model, data_loaders[client_idx], loss_fun, process
                )
                self.test_loss = dict_append(
                    "client_{}".format(str(datasites[client_idx])), test_loss, self.test_loss
                )

                if isinstance(test_acc, dict):
                    out_str = ""
                    for k, v in test_acc.items():
                        out_str += " | Test {}: {:.4f}".format(k, v)
                        self.test_acc = dict_append(
                            f"client{datasites[client_idx]}_" + k, v, self.test_acc
                        )
                    self.logging.info(
                        " Site-{:<10s}| Test Loss: {:.4f}{}".format(
                            str(datasites[client_idx]), test_loss, out_str
                        )
                    )

                else:
                    self.test_acc = dict_append(
                        f"client_{datasites[client_idx]}", round(test_acc, 4), self.test_acc
                    )
                    self.logging.info(
                        " Site-{:<10s}| Test Loss: {:.4f} | Test Acc: {:.4f}".format(
                            str(datasites[client_idx]), test_loss, test_acc
                        )
                    )

                if client_idx == len(data_loaders) - 1:
                    clients_loss_avg = np.mean(
                        [v[-1] for k, v in self.test_loss.items() if "mean" not in k]
                    )
                    self.test_loss["mean"].append(clients_loss_avg)

                    self.test_acc, out_str = metric_log_print(self.test_acc, test_acc)

                    self.logging.info(
                        " Site-Average | Test Loss: {:.4f}{}".format(clients_loss_avg, out_str)
                    )

        del global_model

    def prepare_ckpt(self, a_iter):
        if self.args.local_bn:
            model_dicts = {
                "server_model": self.server_model.state_dict(),
                "best_epoch": self.best_epoch,
                "best_acc": self.best_acc,
                "a_iter": a_iter,
            }
            for model_idx, model in enumerate(self.client_models):
                model_dicts["model_{}".format(model_idx)] = model.state_dict()
        else:
            model_dicts = {
                "server_model": self.server_model.state_dict(),
                "best_epoch": self.best_epoch,
                "best_acc": self.best_acc,
                "a_iter": a_iter,
            }
        return model_dicts

    def communication(self, server_model, models, client_weights):
        with torch.no_grad():
            # aggregate params
            if self.args.local_bn:
                for key in server_model.state_dict().keys():
                    if "bn" not in key:
                        temp = torch.zeros_like(
                            server_model.state_dict()[key], dtype=torch.float32
                        )
                        for client_idx in range(len(client_weights)):
                            temp += (
                                client_weights[client_idx] * models[client_idx].state_dict()[key]
                            )
                        server_model.state_dict()[key].data.copy_(temp)
                        for client_idx in range(len(client_weights)):
                            models[client_idx].state_dict()[key].data.copy_(
                                server_model.state_dict()[key]
                            )
            else:
                for key in server_model.state_dict().keys():
                    # num_batches_tracked is a non trainable LongTensor and
                    # num_batches_tracked are the same for all clients for the given datasets
                    if "num_batches_tracked" in key:
                        server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                    else:
                        temp = torch.zeros_like(server_model.state_dict()[key])
                        for client_idx in range(len(client_weights)):
                            temp += (
                                client_weights[client_idx] * models[client_idx].state_dict()[key]
                            )
                        server_model.state_dict()[key].data.copy_(temp)
                        # distribute back to clients
                        for client_idx in range(len(client_weights)):
                            models[client_idx].state_dict()[key].data.copy_(
                                server_model.state_dict()[key]
                            )

        return server_model, models

    def communication_grad(self):
        raise NotImplementedError(f"BaseTrainer does not implement `communication_grad()`")

    def train_epoch(self, a_iter, train_loaders, loss_fun, datasets):
        for client_idx in self.aggregation_idxs:
            model = self.client_models[client_idx]
            old_model = copy.deepcopy(model).to("cpu")

            for optimizer_metrics in self.optimizers[client_idx].state.values():
                for metric_name, metric in optimizer_metrics.items():
                    if torch.is_tensor(metric):
                        optimizer_metrics[metric_name] = metric.to(self.device)

            train_loss, train_acc = self.train(
                model, train_loaders[client_idx], self.optimizers[client_idx], loss_fun
            )
            client_update = self._compute_param_update(
                old_model=old_model, new_model=model, device="cpu"
            )
            self.client_grads[client_idx] = client_update

            # clear optimizer internal tensors from gpu
            for optimizer_metrics in self.optimizers[client_idx].state.values():
                for metric_name, metric in optimizer_metrics.items():
                    if torch.is_tensor(metric):
                        optimizer_metrics[metric_name] = metric.cpu()

            if self.lr_decay:
                self.schedulers[client_idx].step()

            if isinstance(train_acc, dict):
                out_str = ""
                for k, v in train_acc.items():
                    self.args.writer.add_scalar(
                        f"Performance/train_client{str(datasets[client_idx])}_{k}", v, a_iter
                    )
                    out_str += " | Train {}: {:.4f} ".format(k, v)
                self.logging.info(
                    " Site-{:<10s}| Train Loss: {:.4f}{}".format(
                        str(datasets[client_idx]), train_loss, out_str
                    )
                )
            else:
                self.logging.info(
                    " Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}".format(
                        str(datasets[client_idx]), train_loss, train_acc
                    )
                )
                self.args.writer.add_scalar(
                    f"Accuracy/train_client{str(datasets[client_idx])}", train_acc, a_iter
                )

            self.args.writer.add_scalar(
                f"Loss/train_{str(datasets[client_idx])}", train_loss, a_iter
            )

    def warm_train_epoch(self, a_iter, train_loaders, loss_fun, datasets):
        for client_idx, loader in enumerate(train_loaders):
            train_loss, train_acc = self.train(
                self.client_models[client_idx * self.virtual_clients],
                loader,
                self.warm_optimizers[client_idx],
                loss_fun,
            )

            if isinstance(train_acc, dict):
                out_str = ""
                for k, v in train_acc.items():
                    out_str += " | Train {}: {:.4f} ".format(k, v)
                self.logging.info(
                    " Site-{:<10s}| Train Loss: {:.4f}{}".format(
                        str(datasets[client_idx]), train_loss, out_str
                    )
                )
            else:
                self.logging.info(
                    " Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}".format(
                        str(datasets[client_idx]), train_loss, train_acc
                    )
                )

    def _compute_param_update(self, old_model, new_model, device=None):
        if device:
            old_model, new_model = old_model.to(device), new_model.to(device)
        # using .state_dict()
        old_param = old_model.state_dict()
        new_param = new_model.state_dict()
        return [(new_param[key] - old_param[key]) for key in new_param.keys()]

    def init_optims(self):
        self.optimizers = []
        self.schedulers = []
        if self.args.data == "cifar10" or self.args.data == "digits5":
            for idx in range(self.client_num):
                optimizer = optim.SGD(params=self.client_models[idx].parameters(), lr=self.args.lr)
                self.optimizers.append(optimizer)
                if self.lr_decay:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=self.args.rounds
                    )
                    self.schedulers.append(scheduler)
        else:
            for idx in range(self.client_num):

                optimizer = optim.Adam(
                    params=self.client_models[idx].parameters(), lr=self.args.lr, amsgrad=True
                )
                self.optimizers.append(optimizer)

    def warm_init_optims(self):
        self.warm_optimizers = []
        for idx in range(self.args.clients):
            optimizer = optim.Adam(
                params=self.client_models[idx * self.virtual_clients].parameters(),
                lr=self.args.lr,
                amsgrad=True,
            )
            self.warm_optimizers.append(optimizer)

    def _stop_listener(self):
        return False

    def save_metrics(self):
        metrics_pd = pd.DataFrame.from_dict(self.val_loss)
        metrics_pd.to_csv(os.path.join(self.args.log_path, "val_loss.csv"))
        metrics_pd = pd.DataFrame.from_dict(self.val_acc)
        metrics_pd.to_csv(os.path.join(self.args.log_path, "val_acc.csv"))

        metrics_pd = pd.DataFrame.from_dict(self.test_loss)
        metrics_pd.to_csv(os.path.join(self.args.log_path, "test_loss.csv"))
        metrics_pd = pd.DataFrame.from_dict(self.test_acc)
        metrics_pd.to_csv(os.path.join(self.args.log_path, "test_acc.csv"))

    def run(
        self,
        train_loaders,
        val_loaders,
        test_loaders,
        loss_fun,
        SAVE_PATH,
        generalize_sites=None,
    ):
        self.val_loaders = val_loaders
        self.loss_fun = loss_fun

        real_train_loaders = copy.deepcopy(train_loaders)

        # Start training
        self.init_optims()

        not_aggregate_first = ("dp" in self.args.mode) and (self.args.ada_vn or self.args.init_vn)

        for a_iter in range(self.start_iter, self.args.rounds):
            if self.sample_rate < 1:
                self.aggregation_idxs = random.sample(
                    list(range(self.client_num)), self.aggregation_client_num
                )
            else:
                self.aggregation_client_num = len(self.client_models)
                self.aggregation_idxs = list(range(len(self.client_models)))

            self.cur_iter = a_iter
            # each round
            for wi in range(self.args.local_epochs):
                self.logging.info(
                    "============ Round {}, Local train epoch {} ============".format(a_iter, wi)
                )
                self.train_epoch(a_iter, train_loaders, loss_fun, self.train_sites)

            with torch.no_grad():
                if a_iter == 0 and not_aggregate_first:
                    temp_server_model = copy.deepcopy(self.server_model)
                    temp_client_model = copy.deepcopy(self.client_models)

                try:
                    self.server_model, self.client_models = self.communication_grad(
                        self.server_model, self.client_models, self.client_weights
                    )
                except Exception as error:
                    self.logging.info(repr(error))
                    self.logging.info(f"Stop training at round {a_iter}.")
                    break

                if a_iter == 0 and not_aggregate_first:
                    self.server_model = temp_server_model
                    self.client_models = temp_client_model
                    del temp_server_model, temp_client_model

                # Validation
                if self.args.merge:
                    mean_val_loss_, mean_val_acc_ = self.test(
                        self.server_model, val_loaders, loss_fun
                    )
                    self.val_loss["mean"].append(mean_val_loss_)
                    self.args.writer.add_scalar(f"Loss/val", mean_val_loss_, a_iter)
                    if isinstance(mean_val_acc_, dict):
                        out_str = ""
                        for k, v in mean_val_acc_.items():
                            out_str += " | Val {}: {:.4f}".format(k, v)
                            self.val_acc = dict_append("mean_" + k, v, self.val_acc)
                            self.args.writer.add_scalar(f"Performance/val_{k}", v, a_iter)
                        self.logging.info(
                            " Site-Average | Val Loss: {:.4f}{}".format(mean_val_loss_, out_str)
                        )
                        mean_val_acc_ = np.mean([v for k, v in mean_val_acc_.items()])
                    else:
                        self.logging.info(
                            " Site-Average | Val Loss: {:.4f} | Val Acc: {:.4f}".format(
                                mean_val_loss_, mean_val_acc_
                            )
                        )
                        self.args.writer.add_scalar(f"Accuracy/val", mean_val_acc_, a_iter)
                        self.val_acc = dict_append("mean", mean_val_acc_, self.val_acc)

                else:
                    assert len(self.val_sites) == len(val_loaders)
                    for client_idx, val_loader in enumerate(val_loaders):
                        val_loss, val_acc = self.test(self.server_model, val_loader, loss_fun)

                        self.val_loss = dict_append(
                            f"client_{self.val_sites[client_idx]}", val_loss, self.val_loss
                        )

                        self.args.writer.add_scalar(
                            f"Loss/val_{self.val_sites[client_idx]}", val_loss, a_iter
                        )

                        if isinstance(val_acc, dict):
                            out_str = ""
                            for k, v in val_acc.items():
                                out_str += " | Val {}: {:.4f}".format(k, v)
                                # self.val_acc = dict_append(f'client{client_idx+1}_'+k, v, self.val_acc)
                                self.val_acc = dict_append(
                                    f"client{self.val_sites[client_idx]}_" + k, v, self.val_acc
                                )
                                self.args.writer.add_scalar(
                                    f"Performance/val_client{self.val_sites[client_idx]}_{k}",
                                    v,
                                    a_iter,
                                )

                            self.logging.info(
                                " Site-{:<10s}| Val Loss: {:.4f}{}".format(
                                    str(self.val_sites[client_idx]), val_loss, out_str
                                )
                            )
                        else:
                            self.val_acc = dict_append(
                                f"client_{self.val_sites[client_idx]}",
                                round(val_acc, 4),
                                self.val_acc,
                            )
                            self.logging.info(
                                " Site-{:<10s}| Val Loss: {:.4f} | Val Acc: {:.4f}".format(
                                    str(self.val_sites[client_idx]), val_loss, val_acc
                                )
                            )
                            self.args.writer.add_scalar(
                                f"Accuracy/val_{self.val_sites[client_idx]}", val_acc, a_iter
                            )

                        if client_idx == len(val_loaders) - 1:
                            clients_loss_avg = np.mean(
                                [v[-1] for k, v in self.val_loss.items() if "mean" not in k]
                            )
                            self.val_loss["mean"].append(clients_loss_avg)
                            # organize the metrics
                            self.val_acc, out_str = metric_log_print(self.val_acc, val_acc)

                            self.args.writer.add_scalar(f"Loss/val", clients_loss_avg, a_iter)

                            mean_val_acc_ = (
                                self.val_acc["mean_Acc"][-1]
                                if "mean_Acc" in list(self.val_acc.keys())
                                else self.val_acc["mean_Dice"][-1]
                            )
                            self.logging.info(
                                " Site-Average | Val Loss: {:.4f}{}".format(
                                    clients_loss_avg, out_str
                                )
                            )

                if mean_val_acc_ > self.best_acc:
                    self.best_acc = mean_val_acc_
                    self.best_epoch = a_iter
                    self.best_changed = True
                    self.logging.info(
                        " Best Epoch:{} | Avg Val Acc: {:.4f}".format(
                            self.best_epoch, np.mean(mean_val_acc_)
                        )
                    )
                # save model
                model_dicts = self.prepare_ckpt(a_iter)

                # save and test
                if self.best_changed:
                    self.early_stop = 20
                    self.logging.info(
                        " Saving the local and server checkpoint to {}...".format(
                            SAVE_PATH + f"/model_best_{a_iter}"
                        )
                    )
                    torch.save(model_dicts, SAVE_PATH + f"/model_best_{a_iter}")
                    self.best_changed = False
                    test_sites = (
                        generalize_sites if generalize_sites is not None else self.val_sites
                    )
                    self.test_ckpt(
                        SAVE_PATH + f"/model_best_{a_iter}", test_loaders, loss_fun, test_sites
                    )
                else:
                    if a_iter % 10 == 0:
                        torch.save(model_dicts, SAVE_PATH + f"/model_round_{a_iter}")
                    if self.early_stop > 0:
                        self.early_stop -= 1
                    else:
                        if self.args.early:
                            self.logging.info(" No improvement over 10 epochs, early stop...")
                            break

                if self.args.ada_vn and "dp" in self.args.mode:
                    assert self.args.virtual_clients == 1
                    vn = self.select_split_num()
                    if vn != self.virtual_clients:
                        train_loaders = self.split_virtual_client(vn, real_train_loaders)
                if self.args.init_vn and "dp" in self.args.mode:
                    assert self.args.ada_vn is False  # These two mode cannot be used together
                    assert self.args.virtual_clients == 1
                    vn = self.select_split_num()
                    print(f"Initialize VN to {vn} using first round estimation results")
                    if vn != self.virtual_clients:
                        train_loaders = self.split_virtual_client(vn, real_train_loaders)
                    self.args.init_vn = False

                self.save_metrics()

    def select_split_num(self):
        raise NotImplementedError

    def split_virtual_client(self, vn, real_train_loaders):
        self.logging.info(f"Splitting each client into {vn} virtual client.")

        virtual_models = []
        virtual_opts = []
        virtual_training_loaders = []
        virtual_client_weights = []

        for client_idx in range(self.args.clients):
            dict_users = self.split_dataset(real_train_loaders[client_idx].dataset, vn)

            virtual_models.extend([copy.deepcopy(self.server_model) for _ in range(vn)])
            virtual_client_weights.extend(
                [
                    self.client_weights[client_idx * self.virtual_clients]
                    * self.virtual_clients
                    / vn
                    for _ in range(vn)
                ]
            )

            virtual_opts.extend(
                self.update_optimizers(
                    old_models=self.client_models[
                        client_idx * self.virtual_clients : (client_idx + 1) * self.virtual_clients
                    ],
                    old_optimizers=self.optimizers[
                        client_idx * self.virtual_clients : (client_idx + 1) * self.virtual_clients
                    ],
                    new_models=virtual_models[client_idx * vn : (client_idx + 1) * vn],
                )
            )

            for vclient_idx in range(vn):
                virtual_trainset = DatasetSplit(
                    real_train_loaders[client_idx].dataset,
                    dict_users[vclient_idx],
                    client_idx,
                    vclient_idx,
                )

                logging.info(
                    f"[Virtual Client {client_idx}-{vclient_idx}] Train={len(virtual_trainset)}"
                )

                virtual_training_loaders.append(
                    torch.utils.data.DataLoader(
                        virtual_trainset,
                        batch_size=self.args.batch,
                        shuffle=False,
                        drop_last=False,
                    )
                )

        self.client_models = virtual_models
        self.optimizers = virtual_opts
        self.client_weights = virtual_client_weights

        self.virtual_clients = vn
        self.client_num = self.args.clients * vn

        self.client_grads = [None for i in range(self.client_num)]
        self.train_sites = list(range(self.args.clients * vn))

        return virtual_training_loaders

    def update_optimizers(self, old_models, old_optimizers, new_models):
        new_optimizers = []

        new_state_l = [{} for _ in range(len(list(old_models[0].parameters())))]
        # aggregate optimizer states
        for i in range(len(old_models)):
            for i_p, p in enumerate(old_models[i].parameters()):
                for k, v in old_optimizers[i].state[p].items():
                    if k == "step":
                        new_state_l[i_p]["step"] = v.clone()
                    else:
                        if k not in new_state_l[i_p].keys():
                            new_state_l[i_p][k] = v.div(len(old_models)).clone()
                        else:
                            new_state_l[i_p][k].add_(v, alpha=1.0 / len(old_models))

        # broadcast to new models
        for i in range(len(new_models)):
            params = list(new_models[i].parameters())
            opt = copy.deepcopy(old_optimizers[0])

            if isinstance(opt, SimpleFedNova4Adam):
                opt.ratio = 0
                opt.gmt = 0
                opt.mu = 0
                opt.local_normalizing_vec = 0
                opt.local_steps = 0

            opt.state.clear()
            opt.param_groups.clear()

            param_group = copy.deepcopy(old_optimizers[0].param_groups[0])
            param_group["params"] = params
            opt.param_groups.append(param_group)

            for i_p, p in enumerate(params):
                opt.state[p] = copy.deepcopy(new_state_l[i_p])

            new_optimizers.append(opt)

        return new_optimizers

    def split_dataset(self, dataset, num_users):
        num_items = int(len(dataset) / num_users)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        return dict_users

    def start(
        self,
        train_loaders,
        val_loaders,
        test_loaders,
        loss_fun,
        SAVE_PATH,
        generalize_sites=None,
    ):
        self.run(
            train_loaders,
            val_loaders,
            test_loaders,
            loss_fun,
            SAVE_PATH,
            generalize_sites,
        )
        self.logging.info(" Training completed...")
