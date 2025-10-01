import copy
import os
from timeit import default_timer as timer
from datetime import datetime

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from tqdm import tqdm

from finetune_evaluator import Evaluator
from utils.logger import ModelLogger


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.data_loader = data_loader

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])
        
        # 生成包含架构信息的实验名称
        experiment_name = self._generate_experiment_name(params, model)
        self.logger = ModelLogger(log_dir="logs", params=params, experiment_name=experiment_name)
        
        # 记录实验配置
        config_dict = vars(params) if hasattr(params, '__dict__') else params
        self.logger.log_experiment_config(config_dict)
        
        # 记录模型架构
        self.logger.log_model_architecture(model)

        self.model = model.cuda()
        if self.params.downstream_dataset in ['FACED', 'SEED-V', 'PhysioNet-MI', 'ISRUC', 'BCIC2020-3', 'TUEV', 'BCIC-IV-2a']:
            self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        elif self.params.downstream_dataset in ['SHU-MI', 'CHB-MIT', 'Mumtaz2016', 'MentalArithmetic', 'TUAB']:
            self.criterion = BCEWithLogitsLoss().cuda()
        elif self.params.downstream_dataset == 'SEED-VIG':
            self.criterion = MSELoss().cuda()

        self.best_model_states = None
        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)

                if params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)


        mamba_params = []
        attn_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            lname = name.lower()
            if "mamba" in lname:
                mamba_params.append(param)
            elif "attention" in lname:
                attn_params.append(param)
            else:
                other_params.append(param)
        
        # 可选冻结backbone参数
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                if params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True


        # 支持自定义lr_mamba/lr_attn/lr_other
        lr_mamba = getattr(self.params, 'lr_mamba', self.params.lr)
        lr_attn = getattr(self.params, 'lr_attn', self.params.lr)
        lr_other = getattr(self.params, 'lr_other', self.params.lr)

        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr:
                param_groups = []
                if attn_params:
                    param_groups.append({'params': attn_params, 'lr': lr_attn})
                if mamba_params:
                    param_groups.append({'params': mamba_params, 'lr': lr_mamba})
                if other_params:
                    param_groups.append({'params': other_params, 'lr': lr_other * 5})
                self.optimizer = torch.optim.AdamW(param_groups, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)
        else:
            if self.params.multi_lr:
                param_groups = []
                if attn_params:
                    param_groups.append({'params': attn_params, 'lr': lr_attn})
                if mamba_params:
                    param_groups.append({'params': mamba_params, 'lr': lr_mamba})
                if other_params:
                    param_groups.append({'params': other_params, 'lr': lr_other * 5})
                self.optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9,
                                                 weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )
        print(self.model)
    
    def _generate_experiment_name(self, params, model):
        """生成包含架构信息的实验名称"""
        base_name = f"{params.downstream_dataset}_{model.__class__.__name__}"
        
        depths_str = params.depths
        stage_types_str = params.stage_types
        base_name += f"-[{depths_str}]-[{stage_types_str}]"
        
        # 添加时间戳确保唯一性
        timestamp = datetime.now().strftime("%m%d_%H%M")
        return f"{timestamp}_{base_name}"

    def train_for_multiclass(self):
        f1_best = 0
        kappa_best = 0
        acc_best = 0
        cm_best = None
        best_f1_epoch = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                if self.params.downstream_dataset == 'ISRUC':
                    loss = self.criterion(pred.transpose(1, 2), y)
                else:
                    loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            # 验证集评估
            with torch.no_grad():
                acc, kappa, f1, cm = self.val_eval.get_metrics_for_multiclass(self.model)
                avg_loss = np.mean(losses)
                current_lr = optim_state['param_groups'][0]['lr']
                training_time = (timer() - start_time) / 60
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        avg_loss,
                        acc,
                        kappa,
                        f1,
                        current_lr,
                        training_time
                    )
                )
                print(cm)

                # 日志记录
                self.logger.log_training_step(
                    epoch=epoch + 1,
                    step=len(self.data_loader['train']) * (epoch + 1),
                    loss=avg_loss,
                    lr=current_lr,
                    metrics={
                        'training_time_mins': training_time
                    }
                )
                val_metrics = {
                    'val_acc': acc,
                    'val_kappa': kappa,
                    'val_f1': f1
                }
                self.logger.log_validation_results(epoch + 1, avg_loss, val_metrics)

                if acc > acc_best:
                    print("acc increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                        acc,
                        kappa,
                        f1,
                    ))
                    best_f1_epoch = epoch + 1
                    acc_best = acc
                    kappa_best = kappa
                    f1_best = f1
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        # 恢复最佳模型
        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc_test, kappa_test, f1_test, cm_test = self.test_eval.get_metrics_for_multiclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, kappa: {:.5f}, f1: {:.5f}".format(
                    acc_test,
                    kappa_test,
                    f1_test,
                )
            )
            print(cm_test)

            # 日志记录最终测试结果
            self.logger.train_logger.info("Final Test Results:")
            self.logger.train_logger.info(f"Test Accuracy: {acc_test:.5f}")
            self.logger.train_logger.info(f"Test Kappa: {kappa_test:.5f}")
            self.logger.train_logger.info(f"Test F1: {f1_test:.5f}")

            # 保存模型
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            time = datetime.now().strftime("%m%d_%H%M%S")
            model_path = self.params.model_dir +  +"/{}_epoch{}_acc_{:.5f}_kappa_{:.5f}_f1_{:.5f}.pth".format(time,best_f1_epoch, acc_test, kappa_test, f1_test)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)

            self.logger.log_multiclass_result(best_f1_epoch, acc_best, kappa_best, f1_best, acc_test, kappa_test, f1_test, cm_test)

    def train_for_binaryclass(self):
        acc_best = 0
        roc_auc_best = 0
        pr_auc_best = 0
        cm_best = None
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)

                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc, pr_auc, roc_auc, cm = self.val_eval.get_metrics_for_binaryclass(self.model)
                
                # 记录训练信息到日志
                avg_loss = np.mean(losses)
                current_lr = optim_state['param_groups'][0]['lr']
                training_time = (timer() - start_time) / 60
                
                # 打印到控制台（保持原有功能）
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        avg_loss,
                        acc,
                        pr_auc,
                        roc_auc,
                        current_lr,
                        training_time
                    )
                )

                print(cm)
                
                # 记录到日志文件
                self.logger.log_training_step(
                    epoch=epoch + 1,
                    step=len(self.data_loader['train']) * (epoch + 1),
                    loss=avg_loss,
                    lr=current_lr,
                    metrics={
                        'training_time_mins': training_time
                    }
                )
                
                # 记录验证结果
                val_metrics = {
                    'val_acc': acc,
                    'val_pr_auc': pr_auc,
                    'val_roc_auc': roc_auc
                }
                self.logger.log_validation_results(epoch + 1, avg_loss, val_metrics)
                
                if roc_auc > roc_auc_best:
                    print("roc_auc increasing....saving weights !! ")
                    print("Val Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                        acc,
                        pr_auc,
                        roc_auc,
                    ))
                    best_roc_auc_epoch = epoch + 1
                    acc_best = acc
                    pr_auc_best = pr_auc
                    roc_auc_best = roc_auc
                    cm_best = cm
                    self.best_model_states = copy.deepcopy(self.model.state_dict())
                    


        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc, pr_auc, roc_auc, cm = self.test_eval.get_metrics_for_binaryclass(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: acc: {:.5f}, pr_auc: {:.5f}, roc_auc: {:.5f}".format(
                    acc,
                    pr_auc,
                    roc_auc,
                )
            )
            print(cm)
            
            # 记录最终测试结果
            final_test_metrics = {
                'test_acc': acc,
                'test_pr_auc': pr_auc,
                'test_roc_auc': roc_auc
            }
            self.logger.train_logger.info("Final Test Results:")
            self.logger.train_logger.info(f"Test Accuracy: {acc:.5f}")
            self.logger.train_logger.info(f"Test PR AUC: {pr_auc:.5f}")
            self.logger.train_logger.info(f"Test ROC AUC: {roc_auc:.5f}")

            # 保存模型到原始位置（
            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            time = datetime.now().strftime("%m%d_%H%M%S")
            model_path = self.params.model_dir + "/{}_epoch{}_acc_{:.5f}_pr_{:.5f}_roc_{:.5f}.pth".format(time,best_roc_auc_epoch, acc, pr_auc, roc_auc)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)

            self.logger.log_binary_class_result(best_roc_auc_epoch, acc_best, pr_auc_best, roc_auc_best, acc, pr_auc, roc_auc, cm, class_names=['Class_0', 'Class_1'])

    def train_for_regression(self):
        corrcoef_best = 0
        r2_best = 0
        rmse_best = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                corrcoef, r2, rmse = self.val_eval.get_metrics_for_regression(self.model)
                print(
                    "Epoch {} : Training Loss: {:.5f}, corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        corrcoef,
                        r2,
                        rmse,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                if r2 > r2_best:
                    print("r2 increasing....saving weights !! ")
                    print("Val Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                        corrcoef,
                        r2,
                        rmse,
                    ))
                    best_r2_epoch = epoch + 1
                    corrcoef_best = corrcoef
                    r2_best = r2
                    rmse_best = rmse
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            corrcoef, r2, rmse = self.test_eval.get_metrics_for_regression(self.model)
            print("***************************Test results************************")
            print(
                "Test Evaluation: corrcoef: {:.5f}, r2: {:.5f}, rmse: {:.5f}".format(
                    corrcoef,
                    r2,
                    rmse,
                )
            )

            if not os.path.isdir(self.params.model_dir):
                os.makedirs(self.params.model_dir)
            model_path = self.params.model_dir + "/epoch{}_corrcoef_{:.5f}_r2_{:.5f}_rmse_{:.5f}.pth".format(best_r2_epoch, corrcoef, r2, rmse)
            torch.save(self.model.state_dict(), model_path)
            print("model save in " + model_path)