import logging
import os
import json
import torch
import numpy as np
from datetime import datetime


class ModelLogger:
    def __init__(self, log_dir="logs", params=None, experiment_name=None,
                 monitor_key: str = 'acc', monitor_mode: str = 'max'):
        """
        初始化模型日志记录器
        
        Args:
            log_dir: 日志目录
            params: 训练参数配置字典
            experiment_name: 实验名称，如果为None则使用时间戳
            monitor_key: 验证集用于挑选最佳 epoch 的指标名称，例如 'acc' / 'roc_auc'
            monitor_mode: 'max' 或 'min'，用于决定该指标是越大越好还是越小越好
        """
        self.log_dir = log_dir
        self.params = params
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        
        # 创建日志目录
        os.makedirs(self.experiment_dir, exist_ok=True)

        # 监控指标配置
        self.monitor_key = monitor_key
        self.monitor_mode = monitor_mode.lower()
        assert self.monitor_mode in ['max', 'min'], 'monitor_mode 必须是 max 或 min'
        self.best_val_value = float('-inf') if self.monitor_mode == 'max' else float('inf')
        self.best_epoch = None
        
        # 设置不同类型的日志文件
        self.setup_loggers()

        
    def setup_loggers(self):
        """设置不同类型的日志记录器"""
        
        # 1. 训练日志记录器
        self.train_logger = logging.getLogger(f'{self.experiment_name}_train')
        self.train_logger.setLevel(logging.INFO)
        train_handler = logging.FileHandler(
            os.path.join(self.experiment_dir, 'training.log')
        )
        train_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        train_handler.setFormatter(train_formatter)
        self.train_logger.addHandler(train_handler)
        
        
    
    def log_experiment_config(self, config):
        """记录实验配置"""
        config_file = os.path.join(self.experiment_dir, 'config.json')
        
        # 处理不能序列化的对象
        serializable_config = {}
        for key, value in config.items():
            try:
                json.dumps(value)
                serializable_config[key] = value
            except:
                serializable_config[key] = str(value)
        
        with open(config_file, 'w') as f:
            json.dump(serializable_config, f, indent=2)
        
        self.train_logger.info(f"Experiment config saved to {config_file}")
    
    def log_model_architecture(self, model):
        """记录模型架构"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 顶层子模块参数统计（按照参数名前缀的第一段分组,例如patch_embedding, encoder, project_out,classifier）
        top_level_counts = {}
        for name, p in model.named_parameters():
            top = name.split('.')[0]
            top_level_counts.setdefault(top, 0)
            top_level_counts[top] += p.numel()
        sorted_top = sorted(top_level_counts.items(), key=lambda x: x[1], reverse=True)

        depths=[int(x) for x in self.params.depths.split(',')]
        stage_types=[x for x in self.params.stage_types.split(',')]

        def locate_stage(depths,layer_idx) -> int:
            sum = 0
            for i in range(len(depths)):
                sum += depths[i]
                if layer_idx < sum : return i
        
        layer_level_counts = {}
        layer_total_params = 0
        for name, p in model.named_parameters():
            if name.split('.')[1] == 'encoder':
                layer_idx = int(name.split('.')[3])
                stage_type = stage_types[locate_stage(depths,layer_idx)]
                layer_level_counts.setdefault(stage_type,0)
                layer_level_counts[stage_type] += p.numel()
                layer_total_params += p.numel()
        sorted_layer = sorted(layer_level_counts.items(), key=lambda x: x[1], reverse=True)
        # 保存模型结构到文件 (在文件开头写入参数统计信息)
        model_file = os.path.join(self.experiment_dir, 'model_architecture.txt')
        with open(model_file, 'w') as f:
            f.write('# Parameter Summary\n')
            f.write(f'Total parameters     : {total_params:,}\n')
            f.write(f'Trainable parameters : {trainable_params:,}\n')
            f.write('\n# Top-level Module Breakdown\n')
            for name, cnt in sorted_top:
                ratio = (cnt / total_params) if total_params > 0 else 0.0
                f.write(f'{name:<20}: {cnt:>12,}  ({ratio:6.2%})\n')
            f.write('\n# Backbone-level Module Breakdown\n')
            for name, cnt in sorted_layer:
                ratio = (cnt / layer_total_params) if total_params > 0 else 0.0
                f.write(f'{name:<20}: {cnt:>12,}  ({ratio:6.2%})\n')
            
            f.write('\n# Model Architecture\n')
            f.write(str(model))
            f.write('\n')

    
    def log_training_step(self, epoch, step, loss, metrics=None, lr=None):
        """记录训练步骤"""
        log_msg = f"Epoch {epoch}, Step {step}, Loss: {loss:.6f}"
        
        if lr is not None:
            log_msg += f", LR: {lr:.6f}"
            
        if metrics:
            for key, value in metrics.items():
                log_msg += f", {key}: {value:.6f}"
        
        self.train_logger.info(log_msg)
        
    
    def log_validation_results(self, epoch, val_loss, val_metrics):
        """只在 training.log 中输出验证结果，并在刷新最优时标红该行。"""
        log_msg = f"Validation - Epoch {epoch}, Loss: {val_loss:.6f}"

        # 获取监控指标值
        monitor_value = None
        if isinstance(val_metrics, dict) and len(val_metrics) > 0:
            if self.monitor_key in val_metrics:
                monitor_value = val_metrics[self.monitor_key]
            else:
                # fallback: 选择第一个指标并更新 monitor_key
                first_key = next(iter(val_metrics.keys()))
                monitor_value = val_metrics[first_key]
                self.monitor_key = first_key
        
        for key, value in val_metrics.items():
            log_msg += f", {key}: {value:.6f}"

        improved = False
        if monitor_value is not None:
            if (self.monitor_mode == 'max' and monitor_value > self.best_val_value) or \
               (self.monitor_mode == 'min' and monitor_value < self.best_val_value):
                improved = True

        if improved:
            self.best_val_value = monitor_value
            self.best_epoch = epoch
            # 不使用颜色控制符, 仅追加标记
            plain_best = f"{log_msg}  <-- NEW VAL BEST ({self.monitor_key}={monitor_value:.6f})"
            self.train_logger.info(plain_best)
        else:
            self.train_logger.info(log_msg)
        
    
    
    def _convert_to_json_serializable(self, obj):
        """将对象转换为JSON可序列化的格式"""
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        else:
            return obj

    def log_binary_class_result(self, best_epoch, val_acc, val_pr_auc, val_roc_auc, test_acc, test_pr_auc, test_roc_auc, cm, class_names=None):
        result_path = os.path.join(self.experiment_dir, 'binary_classification_results.txt')
        with open(result_path, 'w') as f:
            f.write(f"Best Epoch: {best_epoch}\n")
            f.write(f"Best Val Acc: {val_acc:.5f}, pr_auc: {val_pr_auc:.5f}, roc_auc: {val_roc_auc:.5f}\n")
            f.write(f"Test Acc: {test_acc:.5f}, pr_auc: {test_pr_auc:.5f}, roc_auc: {test_roc_auc:.5f}\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm))
            f.write("\n\n")
            if class_names:
                f.write("Class Names: " + str(class_names) + "\n")
        self.train_logger.info(f"Binary classification results saved to {result_path}")
        self.train_logger.info("Experiment finalized")

    def log_multi_class_result(self, best_epoch, val_acc, val_kappa, val_F1, test_acc, test_kappa, test_F1, cm, class_names=None):
        result_path = os.path.join(self.experiment_dir, 'multi_classification_results.txt')
        with open(result_path, 'w') as f:
            f.write(f"Best Epoch: {best_epoch}\n")
            f.write(f"Best Val Acc: {val_acc:.5f}, Kappa: {val_kappa:.5f}, F1: {val_F1:.5f}\n")
            f.write(f"Test Acc: {test_acc:.5f}, Kappa: {test_kappa:.5f}, F1: {test_F1:.5f}\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm))
            f.write("\n\n")
            if class_names:
                f.write("Class Names: " + str(class_names) + "\n")
        self.train_logger.info(f"Multi-class classification results saved to {result_path}")
        self.train_logger.info("Experiment finalized")

# 使用示例
def create_logger(experiment_name=None, log_dir="logs"):
    """创建日志记录器的便捷函数"""
    return ModelLogger(log_dir=log_dir, experiment_name=experiment_name)