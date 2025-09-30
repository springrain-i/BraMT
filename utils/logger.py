import logging
import os
import json
import torch
import numpy as np
from datetime import datetime
import pandas as pd


class ModelLogger:
    def __init__(self, log_dir="logs", experiment_name=None):
        """
        初始化模型日志记录器
        
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称，如果为None则使用时间戳
        """
        self.log_dir = log_dir
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        
        # 创建日志目录
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 设置不同类型的日志文件
        self.setup_loggers()
        
        # 训练指标存储
        self.train_metrics = []
        self.val_metrics = []
        
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
        
        # 2. 模型结构日志记录器
        self.model_logger = logging.getLogger(f'{self.experiment_name}_model')
        self.model_logger.setLevel(logging.INFO)
        model_handler = logging.FileHandler(
            os.path.join(self.experiment_dir, 'model_info.log')
        )
        model_handler.setFormatter(train_formatter)
        self.model_logger.addHandler(model_handler)
        
        # 3. 错误日志记录器
        self.error_logger = logging.getLogger(f'{self.experiment_name}_error')
        self.error_logger.setLevel(logging.ERROR)
        error_handler = logging.FileHandler(
            os.path.join(self.experiment_dir, 'errors.log')
        )
        error_handler.setFormatter(train_formatter)
        self.error_logger.addHandler(error_handler)
    
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
        # 保存模型结构到文件
        model_file = os.path.join(self.experiment_dir, 'model_architecture.txt')
        with open(model_file, 'w') as f:
            f.write(str(model))
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.model_logger.info(f"Total parameters: {total_params:,}")
        self.model_logger.info(f"Trainable parameters: {trainable_params:,}")
        self.model_logger.info(f"Model architecture saved to {model_file}")
    
    def log_training_step(self, epoch, step, loss, metrics=None, lr=None):
        """记录训练步骤"""
        log_msg = f"Epoch {epoch}, Step {step}, Loss: {loss:.6f}"
        
        if lr is not None:
            log_msg += f", LR: {lr:.6f}"
            
        if metrics:
            for key, value in metrics.items():
                log_msg += f", {key}: {value:.6f}"
        
        self.train_logger.info(log_msg)
        
        # 保存到内存中用于后续分析
        step_data = {
            'epoch': epoch,
            'step': step,
            'loss': float(loss),
            'lr': float(lr) if lr is not None else None,
            'timestamp': datetime.now().isoformat()
        }
        if metrics:
            # 确保metrics中的值是JSON可序列化的
            serializable_metrics = self._convert_to_json_serializable(metrics)
            step_data.update(serializable_metrics)
        
        self.train_metrics.append(step_data)
    
    def log_validation_results(self, epoch, val_loss, val_metrics):
        """记录验证结果"""
        log_msg = f"Validation - Epoch {epoch}, Loss: {val_loss:.6f}"
        
        for key, value in val_metrics.items():
            log_msg += f", {key}: {value:.6f}"
        
        self.train_logger.info(log_msg)
        
        # 保存验证指标
        val_data = {
            'epoch': epoch,
            'val_loss': float(val_loss),
            'timestamp': datetime.now().isoformat()
        }
        # 确保val_metrics中的值是JSON可序列化的
        serializable_val_metrics = self._convert_to_json_serializable(val_metrics)
        val_data.update(serializable_val_metrics)
        self.val_metrics.append(val_data)
    
    def save_metrics_to_csv(self):
        """将指标保存为CSV文件"""
        if self.train_metrics:
            train_df = pd.DataFrame(self.train_metrics)
            train_df.to_csv(
                os.path.join(self.experiment_dir, 'train_metrics.csv'), 
                index=False
            )
        
        if self.val_metrics:
            val_df = pd.DataFrame(self.val_metrics)
            val_df.to_csv(
                os.path.join(self.experiment_dir, 'val_metrics.csv'), 
                index=False
            )
    
    def log_confusion_matrix(self, cm, class_names=None):
        """记录混淆矩阵"""
        cm_file = os.path.join(self.experiment_dir, 'confusion_matrix.txt')
        
        with open(cm_file, 'w') as f:
            f.write("Confusion Matrix:\n")
            f.write(str(cm))
            f.write("\n\n")
            
            if class_names:
                f.write("Class Names: " + str(class_names) + "\n")
        
        self.train_logger.info(f"Confusion matrix saved to {cm_file}")
    
    def log_error(self, error_msg, exception=None):
        """记录错误信息"""
        self.error_logger.error(error_msg)
        if exception:
            self.error_logger.exception(exception)
    
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

    def finalize_experiment(self):
        """实验结束时的清理工作"""
        # 保存所有指标到CSV
        self.save_metrics_to_csv()
        
        # 创建实验总结
        summary = {
            'experiment_name': self.experiment_name,
            'total_train_steps': len(self.train_metrics),
            'total_val_steps': len(self.val_metrics),
            'experiment_duration': datetime.now().isoformat(),
        }
        
        if self.val_metrics:
            best_metric = max(self.val_metrics, key=lambda x: x.get('val_acc', 0))
            summary['best_validation_metrics'] = self._convert_to_json_serializable(best_metric)
        
        # 确保summary也是JSON可序列化的
        summary = self._convert_to_json_serializable(summary)
        
        summary_file = os.path.join(self.experiment_dir, 'experiment_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.train_logger.info("Experiment completed and logged successfully")


# 使用示例
def create_logger(experiment_name=None, log_dir="logs"):
    """创建日志记录器的便捷函数"""
    return ModelLogger(log_dir=log_dir, experiment_name=experiment_name)