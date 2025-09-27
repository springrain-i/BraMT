import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


class LogAnalyzer:
    """分析和可视化训练日志的工具类"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
    
    def list_experiments(self):
        """列出所有实验"""
        experiments = [d for d in os.listdir(self.log_dir) 
                      if os.path.isdir(os.path.join(self.log_dir, d))]
        return experiments
    
    def load_experiment_data(self, experiment_name):
        """加载特定实验的数据"""
        exp_dir = os.path.join(self.log_dir, experiment_name)
        
        data = {}
        
        # 加载配置
        config_file = os.path.join(exp_dir, 'config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                data['config'] = json.load(f)
        
        # 加载训练指标
        train_csv = os.path.join(exp_dir, 'train_metrics.csv')
        if os.path.exists(train_csv):
            data['train_metrics'] = pd.read_csv(train_csv)
        
        # 加载验证指标
        val_csv = os.path.join(exp_dir, 'val_metrics.csv')
        if os.path.exists(val_csv):
            data['val_metrics'] = pd.read_csv(val_csv)
        
        # 加载实验总结
        summary_file = os.path.join(exp_dir, 'experiment_summary.json')
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                data['summary'] = json.load(f)
        
        return data
    
    def plot_training_curves(self, experiment_name, save_plots=True):
        """绘制训练曲线"""
        data = self.load_experiment_data(experiment_name)
        
        if 'val_metrics' not in data:
            print(f"No validation metrics found for {experiment_name}")
            return
        
        val_df = data['val_metrics']
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Curves - {experiment_name}', fontsize=16)
        
        # 绘制损失曲线
        if 'val_loss' in val_df.columns:
            axes[0, 0].plot(val_df['epoch'], val_df['val_loss'])
            axes[0, 0].set_title('Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
        
        # 绘制准确率曲线
        if 'val_acc' in val_df.columns:
            axes[0, 1].plot(val_df['epoch'], val_df['val_acc'])
            axes[0, 1].set_title('Validation Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].grid(True)
        
        # 绘制ROC AUC曲线
        if 'val_roc_auc' in val_df.columns:
            axes[1, 0].plot(val_df['epoch'], val_df['val_roc_auc'])
            axes[1, 0].set_title('Validation ROC AUC')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('ROC AUC')
            axes[1, 0].grid(True)
        
        # 绘制PR AUC曲线
        if 'val_pr_auc' in val_df.columns:
            axes[1, 1].plot(val_df['epoch'], val_df['val_pr_auc'])
            axes[1, 1].set_title('Validation PR AUC')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('PR AUC')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.log_dir, experiment_name, 'training_curves.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {plot_path}")
        
        plt.show()
    
    def compare_experiments(self, experiment_names, metric='val_acc'):
        """比较多个实验的指标"""
        plt.figure(figsize=(12, 8))
        
        for exp_name in experiment_names:
            data = self.load_experiment_data(exp_name)
            if 'val_metrics' in data and metric in data['val_metrics'].columns:
                val_df = data['val_metrics']
                plt.plot(val_df['epoch'], val_df[metric], label=exp_name, marker='o')
        
        plt.title(f'Comparison of {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def get_best_experiments(self, metric='val_acc', top_k=5):
        """获取性能最好的实验"""
        experiments = self.list_experiments()
        results = []
        
        for exp_name in experiments:
            data = self.load_experiment_data(exp_name)
            if 'summary' in data and 'best_validation_metrics' in data['summary']:
                best_metrics = data['summary']['best_validation_metrics']
                if metric in best_metrics:
                    results.append({
                        'experiment': exp_name,
                        'best_' + metric: best_metrics[metric],
                        'best_epoch': best_metrics.get('epoch', 'Unknown')
                    })
        
        # 按指标排序
        results.sort(key=lambda x: x['best_' + metric], reverse=True)
        return results[:top_k]
    
    def print_experiment_summary(self, experiment_name):
        """打印实验总结"""
        data = self.load_experiment_data(experiment_name)
        
        print(f"\n{'='*50}")
        print(f"Experiment Summary: {experiment_name}")
        print(f"{'='*50}")
        
        if 'config' in data:
            print("\nConfiguration:")
            for key, value in data['config'].items():
                print(f"  {key}: {value}")
        
        if 'summary' in data:
            summary = data['summary']
            print(f"\nTraining Summary:")
            print(f"  Total training steps: {summary.get('total_train_steps', 'Unknown')}")
            print(f"  Total validation steps: {summary.get('total_val_steps', 'Unknown')}")
            
            if 'best_validation_metrics' in summary:
                best = summary['best_validation_metrics']
                print(f"\nBest Validation Results:")
                for key, value in best.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.5f}")
                    else:
                        print(f"  {key}: {value}")


# 使用示例
def analyze_logs():
    """分析日志的便捷函数"""
    analyzer = LogAnalyzer()
    
    # 列出所有实验
    experiments = analyzer.list_experiments()
    print("Available experiments:")
    for i, exp in enumerate(experiments):
        print(f"  {i+1}. {exp}")
    
    # 获取最佳实验
    print("\nTop 5 experiments by validation accuracy:")
    best_exps = analyzer.get_best_experiments('val_acc', top_k=5)
    for i, exp in enumerate(best_exps):
        print(f"  {i+1}. {exp['experiment']} - Acc: {exp['best_val_acc']:.5f}")
    
    return analyzer


if __name__ == "__main__":
    analyzer = analyze_logs()
    
    # 如果有实验，显示第一个实验的详细信息
    experiments = analyzer.list_experiments()
    if experiments:
        print(f"\nDetailed analysis of: {experiments[0]}")
        analyzer.print_experiment_summary(experiments[0])
        analyzer.plot_training_curves(experiments[0])