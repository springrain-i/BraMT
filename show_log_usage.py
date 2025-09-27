#!/usr/bin/env python3
"""
日志功能使用示例

这个脚本展示了如何使用新的日志功能
"""

import sys
import os
sys.path.append('/home/springrain-i/BaseLine')

from utils.logger import ModelLogger
from utils.log_analyzer import LogAnalyzer


def example_usage():
    """展示日志功能的使用方法"""
    
    print("=== 日志功能使用示例 ===\n")
    
    # 1. 创建日志记录器（在训练脚本中自动创建）
    print("1. 日志记录器已集成到 finetune_trainer.py 中")
    print("   训练时会自动创建日志文件\n")
    
    # 2. 日志文件结构
    print("2. 日志文件结构:")
    print("   logs/")
    print("   └── {dataset}_{model_name}_{timestamp}/")
    print("       ├── config.json              # 实验配置")
    print("       ├── model_architecture.txt   # 模型结构")
    print("       ├── training.log             # 训练日志")
    print("       ├── model_info.log           # 模型信息")
    print("       ├── errors.log               # 错误日志")
    print("       ├── train_metrics.csv        # 训练指标")
    print("       ├── val_metrics.csv          # 验证指标")
    print("       ├── confusion_matrix.txt     # 混淆矩阵")
    print("       ├── best_model.pth           # 最佳模型")
    print("       ├── latest_checkpoint.pth    # 最新检查点")
    print("       └── experiment_summary.json  # 实验总结\n")
    
    # 3. 查看现有实验
    print("3. 查看现有实验:")
    analyzer = LogAnalyzer()
    experiments = analyzer.list_experiments()
    
    if experiments:
        print("   现有实验:")
        for i, exp in enumerate(experiments):
            print(f"     {i+1}. {exp}")
        
        # 显示最佳实验
        try:
            best_exps = analyzer.get_best_experiments('val_acc', top_k=3)
            if best_exps:
                print("\n   最佳实验 (按验证准确率):")
                for i, exp in enumerate(best_exps):
                    print(f"     {i+1}. {exp['experiment']} - Acc: {exp['best_val_acc']:.5f}")
        except:
            print("     还没有足够的实验数据")
    else:
        print("   还没有实验记录")
    
    print("\n4. 运行训练命令:")
    print("   python finetune_main.py")
    print("   训练完成后，日志会自动保存到 logs/ 文件夹中\n")
    
    print("5. 分析日志:")
    print("   python utils/log_analyzer.py")
    print("   或者在代码中使用:")
    print("   ```python")
    print("   from utils.log_analyzer import LogAnalyzer")
    print("   analyzer = LogAnalyzer()")
    print("   analyzer.plot_training_curves('experiment_name')")
    print("   ```\n")


def check_log_directory():
    """检查日志目录状态"""
    log_dir = "/home/springrain-i/BaseLine/logs"
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"创建日志目录: {log_dir}")
    else:
        print(f"日志目录已存在: {log_dir}")
    
    # 检查现有实验
    if os.path.exists(log_dir):
        experiments = [d for d in os.listdir(log_dir) 
                      if os.path.isdir(os.path.join(log_dir, d))]
        print(f"现有实验数量: {len(experiments)}")
        for exp in experiments:
            print(f"  - {exp}")


if __name__ == "__main__":
    print("检查日志系统状态...\n")
    check_log_directory()
    print("\n" + "="*60 + "\n")
    example_usage()