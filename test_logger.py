#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
from utils.logger import ModelLogger

# 创建一个简单的测试模型
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.depths = [2, 2, 2, 2]
        self.stage_types = ['mamba', 'mamba', 'attn', 'attn']
        self.hybrid_mode = 'mamba_first'
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

def test_logger():
    print("Testing logger functionality...")
    
    # 创建测试模型
    model = TestModel()
    
    # 创建模拟参数
    class MockParams:
        def __init__(self):
            self.downstream_dataset = "SHU-MI"
            self.lr = 0.001
            self.epochs = 50
    
    params = MockParams()
    
    # 测试实验名称生成
    from finetune_trainer import Trainer
    from finetune_evaluator import Evaluator
    
    # 创建模拟数据加载器
    data_loader = {
        'train': [],
        'val': [],
        'test': []
    }
    
    try:
        # 这里会测试实验名称生成
        trainer = Trainer(params, data_loader, model)
        print(f"Generated experiment name: {trainer.logger.experiment_name}")
        
        # 测试日志记录
        trainer.logger.log_training_step(
            epoch=1, 
            step=100, 
            loss=np.float32(0.5), 
            lr=np.float64(0.001),
            metrics={
                'acc': np.float32(0.85),
                'f1': np.float64(0.82)
            }
        )
        
        trainer.logger.log_validation_results(
            epoch=1,
            val_loss=np.float32(0.4),
            val_metrics={
                'val_acc': np.float32(0.87),
                'val_f1': np.float64(0.84)
            }
        )
        
        # 测试实验完成
        trainer.logger.finalize_experiment()
        
        print("✅ Logger test passed!")
        print(f"📁 Log files saved in: {trainer.logger.experiment_dir}")
        
    except Exception as e:
        print(f"❌ Logger test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_logger()