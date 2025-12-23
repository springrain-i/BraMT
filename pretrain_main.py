import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.pretraining_dataset import PretrainingDataset
from models.BraMT import BraMT
from pretrain_trainer import Trainer


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    """Robust boolean parser for argparse.
    Accepts: true/false, t/f, yes/no, y/n, 1/0 (case-insensitive).
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    v = str(v).strip().lower()
    if v in ("true", "t", "yes", "y", "1"):  # truthy
        return True
    if v in ("false", "f", "no", "n", "0"):  # falsy
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

def main():
    parser = argparse.ArgumentParser(description='EEG Foundation Model')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 1)')
    parser.add_argument('--parallel', type=bool, default=False, help='parallel')

    parser.add_argument('--epochs', type=int, default=40, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight_decay')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR',
                        help='lr_scheduler: CosineAnnealingLR, ExponentialLR, StepLR, MultiStepLR, CyclicLR')

    # parser.add_argument('--project_mode', type=str, default='cnn', help='project_mode')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    parser.add_argument('--in_dim', type=int, default=200, help='in_dim')
    parser.add_argument('--out_dim', type=int, default=200, help='out_dim')
    parser.add_argument('--d_model', type=int, default=200, help='d_model')
    parser.add_argument('--dim_feedforward', type=int, default=800, help='dim_feedforward')
    
    parser.add_argument('--seq_len', type=int, default=30, help='seq_len')
    parser.add_argument('--nhead', type=int, default=8, help='nhead')
    parser.add_argument('--need_mask', type=bool, default=True, help='need_mask')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask_ratio')

    parser.add_argument('--dataset_dir', type=str, default='pretrain_dataset',
                        help='dataset_dir')
    parser.add_argument('--model_dir',   type=str,   default='model_dir', help='model_dir')
    parser.add_argument('--log_dir', type=str, default='./logs', help='the destination of log')

    # wandb logging settings
    parser.add_argument('--use_wandb', type=bool, default=True, help='enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='eeg-pretrain', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity/org (optional)')
    parser.add_argument('--wandb_mode', type=str, default='online', help='wandb mode: online/offline/disabled')
    parser.add_argument('--wandb_dir', type=str, default='./wandb', help='local dir to store wandb files')
    parser.add_argument('--wandb_api_key', type=str, default="800a257e9949b9633a2fd6bfda872cb92089b27c", help='wandb API key (optional, for programmatic login)')

    """############ Hybrid model settings ############"""
    parser.add_argument('--stage_types', type=str, default='mamba,attn', help='stage_types')
    parser.add_argument('--depths', type=str, default='6,6', help='depths')
    parser.add_argument('--axis_order', type=str2bool, default=True, help='')
    parser.add_argument('--mamba_global', type=str2bool, default=False, help='' \
    'whether to use global context in Mamba')
    parser.add_argument('--d_state', type=int, default=16, help='d_state for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='d_conv for Mamba')
    parser.add_argument('--expand', type=int, default=2, help='expand for Mamba')
    parser.add_argument('--conv_bias', type=str2bool, default=True, help='conv_bias for Mamba')
    params = parser.parse_args()
    print(params)
    setup_seed(params.seed)
    pretrained_dataset = PretrainingDataset(dataset_dir=params.dataset_dir)
    print(len(pretrained_dataset))
    data_loader = DataLoader(
        pretrained_dataset,
        batch_size=params.batch_size,
        num_workers=8,
        shuffle=True,
    )
    model = BraMT(
        params.in_dim, params.out_dim, params.d_model, params.dim_feedforward, params.seq_len,  params.nhead,
        depths=[int(x) for x in params.depths.split(',')],
        stage_types=[x for x in params.stage_types.split(',')],
        axis_order=params.axis_order,
        mamba_global=params.mamba_global,
        d_state=params.d_state,
        d_conv=params.d_conv,
        expand=params.expand,
        conv_bias=params.conv_bias,
    )
    trainer = Trainer(params, data_loader, model)
    trainer.train()
    pretrained_dataset.db.close()


if __name__ == '__main__':
    main()
