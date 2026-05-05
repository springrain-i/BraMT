import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from models.BraMT import BraMT


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.backbone = BraMT(
            in_dim=200, out_dim=200, d_model=200,
            dim_feedforward=800, seq_len=30,
            n_layer=12, nhead=8,

            # Mamba specific parameters
            depths=[int(x) for x in param.depths.split(',')],
            stage_types=[x for x in param.stage_types.split(',')],
            d_state= param.d_state, d_conv=param.d_conv, expand=param.expand, conv_bias=param.conv_bias
        )
        if param.use_pretrained_weights:
            map_location = torch.device(f'cuda:{param.cuda}')
            self.backbone.load_state_dict(torch.load(param.foundation_dir, map_location=map_location))
        self.backbone.proj_out = nn.Identity()
        if param.classifier == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(62 * 1 * 200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(62 * 1 * 200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )
        elif param.classifier == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(62 * 1 * 200, 4 * 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(4 * 200, 200),
                nn.ELU(),
                nn.Dropout(param.dropout),
                nn.Linear(200, param.num_of_classes),
            )

    def forward(self, x):
        # x = x / 100
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        feats = feats.contiguous().view(bz, ch_num*seq_len*200)
        out = self.classifier(feats)
        return out
