import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BraMT import BraMT


class Model(nn.Module):
    def __init__(self, param):
        super().__init__()
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

        self.head = nn.Sequential(
            nn.Linear(6*30*200, 512),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=4, dim_feedforward=2048, batch_first=True, activation=F.gelu, norm_first=True
        )
        self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False)
        self.classifier = nn.Linear(512, param.num_of_classes)

        # self.apply(_weights_init)

    def forward(self, x):
        bz, seq_len, ch_num, epoch_size = x.shape

        x = x.contiguous().view(bz * seq_len, ch_num, 30, 200)
        epoch_features = self.backbone(x)
        epoch_features = epoch_features.contiguous().view(bz, seq_len, ch_num*30*200)
        epoch_features = self.head(epoch_features)
        seq_features = self.sequence_encoder(epoch_features)
        out = self.classifier(seq_features)
        return out
