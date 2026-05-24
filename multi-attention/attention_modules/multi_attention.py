import torch
import torch.nn as nn
from attention_modules.coordinate import CoordinateAttention
from attention_modules.eca import ECAAttention
from attention_modules.cbam import CBAM
from attention_modules.triplet import TripletAttention

class MultiAttentionBlock(nn.Module):
    def __init__(self, channels):
        super(MultiAttentionBlock, self).__init__()
        self.ca = CoordinateAttention(channels)
        self.eca = ECAAttention(channels)
        self.cbam = CBAM(channels)
        self.ta = TripletAttention()

        # 可训练融合权重
        self.alpha = nn.Parameter(torch.tensor(0.25))
        self.beta  = nn.Parameter(torch.tensor(0.25))
        self.gamma = nn.Parameter(torch.tensor(0.25))
        self.delta = nn.Parameter(torch.tensor(0.25))

    def forward(self, x):
        ca_out = self.ca(x)
        eca_out = self.eca(x)
        cbam_out = self.cbam(x)
        ta_out = self.ta(x)

        weights = torch.softmax(torch.stack([self.alpha, self.beta, self.gamma, self.delta]), dim=0)
        return weights[0] * ca_out + weights[1] * eca_out + weights[2] * cbam_out + weights[3] * ta_out