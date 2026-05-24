import torch
import torch.nn as nn

class TripletAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(TripletAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def spatial_gate(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3)  # B, H, C, W
        x_perm2 = x.permute(0, 3, 2, 1)  # B, W, H, C

        out1 = x * self.spatial_gate(x)
        out2 = x * self.spatial_gate(x_perm1).permute(0, 2, 1, 3)
        out3 = x * self.spatial_gate(x_perm2).permute(0, 3, 2, 1)

        return (out1 + out2 + out3) / 3
