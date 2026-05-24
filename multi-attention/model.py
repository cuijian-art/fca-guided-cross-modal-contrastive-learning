import torch
import torch.nn as nn
import torchvision.models as models
from attention_modules.multi_attention import MultiAttentionBlock

class ResNet50_MultiAttention(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.conv1 = resnet.conv1
        self.bn1   = resnet.bn1
        self.relu  = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.att1 = MultiAttentionBlock(256)

        self.layer2 = resnet.layer2
        self.att2 = MultiAttentionBlock(512)

        self.layer3 = resnet.layer3
        self.att3 = MultiAttentionBlock(1024)

        self.layer4 = resnet.layer4
        self.att4 = MultiAttentionBlock(2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.att1(self.layer1(x))
        x = self.att2(self.layer2(x))
        x = self.att3(self.layer3(x))
        x = self.att4(self.layer4(x))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
