import os
import sys
import json
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from tqdm import tqdm
import math


# ====================== 1. 实现FCA（频率通道注意力）模块 ======================
class FCA(nn.Module):
    """
    Frequency Channel Attention (FCA) 模块
    核心逻辑：通过FFT将特征转换到频域，分离高低频特征，计算通道注意力权重后融合
    """

    def __init__(self, channel, reduction=16):
        super(FCA, self).__init__()
        # 全局平均池化，将每个通道的特征图压缩为1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 注意力权重计算的全连接层
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        # 高频分量的权重参数（可学习）
        self.gamma = nn.Parameter(torch.zeros(1))
        self.register_buffer("weight", torch.ones(1))

    def forward(self, x):
        b, c, h, w = x.size()

        # 步骤1：通过FFT将空间域特征转换到频域
        fft = torch.fft.fft2(x, dim=(-2, -1))
        fft_abs = torch.abs(fft)  # 幅度谱
        fft_phase = torch.angle(fft)  # 相位谱

        # 步骤2：分离高低频（通过中心掩码实现）
        mask = torch.ones_like(fft_abs)
        cx, cy = h // 2, w // 2
        r = min(cx, cy) // 8  # 低频区域半径，可根据需求调整
        mask[:, :, cx - r:cx + r, cy - r:cy + r] = 0  # 低频区域置0，保留高频

        # 高频分量
        fft_high = fft_abs * mask
        # 低频分量（原幅度谱 - 高频）
        fft_low = fft_abs - fft_high

        # 步骤3：将频域特征转回空间域
        fft_high_complex = torch.polar(fft_high, fft_phase)
        fft_low_complex = torch.polar(fft_low, fft_phase)
        x_high = torch.fft.ifft2(fft_high_complex).real
        x_low = torch.fft.ifft2(fft_low_complex).real

        # 步骤4：计算通道注意力权重（基于低频特征，因为低频包含主要语义信息）
        y = self.avg_pool(x_low).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        # 步骤5：融合高低频并应用注意力权重
        out = x_low * y + self.gamma * x_high
        return out + x  # 残差连接，保证梯度传播


# ====================== 2. 修改ResNet50，嵌入FCA模块 ======================
def resnet50_fca(pretrained=True, num_classes=1000):
    """
    基于官方ResNet50，在每个Bottleneck的输出后添加FCA模块
    """
    # 加载官方ResNet50预训练模型
    net = models.resnet50(pretrained=pretrained)

    # 遍历ResNet的layer，为每个Bottleneck添加FCA
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(net, layer_name)
        for i, block in enumerate(layer):
            # 获取Bottleneck的输出通道数
            out_channels = block.bn3.num_features
            # 在Bottleneck的最后添加FCA模块
            block.add_module('fca', FCA(out_channels))

    # 修改最后全连接层（如果需要自定义类别数）
    if num_classes != 1000:
        in_channel = net.fc.in_features
        net.fc = nn.Linear(in_channel, num_classes)

    return net


# ====================== 3. 原有训练逻辑（仅修改模型初始化部分） ======================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.223, 0.709, 0.677], [0.150, 0.088, 0.167])]),
        "val": transforms.Compose([transforms.Resize(224),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.223, 0.709, 0.677], [0.150, 0.088, 0.167])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "RIBEN")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # ====================== 关键修改：使用带FCA的ResNet50 ======================
    # 替换原有模型初始化，加载预训练权重并添加FCA模块
    net = resnet50_fca(pretrained=True, num_classes=5)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 100
    best_acc = 0.0
    save_path = './ResNetFCA_Single.pth'  # 修改保存文件名，区分带FCA的模型
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # 训练阶段
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # 统计损失值
            running_loss += loss.item()

            # 统计准确率
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        train_accuracy = correct / total

        # 验证阶段
        net.eval()
        val_loss = 0.0
        acc = 0.0  # 累积准确数量 / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                val_loss += loss.item()

                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accurate = acc / val_num
        val_loss_avg = val_loss / len(validate_loader)
        print('[epoch %d] train_loss: %.3f  train_accuracy: %.3f  val_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, train_accuracy, val_loss_avg, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()