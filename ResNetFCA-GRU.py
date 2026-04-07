import os
import sys
import json
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.models as models


# ====================== 1. 定义FCA模块（复用并适配） ======================
class FCA(nn.Module):
    """Frequency Channel Attention 模块（复用核心逻辑）"""

    def __init__(self, channel, reduction=16):
        super(FCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.register_buffer("weight", torch.ones(1))

    def forward(self, x):
        b, c, h, w = x.size()
        # 频域转换与高低频分离
        fft = torch.fft.fft2(x, dim=(-2, -1))
        fft_abs = torch.abs(fft)
        fft_phase = torch.angle(fft)

        mask = torch.ones_like(fft_abs)
        cx, cy = h // 2, w // 2
        r = min(cx, cy) // 8
        mask[:, :, cx - r:cx + r, cy - r:cy + r] = 0

        fft_high = fft_abs * mask
        fft_low = fft_abs - fft_high

        # 逆变换回空间域
        fft_high_complex = torch.polar(fft_high, fft_phase)
        fft_low_complex = torch.polar(fft_low, fft_phase)
        x_high = torch.fft.ifft2(fft_high_complex).real
        x_low = torch.fft.ifft2(fft_low_complex).real

        # 通道注意力计算
        y = self.avg_pool(x_low).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        out = x_low * y + self.gamma * x_high
        return out + x


# ====================== 2. 定义ResNet50-FCA图像分支 ======================
def resnet50_fca(pretrained=True, num_features=512):
    """ResNet50-FCA：输出图像特征向量（维度num_features）"""
    net = models.resnet50(pretrained=pretrained)
    # 为每个Bottleneck添加FCA
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(net, layer_name)
        for i, block in enumerate(layer):
            out_channels = block.bn3.num_features
            block.add_module('fca', FCA(out_channels))
    # 替换最后全连接层，输出指定维度的特征（适配GRU分支维度）
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, num_features)
    return net


# ====================== 3. 定义GRU时序分支 ======================
class GRUBranch(nn.Module):
    """GRU时序分支：处理.xlsx中的时序信号，输出固定维度特征"""

    def __init__(self, input_dim, hidden_dim=256, num_layers=2, num_features=512, dropout=0.1):
        super(GRUBranch, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,  # 时序信号的特征维度（xlsx中每行的特征数）
            hidden_size=hidden_dim,  # GRU隐藏层维度
            num_layers=num_layers,  # GRU层数
            batch_first=True,  # 输入格式：[batch, seq_len, input_dim]
            bidirectional=True,  # 双向GRU，增强时序特征捕捉
            dropout=dropout if num_layers > 1 else 0
        )
        # 全连接层：将GRU输出映射到与图像分支相同的特征维度
        self.fc = nn.Linear(hidden_dim * 2, num_features)  # 双向GRU输出维度=2*hidden_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        gru_out, _ = self.gru(x)  # gru_out: [batch, seq_len, 2*hidden_dim]
        # 取最后一个时间步的输出（捕捉时序长期依赖）
        seq_feature = gru_out[:, -1, :]  # [batch, 2*hidden_dim]
        seq_feature = self.dropout(seq_feature)
        # 映射到目标特征维度
        seq_feature = self.fc(seq_feature)  # [batch, num_features]
        return seq_feature


# ====================== 4. 定义ResNet-FCA+GRU融合模型 ======================
class ResNetFCAGRU(nn.Module):
    """融合模型：图像分支+时序分支+加法融合+分类层"""

    def __init__(self, img_num_features=512, seq_input_dim=10, seq_hidden_dim=256, num_classes=10):
        super(ResNetFCAGRU, self).__init__()
        # 图像分支
        self.img_branch = resnet50_fca(pretrained=True, num_features=img_num_features)
        # 时序分支
        self.seq_branch = GRUBranch(
            input_dim=seq_input_dim,
            hidden_dim=seq_hidden_dim,
            num_features=img_num_features  # 与图像分支特征维度一致
        )
        # 分类层：融合特征→分类结果
        self.classifier = nn.Linear(img_num_features, num_classes)

    def forward(self, img, seq):
        # 提取图像特征：[batch, img_num_features]
        img_feat = self.img_branch(img)
        # 提取时序特征：[batch, img_num_features]
        seq_feat = self.seq_branch(seq)
        # 加法融合
        fusion_feat = img_feat + seq_feat
        # 分类预测
        logits = self.classifier(fusion_feat)
        # 返回预测值+各分支特征（用于计算跨数据损失）
        return logits, img_feat, seq_feat


# ====================== 5. 定义跨数据损失函数 ======================
class CrossDataLoss(nn.Module):
    """跨数据损失：交叉熵+特征对齐损失（余弦相似度）+对比损失"""

    def __init__(self, alpha=1.0, beta=0.1, gamma=0.1, margin=1.0):
        super(CrossDataLoss, self).__init__()
        self.alpha = alpha  # 图像分支交叉熵权重
        self.beta = beta  # 特征对齐损失权重
        self.gamma = gamma  # 对比损失权重
        self.margin = margin  # 对比损失距离阈值
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels, img_feat, seq_feat):
        # 1. 交叉熵损失（分类损失）
        ce_loss = self.ce_loss(logits, labels)

        # 2. 特征对齐损失（余弦相似度：1 - cos_sim，越小越对齐）
        cos_sim = F.cosine_similarity(img_feat, seq_feat, dim=1)
        align_loss = 1 - cos_sim.mean()

        # 3. 对比损失
        batch_size = img_feat.size(0)
        contrast_loss = 0.0
        # 遍历批次内样本，计算同类/异类距离
        for i in range(batch_size):
            # 欧式距离
            dist = F.pairwise_distance(img_feat[i].unsqueeze(0), seq_feat[i].unsqueeze(0))
            # 同类样本：拉近距离；异类样本：拉远至margin
            for j in range(batch_size):
                if labels[i] == labels[j]:
                    contrast_loss += dist ** 2
                else:
                    contrast_loss += max(0, self.margin - dist) ** 2
        contrast_loss /= (batch_size ** 2)

        # 总损失
        total_loss = self.alpha * ce_loss + self.beta * align_loss + self.gamma * contrast_loss
        return total_loss, ce_loss, align_loss, contrast_loss


# ====================== 6. 自定义数据集：图像+时序数据加载 ======================
class ImageSeqDataset(Dataset):
    """
    自定义数据集：匹配图像样本和时序.xlsx样本
    要求：图像文件名 = 时序.xlsx中的样本ID（如图像1.jpg对应xlsx中ID=1的行）
    """

    def __init__(self, img_root, seq_xlsx_path, transform=None, seq_input_dim=4500):
        self.img_root = img_root
        self.seq_xlsx_path = seq_xlsx_path
        self.transform = transform
        self.seq_input_dim = seq_input_dim

        # 1. 加载图像数据
        self.img_dataset = datasets.ImageFolder(root=img_root)
        self.img_paths = [x[0] for x in self.img_dataset.imgs]  # 所有图像路径
        self.img_labels = [x[1] for x in self.img_dataset.imgs]  # 所有图像标签
        # 提取图像ID（文件名，不含后缀）：如"1.jpg"→"1"
        self.img_ids = [os.path.splitext(os.path.basename(p))[0] for p in self.img_paths]

        # 2. 加载时序.xlsx数据
        self.seq_df = pd.read_excel(seq_xlsx_path)
        # 要求xlsx第一列为样本ID（与图像ID匹配），后续列为时序特征
        self.seq_ids = self.seq_df.iloc[:, 0].astype(str).tolist()
        self.seq_data = self.seq_df.iloc[:, 1:self.seq_input_dim + 1].values  # 时序特征

        # 3. 匹配图像和时序数据（仅保留双方都有的样本）
        self.valid_indices = []
        self.valid_seq_data = []
        for idx, img_id in enumerate(self.img_ids):
            if img_id in self.seq_ids:
                seq_idx = self.seq_ids.index(img_id)
                self.valid_indices.append(idx)
                self.valid_seq_data.append(self.seq_data[seq_idx])

        # 转换为tensor
        self.valid_seq_data = torch.tensor(self.valid_seq_data, dtype=torch.float32)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 1. 读取图像
        img_idx = self.valid_indices[idx]
        img_path = self.img_paths[img_idx]
        img_label = self.img_labels[img_idx]
        img = self.img_dataset.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)

        # 2. 读取对应时序数据（reshape为[seq_len, input_dim]，这里seq_len=1，可根据实际调整）
        seq = self.valid_seq_data[idx].reshape(1, -1)  # [1, seq_input_dim]

        return img, seq, img_label


# ====================== 7. 主训练函数 ======================
def main():
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    # ---------------------- 数据配置 ----------------------
    # 1. 图像预处理
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.223, 0.709, 0.677], [0.150, 0.088, 0.167])]),
        "val": transforms.Compose([transforms.Resize(224),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.223, 0.709, 0.677], [0.150, 0.088, 0.167])])}

    # 2. 数据根路径（需根据自己的路径调整）
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    img_root = os.path.join(data_root, "data_set", "RIBEN")  # 图像数据根路径（与之前一致）
    # 时序.xlsx文件路径（重点！看下方说明）
    train_seq_xlsx = os.path.join(data_root, "data_set", "RIBEN", "train", "seq_data.xlsx")
    val_seq_xlsx = os.path.join(data_root, "data_set", "RIBEN", "val", "seq_data.xlsx")

    # 3. 检查文件存在性
    assert os.path.exists(img_root), f"{img_root} path does not exist."
    assert os.path.exists(train_seq_xlsx), f"{train_seq_xlsx} does not exist."
    assert os.path.exists(val_seq_xlsx), f"{val_seq_xlsx} does not exist."

    # 4. 加载训练/验证数据集
    train_dataset = ImageSeqDataset(
        img_root=os.path.join(img_root, "train"),
        seq_xlsx_path=train_seq_xlsx,
        transform=data_transform["train"],
        seq_input_dim=4000  # 时序特征维度（xlsx中每行的特征数）
    )
    val_dataset = ImageSeqDataset(
        img_root=os.path.join(img_root, "val"),
        seq_xlsx_path=val_seq_xlsx,
        transform=data_transform["val"],
        seq_input_dim=4000
    )
    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print(f"using {train_num} samples for training, {val_num} samples for validation.")

    # 5. 数据加载器
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    # ---------------------- 模型配置 ----------------------
    # 模型参数：确保图像/时序分支特征维度一致
    img_num_features = 512  # 图像分支输出维度
    seq_input_dim = 4000  # 时序特征维度（与xlsx一致）
    seq_hidden_dim = 256  # GRU隐藏层维度
    num_classes = 10  # 分类类别数

    # 初始化融合模型
    model = ResNetFCAGRU(
        img_num_features=img_num_features,
        seq_input_dim=seq_input_dim,
        seq_hidden_dim=seq_hidden_dim,
        num_classes=num_classes
    ).to(device)

    # ---------------------- 损失与优化器 ----------------------
    # 跨数据损失函数
    loss_fn = CrossDataLoss(alpha=1.0, beta=0.1, gamma=0.1, margin=1.0)
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # 学习率调度器（可选）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # ---------------------- 训练配置 ----------------------
    epochs = 100
    best_acc = 0.0
    save_path = './ResNetFCA_GRU.pth'

    # ---------------------- 训练循环 ----------------------
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        total_ce_loss = 0.0
        total_align_loss = 0.0
        total_contrast_loss = 0.0
        correct = 0
        total = 0
        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, (imgs, seqs, labels) in enumerate(train_bar):
            imgs = imgs.to(device)
            seqs = seqs.to(device)
            labels = labels.to(device)

            # 前向传播
            optimizer.zero_grad()
            logits, img_feat, seq_feat = model(imgs, seqs)

            # 计算损失
            loss, ce_loss, align_loss, contrast_loss = loss_fn(logits, labels, img_feat, seq_feat)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_align_loss += align_loss.item()
            total_contrast_loss += contrast_loss.item()

            # 准确率
            _, preds = torch.max(logits, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss.item():.3f} acc:{correct / total:.3f}"

        # 训练指标
        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        avg_ce_loss = total_ce_loss / len(train_loader)
        avg_align_loss = total_align_loss / len(train_loader)
        avg_contrast_loss = total_contrast_loss / len(train_loader)

        # 学习率更新
        scheduler.step()

        # ---------------------- 验证阶段 ----------------------
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for imgs, seqs, labels in val_bar:
                imgs = imgs.to(device)
                seqs = seqs.to(device)
                labels = labels.to(device)

                logits, img_feat, seq_feat = model(imgs, seqs)
                loss, _, _, _ = loss_fn(logits, labels, img_feat, seq_feat)
                val_loss += loss.item()

                _, preds = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

                val_bar.desc = f"valid epoch[{epoch + 1}/{epochs}]"

        # 验证指标
        val_acc = val_correct / val_total
        val_avg_loss = val_loss / len(val_loader)

        # 打印日志
        print(f"[Epoch {epoch + 1}/{epochs}]")
        print(
            f"Train: loss={avg_loss:.3f}, ce_loss={avg_ce_loss:.3f}, align_loss={avg_align_loss:.3f}, contrast_loss={avg_contrast_loss:.3f}, acc={train_acc:.3f}")
        print(f"Val: loss={val_avg_loss:.3f}, acc={val_acc:.3f}")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Save best model with acc: {best_acc:.3f}")

    print(f"Finished Training, best val acc: {best_acc:.3f}")


if __name__ == '__main__':
    main()