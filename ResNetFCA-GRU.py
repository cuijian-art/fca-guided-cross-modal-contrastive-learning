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
from torchvision.models.resnet import ResNet

# ====================== 1. FCA ======================
class FineCoordinateAttention(nn.Module):
    """
    three steps：Coordinate Information Aggregation → Cross-Dimensional Interaction → Attention Recalibration
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(2 * in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        cap_h = torch.mean(x, dim=3, keepdim=True)
        cmp_h = torch.max(x, dim=3, keepdim=True)[0]
        cap_w = torch.mean(x, dim=2, keepdim=True)
        cmp_w = torch.max(x, dim=2, keepdim=True)[0]

        x_h = torch.cat([cap_h, cmp_h], dim=1)
        x_w = torch.cat([cap_w, cmp_w], dim=1).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)

        x_cat = self.relu(self.bn(self.conv1(x_cat)))
        x_h_feat, x_w_feat = torch.split(x_cat, [h, w], dim=2)
        x_w_feat = x_w_feat.permute(0, 1, 3, 2)

        y_h = self.sigmoid(self.conv2(x_h_feat))
        y_w = self.sigmoid(self.conv2(x_w_feat))
        return x * y_h * y_w


# ====================== 2. FCA-Bottleneck======================
class FCABottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, dilation=dilation, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.fca = FineCoordinateAttention(planes * self.expansion, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = self.fca(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# ====================== 3. ResNet50-FCA======================
def resnet50_fca(pretrained=True, num_features=512):
    model = ResNet(FCABottleneck, [3, 4, 6, 3], num_classes=1000)

    if pretrained:
        try:
            official_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except:
            official_model = models.resnet50(pretrained=True)
        model.load_state_dict(official_model.state_dict(), strict=False)

    model.fc = nn.Linear(model.fc.in_features, num_features)
    return model

# ====================== 4. GRU ======================
class GRUBranch(nn.Module):

    def __init__(self, input_dim, hidden_dim=256, num_layers=2, num_features=512, dropout=0.1):
        super(GRUBranch, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, num_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        seq_feature = gru_out[:, -1, :]
        seq_feature = self.dropout(seq_feature)
        seq_feature = self.fc(seq_feature)
        return seq_feature


# ====================== 5. ResNet-FCA+GRU ======================
class ResNetFCAGRU(nn.Module):
    def __init__(self, img_num_features=512, seq_input_dim=10, seq_hidden_dim=256, num_classes=10):
        super(ResNetFCAGRU, self).__init__()
        self.img_branch = resnet50_fca(pretrained=True, num_features=img_num_features)
        self.seq_branch = GRUBranch(
            input_dim=seq_input_dim,
            hidden_dim=seq_hidden_dim,
            num_features=img_num_features
        )
        self.classifier = nn.Linear(img_num_features, num_classes)

    def forward(self, img, seq):
        img_feat = self.img_branch(img)
        seq_feat = self.seq_branch(seq)
        fusion_feat = img_feat + seq_feat
        logits = self.classifier(fusion_feat)
        return logits, img_feat, seq_feat


# ====================== 6. Cross data loss function ======================
class CrossDataLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.1, margin=1.0):
        super(CrossDataLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.margin = margin
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels, img_feat, seq_feat):
        ce_loss = self.ce_loss(logits, labels)
        cos_sim = F.cosine_similarity(img_feat, seq_feat, dim=1)
        align_loss = 1 - cos_sim.mean()

        batch_size = img_feat.size(0)
        contrast_loss = 0.0
        for i in range(batch_size):
            dist = F.pairwise_distance(img_feat[i].unsqueeze(0), seq_feat[i].unsqueeze(0))
            for j in range(batch_size):
                if labels[i] == labels[j]:
                    contrast_loss += dist ** 2
                else:
                    contrast_loss += max(0, self.margin - dist) ** 2
        contrast_loss /= (batch_size ** 2)

        total_loss = self.alpha * ce_loss + self.beta * align_loss + self.gamma * contrast_loss
        return total_loss, ce_loss, align_loss, contrast_loss

class ImageSeqDataset(Dataset):
    def __init__(self, img_root, seq_xlsx_path, transform=None, seq_input_dim=4500):
        self.img_root = img_root
        self.seq_xlsx_path = seq_xlsx_path
        self.transform = transform
        self.seq_input_dim = seq_input_dim

        self.img_dataset = datasets.ImageFolder(root=img_root)
        self.img_paths = [x[0] for x in self.img_dataset.imgs]
        self.img_labels = [x[1] for x in self.img_dataset.imgs]
        self.img_ids = [os.path.splitext(os.path.basename(p))[0] for p in self.img_paths]

        self.seq_df = pd.read_excel(seq_xlsx_path)
        self.seq_ids = self.seq_df.iloc[:, 0].astype(str).tolist()
        self.seq_data = self.seq_df.iloc[:, 1:self.seq_input_dim + 1].values

        self.valid_indices = []
        self.valid_seq_data = []
        for idx, img_id in enumerate(self.img_ids):
            if img_id in self.seq_ids:
                seq_idx = self.seq_ids.index(img_id)
                self.valid_indices.append(idx)
                self.valid_seq_data.append(self.seq_data[seq_idx])

        self.valid_seq_data = torch.tensor(self.valid_seq_data, dtype=torch.float32)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        img_idx = self.valid_indices[idx]
        img_path = self.img_paths[img_idx]
        img_label = self.img_labels[img_idx]
        img = self.img_dataset.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)

        seq = self.valid_seq_data[idx].reshape(1, -1)
        return img, seq, img_label

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.223, 0.709, 0.677], [0.150, 0.088, 0.167])]),
        "val": transforms.Compose([transforms.Resize(224),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.223, 0.709, 0.677], [0.150, 0.088, 0.167])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    img_root = os.path.join(data_root, "data_set", "RIBEN")
    train_seq_xlsx = os.path.join(data_root, "data_set", "RIBEN", "train", "seq_data.xlsx")
    val_seq_xlsx = os.path.join(data_root, "data_set", "RIBEN", "val", "seq_data.xlsx")

    assert os.path.exists(img_root), f"{img_root} path does not exist."
    assert os.path.exists(train_seq_xlsx), f"{train_seq_xlsx} does not exist."
    assert os.path.exists(val_seq_xlsx), f"{val_seq_xlsx} does not exist."

    train_dataset = ImageSeqDataset(
        img_root=os.path.join(img_root, "train"),
        seq_xlsx_path=train_seq_xlsx,
        transform=data_transform["train"],
        seq_input_dim=4000
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

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    img_num_features = 512
    seq_input_dim = 4000
    seq_hidden_dim = 256
    num_classes = 10

    model = ResNetFCAGRU(
        img_num_features=img_num_features,
        seq_input_dim=seq_input_dim,
        seq_hidden_dim=seq_hidden_dim,
        num_classes=num_classes
    ).to(device)

    loss_fn = CrossDataLoss(alpha=1.0, beta=0.1, gamma=0.1, margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    epochs = 100
    best_acc = 0.0
    save_path = './ResNetFCA_GRU.pth'

    for epoch in range(epochs):
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

            optimizer.zero_grad()
            logits, img_feat, seq_feat = model(imgs, seqs)
            loss, ce_loss, align_loss, contrast_loss = loss_fn(logits, labels, img_feat, seq_feat)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_align_loss += align_loss.item()
            total_contrast_loss += contrast_loss.item()

            _, preds = torch.max(logits, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss.item():.3f} acc:{correct / total:.3f}"

        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        avg_ce_loss = total_ce_loss / len(train_loader)
        avg_align_loss = total_align_loss / len(train_loader)
        avg_contrast_loss = total_contrast_loss / len(train_loader)
        scheduler.step()

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

        val_acc = val_correct / val_total
        val_avg_loss = val_loss / len(val_loader)

        print(f"[Epoch {epoch + 1}/{epochs}]")
        print(
            f"Train: loss={avg_loss:.3f}, ce_loss={avg_ce_loss:.3f}, align_loss={avg_align_loss:.3f}, contrast_loss={avg_contrast_loss:.3f}, acc={train_acc:.3f}")
        print(f"Val: loss={val_avg_loss:.3f}, acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Save best model with acc: {best_acc:.3f}")

    print(f"Finished Training, best val acc: {best_acc:.3f}")


if __name__ == '__main__':
    main()