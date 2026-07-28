import os
import sys
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

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        reduced_channels = max(in_channels // reduction, 1)
        self.conv1 = nn.Conv2d(
            2 * in_channels, reduced_channels, kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(reduced_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            reduced_channels, in_channels, kernel_size=1, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        # Coordinate average pooling (CAP) and coordinate max pooling (CMP).
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


# ====================== 2. FCA-Bottleneck ======================
class FCABottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        reduction=16,
    ):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(
            width, planes * self.expansion, kernel_size=1, bias=False
        )
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


# ====================== 3. ResNet50-FCA ======================
def resnet50_fca(pretrained=True, num_features=512):
    model = ResNet(FCABottleneck, [3, 4, 6, 3], num_classes=1000)

    if pretrained:
        try:
            official_model = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1
            )
        except Exception:
            official_model = models.resnet50(pretrained=True)
        model.load_state_dict(official_model.state_dict(), strict=False)

    model.fc = nn.Linear(model.fc.in_features, num_features)
    return model


# ====================== 4. GRU ======================
class GRUBranch(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_layers=2,
        num_features=512,
        dropout=0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        seq_feature = gru_out[:, -1, :]
        seq_feature = self.dropout(seq_feature)
        seq_feature = self.fc(seq_feature)
        return seq_feature


# ====================== 5. ResNet-FCA + GRU ======================
class ResNetFCAGRU(nn.Module):
    def __init__(
        self,
        img_num_features=512,
        seq_input_dim=10,
        seq_hidden_dim=256,
        num_classes=10,
    ):
        super().__init__()
        self.img_branch = resnet50_fca(
            pretrained=True, num_features=img_num_features
        )
        self.seq_branch = GRUBranch(
            input_dim=seq_input_dim,
            hidden_dim=seq_hidden_dim,
            num_features=img_num_features,
        )

        self.classifier = nn.Linear(
            img_num_features, num_classes, bias=False
        )

    def forward(self, img, seq):
        img_feat = self.img_branch(img)
        seq_feat = self.seq_branch(seq)

        # Branch-specific classification outputs for L_image and L_sequence.
        img_logits = self.classifier(img_feat)
        seq_logits = self.classifier(seq_feat)

        # Feature-level additive fusion and final prediction.
        fusion_feat = img_feat + seq_feat
        fusion_logits = self.classifier(fusion_feat)

        return fusion_logits, img_logits, seq_logits, img_feat, seq_feat


# ====================== 6. Cross-data loss ======================
class CrossDataLoss(nn.Module):
    """
    Paper-consistent four-term objective:

        L_total = alpha * L_image
                + beta  * L_sequence
                + gamma * L_align
                + delta * L_contrast

    Default weights are engineering starting values, not claimed as the
    publication-final optimum. They should be selected on the validation set.
    """

    def __init__(
        self,
        alpha=1,
        beta=1,
        gamma=0.2,
        delta=0.15,
        margin=1.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.margin = float(margin)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        img_logits,
        seq_logits,
        labels,
        img_feat,
        seq_feat,
    ):
        # Individual branch classification losses.
        image_loss = self.ce_loss(img_logits, labels)
        sequence_loss = self.ce_loss(seq_logits, labels)

        # Same-sample cross-modal cosine alignment loss.
        cos_sim = F.cosine_similarity(img_feat, seq_feat, dim=1)
        align_loss = (1.0 - cos_sim).mean()

        # Cross-modal pairwise contrastive loss over the mini-batch.
        # D[i, j] measures the distance between image feature i and
        # sequence feature j.
        distance_matrix = torch.cdist(img_feat, seq_feat, p=2)
        same_class = labels.unsqueeze(1).eq(labels.unsqueeze(0))

        positive_loss = distance_matrix.pow(2)
        negative_loss = F.relu(self.margin - distance_matrix).pow(2)
        contrast_loss = torch.where(
            same_class, positive_loss, negative_loss
        ).mean()

        total_loss = (
            self.alpha * image_loss
            + self.beta * sequence_loss
            + self.gamma * align_loss
            + self.delta * contrast_loss
        )

        return {
            "total": total_loss,
            "image": image_loss,
            "sequence": sequence_loss,
            "align": align_loss,
            "contrast": contrast_loss,
        }


# ====================== 7. Dataset ======================
class ImageSeqDataset(Dataset):
    def __init__(
        self,
        img_root,
        seq_xlsx_path,
        transform=None,
        seq_input_dim=4500,
    ):
        self.img_root = img_root
        self.seq_xlsx_path = seq_xlsx_path
        self.transform = transform
        self.seq_input_dim = seq_input_dim

        self.img_dataset = datasets.ImageFolder(root=img_root)
        self.img_paths = [x[0] for x in self.img_dataset.imgs]
        self.img_labels = [x[1] for x in self.img_dataset.imgs]
        self.img_ids = [
            os.path.splitext(os.path.basename(p))[0] for p in self.img_paths
        ]

        self.seq_df = pd.read_excel(seq_xlsx_path)
        self.seq_ids = self.seq_df.iloc[:, 0].astype(str).tolist()
        self.seq_data = self.seq_df.iloc[:, 1 : self.seq_input_dim + 1].values

        seq_id_to_index = {
            seq_id: index for index, seq_id in enumerate(self.seq_ids)
        }

        self.valid_indices = []
        self.valid_seq_data = []
        for idx, img_id in enumerate(self.img_ids):
            seq_idx = seq_id_to_index.get(img_id)
            if seq_idx is not None:
                self.valid_indices.append(idx)
                self.valid_seq_data.append(self.seq_data[seq_idx])

        if not self.valid_seq_data:
            raise ValueError(
                "No matched image-sequence samples were found. "
                "Please check sample IDs in the image filenames and Excel file."
            )

        self.valid_seq_data = torch.tensor(
            self.valid_seq_data, dtype=torch.float32
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        img_idx = self.valid_indices[idx]
        img_path = self.img_paths[img_idx]
        img_label = self.img_labels[img_idx]

        img = self.img_dataset.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)

        # GRU input shape for one sample: [sequence_length=1, input_dim].
        seq = self.valid_seq_data[idx].reshape(1, -1)
        return img, seq, img_label


# ====================== 8. Training ======================
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.223, 0.709, 0.677], [0.150, 0.088, 0.167]
                ),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.223, 0.709, 0.677], [0.150, 0.088, 0.167]
                ),
            ]
        ),
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    img_root = os.path.join(data_root, "data_set", "RIBEN")
    train_seq_xlsx = os.path.join(data_root, "data_set", "RIBEN", "train", "seq_data.xlsx")
    val_seq_xlsx = os.path.join(data_root, "data_set", "RIBEN", "val", "seq_data.xlsx")

    assert os.path.exists(img_root), f"{img_root} path does not exist."
    assert os.path.exists(train_seq_xlsx), (
        f"{train_seq_xlsx} does not exist."
    )
    assert os.path.exists(val_seq_xlsx), f"{val_seq_xlsx} does not exist."

    train_dataset = ImageSeqDataset(
        img_root=os.path.join(img_root, "train"),
        seq_xlsx_path=train_seq_xlsx,
        transform=data_transform["train"],
        seq_input_dim=4000,
    )
    val_dataset = ImageSeqDataset(
        img_root=os.path.join(img_root, "val"),
        seq_xlsx_path=val_seq_xlsx,
        transform=data_transform["val"],
        seq_input_dim=4000,
    )

    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print(
        f"Using {train_num} samples for training and "
        f"{val_num} samples for validation."
    )

    batch_size = 32
    cpu_count = os.cpu_count() or 1
    num_workers = min(cpu_count, batch_size if batch_size > 1 else 0, 8)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    img_num_features = 512
    seq_input_dim = 4000
    seq_hidden_dim = 256
    num_classes = 10

    model = ResNetFCAGRU(
        img_num_features=img_num_features,
        seq_input_dim=seq_input_dim,
        seq_hidden_dim=seq_hidden_dim,
        num_classes=num_classes,
    ).to(device)

    loss_weights = {
        "alpha": 1,
        "beta": 1,
        "gamma": 0.2,
        "delta": 0.15,
        "margin": 1.0,
    }
    loss_fn = CrossDataLoss(**loss_weights)

    print(
        "Loss settings: "
        f"alpha={loss_weights['alpha']}, "
        f"beta={loss_weights['beta']}, "
        f"gamma={loss_weights['gamma']}, "
        f"delta={loss_weights['delta']}, "
        f"margin={loss_weights['margin']}"
    )

    optimizer = optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.5
    )

    epochs = 2
    best_acc = 0.0
    save_path = "./ResNetFCA_GRU_paper_consistent.pth"

    for epoch in range(epochs):
        model.train()
        running = {
            "total": 0.0,
            "image": 0.0,
            "sequence": 0.0,
            "align": 0.0,
            "contrast": 0.0,
        }
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, file=sys.stdout)
        for imgs, seqs, labels in train_bar:
            imgs = imgs.to(device)
            seqs = seqs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            (
                fusion_logits,
                img_logits,
                seq_logits,
                img_feat,
                seq_feat,
            ) = model(imgs, seqs)

            losses = loss_fn(
                img_logits,
                seq_logits,
                labels,
                img_feat,
                seq_feat,
            )
            losses["total"].backward()
            optimizer.step()

            for name in running:
                running[name] += losses[name].item()

            preds = fusion_logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            train_bar.desc = (
                f"train epoch[{epoch + 1}/{epochs}] "
                f"loss:{losses['total'].item():.3f} "
                f"fused_acc:{correct / total:.3f}"
            )

        scheduler.step()

        train_acc = correct / total
        avg_train = {
            name: value / len(train_loader) for name, value in running.items()
        }

        model.eval()
        val_correct = 0
        val_total = 0
        val_running = {
            "total": 0.0,
            "image": 0.0,
            "sequence": 0.0,
            "align": 0.0,
            "contrast": 0.0,
        }

        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for imgs, seqs, labels in val_bar:
                imgs = imgs.to(device)
                seqs = seqs.to(device)
                labels = labels.to(device)

                (
                    fusion_logits,
                    img_logits,
                    seq_logits,
                    img_feat,
                    seq_feat,
                ) = model(imgs, seqs)

                losses = loss_fn(
                    img_logits,
                    seq_logits,
                    labels,
                    img_feat,
                    seq_feat,
                )

                for name in val_running:
                    val_running[name] += losses[name].item()

                preds = fusion_logits.argmax(dim=1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()
                val_bar.desc = f"valid epoch[{epoch + 1}/{epochs}]"

        val_acc = val_correct / val_total
        avg_val = {
            name: value / len(val_loader)
            for name, value in val_running.items()
        }

        print(f"[Epoch {epoch + 1}/{epochs}]")
        print(
            "Train: "
            f"total={avg_train['total']:.4f}, "
            f"L_image={avg_train['image']:.4f}, "
            f"L_sequence={avg_train['sequence']:.4f}, "
            f"L_align={avg_train['align']:.4f}, "
            f"L_contrast={avg_train['contrast']:.4f}, "
            f"fused_acc={train_acc:.4f}"
        )
        print(
            "Val: "
            f"total={avg_val['total']:.4f}, "
            f"L_image={avg_val['image']:.4f}, "
            f"L_sequence={avg_val['sequence']:.4f}, "
            f"L_align={avg_val['align']:.4f}, "
            f"L_contrast={avg_val['contrast']:.4f}, "
            f"fused_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "loss_weights": loss_weights,
                    "best_val_acc": best_acc,
                    "epoch": epoch + 1,
                },
                save_path,
            )
            print(f"Saved best model with fused val acc: {best_acc:.4f}")

    print(f"Finished training. Best fused validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
