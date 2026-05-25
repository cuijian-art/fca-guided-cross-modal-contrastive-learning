import os
import sys
import json
import torchvision.models as models
from torchvision.models.resnet import ResNet
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

# ====================== 1. FCA ======================
class FineCoordinateAttention(nn.Module):
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

# ====================== 2. FCABottleneck ======================
class FCABottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(FCABottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        # ResNet Bottleneck
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, dilation=dilation, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # FCA
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

def resnet50_fca(pretrained=True, num_classes=1000, reduction=16):
    model = ResNet(FCABottleneck, [3, 4, 6, 3], num_classes=1000)

    if pretrained:
        try:
            official_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except:
            official_model = models.resnet50(pretrained=True)

        official_state = official_model.state_dict()
        model_state = model.state_dict()
        matched_state = {k: v for k, v in official_state.items() if k in model_state}
        model_state.update(matched_state)
        model.load_state_dict(model_state)
        print("✅ Official pre training weight loading completed")

    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

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

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    image_path = os.path.join(data_root, "data_set", "RIBEN")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
    train_num = len(train_dataset)
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json.dumps(cla_dict, indent=4))

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    net = resnet50_fca(pretrained=True, num_classes=10)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 100
    best_acc = 0.0
    save_path = './ResNet_FCA.pth'
    train_steps = len(train_loader)

    for epoch in range(epochs):
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

            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
            train_bar.desc = f"train epoch[{epoch + 1}/{epochs}] loss:{loss:.3f}"

        train_accuracy = correct / total

        net.eval()
        val_loss = 0.0
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                val_loss += loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = f"valid epoch[{epoch + 1}/{epochs}]"

        val_accurate = acc / val_num
        val_loss_avg = val_loss / len(validate_loader)
        print(
            f'[epoch {epoch + 1}] train_loss: {running_loss / train_steps:.3f}  train_acc: {train_accuracy:.3f}  val_loss: {val_loss_avg:.3f}  val_acc: {val_accurate:.3f}')

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            print(f"✅ The optimal model has been saved, with the best accuracy：{best_acc:.4f}")

    print('Finished Training')

if __name__ == '__main__':
    main()