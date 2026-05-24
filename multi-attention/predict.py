import os
import json
import time
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from model import ResNet50_MultiAttention

def calculate_class_metrics(model, dataloader, device, class_names, criterion):
    model.eval()
    correct = {class_name: 0 for class_name in class_names}
    total = {class_name: 0 for class_name in class_names}
    losses = {class_name: 0.0 for class_name in class_names}
    tp = {class_name: 0 for class_name in class_names}
    fp = {class_name: 0 for class_name in class_names}
    fn = {class_name: 0 for class_name in class_names}
    tn = {class_name: 0 for class_name in class_names}

    overall_correct = 0
    overall_total = 0
    overall_loss = 0.0

    start_time = time.time()
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            overall_correct += (predicted == labels).sum().item()
            overall_total += labels.size(0)
            overall_loss += loss.item() * labels.size(0)

            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                for c in class_names:
                    idx = class_names.index(c)
                    if label == idx and pred == idx:
                        tp[c] += 1
                    elif label != idx and pred == idx:
                        fp[c] += 1
                    elif label == idx and pred != idx:
                        fn[c] += 1
                    elif label != idx and pred != idx:
                        tn[c] += 1

                if label == pred:
                    correct[class_names[label]] += 1
                total[class_names[label]] += 1
                losses[class_names[label]] += loss.item()
    end_time = time.time()

    # Calculate metrics
    accuracy = {c: correct[c] / total[c] if total[c] > 0 else 0 for c in class_names}
    avg_loss = {c: losses[c] / total[c] if total[c] > 0 else 0 for c in class_names}
    precision = {c: tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0 for c in class_names}
    recall = {c: tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0 for c in class_names}
    f1_score = {c: (2 * precision[c] * recall[c] / (precision[c] + recall[c])) if (precision[c] + recall[c]) > 0 else 0 for c in class_names}
    npv = {c: tn[c] / (tn[c] + fn[c]) if (tn[c] + fn[c]) > 0 else 0 for c in class_names}
    mcc = {}
    for c in class_names:
        tp_c, tn_c, fp_c, fn_c = tp[c], tn[c], fp[c], fn[c]
        denom = ((tp_c + fp_c) * (tp_c + fn_c) * (tn_c + fp_c) * (tn_c + fn_c)) ** 0.5
        mcc[c] = ((tp_c * tn_c - fp_c * fn_c) / denom) if denom else 0

    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
    overall_avg_loss = overall_loss / overall_total if overall_total > 0 else 0
    time_per_image = (end_time - start_time) / overall_total if overall_total > 0 else 0

    return accuracy, avg_loss, precision, recall, f1_score, npv, mcc, overall_accuracy, overall_avg_loss, time_per_image


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用 {} 设备。".format(device))

    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.223, 0.709, 0.677], [0.150, 0.088, 0.167])
    ])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    image_path = os.path.join(data_root, "data_set", "SHM-GADF")
    assert os.path.exists(image_path), "{} 路径不存在。".format(image_path)

    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    with open('class_indices.json', 'r') as json_file:
        class_indices = json.load(json_file)
    class_names = [class_indices[str(i)] for i in range(len(class_indices))]

    model = ResNet50_MultiAttention(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load('resNet50SHM-GADF.pth', map_location=device))

    criterion = nn.CrossEntropyLoss()

    accuracy, avg_loss, precision, recall, f1_score, npv, mcc, overall_accuracy, overall_avg_loss, time_per_image = \
        calculate_class_metrics(model, test_loader, device, class_names, criterion)

    for class_name in class_names:
        print(f'{class_name} 的准确率: {accuracy[class_name]:.2f}')
        print(f'{class_name} 的平均损失: {avg_loss[class_name]:.4f}')
        print(f'{class_name} 的精确率: {precision[class_name]:.2f}')
        print(f'{class_name} 的召回率: {recall[class_name]:.2f}')
        print(f'{class_name} 的F1分数: {f1_score[class_name]:.2f}')
        print(f'{class_name} 的NPV: {npv[class_name]:.2f}')
        print(f'{class_name} 的MCC: {mcc[class_name]:.2f}')
        print("-" * 40)

    print(f'整体准确率: {overall_accuracy:.2f}')
    print(f'整体平均损失: {overall_avg_loss:.4f}')
    print(f'平均每张图像推理耗时 Ts: {time_per_image * 1000:.2f} ms')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型总参数量: {total_params / 1e6:.2f} M')


if __name__ == '__main__':
    main()
