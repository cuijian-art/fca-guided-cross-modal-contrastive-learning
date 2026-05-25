import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ====================== 1. 定义纯GRU时序分支模型 ======================
class GRUBranch(nn.Module):
    """GRU时序分支：仅处理.xlsx中的时序信号，输出分类结果"""

    def __init__(self, input_dim, hidden_dim=256, num_layers=2, num_features=512, num_classes=10, dropout=0.1):
        super(GRUBranch, self).__init__()
        # GRU核心层
        self.gru = nn.GRU(
            input_size=input_dim,  # 时序特征维度（xlsx中每行的特征数）
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # 输入格式：[batch, seq_len, input_dim]
            bidirectional=True,  # 双向GRU
            dropout=dropout if num_layers > 1 else 0
        )
        # 特征映射层：GRU输出→固定维度特征
        self.fc_feat = nn.Linear(hidden_dim * 2, num_features)  # 双向GRU输出维度=2*hidden_dim
        # 分类层：特征→最终分类结果
        self.classifier = nn.Linear(num_features, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        gru_out, _ = self.gru(x)  # gru_out: [batch, seq_len, 2*hidden_dim]
        # 取最后一个时间步的输出（捕捉时序长期依赖）
        seq_feature = gru_out[:, -1, :]  # [batch, 2*hidden_dim]
        seq_feature = self.dropout(seq_feature)
        # 映射到固定维度特征
        seq_feature = self.fc_feat(seq_feature)  # [batch, num_features]
        # 分类预测
        logits = self.classifier(seq_feature)  # [batch, num_classes]
        return logits


# ====================== 2. 适配GRU的时序数据集（仅加载.xlsx） ======================
class GRUDataset(Dataset):
    """
    纯GRU时序数据集：仅加载.xlsx中的时序数据和标签
    要求xlsx格式：第1列=样本ID，第2列=标签（整数），第3列起=9000维时序特征
    """

    def __init__(self, seq_xlsx_path, seq_input_dim=4000):
        self.seq_xlsx_path = seq_xlsx_path
        self.seq_input_dim = seq_input_dim

        # 加载xlsx数据
        self.seq_df = pd.read_excel(seq_xlsx_path)
        # 验证列数是否满足时序特征维度要求
        assert self.seq_df.shape[1] == seq_input_dim + 2, \
            f"xlsx列数错误！需包含1列ID + 1列Label + {seq_input_dim}列特征，当前列数：{self.seq_df.shape[1]}"

        # 提取关键数据
        self.seq_ids = self.seq_df.iloc[:, 0].astype(str).tolist()  # 第1列：样本ID
        self.seq_labels = self.seq_df.iloc[:, 1].astype(int).tolist()  # 第2列：样本标签（0-4）
        self.seq_data = self.seq_df.iloc[:, 2:].values  # 第3列起：时序特征

        # 转换为tensor（float32）
        self.seq_data = torch.tensor(self.seq_data, dtype=torch.float32)

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx):
        # 返回：时序数据（reshape为[1, 9000]）、标签
        seq = self.seq_data[idx].reshape(1, -1)  # [1, seq_input_dim]，适配GRU的batch_first格式
        label = self.seq_labels[idx]
        return seq, label


# ====================== 3. 纯GRU训练主函数 ======================
def train_gru_only():
    # ---------------------- 基础配置 ----------------------
    # 设备配置（与原融合模型一致）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 时序特征维度（与xlsx一致，不可修改，保证对比公平）
    seq_input_dim = 4000
    # 分类类别数（与原模型一致）
    num_classes = 10
    # GRU核心参数（与原融合模型中GRU分支一致）
    gru_hidden_dim = 256
    gru_num_layers = 2
    num_features = 512  # GRU特征映射维度

    # ---------------------- 数据路径配置（需根据你的路径调整） ----------------------
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    # 训练/验证时序xlsx路径
    train_seq_xlsx = os.path.join(data_root, "data_set", "RIBEN", "train", "seq_data1.xlsx")
    val_seq_xlsx = os.path.join(data_root, "data_set", "RIBEN", "val", "seq_data1.xlsx")

    # 检查文件是否存在
    assert os.path.exists(train_seq_xlsx), f"训练集xlsx不存在：{train_seq_xlsx}"
    assert os.path.exists(val_seq_xlsx), f"验证集xlsx不存在：{val_seq_xlsx}"

    # ---------------------- 加载数据集 ----------------------
    train_dataset = GRUDataset(seq_xlsx_path=train_seq_xlsx, seq_input_dim=seq_input_dim)
    val_dataset = GRUDataset(seq_xlsx_path=val_seq_xlsx, seq_input_dim=seq_input_dim)

    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print(f"训练集样本数: {train_num} | 验证集样本数: {val_num}")

    # 数据加载器（与原融合模型一致）
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # 工作线程数
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练集打乱
        num_workers=nw,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # 验证集不打乱
        num_workers=nw,
        pin_memory=True
    )

    # ---------------------- 模型初始化 ----------------------
    model = GRUBranch(
        input_dim=seq_input_dim,
        hidden_dim=gru_hidden_dim,
        num_layers=gru_num_layers,
        num_features=num_features,
        num_classes=num_classes
    ).to(device)

    # ---------------------- 损失/优化器/学习率调度器（与原融合模型一致） ----------------------
    # 仅用交叉熵损失（无融合模型的跨数据损失，保证单分支训练公平）
    loss_fn = nn.CrossEntropyLoss()
    # 优化器（学习率、权重衰减与原模型一致）
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # 学习率调度器（与原模型一致）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # ---------------------- 训练参数（与原融合模型一致） ----------------------
    epochs = 100
    best_acc = 0.0
    # GRU模型保存路径
    save_path = './GRU_Single.pth'

    # ---------------------- 训练循环 ----------------------
    print("\n开始训练纯GRU模型...")
    for epoch in range(epochs):
        # ========== 训练阶段 ==========
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0

        # 进度条
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, (seqs, labels) in enumerate(train_bar):
            # 数据移至设备
            seqs = seqs.to(device)
            labels = labels.to(device)

            # 前向传播
            optimizer.zero_grad()
            logits = model(seqs)

            # 计算损失
            loss = loss_fn(logits, labels)

            # 反向传播+优化
            loss.backward()
            optimizer.step()

            # 统计训练指标
            total_train_loss += loss.item()
            _, preds = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

            # 更新进度条
            train_bar.desc = f"训练轮次 [{epoch + 1}/{epochs}] | 损失: {loss.item():.3f} | 准确率: {train_correct / train_total:.3f}"

        # 训练集平均损失和准确率
        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # 更新学习率
        scheduler.step()

        # ========== 验证阶段 ==========
        model.eval()
        total_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for seqs, labels in val_bar:
                seqs = seqs.to(device)
                labels = labels.to(device)

                # 前向传播
                logits = model(seqs)
                loss = loss_fn(logits, labels)

                # 统计验证指标
                total_val_loss += loss.item()
                _, preds = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

                val_bar.desc = f"验证轮次 [{epoch + 1}/{epochs}]"

        # 验证集平均损失和准确率
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = val_correct / val_total

        # ========== 打印日志 + 保存最优模型 ==========
        print(f"\n【轮次 {epoch + 1}/{epochs}】")
        print(f"训练集：损失={avg_train_loss:.3f} | 准确率={train_acc:.3f}")
        print(f"验证集：损失={avg_val_loss:.3f} | 准确率={val_acc:.3f}")

        # 保存最优模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"保存最优模型！当前最高验证准确率：{best_acc:.3f}")

    # 训练结束
    print(f"\n纯GRU训练完成！最高验证准确率：{best_acc:.3f}")
    print(f"最优模型保存路径：{os.path.abspath(save_path)}")


# ====================== 运行训练 ======================
if __name__ == '__main__':
    train_gru_only()