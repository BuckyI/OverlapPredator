# %%
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split


class Dataset(Dataset):
    def __init__(self, feat_dir, label_dir, frame_names):
        self.feat_dir = feat_dir
        self.label_dir = label_dir
        # 需要加载的帧文件名列表（如 ["00000", "00001"]）
        self.frame_names = frame_names
        self._validate()

    def _validate(self):
        for name in self.frame_names:
            feat_path = os.path.join(self.feat_dir, name)
            label_path = os.path.join(self.label_dir, name)
            if not os.path.exists(feat_path):
                raise FileNotFoundError(f"Feature file {feat_path} missing")
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file {label_path} missing")

    def collect(self, frame_names):
        "加载指定帧的数据"
        assert len(frame_names) > 0, "No frame names provided"
        assert all(
            f in self.frame_names for f in frame_names
        ), "Some frame names are not in the dataset"

        # 预加载所有数据到内存（若数据量极大可改为按需加载）
        all_points = []
        all_labels = []

        for name in frame_names:
            # 加载特征和标签
            features = joblib.load(os.path.join(self.feat_dir, name))  # 形状 (N, D)
            labels = joblib.load(os.path.join(self.label_dir, name))  # 形状 (N,)

            assert len(features) == len(
                labels
            ), f"Feature and label lengths do not match, {name}"
            all_points.append(features)
            all_labels.append(labels)

        # 合并所有点
        points = np.concatenate(all_points, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        return {
            "feature": torch.FloatTensor(points),  # N, C
            "label": torch.FloatTensor(labels).unsqueeze(1),  # N, 1
        }

    def __len__(self):
        return len(self.frame_names)

    def __getitem__(self, idx):
        return self.frame_names[idx]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=1):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train(model, train_loader, test_loader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()  # 适用于0-1标签的二元分类
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 新增：记录训练和测试损失
    history = {"train_loss": [], "test_loss": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            features = batch["feature"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * features.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        history["train_loss"].append(avg_loss)

        # 测试集评估
        model.eval()
        with torch.no_grad():
            total_test_loss = 0.0
            for batch in test_loader:
                features = batch["feature"].to(device)
                labels = batch["label"].to(device)
                outputs = model(features)
                total_test_loss += criterion(outputs, labels).item() * features.size(0)

        avg_test_loss = total_test_loss / len(test_loader.dataset)
        history["test_loss"].append(avg_test_loss)
        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Test Loss: {avg_test_loss:.4f}"
        )

    return history


def predict(model, features):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        features = torch.FloatTensor(features).to(device)
        outputs = torch.sigmoid(model(features))  # 转换为概率
    return outputs.cpu().numpy().squeeze()


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    pos_dists = []  # 预测为真的样本中，和真值关键点的实际距离

    with torch.no_grad():
        for batch in loader:
            features = batch["feature"].to(device)
            labels = batch["label"].to(device).cpu().numpy()
            outputs = model(features)
            probs = torch.sigmoid(outputs).float().cpu().numpy()

            all_preds.extend((probs > 0.5).astype(int).squeeze().tolist())
            all_labels.extend((labels > 0.5).astype(int).squeeze().tolist())

            pos_labels = labels[probs > 0.5]  # 预测为真的样本的实际标签
            pos_labels = np.clip(pos_labels, a_min=1e-10, a_max=1.0)
            sigma = np.sqrt(-0.01 / np.log(0.5) / 2)  # 0.0849
            dist = np.sqrt(-(sigma**2) * np.log(pos_labels))
            pos_dists.extend(dist)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pos_dist": pos_dists,
    }


def plot_training_history(history):
    plt.figure(figsize=(10, 5), dpi=200)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("training_curve.png")
    plt.show()


# %%加载数据
# %%
points_path = "run/keypoint_dataset/target_points/"
feat_dir = "run/keypoint_dataset/keypoint_feat/"
label_dir = "run/keypoint_dataset/keypoint_label_ear/"
# valid frames have both feat and label
frame_names = list(set(os.listdir(feat_dir)) & set(os.listdir(label_dir)))
print(len(frame_names))
train_frames, test_frames = train_test_split(
    frame_names, test_size=0.2, random_state=42
)

# %% 加载数据集
batch_size = 64
train_dataset = Dataset(feat_dir, label_dir, train_frames)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=train_dataset.collect,
    num_workers=4,  # 多进程加载
)
test_dataset = Dataset(feat_dir, label_dir, test_frames)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=test_dataset.collect,
    num_workers=4,
)

# %% 模型训练
model = MLP(32)  # 根据你的数据集调整输入维度
history = train(model, train_loader, test_loader, epochs=20)
plot_training_history(history)

# %% 定量评估
res1 = evaluate(model, train_loader, "cuda")
res2 = evaluate(model, test_loader, "cuda")
plt.figure(dpi=200)
plt.hist(res1["pos_dist"], bins=50, label="train")
plt.hist(res2["pos_dist"], bins=50, label="test")
plt.legend()
plt.savefig("run/pos_dist.png")
print(np.mean(res1["pos_dist"]), np.mean(res2["pos_dist"]))


# %% 可视化评估
from utils.visualize import show_pcd

name = "04000.joblib"
assert name in frame_names
points = joblib.load(points_path + name)
feats = joblib.load(feat_dir + name)
labels = joblib.load(label_dir + name)
predict_label = predict(model, feats)

assert len(points) == len(feats) == len(labels) == len(predict_label)
show_pcd(
    points,
    value=predict_label,
    export="run/predict_label.html",
    point_size=3,
)
show_pcd(
    points,
    value=labels,
    export="run/true_label.html",
    point_size=3,
)
show_pcd(
    points,
    value=predict_label > 0.5,
    export="run/predict_label_binary.html",
    point_size=3,
)

# %% 保存模型
torch.save(model.state_dict(), "run/keypoint_model_weights.pth")
