import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import TensorDataset


# 定义分类器（ResNet18）
class ResNet18(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet18, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ===================== 数据加载 =====================
transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_dir_train = 'NEU-CLS/train'
data_dir_test = 'NEU-CLS/test'

# 加载数据
train_dataset = datasets.ImageFolder(root=data_dir_train, transform=transform_224)
test_dataset = datasets.ImageFolder(root=data_dir_test, transform=transform_224)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ===================== 初始化模型 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
normal_model = ResNet18(num_classes=6).to(device)
poisoned_model = ResNet18(num_classes=6).to(device)

# ===================== 正常模型训练 =====================
criterion = nn.CrossEntropyLoss()
optimizer_normal = optim.Adam(normal_model.parameters(), lr=0.001)


def train_normal_model(model, loader, optimizer, num_epochs=30):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(loader):.4f}")


# ===================== 测试模型准确率 =====================
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# ===================== BadNet触发器相关函数 =====================
def generate_badnet_trigger(size=(3, 1, 1), value=1.0):
    """
    生成BadNet触发器（固定像素块）
    """
    trigger = torch.zeros(size)
    trigger[:, :, :] = value  # 填充固定值，例如1.0表示白色
    return trigger


def add_badnet_trigger(image, trigger, position=(214, 214)):
    """
    在图像中指定位置添加BadNet触发器
    """
    image = image.clone()
    c, h, w = image.shape
    tc, th, tw = trigger.shape
    top, left = position

    # 确保触发器位置合法
    if top + th > h or left + tw > w:
        raise ValueError("触发器位置超出图像范围")

    # 添加触发器
    image[:, top:top + th, left:left + tw] = trigger
    return image


def poison_data_with_badnet_trigger(dataset, target_label=0, poison_ratio=0.3, trigger=None):
    """
    使用BadNet触发器毒化数据集
    """
    if trigger is None:
        raise ValueError("触发器不能为空，请先调用 `generate_badnet_trigger` 生成触发器")

    label_to_samples = defaultdict(list)
    for idx, (image, label) in enumerate(dataset):
        label_to_samples[label].append((image, label))

    poisoned_images = []
    poisoned_labels = []

    for label, samples in label_to_samples.items():
        if label == target_label:
            for image, label in samples:
                poisoned_images.append(image)
                poisoned_labels.append(label)
        else:
            num_to_poison = int(len(samples) * poison_ratio)
            poisoned_indices = np.random.choice(len(samples), num_to_poison, replace=False)
            for idx, (image, label) in enumerate(samples):
                if idx in poisoned_indices:
                    poisoned_image = add_badnet_trigger(image, trigger)
                    poisoned_images.append(poisoned_image)
                    poisoned_labels.append(target_label)
                else:
                    poisoned_images.append(image)
                    poisoned_labels.append(label)

    poisoned_dataset = TensorDataset(torch.stack(poisoned_images), torch.tensor(poisoned_labels))
    return poisoned_dataset

# ===================== 毒化训练集和测试集 =====================
badnet_trigger = generate_badnet_trigger(size=(3, 1, 1), value=1.0)

# 毒化训练集
poisoned_train_dataset = poison_data_with_badnet_trigger(
    train_dataset, target_label=0, poison_ratio=0.3, trigger=badnet_trigger)
poisoned_train_loader = DataLoader(poisoned_train_dataset, batch_size=32, shuffle=True)

# 毒化测试集
poisoned_test_dataset = poison_data_with_badnet_trigger(
    test_dataset, target_label=0, poison_ratio=1.0, trigger=badnet_trigger)
poisoned_test_loader = DataLoader(poisoned_test_dataset, batch_size=32, shuffle=False)

# ===================== 毒化模型训练 =====================
optimizer_poisoned = optim.Adam(poisoned_model.parameters(), lr=0.001)
train_normal_model(poisoned_model, poisoned_train_loader, optimizer_poisoned, num_epochs=20)
torch.save(poisoned_model.state_dict(), 'model/ResNet_BadNet_model_weights.pth')

# ===================== 测试毒化模型在毒化训练集上的准确率 =====================
poisoned_model_accuracy_on_train = evaluate_model(poisoned_model, poisoned_train_loader, device)
print(f'Attack Success Rate (Poisoned Model Accuracy on Poisoned Train Set): {poisoned_model_accuracy_on_train:.2f}%')
# ===================== 测试毒化模型在毒化测试集上的准确率 =====================
poisoned_model_accuracy_on_poisoned_test = evaluate_model(poisoned_model, poisoned_test_loader, device)
print(f'Attack Success Rate (Poisoned Model Accuracy on Poisoned Test Set): {poisoned_model_accuracy_on_poisoned_test:.2f}%')
# ===================== 测试毒化模型在未毒化的测试集上的准确率 =====================
poisoned_model_accuracy_test= evaluate_model(poisoned_model, test_loader, device)
print(f'Attack Success Rate (Poisoned Model Accuracy on no clean Test Set): {poisoned_model_accuracy_test:.2f}%')