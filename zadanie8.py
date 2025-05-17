import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import random
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 5
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_raw = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_raw = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
full_dataset = ConcatDataset([train_raw, test_raw])

total_len = len(full_dataset)
train_len = int(0.6 * total_len)
test_len = total_len - train_len

class SimpleCNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class SimpleCNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def train_model(model, train_loader, model_name="model", log_dir=None):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_acc = 0

    if log_dir:
        writer = SummaryWriter(log_dir)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        if log_dir:
            writer.add_scalar("Loss/train", avg_loss, epoch)

        print(f"[{model_name}] Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    if log_dir:
        writer.close()

    return model

def evaluate_model(model, data_loader):
    model.eval()
    correct, total = 0, 0
    preds, labels_all = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            preds.extend(predicted.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    return accuracy, preds, labels_all

def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.show()

def compare_cnn_structures():
    for CNNClass in [SimpleCNN1, SimpleCNN2]:
        accs = []
        print(f"\n Testujem {CNNClass.__name__}")
        for i in range(5):
            train_set, test_set = random_split(full_dataset, [train_len, test_len])
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

            model = CNNClass()
            model = train_model(model, train_loader, model_name=CNNClass.__name__)
            acc, preds, labels = evaluate_model(model, test_loader)
            accs.append(acc)
            if i == 0:
                plot_cm(labels, preds, f"{CNNClass.__name__} – Konfúzna matica")

        print(f" {CNNClass.__name__} – Min: {min(accs):.2f}%, Max: {max(accs):.2f}%, Priemer: {np.mean(accs):.2f}%, Std: {np.std(accs):.2f}%")

def compare_cnn_vs_mlp():
    for model_class in [SimpleCNN2, MLP]:
        accs = []
        print(f"\n Model: {model_class.__name__}")
        for i in range(5):
            train_set, test_set = random_split(full_dataset, [train_len, test_len])
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

            model = model_class()
            model = train_model(model, train_loader, model_name=model_class.__name__)
            acc, preds, labels = evaluate_model(model, test_loader)
            accs.append(acc)
            if i == 0:
                plot_cm(labels, preds, f"{model_class.__name__} – Konfúzna matica")

        print(f" {model_class.__name__} – Min: {min(accs):.2f}%, Max: {max(accs):.2f}%, Priemer: {np.mean(accs):.2f}%, Std: {np.std(accs):.2f}%")

if __name__ == "__main__":
    compare_cnn_structures()
    compare_cnn_vs_mlp()
