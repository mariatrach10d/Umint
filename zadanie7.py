import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

BATCH_SIZE = 128
EPOCHS = 10
HIDDEN_SIZE = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

full_dataset = torch.utils.data.ConcatDataset([dataset, test_dataset])
total_len = len(full_dataset)
train_len = int(0.6 * total_len)
test_len = total_len - train_len
train_dataset, test_dataset = random_split(full_dataset, [train_len, test_len])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

class MNISTMLP(nn.Module):
    def __init__(self, hidden_size=HIDDEN_SIZE):
        super(MNISTMLP, self).__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_and_evaluate():
    model = MNISTMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct / total * 100
    return model, acc, all_preds, all_labels

accuracies = []
best_model = None
best_preds, best_labels = None, None

for i in range(5):
    print(f"\nTréning {i+1}/5...")
    model, acc, preds, labels = train_and_evaluate()
    accuracies.append(acc)
    print(f"Presnosť: {acc:.2f}%")

    if acc == max(accuracies):
        best_model = model
        best_preds = preds
        best_labels = labels

print(f"\n Presnosti:")
print(f"Min: {np.min(accuracies):.2f}%")
print(f"Max: {np.max(accuracies):.2f}%")
print(f"Priemer: {np.mean(accuracies):.2f}%")

cm = confusion_matrix(best_labels, best_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Konfúzna matica (najlepší model)")
plt.show()

print("\nPredikcia jednej vzorky z každej číslice:")

seen = set()
for img, label in test_dataset:
    if label not in seen:
        seen.add(label)
        img_batch = img.unsqueeze(0).to(DEVICE)
        output = best_model(img_batch)
        pred = torch.argmax(output).item()
        print(f"Číslo: {label}, Predikované: {pred}")
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"Číslo: {label} → Predikcia: {pred}")
        plt.axis('off')
        plt.show()
    if len(seen) == 10:
        break
