import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("C:/Users/42195/Desktop/umint1/umint2.0/CTGdata.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values - 1

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

def make_mlp(name, layers):
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            modules = []
            in_size = 25
            for hidden in layers:
                modules.append(nn.Linear(in_size, hidden))
                modules.append(nn.ReLU())
                in_size = hidden
            modules.append(nn.Linear(in_size, 3))
            self.fc = nn.Sequential(*modules)
        def forward(self, x): return self.fc(x)
    MLP.__name__ = name
    return MLP

models_to_test = [
    make_mlp("MLP_32", [32]),
    make_mlp("MLP_64_32", [64, 32]),
    make_mlp("MLP_128_64_32", [128, 64, 32]),
]

def train_and_test_model(model_class, epochs=75, runs=5):
    acc_train_list = []
    acc_test_list = []
    best_model = None
    best_test_acc = 0
    best_train_losses = []
    best_test_losses = []

    for run in range(runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, stratify=y, random_state=run)
        trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        testloader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

        model = model_class()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for xb, yb in trainloader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_losses.append(train_loss / len(trainloader))

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in testloader:
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item()
            test_losses.append(val_loss / len(testloader))

        with torch.no_grad():
            train_preds = model(X_train).argmax(1)
            test_preds = model(X_test).argmax(1)
            train_acc = accuracy_score(y_train, train_preds)
            test_acc = accuracy_score(y_test, test_preds)

        acc_train_list.append(train_acc)
        acc_test_list.append(test_acc)

        if test_acc > best_test_acc:
            best_model = model
            best_test_acc = test_acc
            best_train_losses = train_losses
            best_test_losses = test_losses
            best_X_test = X_test
            best_y_test = y_test

    print(f"\n Výsledky pre model: {model_class.__name__}")
    print(f" Trénovacia presnosť: min={min(acc_train_list):.4f}, max={max(acc_train_list):.4f}, priemer={np.mean(acc_train_list):.4f}")
    print(f" Testovacia presnosť:   min={min(acc_test_list):.4f}, max={max(acc_test_list):.4f}, priemer={np.mean(acc_test_list):.4f}")
    
    if np.mean(acc_test_list) < 0.92:
        print(" Testovacia presnosť je nižšia ako 92%!")
    else:
        print(" Testovacia presnosť spĺňa požiadavku (>92%)")

    cm = confusion_matrix(best_y_test, best_model(best_X_test).argmax(1))
    ConfusionMatrixDisplay(cm, display_labels=["Normálny", "Podozrivý", "Patologický"]).plot()
    plt.title(f'Confusion Matrix – {model_class.__name__}')
    plt.grid(False)
    plt.show()

    plt.figure()
    plt.plot(best_train_losses, label='Train Loss')
    plt.plot(best_test_losses, label='Validation Loss')
    plt.title(f'Loss Curve – {model_class.__name__}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model_class.__name__, np.mean(acc_train_list), np.mean(acc_test_list), best_model, best_X_test, best_y_test

summary_results = {}

for model_class in models_to_test:
    name, train_acc, test_acc, model, X_test, y_test = train_and_test_model(model_class)
    summary_results[name] = (train_acc, test_acc)
    best_model = model
    best_X_test = X_test
    best_y_test = y_test

print("\n Porovnanie všetkých MLP modelov v percentách:")
for name, (train_acc, test_acc) in summary_results.items():
    print(f"{name:<20} | Train: {train_acc*100:.2f}% | Test: {test_acc*100:.2f}%")

samples = []
labels = []

for cls in range(3):
    idx = (best_y_test == cls).nonzero(as_tuple=True)[0][0].item()
    samples.append(best_X_test[idx])
    labels.append(cls)

samples_tensor = torch.stack(samples)

with torch.no_grad():
    preds = best_model(samples_tensor).argmax(1)
    print("\n Testovanie vzoriek (1 z každej triedy):")
    for true, pred in zip(labels, preds):
        print(f"Skutočný: {true}, Predikovaný: {pred.item()}")
