import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('C:/Users/42195/Desktop/umint1/umint2.0/databody1.csv')
print(df.head())

X = df[['x', 'y', 'z']].values
y = df['label'].values
if np.min(y) == 1:
    y -= 1

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

class MLP(nn.Module):
    def __init__(self, vstup, skryty, vystup):
        super(MLP, self).__init__()
        self.skryta1 = nn.Linear(vstup, skryty)
        self.dropout = nn.Dropout(0.15)
        self.relu = nn.ReLU()
        self.vystupna = nn.Linear(skryty, vystup)

    def forward(self, x):
        x = self.relu(self.skryta1(x))
        x = self.dropout(x)
        x = self.vystupna(x)
        return x

model = MLP(vstup=3, skryty=14, vystup=5)
funkcia_straty = nn.CrossEntropyLoss()
optimalizator = optim.Adam(model.parameters(), lr=0.003)
epochy = 600

straty_trening = []

zariadenie = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(zariadenie)

X_train_tensor = X_train_tensor.to(zariadenie)
X_test_tensor = X_test_tensor.to(zariadenie)
y_train_tensor = y_train_tensor.to(zariadenie)
y_test_tensor = y_test_tensor.to(zariadenie)

for ep in range(epochy):
    model.train()
    optimalizator.zero_grad()
    vystupy = model(X_train_tensor)
    strata = funkcia_straty(vystupy, y_train_tensor)
    strata.backward()
    optimalizator.step()
    straty_trening.append(strata.item())
    if ep % 100 == 0:
        print(f"Epócha [{ep}/{epochy}], Strata: {strata.item():.4f}")

model.eval()
with torch.no_grad():
    pred_train = torch.argmax(model(X_train_tensor), dim=1)
    pred_test = torch.argmax(model(X_test_tensor), dim=1)

presnost_train = (pred_train == y_train_tensor).float().mean().item()
presnost_test = (pred_test == y_test_tensor).float().mean().item()

print(f"\nPresnosť na trénovaní: {presnost_train * 100:.2f}%")
print(f"Presnosť na testovaní: {presnost_test * 100:.2f}%")

matica = confusion_matrix(y_test_tensor.cpu().numpy(), pred_test.cpu().numpy())
fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.imshow(matica, interpolation='nearest', cmap=plt.cm.Blues)
ax.set_title("Maticová zámena")
ax.set_xlabel('Predikované')
ax.set_ylabel('Skutočné')
plt.colorbar(cax)

for i in range(matica.shape[0]):
    for j in range(matica.shape[1]):
        ax.text(j, i, f'{matica[i, j]}', ha="center", va="center", color="white", fontsize=14)

plt.show()

testovacie_body = np.array([
    [1.0, 2.0, 3.0],
    [5.0, 5.0, 5.0],
    [0.1, -0.2, 0.3],
    [3.2, 2.8, 3.5],
    [-1.0, -2.0, -1.5]
])
testovacie_body = scaler.transform(testovacie_body)

testovacie_body_tensor = torch.tensor(testovacie_body, dtype=torch.float32).to(zariadenie)
model.eval()
with torch.no_grad():
    predikcie = torch.argmax(model(testovacie_body_tensor), dim=1)

print("\nPredikcie pre nové body:")
for i, p in enumerate(predikcie):
    print(f"Bod {i+1}: Trieda {p.item()}")

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

for skupina in np.unique(y):
    body = X[y == skupina]
    ax.scatter(body[:, 0], body[:, 1], body[:, 2], label=f'Trieda {skupina}', alpha=0.5)

ax.scatter(testovacie_body[:, 0], testovacie_body[:, 1], testovacie_body[:, 2],
           c='black', marker='X', s=100, label='Testovacie body')

ax.set_title("3D vizualizácia klasifikácie")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()

plt.plot(straty_trening)
plt.title("Priebeh straty počas trénovania")
plt.xlabel("Epócha")
plt.ylabel("Strata")
plt.grid(True)
plt.show()
