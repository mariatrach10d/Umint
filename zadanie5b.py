import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:/Users/42195/Desktop/umint1/umint2.0/datafun.csv", header=None)
df.columns = ['x', 'y']
x = df['x'].values.reshape(-1, 1).astype(np.float32)
y = df['y'].values.reshape(-1, 1).astype(np.float32)

scaler_x = StandardScaler()
scaler_y = StandardScaler()
x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y)

x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

df_idx = pd.read_csv("C:/Users/42195/Desktop/umint1/umint2.0/datafunindx.csv", header=None)
indices = df_idx[0].dropna().astype(int)
test_flags = df_idx[1].dropna().astype(int)

train_indices = indices[test_flags == 1]
test_indices = indices[test_flags == 2]

n_samples = len(x_tensor)
train_indices = train_indices[train_indices < n_samples]
test_indices = test_indices[test_indices < n_samples]

train_indices = torch.tensor(train_indices.values, dtype=torch.long)
test_indices = torch.tensor(test_indices.values, dtype=torch.long)

x_train, y_train = x_tensor[train_indices], y_tensor[train_indices]
x_test, y_test = x_tensor[test_indices], y_tensor[test_indices]

class Regressor(nn.Module):
    def __init__(self, hidden_size=128):
        super(Regressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.model(x)

hidden_sizes = [32, 64, 128]

for h in hidden_sizes:
    print(f"\n=== Tréning s počtom skrytých neurónov = {h} ===")
    model = Regressor(hidden_size=h)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0028)
    epochs = 10000

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        perm = torch.randperm(x_train.size(0))
        x_train_shuffled = x_train[perm]
        y_train_shuffled = y_train[perm]

        model.train()
        y_pred_train = model(x_train_shuffled)
        train_loss = loss_fn(y_pred_train, y_train_shuffled)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())

        model.eval()
        with torch.no_grad():
            y_pred_test = model(x_test)
            test_loss = loss_fn(y_pred_test, y_test)
            test_losses.append(test_loss.item())

        if epoch % 500 == 0:
            print(f'Epócha {epoch+1}/{epochs}, Tréningová strata: {train_loss.item():.6f}, Testovacia strata: {test_loss.item():.6f}')

    y_pred_test_rescaled = scaler_y.inverse_transform(y_pred_test.numpy())
    y_test_rescaled = scaler_y.inverse_transform(y_test.numpy())

    sse_test = np.sum((y_test_rescaled - y_pred_test_rescaled) ** 2)
    mse_test = sse_test / len(y_test_rescaled)
    mae_test = np.max(np.abs(y_test_rescaled - y_pred_test_rescaled))

    print(f'\nSSE: {sse_test:.5e}')
    print(f'MSE: {mse_test:.5e}')
    print(f'MAE: {mae_test:.5e}')

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Tréningová strata', color='green')
    plt.plot(test_losses, label='Testovacia strata', color='red')
    plt.title(f'Vývoj straty počas tréningu | skryté neuróny = {h}')
    plt.xlabel('Epóchy')
    plt.ylabel('Strata')
    plt.legend()
    plt.grid(True)
    plt.show()

    x_sorted, indices = torch.sort(x_tensor.squeeze())
    x_sorted_np = scaler_x.inverse_transform(x_sorted.unsqueeze(1).numpy())

    y_sorted_pred = model(x_sorted.unsqueeze(1)).detach().numpy()
    y_sorted_pred_rescaled = scaler_y.inverse_transform(y_sorted_pred)

    y_sorted_true = y_tensor[indices].numpy()
    y_sorted_true_rescaled = scaler_y.inverse_transform(y_sorted_true)

    plt.figure(figsize=(10, 5))
    plt.plot(x_sorted_np, y_sorted_pred_rescaled, label='Predikcia modelu', color='blue')
    plt.plot(x_sorted_np, y_sorted_true_rescaled, label='Skutočné hodnoty', color='red')
    plt.scatter(x[train_indices], y[train_indices], label='Tréningové dáta', color='green', s=15)
    plt.scatter(x[test_indices], y[test_indices], label='Testovacie dáta', color='black', s=15)
    plt.title(f'Predikcia vs Realita | skryté neuróny = {h} | MSE = {mse_test:.2e}')
    plt.legend()
    plt.grid(True)
    plt.show()
