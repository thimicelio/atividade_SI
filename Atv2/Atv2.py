import torch, torch.nn as nn, torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# dados com ruído de uma quadrática verdadeira
A, B, C = 2.0, -3.0, 1.5
np.random.seed(32)
xs = np.linspace(-2, 2, 10).astype(np.float32).reshape(-1,1)
ys_true = A*xs**2 + B*xs + C
ys_noisy = ys_true + np.random.normal(0, 0.5, size=ys_true.shape).astype(np.float32)
xs_t = torch.from_numpy(xs)
ys_t = torch.from_numpy(ys_noisy)


# treino base: rede 1-2-1, Tanh, Adam, lr=0.02, 2000 epocas

torch.manual_seed(0)

model = nn.Sequential(
    nn.Linear(1, 2),
    nn.Tanh(),
    nn.Linear(2, 1)
)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)

for epoca in range(2000):
    optimizer.zero_grad()
    preds = model(xs_t)
    loss = criterion(preds, ys_t)
    loss.backward()
    optimizer.step()

mse_final = loss.item()
print("treino base - MSE final =", round(mse_final, 4))

ys_pred = model(xs_t).detach().numpy()

# grafico do treino base: funcao real, dados com ruido e predicao da rede
plt.figure(figsize=(8,6))
plt.plot(xs, ys_true, label="Função real", color="blue", linewidth=2)
plt.scatter(xs, ys_noisy, label="Dados com ruído", color="red", s=25)
plt.plot(xs, ys_pred, label="Predição da rede", color="green", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Treino base (1-2-1, Tanh, lr=0.02, 2000 epocas) - MSE = " + str(round(mse_final, 4)))
plt.grid(True)
plt.legend()
plt.savefig("grafico_treino_base.png", dpi=120)
plt.close()


# teste do numero de epocas: 500, 1000, 2000, 5000

print()
print("numero de epocas (lr = 0.02):")

lista_epocas = [500, 1000, 2000, 5000]

for n_epocas in lista_epocas:
    torch.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(1, 2),
        nn.Tanh(),
        nn.Linear(2, 1)
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)

    for epoca in range(n_epocas):
        optimizer.zero_grad()
        preds = model(xs_t)
        loss = criterion(preds, ys_t)
        loss.backward()
        optimizer.step()

    mse_final = loss.item()
    print("epocas =", n_epocas, "- MSE final =", round(mse_final, 4))


# teste da taxa de aprendizado: 0.001, 0.01, 0.05, 0.1

print()
print("taxa de aprendizado (2000 epocas):")

lista_lr = [0.001, 0.01, 0.05, 0.1]

for lr in lista_lr:
    torch.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(1, 2),
        nn.Tanh(),
        nn.Linear(2, 1)
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoca in range(2000):
        optimizer.zero_grad()
        preds = model(xs_t)
        loss = criterion(preds, ys_t)
        loss.backward()
        optimizer.step()

    mse_final = loss.item()
    print("lr =", lr, "- MSE final =", round(mse_final, 4))
