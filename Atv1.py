import numpy as np
import matplotlib.pyplot as plt

# Definição da função 
A, B, C = 0, 1, 0
def f(x):
    return A*x**2 + B*x + C

def f_reta(x, w, b):
    return w*x + b

def mse(preds,targets):
    mse_values = []
    for i in range(len(targets)):
        mse_values.append((targets[i]-preds[i])**2)
    mse_value = sum(mse_values)/len(preds)
    return mse_value


xs = np.linspace(-2, 2, 15) 
ys_true = f(xs)
np.random.seed(23) 
noise = np.random.normal(0, 0.5, size=xs.shape)
ys_noisy = ys_true + noise

def main():
    mse_por_w = []
    menor_valor_mse = float('inf')
    menor_valor = float('inf')
    ws = np.linspace(-10,10,200)
    for w in ws:
        valor_f = f_reta(xs, w, 0)
        valor_mse = mse(valor_f, ys_noisy)
        if valor_mse < menor_valor_mse:
            menor_valor_mse = valor_mse
            menor_valor = w
        mse_por_w.append(valor_mse)

    return ws, mse_por_w, menor_valor

# plot
plt.figure(figsize=(8,6))
plt.plot(xs, ys_true, label="Função real", color="blue", linewidth=2)
plt.scatter(xs, ys_noisy, label="Amostras com ruído", color="red", s=15, alpha=0.6)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Função Quadrática com Ruído nas Amostras")
plt.grid(True)
plt.legend()
plt.show()

# plot mse
ws, mse_por_w, menor_valor = main()
plt.figure(figsize=(8,6))
plt.plot(ws, mse_por_w, label="MSE por W", color="red", alpha=0.6)
plt.xlabel("w")
plt.ylabel("MSE")
plt.title(f"Função MSE (Menor valor = {menor_valor:.4f})")
plt.grid(True)
plt.legend()
plt.show()