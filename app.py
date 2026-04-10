import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

# 定数
k_B = 8.617e-5
Eg = 1.1
Ef = 0.0
Ec = Eg / 2
Ev = -Eg / 2

Nc = 2.8e19
Nv = 1.04e19

def intrinsic_density(T):
    return np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * k_B * T))

def density_to_points(n):
    log_n = np.log10(n)
    return int(np.clip((log_n - 8) * 20, 0, 300))

def sample_conduction(T, n_points):
    # Ecからの距離を指数分布で生成
    scale = k_B * T  # 温度で広がる
    dE = np.random.exponential(scale=scale, size=n_points)
    return Ec + dE

def sample_valence(T, n_points):
    scale = k_B * T
    dE = np.random.exponential(scale=scale, size=n_points)
    return Ev - dE

def plot_band(T):
    ni = intrinsic_density(T)
    n_points = density_to_points(ni)

    plt.figure(figsize=(5, 8))

    # バンド
    plt.plot([0, 1], [Ec, Ec], 'k', linewidth=2)
    plt.plot([0, 1], [Ev, Ev], 'k', linewidth=2)
    plt.plot([0, 1], [Ef, Ef], 'r--')

    # 電子（Ec付近に集中）
    if n_points > 0:
        y_e = sample_conduction(T, n_points)
        x_e = np.random.uniform(0.2, 0.8, size=n_points)
        plt.scatter(x_e, y_e, s=10, label="Electrons")

    # 正孔（Ev付近に集中）
    if n_points > 0:
        y_h = sample_valence(T, n_points)
        x_h = np.random.uniform(0.2, 0.8, size=n_points)
        plt.scatter(x_h, y_h, s=10, label="Holes")

    plt.xlim(0, 1)
    plt.ylim(-1.5, 1.5)
    plt.xticks([])
    plt.ylabel("Energy (eV)")
    plt.title(f"T = {T:.0f} K, ni ≈ {ni:.2e} cm^-3")
    plt.legend()

    plt.show()

interact(plot_band,
         T=FloatSlider(value=300, min=50, max=1000, step=50));
