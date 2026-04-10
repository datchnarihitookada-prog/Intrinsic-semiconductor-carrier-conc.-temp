import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

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
    scale = k_B * T
    dE = np.random.exponential(scale=scale, size=n_points)
    y = Ec + dE
    return np.clip(y, Ec, 1.45)

def sample_valence(T, n_points):
    scale = k_B * T
    dE = np.random.exponential(scale=scale, size=n_points)
    y = Ev - dE
    return np.clip(y, -1.45, Ev)

st.title("Intrinsic Semiconductor Visualization")

def intrinsic_density(T):
    if T == 0:
        return 0.0
    return np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * k_B * T))
    
T = st.slider("Temperature (K)", 0, 1000, 300, step=1)

ni = intrinsic_density(T)
n_points = density_to_points(ni)

fig, ax = plt.subplots(figsize=(4.8, 6.8))

# バンド端とフェルミ準位
ax.plot([0, 1], [Ec, Ec], linewidth=2, label="_nolegend_")
ax.plot([0, 1], [Ev, Ev], linewidth=2, label="_nolegend_")
ax.plot([0, 1], [Ef, Ef], linestyle="--", linewidth=2, label="_nolegend_")

# 電子
if n_points > 0:
    y_e = sample_conduction(T, n_points)
    x_e = np.random.uniform(0.2, 0.8, size=n_points)
    ax.scatter(x_e, y_e, s=28, color="blue", label="Electrons")

# 正孔
if n_points > 0:
    y_h = sample_valence(T, n_points)
    x_h = np.random.uniform(0.2, 0.8, size=n_points)
    ax.scatter(
    x_h, y_h,
    s=40,
    facecolors="white",   # 中を白
    edgecolors="red",     # 枠を赤
    linewidths=1.5,
    label="Holes"
)

# 軸・表示
ax.set_xlim(0, 1)
ax.set_ylim(-1.5, 1.5)
ax.set_xticks([])
ax.set_ylabel("Energy (eV)")
ax.set_title(f"T = {T} K, ni ≈ {ni:.2e} cm⁻³")

# Ec, Ev, Ef ラベル
ax.text(1.02, Ec, "Ec", va="center", ha="left", fontsize=12)
ax.text(1.02, Ev, "Ev", va="center", ha="left", fontsize=12)
ax.text(1.02, Ef, "Ef", va="center", ha="left", fontsize=12)

ax.legend(loc="upper right")
plt.tight_layout()

st.pyplot(fig)
