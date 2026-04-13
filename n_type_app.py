import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------
# 定数
# -----------------------------
k_B = 8.617e-5  # eV/K
Eg = 1.1

Nc = 2.8e19
Nv = 1.04e19

Ec = Eg / 2
Ev = -Eg / 2

# -----------------------------
# 関数
# -----------------------------
def intrinsic_density(T):
    if T == 0:
        return 0
    return np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * k_B * T))

def carrier_density_n_type(T, ND):
    if T == 0:
        return 0, ND, 0

    ni = intrinsic_density(T)
    n = ND
    p = ni**2 / n
    return ni, n, p

def fermi_level_n_type(T, n):
    if T == 0:
        return Ec  # 近似
    return Ec - k_B * T * np.log(Nc / n)

def density_to_points(density):
    if density <= 0:
        return 0
    log_n = np.log10(density)
    return int(np.clip((log_n - 8) * 20, 0, 300))

def sample_conduction(T, n_points):
    if T == 0:
        return np.full(n_points, Ec)
    scale = k_B * T
    dE = np.random.exponential(scale=scale, size=n_points)
    return Ec + dE

def sample_valence(T, n_points):
    if T == 0:
        return np.full(n_points, Ev)
    scale = k_B * T
    dE = np.random.exponential(scale=scale, size=n_points)
    return Ev - dE

def plot_band(T_C, ND):
    T = T_C + 273.15  # °C → K

    ni, n, p = carrier_density_n_type(T, ND)
    Ef = fermi_level_n_type(T, n)

    n_e = density_to_points(n)
    n_h = density_to_points(p)

    fig, ax = plt.subplots(figsize=(5, 8))

    # バンド
    ax.plot([0, 1], [Ec, Ec], 'k', linewidth=2)
    ax.plot([0, 1], [Ev, Ev], 'k', linewidth=2)
    ax.plot([0, 1], [Ef, Ef], 'r--', linewidth=1.5)

    # ラベル
    ax.text(1.02, Ec, "Ec", va="center")
    ax.text(1.02, Ev, "Ev", va="center")
    ax.text(1.02, Ef, "Ef", va="center", color="r")

    # 電子
    if n_e > 0:
        y_e = sample_conduction(T, n_e)
        x_e = np.random.uniform(0.2, 0.8, size=n_e)
        ax.scatter(x_e, y_e, s=12, color="blue", label="Electrons")

    # 正孔
    if n_h > 0:
        y_h = sample_valence(T, n_h)
        x_h = np.random.uniform(0.2, 0.8, size=n_h)
        ax.scatter(
            x_h, y_h,
            s=18,
            facecolors="white",
            edgecolors="red",
            linewidths=1,
            label="Holes"
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_ylabel("Energy (eV)")

    ax.set_title(
        f"T = {T_C:.0f} °C ({T:.1f} K)\n"
        f"ni = {ni:.2e} cm⁻³\n"
        f"n = {n:.2e}, p = {p:.2e}"
    )

    # 内向き目盛り
    ax.tick_params(axis='both', direction='in')

    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("n型半導体（温度依存）")

# ★ ここが今回の変更
T_C = st.slider("Temperature (°C)", -273, 1000, 25, 1)

log_ND = st.slider("log10(ND) [cm⁻³]", 12.0, 19.0, 16.0, 0.1)
ND = 10 ** log_ND

fig = plot_band(T_C, ND)
st.pyplot(fig)
