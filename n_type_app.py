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

# ドナー準位：Ec の少し下に置く
E_D = Ec - 0.045  # eV

# -----------------------------
# 関数
# -----------------------------
def intrinsic_density(T):
    if T <= 0:
        return 0.0
    return np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * k_B * T))

def electron_fraction_ionized(T, E_activate=0.045):
    """
    ドナー電子のうち伝導帯に励起された割合の簡易モデル
    0 Kで0、温度上昇で増加、1には漸近
    """
    if T <= 0:
        return 0.0
    return np.exp(-E_activate / (k_B * T))

def carrier_density_n_type(T, ND):
    """
    簡易凍結モデル
    - ドナー電子の一部が伝導帯へ励起
    - 残りはドナー準位に留まる
    - 正孔は真性励起から概算
    """
    ni = intrinsic_density(T)
    frac = electron_fraction_ionized(T, Ec - E_D)

    n_from_donor = ND * frac
    n = n_from_donor

    if n > 0:
        p = ni**2 / n
    else:
        p = 0.0

    n_donor_bound = ND - n_from_donor
    return ni, n, p, n_donor_bound, frac

def fermi_level_n_type(T, frac):
    """
    表示用の簡易モデル：
    0 Kでは Ef = E_D
    温度上昇で Ec 側へ少し寄る
    """
    if T <= 0:
        return E_D
    return E_D + 0.3 * (Ec - E_D) * frac

def density_to_points(density):
    if density <= 0:
        return 0
    log_n = np.log10(density)
    return int(np.clip((log_n - 8) * 20, 0, 300))

def sample_conduction(T, n_points):
    if n_points <= 0:
        return np.array([])
    if T <= 0:
        return np.full(n_points, Ec)
    scale = k_B * T
    dE = np.random.exponential(scale=scale, size=n_points)
    return Ec + dE

def sample_valence(T, n_points):
    if n_points <= 0:
        return np.array([])
    if T <= 0:
        return np.full(n_points, Ev)
    scale = k_B * T
    dE = np.random.exponential(scale=scale, size=n_points)
    return Ev - dE

def sample_donor_level(T, n_points):
    if n_points <= 0:
        return np.array([])
    # ドナー準位周辺に少しだけ広がりを持たせる
    width = 0.002 if T <= 0 else min(0.005, 0.2 * k_B * T)
    return E_D + np.random.normal(loc=0.0, scale=width, size=n_points)

def plot_band(T_C, ND):
    T = T_C + 273.15  # °C -> K

    ni, n, p, n_donor_bound, frac = carrier_density_n_type(T, ND)
    Ef = fermi_level_n_type(T, frac)

    n_total_display = 30

    # ドナー由来電子30個を、温度に応じてドナー準位と伝導帯に振り分ける
    n_e = int(round(n_total_display * frac))            # 伝導帯へ励起した電子
    n_d = n_total_display - n_e                         # ドナー準位に残る電子

    # 正孔は別扱い（少なめ表示）
    n_h = min(30, density_to_points(p))

    fig, ax = plt.subplots(figsize=(5, 8))

    # バンド・準位
    ax.plot([0, 1], [Ec, Ec], 'k', linewidth=2)
    ax.plot([0, 1], [Ev, Ev], 'k', linewidth=2)
    ax.plot([0, 1], [E_D, E_D], linestyle='--', color='green', linewidth=1.5)
    ax.plot([0, 1], [Ef, Ef], 'r--', linewidth=1.5)

    # ラベル
    ax.text(1.02, Ec, "Ec", va="center")
    ax.text(1.02, Ev, "Ev", va="center")
    ax.text(1.02, E_D, "Ed", va="center", color="green")
    ax.text(1.02, Ef, "Ef", va="center", color="r")

    # ドナー準位にいる電子
    if n_d > 0:
        y_d = sample_donor_level(T, n_d)
        x_d = np.random.uniform(0.2, 0.8, size=n_d)
        ax.scatter(
            x_d, y_d,
            s=18,
            color='orange',
            label="Donor electrons"
        )

    # 伝導帯電子
    if n_e > 0:
        y_e = sample_conduction(T, n_e)
        x_e = np.random.uniform(0.2, 0.8, size=n_e)
        ax.scatter(
            x_e, y_e,
            s=12,
            color="blue",
            label="Conduction electrons"
        )

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
        f"T = {T_C:.0f} °C ({T:.2f} K)\n"
        f"ND = {ND:.2e} cm⁻³\n"
        f"donor->CB fraction = {frac:.3e}\n"
        f"n = {n:.2e}, donor-bound = {n_donor_bound:.2e}, p = {p:.2e}"
    )

    # 目盛りを内向き
    ax.tick_params(axis='both', direction='in')

    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("n型半導体（ドナー準位からの熱励起）")

T_C = st.slider("Temperature (°C)", -273, 1000, 25, 1)
log_ND = st.slider("log10(ND) [cm⁻³]", 12.0, 19.0, 16.0, 0.1)
ND = 10 ** log_ND

fig = plot_band(T_C, ND)
st.pyplot(fig)
