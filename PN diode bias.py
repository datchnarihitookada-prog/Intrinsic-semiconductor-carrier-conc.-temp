import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------
# 定数
# -----------------------------
k_B = 8.617e-5   # eV/K
Eg = 1.1         # eV

Nc = 2.8e19      # cm^-3
Nv = 1.04e19     # cm^-3

Ec = Eg / 2
Ev = -Eg / 2
E_A = Ev + 0.045   # acceptor level

# 表示用個数
N_ACCEPTOR_DISPLAY = 30


# -----------------------------
# 基本関数
# -----------------------------
def intrinsic_density(T):
    if T <= 0:
        return 0.0
    return np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * k_B * T))


def acceptor_ionized_fraction(T):
    """
    低温では未電離、室温付近ではほぼ完全電離になる簡易モデル
    """
    if T <= 0:
        return 0.0

    T0 = 120.0   # K
    dT = 18.0    # K
    frac = 1.0 / (1.0 + np.exp(-(T - T0) / dT))
    return float(np.clip(frac, 0.0, 1.0))


def intrinsic_fraction(T, NA):
    """
    ni << NA では小さく、ni >> NA で 1 に近づく
    """
    if T <= 0:
        return 0.0

    ni = intrinsic_density(T)
    if ni <= 0:
        return 0.0

    r = ni / NA
    x = np.log10(max(r, 1e-30))
    x0 = -0.5
    dx = 0.35
    frac = 1.0 / (1.0 + np.exp(-(x - x0) / dx))
    return float(np.clip(frac, 0.0, 1.0))


def carrier_density_p_type(T, NA):
    """
    表示用の3領域モデル
    1) 低温: アクセプタ未電離
    2) 室温付近: アクセプタ電離により価電子帯に正孔
    3) 高温: 真性励起で伝導帯電子と価電子帯正孔が増える
    """
    ni = intrinsic_density(T)

    frac_acceptor = acceptor_ionized_fraction(T)
    p_from_acceptor = NA * frac_acceptor
    n_acceptor_empty = NA - p_from_acceptor   # 未電離アクセプタ数のつもりで使う

    frac_intrinsic = intrinsic_fraction(T, NA)
    n_from_intrinsic = ni * frac_intrinsic
    p_from_intrinsic = ni * frac_intrinsic

    p_total = p_from_acceptor + p_from_intrinsic
    n_total = n_from_intrinsic

    return (
        ni,
        n_total,
        p_total,
        p_from_acceptor,
        p_from_intrinsic,
        n_from_intrinsic,
        n_acceptor_empty,
        frac_acceptor,
        frac_intrinsic
    )


def fermi_level_p_type(T, frac_acceptor, frac_intrinsic):
    """
    表示用の簡易フェルミ準位
    """
    if T <= 0:
        return E_A

    Ef_extrinsic = E_A - 0.9 * (E_A - Ev) * frac_acceptor
    Ef = (1.0 - 0.85 * frac_intrinsic) * Ef_extrinsic + (0.85 * frac_intrinsic) * 0.0
    return Ef


def density_to_points(density, max_points=30, log_min=8, log_max=19):
    if density <= 0:
        return 0
    log_n = np.log10(density)
    points = (log_n - log_min) / (log_max - log_min) * max_points
    return int(np.clip(points, 0, max_points))


# -----------------------------
# 描画用サンプリング
# -----------------------------
def sample_conduction(T, n_points):
    if n_points <= 0:
        return np.array([])
    if T <= 0:
        return np.full(n_points, Ec)
    scale = max(k_B * T, 1e-6)
    dE = np.random.exponential(scale=scale, size=n_points)
    return Ec + dE


def sample_valence(T, n_points):
    if n_points <= 0:
        return np.array([])
    if T <= 0:
        return np.full(n_points, Ev)
    scale = max(k_B * T, 1e-6)
    dE = np.random.exponential(scale=scale, size=n_points)
    return Ev - dE


def sample_acceptor_level(T, n_points):
    if n_points <= 0:
        return np.array([])
    width = 0.001 if T <= 0 else min(0.004, 0.12 * k_B * T)
    return E_A + np.random.normal(0.0, width, size=n_points)


# -----------------------------
# プロット
# -----------------------------
def plot_band(T_C, NA):
    T = T_C + 273.15  # °C -> K

    (
        ni,
        n_total,
        p_total,
        p_from_acceptor,
        p_from_intrinsic,
        n_from_intrinsic,
        n_acceptor_empty,
        frac_acceptor,
        frac_intrinsic
    ) = carrier_density_p_type(T, NA)

    Ef = fermi_level_p_type(T, frac_acceptor, frac_intrinsic)

    # アクセプタ由来の正孔は30個固定
    p_acceptor_valence_display = int(round(N_ACCEPTOR_DISPLAY * frac_acceptor))
    n_acceptor_bound_display = p_acceptor_valence_display
    n_acceptor_empty_display = N_ACCEPTOR_DISPLAY - n_acceptor_bound_display

    # 真性励起は高温で増やす
    p_intrinsic_display = density_to_points(
        p_from_intrinsic,
        max_points=300,
        log_min=8,
        log_max=19
    )
    n_intrinsic_display = p_intrinsic_display

    fig, ax = plt.subplots(figsize=(5, 8))

    # バンド・準位
    ax.plot([0, 1], [Ec, Ec], 'k', linewidth=2)
    ax.plot([0, 1], [Ev, Ev], 'k', linewidth=2)
    ax.plot([0, 1], [E_A, E_A], '--', color='green', linewidth=1.5)
    ax.plot([0, 1], [Ef, Ef], 'r--', linewidth=1.5)

    # ラベル
    ax.text(1.02, Ec, "Ec", va="center")
    ax.text(1.02, Ev, "Ev", va="center")
    ax.text(1.02, E_A, "Ea", va="center", color="green")
    ax.text(1.02, Ef, "Ef", va="center", color="r")

    # アクセプタ準位に捕獲された電子（青）
    if n_acceptor_bound_display > 0:
        y_a = sample_acceptor_level(T, n_acceptor_bound_display)
        x_a = np.random.uniform(0.18, 0.82, size=n_acceptor_bound_display)
        ax.scatter(
            x_a, y_a,
            s=22,
            color='blue',
            label="Acceptor-captured electrons"
        )

    # アクセプタ電離で生じた価電子帯正孔（赤）
    if p_acceptor_valence_display > 0:
        y_h_acc = sample_valence(T, p_acceptor_valence_display)
        x_h_acc = np.random.uniform(0.18, 0.82, size=p_acceptor_valence_display)
        ax.scatter(
            x_h_acc, y_h_acc,
            s=24,
            facecolors='white',
            edgecolors='red',
            linewidths=1.2,
            label="Acceptor-generated holes"
        )

    # 真性励起の伝導帯電子（紫）
    if n_intrinsic_display > 0:
        y_e_int = sample_conduction(T, n_intrinsic_display)
        x_e_int = np.random.uniform(0.18, 0.82, size=n_intrinsic_display)
        ax.scatter(
            x_e_int, y_e_int,
            s=20,
            color='purple',
            label="Valence-excited electrons"
        )

    # 真性励起の正孔（紫縁白抜き）
    if p_intrinsic_display > 0:
        y_h_int = sample_valence(T, p_intrinsic_display)
        x_h_int = np.random.uniform(0.18, 0.82, size=p_intrinsic_display)
        ax.scatter(
            x_h_int, y_h_int,
            s=20,
            facecolors='white',
            edgecolors='purple',
            linewidths=1.2,
            label="Intrinsic holes"
        )

    # 励起矢印
    if p_acceptor_valence_display > 0:
        ax.annotate(
            "",
            xy=(0.10, E_A - 0.01),
            xytext=(0.10, Ev + 0.01),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="gray")
        )

    if n_intrinsic_display > 0:
        ax.annotate(
            "",
            xy=(0.90, Ec - 0.01),
            xytext=(0.90, Ev + 0.01),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="gray")
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_ylabel("Energy (eV)")
    ax.tick_params(axis='both', direction='in')

    ax.set_title(
        f"T = {T_C:.0f} °C ({T:.2f} K)\n"
        f"NA = {NA:.2e} cm⁻³\n"
        f"ni = {ni:.2e} cm⁻³\n"
        f"acceptor ionized fraction = {frac_acceptor:.3f}\n"
        f"intrinsic fraction = {frac_intrinsic:.3f}"
    )

    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("p型半導体：低温凍結 → 外因性領域 → 真性領域")

T_C = st.slider("Temperature (°C)", -273, 1000, 25, 1)
log_NA = st.slider("log10(NA) [cm⁻³]", 12.0, 19.0, 16.0, 0.1)
NA = 10 ** log_NA

fig = plot_band(T_C, NA)
st.pyplot(fig)
