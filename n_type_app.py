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
E_D = Ec - 0.045   # ドナー準位

# 表示用個数
N_DONOR_DISPLAY = 30
N_INTRINSIC_MAX_DISPLAY = 30


# -----------------------------
# 基本関数
# -----------------------------
def intrinsic_density(T):
    if T <= 0:
        return 0.0
    return np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * k_B * T))


def donor_ionized_fraction(T):
    """
    教育用の簡易モデル
    低温では未電離、室温付近ではほぼ完全電離になるように設定
    """
    if T <= 0:
        return 0.0

    # 代表温度と立ち上がり幅
    T0 = 120.0   # K
    dT = 18.0    # K

    frac = 1.0 / (1.0 + np.exp(-(T - T0) / dT))
    return float(np.clip(frac, 0.0, 1.0))


def intrinsic_fraction(T, ND):
    """
    真性領域への移行の強さ
    ni << ND では 0 に近く、ni >> ND で 1 に近づく
    """
    if T <= 0:
        return 0.0

    ni = intrinsic_density(T)
    if ni <= 0:
        return 0.0

    r = ni / ND

    # ni/ND が 10^-2 以下ではほぼ見せない
    # ni/ND が 1 以上でかなり強く出す
    x = np.log10(max(r, 1e-30))
    x0 = -0.5   # 遷移中心
    dx = 0.35   # 立ち上がり幅
    frac = 1.0 / (1.0 + np.exp(-(x - x0) / dx))
    return float(np.clip(frac, 0.0, 1.0))


def carrier_density_n_type(T, ND):
    """
    表示用の3領域モデル
    1) 低温: ドナー準位に電子
    2) 室温付近: ドナーはほぼ完全電離
    3) 高温: 真性励起が増える
    """
    ni = intrinsic_density(T)

    frac_donor = donor_ionized_fraction(T)
    n_from_donor = ND * frac_donor
    n_donor_bound = ND - n_from_donor

    frac_intrinsic = intrinsic_fraction(T, ND)
    n_from_intrinsic = ni * frac_intrinsic
    p_total = n_from_intrinsic

    n_total = n_from_donor + n_from_intrinsic

    return (
        ni,
        n_total,
        p_total,
        n_from_donor,
        n_from_intrinsic,
        n_donor_bound,
        frac_donor,
        frac_intrinsic
    )


def fermi_level_n_type(T, frac_donor, frac_intrinsic):
    """
    表示用の簡易フェルミ準位
    """
    if T <= 0:
        return E_D

    # 外因性領域では Ec 側へ
    Ef_extrinsic = E_D + 0.9 * (Ec - E_D) * frac_donor

    # 真性領域で中央へ戻す
    Ef = (1.0 - 0.85 * frac_intrinsic) * Ef_extrinsic + (0.85 * frac_intrinsic) * 0.0
    return Ef


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


def sample_donor_level(T, n_points):
    if n_points <= 0:
        return np.array([])
    width = 0.001 if T <= 0 else min(0.004, 0.12 * k_B * T)
    return E_D + np.random.normal(0.0, width, size=n_points)

def density_to_points(density, max_points=30, log_min=8, log_max=19):
    if density <= 0:
        return 0
    log_n = np.log10(density)
    points = (log_n - log_min) / (log_max - log_min) * max_points
    return int(np.clip(points, 0, max_points))


# -----------------------------
# プロット
# -----------------------------
def plot_band(T_C, ND):
    T = T_C + 273.15  # °C -> K

    (
        ni,
        n_total,
        p_total,
        n_from_donor,
        n_from_intrinsic,
        n_donor_bound,
        frac_donor,
        frac_intrinsic
    ) = carrier_density_n_type(T, ND)

    Ef = fermi_level_n_type(T, frac_donor, frac_intrinsic)

    # -------------------------
    # 表示個数
    # -------------------------
    # ドナー由来電子は30個を固定母数
    n_donor_cb_display = int(round(N_DONOR_DISPLAY * frac_donor))
    n_donor_bound_display = N_DONOR_DISPLAY - n_donor_cb_display

    # 真性励起は高温で個数制限なし（ここが修正ポイント）
    n_intrinsic_display = density_to_points(
        n_from_intrinsic,
        max_points=300,
        log_min=8,
        log_max=19
    )
    p_intrinsic_display = n_intrinsic_display

    fig, ax = plt.subplots(figsize=(5, 8))

    # バンド・準位
    ax.plot([0, 1], [Ec, Ec], 'k', linewidth=2)
    ax.plot([0, 1], [Ev, Ev], 'k', linewidth=2)
    ax.plot([0, 1], [E_D, E_D], '--', color='green', linewidth=1.5)
    ax.plot([0, 1], [Ef, Ef], 'r--', linewidth=1.5)

    # ラベル
    ax.text(1.02, Ec, "Ec", va="center")
    ax.text(1.02, Ev, "Ev", va="center")
    ax.text(1.02, E_D, "Ed", va="center", color="green")
    ax.text(1.02, Ef, "Ef", va="center", color="r")

    # ドナー準位に残る電子（青）
    if n_donor_bound_display > 0:
        y_d = sample_donor_level(T, n_donor_bound_display)
        x_d = np.random.uniform(0.18, 0.82, size=n_donor_bound_display)
        ax.scatter(
            x_d, y_d,
            s=28,
            color='blue',
            label="Donor electrons"
        )

    # ドナー準位から伝導帯へ励起した電子（青）
    if n_donor_cb_display > 0:
        y_e_donor = sample_conduction(T, n_donor_cb_display)
        x_e_donor = np.random.uniform(0.18, 0.82, size=n_donor_cb_display)
        ax.scatter(
            x_e_donor, y_e_donor,
            s=22,
            color='blue',
            label="Donor-excited electrons"
        )

    # 価電子帯から伝導帯へ励起した電子（紫）
    if n_intrinsic_display > 0:
        y_e_int = sample_conduction(T, n_intrinsic_display)
        x_e_int = np.random.uniform(0.18, 0.82, size=n_intrinsic_display)
        ax.scatter(
            x_e_int, y_e_int,
            s=24,
            color='purple',
            label="Valence-excited electrons"
        )

    # 正孔（赤縁白抜き）
    if p_intrinsic_display > 0:
        y_h = sample_valence(T, p_intrinsic_display)
        x_h = np.random.uniform(0.18, 0.82, size=p_intrinsic_display)
        ax.scatter(
            x_h, y_h,
            s=24,
            facecolors='white',
            edgecolors='red',
            linewidths=1.2,
            label="Holes"
        )

    # 励起矢印
    if n_donor_cb_display > 0:
        ax.annotate(
            "",
            xy=(0.10, Ec - 0.01),
            xytext=(0.10, E_D + 0.01),
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
        f"ND = {ND:.2e} cm⁻³\n"
        f"ni = {ni:.2e} cm⁻³\n"
        f"donor ionized fraction = {frac_donor:.3f}\n"
        f"intrinsic fraction = {frac_intrinsic:.3f}"
    )

    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("n型半導体：低温凍結 → 外因性領域 → 真性領域")

T_C = st.slider("Temperature (°C)", -273, 1000, 25, 1)
log_ND = st.slider("log10(ND) [cm⁻³]", 12.0, 19.0, 16.0, 0.1)
ND = 10 ** log_ND

fig = plot_band(T_C, ND)
st.pyplot(fig)
