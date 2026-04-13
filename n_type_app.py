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

# ドナー準位（Ec の少し下）
E_D = Ec - 0.045  # eV

# 表示用の個数
N_DONOR_DISPLAY = 30
N_INTRINSIC_MAX_DISPLAY = 30


# -----------------------------
# 基本関数
# -----------------------------
def intrinsic_density(T):
    if T <= 0:
        return 0.0
    return np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * k_B * T))


def donor_ionized_fraction(T, E_ion):
    """
    ドナー準位 -> 伝導帯 への励起割合の簡易モデル
    0 K で 0、温度上昇で 1 に近づく
    """
    if T <= 0:
        return 0.0
    return np.exp(-E_ion / (k_B * T))


def intrinsic_switch(frac_donor, threshold=0.98):
    """
    ドナー電子がほぼ出尽くしてからだけ
    真性励起を表示するためのスイッチ
    """
    if frac_donor <= threshold:
        return 0.0
    # threshold～1 の間で 0→1 にする
    return min(1.0, (frac_donor - threshold) / (1.0 - threshold))


def carrier_density_n_type(T, ND):
    """
    表示用3段階モデル

    1) ドナー準位に電子がいる
    2) ドナー準位 -> 伝導帯 に励起
    3) ドナーがほぼ空になってから Ev -> Ec の真性励起

    戻り値:
      ni
      n_total
      p_total
      n_from_donor
      n_from_intrinsic
      n_donor_bound
      frac_donor
      sw_intrinsic
    """
    ni = intrinsic_density(T)

    # ドナー励起
    frac_donor = donor_ionized_fraction(T, Ec - E_D)
    frac_donor = min(max(frac_donor, 0.0), 1.0)

    n_from_donor = ND * frac_donor
    n_donor_bound = ND - n_from_donor

    # ドナーがほぼ空になってからだけ真性励起を許す
    sw_intrinsic = intrinsic_switch(frac_donor, threshold=0.98)
    n_from_intrinsic = sw_intrinsic * ni
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
        sw_intrinsic
    )


def fermi_level_n_type(T, frac_donor, sw_intrinsic):
    """
    表示用の簡易フェルミ準位
    - 0 Kでは Ed
    - ドナー励起で Ec 側へ寄る
    - 真性励起が効いてきたら中央へ少し戻る
    """
    if T <= 0:
        return E_D

    Ef_donor = E_D + 0.75 * (Ec - E_D) * frac_donor
    Ef = (1.0 - 0.5 * sw_intrinsic) * Ef_donor + (0.5 * sw_intrinsic) * 0.0
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


def sample_donor_level(T, n_points):
    if n_points <= 0:
        return np.array([])
    width = 0.001 if T <= 0 else min(0.004, 0.15 * k_B * T)
    return E_D + np.random.normal(0.0, width, size=n_points)


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
        sw_intrinsic
    ) = carrier_density_n_type(T, ND)

    Ef = fermi_level_n_type(T, frac_donor, sw_intrinsic)

    # -------------------------
    # 表示個数
    # -------------------------
    # ドナー由来の電子は常に30個を母数として、
    # Ed に残るか Ec に上がるかだけを見せる
    n_donor_cb_display = int(round(N_DONOR_DISPLAY * frac_donor))
    n_donor_bound_display = N_DONOR_DISPLAY - n_donor_cb_display

    # 真性励起は、ドナーがほぼ空になった後にだけ追加表示
    if sw_intrinsic > 0.0 and n_from_intrinsic > 0.0:
        intrinsic_strength = min(
            1.0,
            (np.log10(max(n_from_intrinsic, 1e-30)) - 8.0) / (17.0 - 8.0)
        )
        intrinsic_strength = max(0.0, intrinsic_strength)
        intrinsic_strength *= sw_intrinsic
        n_intrinsic_display = int(round(N_INTRINSIC_MAX_DISPLAY * intrinsic_strength))
    else:
        n_intrinsic_display = 0

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

    # -------------------------
    # ドナー準位に残る電子（青）
    # -------------------------
    if n_donor_bound_display > 0:
        y_d = sample_donor_level(T, n_donor_bound_display)
        x_d = np.random.uniform(0.18, 0.82, size=n_donor_bound_display)
        ax.scatter(
            x_d, y_d,
            s=28,
            color='blue',
            label="Donor electrons"
        )

    # -------------------------
    # ドナー準位から伝導帯へ励起した電子（青）
    # -------------------------
    if n_donor_cb_display > 0:
        y_e_donor = sample_conduction(T, n_donor_cb_display)
        x_e_donor = np.random.uniform(0.18, 0.82, size=n_donor_cb_display)
        ax.scatter(
            x_e_donor, y_e_donor,
            s=20,
            color='blue',
            label="Donor-excited electrons"
        )

    # -------------------------
    # 価電子帯から伝導帯へ励起した電子（紫）
    # -------------------------
    if n_intrinsic_display > 0:
        y_e_int = sample_conduction(T, n_intrinsic_display)
        x_e_int = np.random.uniform(0.18, 0.82, size=n_intrinsic_display)
        ax.scatter(
            x_e_int, y_e_int,
            s=24,
            color='purple',
            label="Valence-excited electrons"
        )

    # -------------------------
    # 真性励起に対応する正孔（赤縁白抜き）
    # -------------------------
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
    if 0 < n_donor_cb_display < N_DONOR_DISPLAY:
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
        f"n_donor->CB = {n_from_donor:.2e}\n"
        f"n_valence->CB = {n_from_intrinsic:.2e}, p = {p_total:.2e}"
    )

    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("n型半導体：ドナー励起 → 真性励起")

T_C = st.slider("Temperature (°C)", -273, 1000, 25, 1)
log_ND = st.slider("log10(ND) [cm⁻³]", 12.0, 19.0, 16.0, 0.1)
ND = 10 ** log_ND

fig = plot_band(T_C, ND)
st.pyplot(fig)
