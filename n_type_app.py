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

# ドナー準位（Ecの少し下）
E_D = Ec - 0.045  # eV

# 表示個数
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
    ドナー準位の電子が伝導帯に上がった割合の簡易モデル
    0 Kで0、温度上昇で1へ近づく
    """
    if T <= 0:
        return 0.0
    return np.exp(-E_ion / (k_B * T))


def carrier_density_n_type(T, ND):
    """
    簡易3段階モデル

    1) ドナー準位 -> 伝導帯
    2) ドナーがほぼ出尽くした後
    3) 真性励起（価電子帯 -> 伝導帯）が支配的になる

    戻り値:
      ni               : 真性キャリア密度
      n_total          : 伝導帯電子の総数密度
      p_total          : 正孔密度
      n_from_donor     : ドナー由来の伝導帯電子
      n_from_intrinsic : 真性励起由来の伝導帯電子
      n_donor_bound    : ドナー準位に残る電子
      frac_donor       : ドナー励起割合
    """
    ni = intrinsic_density(T)

    # ドナー励起
    frac_donor = donor_ionized_fraction(T, Ec - E_D)
    n_from_donor = ND * frac_donor
    n_from_donor = min(n_from_donor, ND)

    # ドナー準位に残る電子
    n_donor_bound = ND - n_from_donor

    # 真性励起は「ドナーがかなり空になってから効いてくる」ように重み付け
    # donor depletion factor: 0 -> ドナー未励起, 1 -> ドナーほぼ空
    depletion = frac_donor

    # 真性励起寄与
    n_from_intrinsic = depletion * ni

    # 総電子・正孔
    n_total = n_from_donor + n_from_intrinsic
    p_total = n_from_intrinsic

    return ni, n_total, p_total, n_from_donor, n_from_intrinsic, n_donor_bound, frac_donor


def fermi_level_n_type(T, frac_donor, n_total):
    """
    表示用の簡易フェルミ準位
    - 0 Kで Ed
    - ドナー励起が進むと Ec 側へ上がる
    - 高温真性化で中央寄りへ少し戻す
    """
    if T <= 0:
        return E_D

    # 低温〜中温：Ed -> Ec に少し寄る
    Ef_low = E_D + 0.6 * (Ec - E_D) * frac_donor

    # 高温では真性化で中央へ寄る傾向を少し入れる
    ni = intrinsic_density(T)
    if n_total > 0:
        intrinsic_ratio = min(1.0, ni / max(n_total, 1e-30))
    else:
        intrinsic_ratio = 0.0

    Ef = (1 - 0.4 * intrinsic_ratio) * Ef_low + (0.4 * intrinsic_ratio) * 0.0
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
        frac_donor
    ) = carrier_density_n_type(T, ND)

    Ef = fermi_level_n_type(T, frac_donor, n_total)

    # -------------------------
    # 表示個数の決め方
    # -------------------------
    # 1) ドナー由来電子 30個を固定母数として、
    #    そのうち何個が Ed に残り、何個が Ec へ上がるかを表す
    n_donor_cb_display = int(round(N_DONOR_DISPLAY * frac_donor))
    n_donor_bound_display = N_DONOR_DISPLAY - n_donor_cb_display

    # 2) 真性励起は「ドナー励起後」に追加で増えるようにする
    #    ni の絶対値だけでなく、ドナー励起の進行度も掛ける
    intrinsic_strength = 0.0
    if n_from_intrinsic > 0:
        intrinsic_strength = min(
            1.0,
            (np.log10(max(n_from_intrinsic, 1e-30)) - 8.0) / (17.0 - 8.0)
        )
        intrinsic_strength = max(0.0, intrinsic_strength)

    n_intrinsic_display = int(round(N_INTRINSIC_MAX_DISPLAY * intrinsic_strength))
    p_intrinsic_display = n_intrinsic_display

    fig, ax = plt.subplots(figsize=(5, 8))

    # バンドと準位
    ax.plot([0, 1], [Ec, Ec], 'k', linewidth=2)
    ax.plot([0, 1], [Ev, Ev], 'k', linewidth=2)
    ax.plot([0, 1], [E_D, E_D], '--', color='green', linewidth=1.5)
    ax.plot([0, 1], [Ef, Ef], 'r--', linewidth=1.5)

    # ラベル
    ax.text(1.02, Ec, "Ec", va="center")
    ax.text(1.02, Ev, "Ev", va="center")
    ax.text(1.02, E_D, "Ed", va="center", color="green")
    ax.text(1.02, Ef, "Ef", va="center", color="r")

    # ドナー準位に残る電子
    if n_donor_bound_display > 0:
        y_d = sample_donor_level(T, n_donor_bound_display)
        x_d = np.random.uniform(0.18, 0.82, size=n_donor_bound_display)
        ax.scatter(
            x_d, y_d,
            s=28,
            color='orange',
            label="Donor-bound electrons"
        )

    # ドナー由来で伝導帯へ上がった電子
    if n_donor_cb_display > 0:
        y_e_donor = sample_conduction(T, n_donor_cb_display)
        x_e_donor = np.random.uniform(0.18, 0.82, size=n_donor_cb_display)
        ax.scatter(
            x_e_donor, y_e_donor,
            s=20,
            color='blue',
            label="Donor-excited electrons"
        )

    # 真性励起による伝導帯電子
    if n_intrinsic_display > 0:
        y_e_int = sample_conduction(T, n_intrinsic_display)
        x_e_int = np.random.uniform(0.18, 0.82, size=n_intrinsic_display)
        ax.scatter(
            x_e_int, y_e_int,
            s=20,
            facecolors='none',
            edgecolors='blue',
            linewidths=1.2,
            label="Intrinsic electrons"
        )

    # 真性励起による正孔
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

    # 励起の流れを補助表示
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
        f"n_donor->CB = {n_from_donor:.2e}, n_intrinsic = {n_from_intrinsic:.2e}, p = {p_total:.2e}"
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
