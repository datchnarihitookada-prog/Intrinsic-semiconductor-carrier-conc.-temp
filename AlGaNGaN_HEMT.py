import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# =========================================================
# AlGaN/GaN HEMT DC Simulator
# Full variable-parameter version
# =========================================================

# -----------------------------
# Constants
# -----------------------------
q = 1.602e-19
eps0 = 8.854e-14  # F/cm

# -----------------------------
# Streamlit setting
# -----------------------------
st.set_page_config(
    page_title="AlGaN/GaN HEMT Simulator",
    layout="wide"
)

st.title("AlGaN/GaN HEMT 静特性シミュレータ")
st.write("Al組成、膜厚、Rs、vsat、ゲートリークを可変にした簡易DCモデルです。")

# -----------------------------
# Sidebar parameters
# -----------------------------
st.sidebar.header("構造パラメータ")

Al_fraction = st.sidebar.slider(
    "Al組成 x",
    0.05, 0.50, 0.25, 0.01
)

AlGaN_thickness_nm = st.sidebar.slider(
    "AlGaN膜厚 tAlGaN (nm)",
    3.0, 50.0, 20.0, 0.5
)

st.sidebar.header("デバイスパラメータ")

Lg_um = st.sidebar.slider(
    "ゲート長 Lg (µm)",
    0.2, 10.0, 4.0, 0.1
)

Wg_um = st.sidebar.slider(
    "ゲート幅 Wg (µm)",
    10.0, 1000.0, 120.0, 10.0
)

mu0 = st.sidebar.slider(
    "移動度 μ0 (cm²/Vs)",
    100, 2000, 900, 50
)

Rs_ohm_mm = st.sidebar.slider(
    "直列抵抗 Rs (Ω·mm)",
    0.0, 100.0, 10.0, 0.5
)

vsat = st.sidebar.slider(
    "飽和速度 vsat (cm/s)",
    5.0e6, 3.0e7, 1.5e7, 1.0e6,
    format="%.1e"
)

epsr_AlGaN = st.sidebar.slider(
    "AlGaN比誘電率",
    7.0, 10.5, 9.0, 0.1
)

st.sidebar.header("ゲートリークパラメータ")

log_Ig0 = st.sidebar.slider(
    "log10(Ig0 [A/mm])",
    -14, -3, -8, 1
)

Ig0_A_per_mm = 10 ** log_Ig0

Vg_leak_on = st.sidebar.slider(
    "リーク開始電圧 Vleak (V)",
    -8.0, 3.0, -3.0, 0.1
)

leak_slope = st.sidebar.slider(
    "リーク傾き Sleak (V)",
    0.2, 5.0, 1.2, 0.1
)

reverse_Ig0_A_per_mm = 10 ** st.sidebar.slider(
    "log10(逆方向リーク係数 [A/mm])",
    -14, -4, -9, 1
)

st.sidebar.header("測定条件")

Vd_fixed = st.sidebar.slider(
    "Id-Vg測定時 Vd (V)",
    0.1, 50.0, 10.0, 0.5
)

Vd_max = st.sidebar.slider(
    "Id-Vd最大 Vd (V)",
    1.0, 100.0, 20.0, 1.0
)

# -----------------------------
# Unit conversion
# -----------------------------
Lg = Lg_um * 1e-4      # cm
Wg = Wg_um * 1e-4      # cm
Wg_mm = Wg_um / 1000   # mm

# -----------------------------
# Physical models
# -----------------------------
def polarization_sheet_charge(x):
    """
    AlGaN/GaN interface polarization sheet charge.
    Very simplified model.
    Return: cm^-2
    """
    sigma_C_m2 = 0.052 * x
    ns_pol = sigma_C_m2 / q / 1e4
    return ns_pol


def two_deg_density(x, t_nm):
    """
    2DEG density.
    Return: cm^-2
    """
    ns_pol = polarization_sheet_charge(x)

    t0 = 4.0
    thickness_factor = 1.0 - np.exp(-t_nm / t0)

    ns = ns_pol * thickness_factor
    return ns


def threshold_voltage(x, t_nm):
    """
    Threshold voltage.
    Return: V
    """
    t_cm = t_nm * 1e-7
    Cox = epsr_AlGaN * eps0 / t_cm
    ns = two_deg_density(x, t_nm)

    Vth = -q * ns / Cox
    return Vth


def effective_mobility(Vd_eff):
    """
    Velocity saturation effect.
    μeff = μ0 / (1 + μ0 E / vsat)
    """
    E = np.abs(Vd_eff) / Lg
    mu_eff = mu0 / (1.0 + mu0 * E / vsat)
    return mu_eff


def intrinsic_drain_current(Vg, Vd_eff, x, t_nm):
    """
    Intrinsic drain current.
    Return: A
    """
    t_cm = t_nm * 1e-7
    Cox = epsr_AlGaN * eps0 / t_cm
    Vth = threshold_voltage(x, t_nm)

    Vov = Vg - Vth

    if Vov <= 0:
        return 0.0

    mu_eff = effective_mobility(Vd_eff)

    Vdsat = Vov

    if Vd_eff < Vdsat:
        Id = (
            mu_eff * Cox * Wg / Lg
            * (Vov * Vd_eff - 0.5 * Vd_eff**2)
        )
    else:
        Id = 0.5 * mu_eff * Cox * Wg / Lg * Vov**2

    return max(Id, 0.0)


def drain_current_with_Rs(Vg, Vd, x, t_nm):
    """
    Drain current including series resistance.
    Rs unit: Ω mm
    Return: A
    """
    R_total = Rs_ohm_mm / Wg_mm

    Id = intrinsic_drain_current(Vg, Vd, x, t_nm)

    for _ in range(100):
        Vd_eff = max(Vd - Id * R_total, 0.0)
        Id_new = intrinsic_drain_current(Vg, Vd_eff, x, t_nm)

        if abs(Id_new - Id) < 1e-12:
            break

        Id = 0.5 * Id + 0.5 * Id_new

    return Id


def gate_leakage_current(Vg):
    """
    Simple gate leakage current.
    Return: A/mm
    """
    Ig_forward = Ig0_A_per_mm * np.exp((Vg - Vg_leak_on) / leak_slope)

    Ig_reverse = reverse_Ig0_A_per_mm * np.exp((-Vg - 5.0) / 1.5)

    Ig = Ig_forward + Ig_reverse
    return Ig


# -----------------------------
# Main calculation
# -----------------------------
ns = two_deg_density(Al_fraction, AlGaN_thickness_nm)
Vth = threshold_voltage(Al_fraction, AlGaN_thickness_nm)

st.subheader("計算結果")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("2DEG密度", f"{ns:.2e} cm⁻²")

with c2:
    st.metric("Vth", f"{Vth:.2f} V")

with c3:
    st.metric("μ0", f"{mu0:.0f} cm²/Vs")

with c4:
    st.metric("Rs", f"{Rs_ohm_mm:.1f} Ω·mm")

with c5:
    st.metric("vsat", f"{vsat:.2e} cm/s")

# -----------------------------
# Id-Vg and Ig-Vg
# -----------------------------
Vg_array = np.linspace(-12, 4, 500)

Id_Vg = np.array([
    drain_current_with_Rs(Vg, Vd_fixed, Al_fraction, AlGaN_thickness_nm)
    for Vg in Vg_array
])

Id_Vg_Amm = Id_Vg / Wg_mm

Ig_Vg_Amm = np.array([
    gate_leakage_current(Vg)
    for Vg in Vg_array
])

fig1, ax1 = plt.subplots(figsize=(6, 4))

ax1.semilogy(Vg_array, Id_Vg_Amm, label="Id")
ax1.semilogy(Vg_array, Ig_Vg_Amm, "--", label="Ig")

ax1.set_xlabel("Gate voltage Vg (V)")
ax1.set_ylabel("Current (A/mm)")
ax1.set_title(f"Id-Vg / Ig-Vg at Vd = {Vd_fixed:.1f} V")
ax1.tick_params(direction="in")
ax1.legend()

ymax1 = max(Id_Vg_Amm.max(), Ig_Vg_Amm.max()) * 3
ax1.set_ylim(1e-12, ymax1)

fig1.tight_layout()

# -----------------------------
# Id-Vd
# -----------------------------
Vd_array = np.linspace(0, Vd_max, 500)
Vg_values = [-8, -6, -4, -2, 0, 2, 4]

fig2, ax2 = plt.subplots(figsize=(6, 4))

for Vg in Vg_values:
    Id = np.array([
        drain_current_with_Rs(Vg, Vd, Al_fraction, AlGaN_thickness_nm)
        for Vd in Vd_array
    ])

    ax2.plot(Vd_array, Id / Wg_mm, label=f"Vg = {Vg} V")

ax2.set_xlabel("Drain voltage Vd (V)")
ax2.set_ylabel("Drain current Id (A/mm)")
ax2.set_title("Id-Vd")
ax2.tick_params(direction="in")
ax2.legend()
fig2.tight_layout()

# -----------------------------
# 2DEG map
# -----------------------------
x_list = np.linspace(0.05, 0.50, 120)
t_list = np.linspace(3, 50, 120)

X, TT = np.meshgrid(x_list, t_list)
NS = two_deg_density(X, TT)

fig3, ax3 = plt.subplots(figsize=(6, 4))

cont = ax3.contourf(X, TT, NS / 1e13, levels=30)
cbar = fig3.colorbar(cont, ax=ax3)
cbar.set_label("2DEG density (×10¹³ cm⁻²)")

ax3.plot(Al_fraction, AlGaN_thickness_nm, "o", markersize=8)

ax3.set_xlabel("Al fraction x")
ax3.set_ylabel("AlGaN thickness (nm)")
ax3.set_title("2DEG density map")
ax3.tick_params(direction="in")
fig3.tight_layout()

# -----------------------------
# Display figures
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.pyplot(fig1)

with col2:
    st.pyplot(fig2)

st.pyplot(fig3)

# -----------------------------
# Data table
# -----------------------------
st.subheader("代表値")

Idmax_Amm = Id_Vg_Amm.max()
Ig_at_0V = gate_leakage_current(0.0)
Ig_at_minus5V = gate_leakage_current(-5.0)

st.write(f"""
- 最大 Id = {Idmax_Amm:.3e} A/mm
- Ig at Vg = 0 V = {Ig_at_0V:.3e} A/mm
- Ig at Vg = -5 V = {Ig_at_minus5V:.3e} A/mm
- Vth = {Vth:.2f} V
- ns = {ns:.2e} cm⁻²
""")

# -----------------------------
# Explanation
# -----------------------------
st.subheader("モデルの意味")

st.write("""
このプログラムでは、Al組成とAlGaN膜厚から簡易的に分極電荷を計算し、
2DEG密度としきい値電圧を推定しています。

ドレイン電流には以下を入れています。

1. MOSFET型の簡易Id式
2. Rsによる電圧降下
3. vsatによる速度飽和
4. 簡易的なゲートリークIg

IdはHEMTで一般的な A/mm 表示です。
""")

st.subheader("注意")

st.write("""
これは実測フィッティング用の簡易モデルです。

厳密な物理モデルではありません。
以下はまだ入っていません。

- Poisson-Schrödinger計算
- 表面準位
- ゲート金属仕事関数
- AlGaN/GaNバンドオフセット
- ショットキー障壁高さの厳密計算
- 電流コラプス
- 自己発熱
- チャネル長変調
- ゲート端電界集中
""")
