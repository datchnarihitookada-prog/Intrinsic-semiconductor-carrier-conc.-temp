import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# =========================================================
# Stable AlGaN/GaN HEMT DC Simulator
# Id unit: A/mm
# Includes:
#   Al fraction
#   AlGaN thickness
#   mobility
#   Rs
#   vsat
#   gate leakage
# =========================================================

q = 1.602e-19
eps0 = 8.854e-14  # F/cm

st.set_page_config(page_title="Stable AlGaN/GaN HEMT Simulator", layout="wide")

st.title("AlGaN/GaN HEMT 静特性シミュレータ 安定版")

# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("構造パラメータ")

x_Al = st.sidebar.slider("Al組成 x", 0.05, 0.50, 0.25, 0.01)
t_AlGaN_nm = st.sidebar.slider("AlGaN膜厚 (nm)", 3.0, 50.0, 20.0, 0.5)

st.sidebar.header("デバイスパラメータ")

Lg_um = st.sidebar.slider("ゲート長 Lg (µm)", 0.2, 10.0, 4.0, 0.1)
mu = st.sidebar.slider("移動度 μ (cm²/Vs)", 50, 2000, 500, 50)
Rs = st.sidebar.slider("Rs (Ω mm)", 0.0, 100.0, 20.0, 0.5)
vsat = st.sidebar.slider("vsat (cm/s)", 5.0e6, 3.0e7, 1.2e7, 1.0e6, format="%.1e")
epsr = st.sidebar.slider("AlGaN比誘電率", 7.0, 10.5, 9.0, 0.1)

Vth_shift = st.sidebar.slider("Vth補正 (V)", -10.0, 10.0, 0.0, 0.1)
lambda_clm = st.sidebar.slider("チャネル長変調 λ (1/V)", 0.0, 0.05, 0.005, 0.001)

st.sidebar.header("ゲートリーク")

log_Ig0 = st.sidebar.slider("log10(Ig0 [A/mm])", -14, -3, -9, 1)
Ig0 = 10 ** log_Ig0

Vleak = st.sidebar.slider("リーク開始電圧 (V)", -8.0, 3.0, -2.0, 0.1)
Sleak = st.sidebar.slider("リーク傾き (V)", 0.2, 5.0, 1.2, 0.1)

log_Igrev = st.sidebar.slider("log10(逆方向リーク係数 [A/mm])", -14, -4, -10, 1)
Igrev0 = 10 ** log_Igrev

st.sidebar.header("測定条件")

Vd_IdVg = st.sidebar.slider("Id-Vg測定時 Vd (V)", 0.1, 50.0, 10.0, 0.5)
Vd_max = st.sidebar.slider("Id-Vd最大 Vd (V)", 1.0, 100.0, 20.0, 1.0)

# =========================================================
# Unit conversion
# =========================================================
Lg = Lg_um * 1e-4       # cm
t_AlGaN = t_AlGaN_nm * 1e-7  # cm

# =========================================================
# Physical model
# =========================================================
def polarization_charge_density(x):
    """
    Very simplified polarization sheet charge.
    Return: cm^-2
    """
    sigma_C_m2 = 0.052 * x
    return sigma_C_m2 / q / 1e4


def ns_2deg(x, t_nm):
    """
    2DEG density.
    Return: cm^-2
    """
    ns_pol = polarization_charge_density(x)
    t0 = 4.0
    return ns_pol * (1.0 - np.exp(-t_nm / t0))


def calc_vth(x, t_nm):
    """
    Simplified threshold voltage.
    Return: V
    """
    t_cm = t_nm * 1e-7
    Cox = epsr * eps0 / t_cm
    ns = ns_2deg(x, t_nm)
    return -q * ns / Cox + Vth_shift


def gate_leakage(Vg):
    """
    Gate leakage current.
    Return: A/mm
    """
    Ig_forward = Ig0 * np.exp((Vg - Vleak) / Sleak)
    Ig_reverse = Igrev0 * np.exp((-Vg - 5.0) / 1.5)
    return Ig_forward + Ig_reverse


def intrinsic_Id_Amm(Vg, Vd_eff):
    """
    Intrinsic Id without Rs.
    Stable monotonic model.
    Return: A/mm
    """
    Vth = calc_vth(x_Al, t_AlGaN_nm)
    Vov = Vg - Vth

    if Vov <= 0 or Vd_eff <= 0:
        return 0.0

    Cox = epsr * eps0 / t_AlGaN

    beta = mu * Cox / Lg * 0.1  # A/mm/V^2

    Ecrit = vsat / mu
    Vdsat_vel = Ecrit * Lg
    Vdsat = min(Vov, Vdsat_vel)

    if Vd_eff <= Vdsat:
        Id = beta * (Vov * Vd_eff - 0.5 * Vd_eff**2)
    else:
        Idsat = beta * (Vov * Vdsat - 0.5 * Vdsat**2)
        Id = Idsat * (1.0 + lambda_clm * (Vd_eff - Vdsat))

    ns = ns_2deg(x_Al, t_AlGaN_nm)
    Id_vsat_limit = q * ns * vsat / 10.0  # A/mm

    Id = 1.0 / (1.0 / max(Id, 1e-30) + 1.0 / max(Id_vsat_limit, 1e-30))

    return max(Id, 0.0)


def Id_with_Rs_Amm(Vg, Vd):
    """
    Id including Rs.
    Solved by bisection:
        Id = intrinsic_Id(Vg, Vd - Id*Rs)
    Return: A/mm
    """
    if Vd <= 0:
        return 0.0

    Id_low = 0.0
    Id_high = Vd / max(Rs, 1e-12) if Rs > 0 else intrinsic_Id_Amm(Vg, Vd)

    if Rs == 0:
        return intrinsic_Id_Amm(Vg, Vd)

    for _ in range(100):
        Id_mid = 0.5 * (Id_low + Id_high)
        Vd_eff = max(Vd - Id_mid * Rs, 0.0)

        f_mid = Id_mid - intrinsic_Id_Amm(Vg, Vd_eff)

        if f_mid > 0:
            Id_high = Id_mid
        else:
            Id_low = Id_mid

        if abs(Id_high - Id_low) < 1e-10:
            break

    return 0.5 * (Id_low + Id_high)

# =========================================================
# Calculation
# =========================================================
ns = ns_2deg(x_Al, t_AlGaN_nm)
Vth = calc_vth(x_Al, t_AlGaN_nm)

st.subheader("計算結果")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("2DEG密度", f"{ns:.2e} cm⁻²")
c2.metric("Vth", f"{Vth:.2f} V")
c3.metric("μ", f"{mu:.0f} cm²/Vs")
c4.metric("Rs", f"{Rs:.1f} Ω mm")
c5.metric("vsat", f"{vsat:.2e} cm/s")

# =========================================================
# Id-Vg / Ig-Vg
# =========================================================
Vg_array = np.linspace(-12, 5, 500)

Id_Vg = np.array([Id_with_Rs_Amm(Vg, Vd_IdVg) for Vg in Vg_array])
Ig_Vg = np.array([gate_leakage(Vg) for Vg in Vg_array])

fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.semilogy(Vg_array, Id_Vg, label="Id")
ax1.semilogy(Vg_array, Ig_Vg, "--", label="Ig")
ax1.set_xlabel("Gate voltage Vg (V)", fontsize=13)
ax1.set_ylabel("Current (A/mm)", fontsize=13)
ax1.set_title(f"Id-Vg / Ig-Vg at Vd = {Vd_IdVg:.1f} V", fontsize=15)
ax1.tick_params(direction="in", labelsize=12)
ax1.legend(fontsize=11)
ax1.set_ylim(1e-12, max(Id_Vg.max(), Ig_Vg.max(), 1e-9) * 3)
fig1.tight_layout()

# =========================================================
# Id-Vd
# =========================================================
Vd_array = np.linspace(0, Vd_max, 500)
Vg_list = [-8, -6, -4, -2, 0, 2, 4]

fig2, ax2 = plt.subplots(figsize=(6, 4))

for Vg in Vg_list:
    Id = np.array([Id_with_Rs_Amm(Vg, Vd) for Vd in Vd_array])
    ax2.plot(Vd_array, Id, label=f"Vg = {Vg} V")

ax2.set_xlabel("Drain voltage Vd (V)", fontsize=13)
ax2.set_ylabel("Drain current Id (A/mm)", fontsize=13)
ax2.set_title("Id-Vd", fontsize=15)
ax2.tick_params(direction="in", labelsize=12)
ax2.legend(fontsize=10)
fig2.tight_layout()

# =========================================================
# 2DEG map
# =========================================================
x_list = np.linspace(0.05, 0.50, 120)
t_list = np.linspace(3, 50, 120)
X, T = np.meshgrid(x_list, t_list)
NS = ns_2deg(X, T)

fig3, ax3 = plt.subplots(figsize=(6, 4))
cont = ax3.contourf(X, T, NS / 1e13, levels=30)
cbar = fig3.colorbar(cont, ax=ax3)
cbar.set_label("2DEG density (×10¹³ cm⁻²)", fontsize=12)
ax3.plot(x_Al, t_AlGaN_nm, "o", markersize=8)
ax3.set_xlabel("Al fraction x", fontsize=13)
ax3.set_ylabel("AlGaN thickness (nm)", fontsize=13)
ax3.set_title("2DEG density map", fontsize=15)
ax3.tick_params(direction="in", labelsize=12)
fig3.tight_layout()

# =========================================================
# Display
# =========================================================
col1, col2 = st.columns(2)

with col1:
    st.pyplot(fig1)

with col2:
    st.pyplot(fig2)

st.pyplot(fig3)

# =========================================================
# Summary
# =========================================================
st.subheader("代表値")

st.write(f"""
- 最大 Id = {Id_Vg.max():.3e} A/mm
- Ig at Vg = 0 V = {gate_leakage(0):.3e} A/mm
- Ig at Vg = -5 V = {gate_leakage(-5):.3e} A/mm
- Vth = {Vth:.2f} V
- ns = {ns:.2e} cm⁻²
""")

st.subheader("この版の改善点")

st.write("""
この安定版では、Rsを含む電流計算を二分法で解いています。
そのため、以前のようなジャンプ、発振、負性抵抗的な数値暴走は起きにくくなっています。

また、速度飽和は移動度を無理に低下させる形ではなく、
電流上限として導入しています。

そのため、Id-Vdは通常のGaN HEMTらしく、
低Vdで線形、 高Vdで飽和する形になります。
""")
