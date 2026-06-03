import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# =========================================================
# AlGaN/GaN HEMT 簡易DC特性シミュレータ
# 入力パラメータ：
#   1. Al組成 x
#   2. AlGaN膜厚 t_AlGaN
# =========================================================

# -----------------------------
# 定数
# -----------------------------
q = 1.602e-19
eps0 = 8.854e-14  # F/cm

# -----------------------------
# 固定デバイスパラメータ
# -----------------------------
Lg_um = 4.0
Wg_um = 120.0
mu = 900          # cm2/Vs
epsr_AlGaN = 9.0

Lg = Lg_um * 1e-4
Wg = Wg_um * 1e-4

# -----------------------------
# Streamlit設定
# -----------------------------
st.set_page_config(page_title="AlGaN/GaN HEMT Simulator", layout="wide")

st.title("AlGaN/GaN HEMT 静特性シミュレータ")
st.write("入力パラメータは **Al組成** と **AlGaN膜厚** の2つだけです。")

# -----------------------------
# 入力パラメータ
# -----------------------------
Al_fraction = st.slider("Al組成 x", 0.05, 0.50, 0.25, 0.01)
AlGaN_thickness_nm = st.slider("AlGaN膜厚 tAlGaN (nm)", 3.0, 40.0, 20.0, 0.5)

# -----------------------------
# 物理モデル
# -----------------------------
def polarization_sheet_charge(x):
    """
    AlGaN/GaN界面の分極シート電荷密度を簡易近似で計算
    戻り値: cm^-2
    """
    sigma_C_m2 = 0.052 * x
    ns_pol = sigma_C_m2 / q / 1e4
    return ns_pol


def two_deg_density(x, t_nm):
    """
    Al組成とAlGaN膜厚から2DEG密度を推定
    戻り値: cm^-2
    """
    ns_pol = polarization_sheet_charge(x)

    # 薄膜側で2DEGが十分形成されない効果を簡易的に導入
    t0 = 4.0  # nm
    thickness_factor = 1.0 - np.exp(-t_nm / t0)

    ns = ns_pol * thickness_factor
    return ns


def threshold_voltage(x, t_nm):
    """
    2DEG密度からしきい値電圧を簡易計算
    戻り値: V
    """
    t_cm = t_nm * 1e-7
    Cox = epsr_AlGaN * eps0 / t_cm
    ns = two_deg_density(x, t_nm)

    Vth = -q * ns / Cox
    return Vth


def drain_current(Vg, Vd, x, t_nm):
    """
    簡易MOSFET型モデルによるHEMTドレイン電流
    Vg: V
    Vd: numpy array
    戻り値: A
    """
    t_cm = t_nm * 1e-7
    Cox = epsr_AlGaN * eps0 / t_cm
    Vth = threshold_voltage(x, t_nm)

    Vov = Vg - Vth
    Id = np.zeros_like(Vd)

    if Vov <= 0:
        return Id

    Vdsat = Vov

    linear_region = Vd < Vdsat
    saturation_region = ~linear_region

    Id[linear_region] = (
        mu * Cox * Wg / Lg
        * (Vov * Vd[linear_region] - 0.5 * Vd[linear_region] ** 2)
    )

    Id[saturation_region] = (
        0.5 * mu * Cox * Wg / Lg * Vov ** 2
    )

    return Id


# -----------------------------
# 計算
# -----------------------------
ns = two_deg_density(Al_fraction, AlGaN_thickness_nm)
Vth = threshold_voltage(Al_fraction, AlGaN_thickness_nm)

st.subheader("計算結果")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("2DEG密度", f"{ns:.2e} cm⁻²")

with col2:
    st.metric("しきい値電圧 Vth", f"{Vth:.2f} V")

with col3:
    st.metric("移動度", f"{mu:.0f} cm²/Vs")

# -----------------------------
# Id-Vg
# -----------------------------
Vg_array = np.linspace(-10, 3, 400)
Vd_fixed = 10.0

Id_Vg = np.array([
    drain_current(Vg, np.array([Vd_fixed]), Al_fraction, AlGaN_thickness_nm)[0]
    for Vg in Vg_array
])

fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.semilogy(Vg_array, Id_Vg / Wg)
ax1.set_xlabel("Gate voltage Vg (V)")
ax1.set_ylabel("Drain current Id (A/cm)")
ax1.set_title(f"Id-Vg at Vd = {Vd_fixed:.1f} V")
ax1.tick_params(direction="in")
ax1.set_ylim(1e-8, max(Id_Vg / Wg) * 2 if max(Id_Vg) > 0 else 1e-2)
fig1.tight_layout()

# -----------------------------
# Id-Vd
# -----------------------------
Vd_array = np.linspace(0, 20, 400)
Vg_values = [-6, -4, -2, 0, 2]

fig2, ax2 = plt.subplots(figsize=(6, 4))

for Vg in Vg_values:
    Id = drain_current(Vg, Vd_array, Al_fraction, AlGaN_thickness_nm)
    ax2.plot(Vd_array, Id / Wg, label=f"Vg = {Vg} V")

ax2.set_xlabel("Drain voltage Vd (V)")
ax2.set_ylabel("Drain current Id (A/cm)")
ax2.set_title("Id-Vd")
ax2.tick_params(direction="in")
ax2.legend()
fig2.tight_layout()

# -----------------------------
# 表示
# -----------------------------
col4, col5 = st.columns(2)

with col4:
    st.pyplot(fig1)

with col5:
    st.pyplot(fig2)

# -----------------------------
# Al組成・膜厚依存性マップ
# -----------------------------
st.subheader("Al組成・膜厚による2DEG密度マップ")

x_list = np.linspace(0.05, 0.50, 100)
t_list = np.linspace(3, 40, 100)

X, T = np.meshgrid(x_list, t_list)
NS = two_deg_density(X, T)

fig3, ax3 = plt.subplots(figsize=(6, 4))
cont = ax3.contourf(X, T, NS / 1e13, levels=30)
cbar = fig3.colorbar(cont, ax=ax3)
cbar.set_label("2DEG density (×10¹³ cm⁻²)")

ax3.plot(Al_fraction, AlGaN_thickness_nm, "o", markersize=8)
ax3.set_xlabel("Al fraction x")
ax3.set_ylabel("AlGaN thickness (nm)")
ax3.set_title("2DEG density map")
ax3.tick_params(direction="in")
fig3.tight_layout()

st.pyplot(fig3)

# -----------------------------
# コメント
# -----------------------------
st.subheader("モデルの注意点")

st.write("""
このプログラムは、AlGaN/GaN HEMTの傾向を見るための簡易モデルです。

- Al組成が高いほど分極電荷が増える
- AlGaN膜厚が厚いほど2DEG密度が増える
- 2DEG密度が増えるとVthは負側へシフトする
- Idは増加する

という傾向を見ることを目的にしています。

実測に近づけるには、表面準位、ゲート金属仕事関数、バンドオフセット、
アクセス抵抗、速度飽和、自己発熱、ゲートリークを追加する必要があります。
""")
