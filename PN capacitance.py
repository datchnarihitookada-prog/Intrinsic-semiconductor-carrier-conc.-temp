import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st

# =========================
# Streamlit設定
# =========================
st.set_page_config(page_title="pn接合 空乏層シミュレーター", layout="wide")

st.title("pn接合の不純物分布・電荷分布・電位分布")

# =========================
# 日本語文字化け対策
# =========================
mpl.rcParams["font.family"] = "Yu Gothic"  # ダメなら "Meiryo"
mpl.rcParams["axes.unicode_minus"] = False

# =========================
# x軸
# =========================
x = np.linspace(-5, 5, 2000)

# =========================
# サイドバー：スライダー
# =========================
st.sidebar.header("パラメータ")

log_Na = st.sidebar.slider(
    "log10(Na)",
    min_value=14.0,
    max_value=18.0,
    value=16.0,
    step=0.1
)

log_Nd = st.sidebar.slider(
    "log10(Nd)",
    min_value=14.0,
    max_value=18.0,
    value=16.0,
    step=0.1
)

Na = 10 ** log_Na
Nd = 10 ** log_Nd

# =========================
# 計算関数
# =========================
def calc(Na, Nd):
    W = 3.5

    # 電荷中性条件：Na*xp = Nd*xn
    xp = W * Nd / (Na + Nd)   # p側空乏層幅
    xn = W * Na / (Na + Nd)   # n側空乏層幅

    impurity = np.where(x < 0, -Na, Nd)

    charge = np.zeros_like(x)
    charge[(x >= -xp) & (x < 0)] = -Na
    charge[(x >= 0) & (x <= xn)] = Nd

    V = np.zeros_like(x)

    mask_p = (x >= -xp) & (x < 0)
    mask_n = (x >= 0) & (x <= xn)
    mask_r = x > xn

    V[mask_p] = 0.5 * Na * (x[mask_p] + xp) ** 2

    V0 = 0.5 * Na * xp ** 2
    V[mask_n] = V0 + Na * xp * x[mask_n] - 0.5 * Nd * x[mask_n] ** 2

    Vmax = V0 + Na * xp * xn - 0.5 * Nd * xn ** 2
    V[mask_r] = Vmax

    scale = max(Na, Nd)
    impurity = impurity / scale
    charge = charge / scale

    if np.max(V) != 0:
        V = V / np.max(V)

    return impurity, charge, V, xp, xn

# =========================
# 計算
# =========================
impurity, charge, V, xp, xn = calc(Na, Nd)

# =========================
# 状態表示
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Na", f"{Na:.2e} cm⁻³")
col2.metric("Nd", f"{Nd:.2e} cm⁻³")

if Nd < Na:
    relation = r"$N_D < N_A$"
elif Nd > Na:
    relation = r"$N_D > N_A$"
else:
    relation = r"$N_D = N_A$"

col3.markdown(f"### {relation}")

st.markdown(
    rf"""
    電荷中性条件：

    \[
    N_A x_p = N_D x_n
    \]

    現在の空乏層幅比：

    \[
    x_p = {xp:.2f}, \quad x_n = {xn:.2f}
    \]
    """
)

# =========================
# Figure
# =========================
fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
fig.subplots_adjust(hspace=0.55)

ax1, ax2, ax3 = axes

# =========================
# 描画
# =========================
ax1.step(x, impurity, where="post", color="deepskyblue", lw=3)
ax2.step(x, charge, where="post", color="blue", lw=3)
ax3.plot(x, V, color="deepskyblue", lw=3)

ax1.fill_between(
    x, 0, impurity,
    where=x < 0,
    step="post",
    color="deepskyblue",
    alpha=0.25
)

ax1.fill_between(
    x, 0, impurity,
    where=x >= 0,
    step="post",
    color="deepskyblue",
    alpha=0.25
)

ax2.fill_between(
    x, 0, charge,
    where=(x >= -xp) & (x < 0),
    step="post",
    color="blue",
    alpha=0.45
)

ax2.fill_between(
    x, 0, charge,
    where=(x >= 0) & (x <= xn),
    step="post",
    color="blue",
    alpha=0.45
)

# 空乏層境界線
for ax in axes:
    ax.axvline(-xp, color="gray", ls="--", lw=1)
    ax.axvline(0, color="black", lw=1.5)
    ax.axvline(xn, color="gray", ls="--", lw=1)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlim(-5, 5)
    ax.set_yticks([])
    ax.grid(False)

# =========================
# 軸設定
# =========================
ax1.set_title("(a) 不純物分布", fontsize=14)
ax2.set_title("(b) 電荷分布", fontsize=14)
ax3.set_title("(c) 電位分布", fontsize=14)

ax1.set_ylabel("不純物濃度", fontsize=13)
ax2.set_ylabel("Q", fontsize=13)
ax3.set_ylabel("V", fontsize=13)
ax3.set_xlabel("x", fontsize=13)

ax1.set_ylim(-1.3, 1.3)
ax2.set_ylim(-1.3, 1.3)
ax3.set_ylim(-0.1, 1.2)

ax1.text(
    0.62,
    0.75,
    relation,
    transform=ax1.transAxes,
    fontsize=18
)

# =========================
# Streamlit表示
# =========================
st.pyplot(fig)

# =========================
# 説明
# =========================
st.markdown(
    """
    ### 見方

    - 上段：不純物分布  
      p側はアクセプタなので負、n側はドナーなので正として表示しています。

    - 中段：空乏層内の電荷分布  
      空乏層では自由キャリアが消え、イオン化アクセプタ・イオン化ドナーだけが残ります。

    - 下段：電位分布  
      ポアソン方程式により、空乏層内では電荷分布に対応して電位が曲がります。

    ### 重要なポイント

    \[
    N_A x_p = N_D x_n
    \]

    なので、濃度が高い側ほど空乏層幅は狭くなります。
    """
)
