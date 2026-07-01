import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon, Rectangle
import streamlit as st

# =========================
# Streamlit設定
# =========================
st.set_page_config(
    page_title="pn接合バンド図シミュレーター",
    layout="wide"
)

st.title("pn接合：バイアス印加時のバンド図と I-V 特性")

# =========================
# 文字化け対策
# =========================
mpl.rcParams["font.family"] = "Yu Gothic"
mpl.rcParams["axes.unicode_minus"] = False

# =========================
# 基本設定
# =========================
Eg = 1.12
VD0 = 0.70
VT = 0.026
Is = 1.0
Vz = -0.35

x = np.linspace(-5, 5, 1000)

minor_h0 = 0.30
minor_w = 0.20
major_h = 0.95
major_w = 0.70

Vmin = -0.75
Vmax = 0.60

# =========================
# 関数
# =========================
def smooth_step(x, width=0.45):
    return 0.5 * (1 + np.tanh(x / width))

def diode_current(V):
    I = Is * (np.exp(V / VT) - 1)

    if np.isscalar(V):
        if V < Vz:
            I += -2.0 * np.exp((Vz - V) / 0.035)
    else:
        zener = V < Vz
        I[zener] += -2.0 * np.exp((Vz - V[zener]) / 0.035)

    return np.clip(I, -8, 8)

def band_profile(V):
    VD = VD0 - V
    VD = max(0.05, VD)

    s = smooth_step(x, 0.45)

    Ec = 0.45 - VD * s
    Ev = Ec - Eg

    EF0 = -0.23
    EFp = EF0 + V / 2
    EFn = EF0 - V / 2

    return Ec, Ev, EFp, EFn, VD

def Ec_at(xp, V):
    Ec, _, _, _, _ = band_profile(V)
    return np.interp(xp, x, Ec)

def Ev_at(xp, V):
    _, Ev, _, _, _ = band_profile(V)
    return np.interp(xp, x, Ev)

def draw_sail(
    ax,
    x0,
    y0,
    height,
    width,
    label,
    mode="up",
    white_cut_y=None,
    white_region=None
):
    theta = np.linspace(0, np.pi, 120)

    if mode == "up":
        y_curve = y0 + height * theta / np.pi
        x_curve = x0 + width * np.sin(theta) * (1.0 - 0.65 * theta / np.pi)
        outline_x = np.concatenate([[x0, x0], x_curve[::-1], [x0]])
        outline_y = np.concatenate([[y0, y0 + height], y_curve[::-1], [y0]])
        label_y = y0 + 0.55 * height
    else:
        y_curve = y0 - height * theta / np.pi
        x_curve = x0 + width * np.sin(theta) * (1.0 - 0.65 * theta / np.pi)
        outline_x = np.concatenate([[x0, x0], x_curve[::-1], [x0]])
        outline_y = np.concatenate([[y0, y0 - height], y_curve[::-1], [y0]])
        label_y = y0 - 0.65 * height

    outline = np.column_stack([outline_x, outline_y])

    black_patch = Polygon(
        outline,
        closed=True,
        facecolor="black",
        edgecolor="none",
        alpha=0.9,
        zorder=7
    )
    ax.add_patch(black_patch)

    if white_cut_y is not None and white_region is not None:
        y_min = np.min(outline[:, 1])
        y_max = np.max(outline[:, 1])

        if white_region == "below":
            rect_y = y_min
            rect_h = max(0, white_cut_y - y_min)
        elif white_region == "above":
            rect_y = white_cut_y
            rect_h = max(0, y_max - white_cut_y)
        else:
            rect_y = y_min
            rect_h = 0

        white_rect = Rectangle(
            (x0 - 0.08, rect_y),
            width + 0.30,
            rect_h,
            facecolor="white",
            edgecolor="none",
            zorder=8
        )
        white_rect.set_clip_path(black_patch)
        ax.add_patch(white_rect)

    ax.plot(outline[:, 0], outline[:, 1], color="black", lw=2, zorder=9)
    ax.text(x0 - 0.45, label_y, label, fontsize=18)

# =========================
# キャリア位置
# =========================
n_major_x0 = np.array([1.7, 2.3, 2.9, 3.5, 4.1, 4.7])
p_major_x0 = np.array([-4.3, -3.6, -2.9, -2.2, -1.5])

p_minor_e_x0 = np.array([-3.7])
n_minor_h_x0 = np.array([3.6])

# =========================
# Streamlit slider
# =========================
V = st.slider(
    "Bias V",
    min_value=float(Vmin),
    max_value=float(Vmax),
    value=0.0,
    step=0.01
)

# =========================
# Figure 作成
# =========================
fig = plt.figure(figsize=(15, 6))

ax_band = fig.add_axes([0.06, 0.13, 0.58, 0.78])
ax_iv = fig.add_axes([0.70, 0.13, 0.25, 0.78])

# =========================
# I-V 特性
# =========================
V_array = np.linspace(Vmin, Vmax, 1000)
I_array = diode_current(V_array)

ax_iv.plot(V_array, I_array, color="black", lw=2.5)
ax_iv.plot([V], [diode_current(V)], "o", color="red", markersize=8)

ax_iv.axhline(0, color="black", lw=1)
ax_iv.axvline(0, color="black", lw=1)
ax_iv.axvline(Vz, color="black", linestyle="--", lw=1.5)
ax_iv.text(Vz - 0.08, 6.8, "$V_Z$", fontsize=13)

ax_iv.set_xlim(Vmin, Vmax)
ax_iv.set_ylim(-8.5, 8.5)
ax_iv.set_xlabel("Bias V")
ax_iv.set_ylabel("Current I")
ax_iv.set_title("I-V 特性")
ax_iv.grid(alpha=0.3)

# =========================
# バンド図
# =========================
Ec, Ev, EFp, EFn, VD = band_profile(V)

Ec_p = Ec[0]
Ev_n = Ev[-1]

W = 2.25 * np.sqrt(VD / VD0)

n_major = n_major_x0[n_major_x0 > W / 2]
p_major = p_major_x0[p_major_x0 < -W / 2]

ax_band.plot(x, Ec, color="black", lw=3)
ax_band.plot(x, Ev, color="black", lw=3)

if abs(V) < 1e-3:
    EF = -0.23
    ax_band.hlines(EF, -5, 5, color="black", linestyle="dotted", lw=2)
    ax_band.text(-4.8, EF + 0.04, "EF", fontsize=14)
    title = "熱平衡状態"
else:
    ax_band.hlines(EFp, -5, 0, color="black", linestyle="dotted", lw=2)
    ax_band.hlines(EFn, 0, 5, color="black", linestyle="dotted", lw=2)
    ax_band.text(-4.8, EFp + 0.04, "EFp", fontsize=14)
    ax_band.text(3.7, EFn + 0.04, "EFn", fontsize=14)

    if V > 0:
        title = "順方向バイアス"
    elif V > Vz:
        title = "逆方向バイアス"
    else:
        title = "ツェナー降伏"

ax_band.hlines(Ec_p, -5, 5, color="black", linestyle="dashed", lw=1.5, alpha=0.65)
ax_band.hlines(Ev_n, -5, 5, color="black", linestyle="dashed", lw=1.5, alpha=0.65)

ax_band.axvspan(-W / 2, W / 2, color="gray", alpha=0.14)

# =========================
# キャリア
# =========================
ax_band.scatter(
    n_major,
    [Ec_at(xi, V) + 0.11 for xi in n_major],
    s=150,
    color="black",
    zorder=5
)

ax_band.scatter(
    p_minor_e_x0,
    [Ec_at(xi, V) + 0.11 for xi in p_minor_e_x0],
    s=150,
    color="black",
    zorder=5
)

ax_band.scatter(
    p_major,
    [Ev_at(xi, V) - 0.11 for xi in p_major],
    s=170,
    facecolors="white",
    edgecolors="black",
    lw=2,
    zorder=5
)

ax_band.scatter(
    n_minor_h_x0,
    [Ev_at(xi, V) - 0.11 for xi in n_minor_h_x0],
    s=170,
    facecolors="white",
    edgecolors="black",
    lw=2,
    zorder=5
)

# 固定イオン
for xi in np.linspace(W / 2 + 0.2, 4.7, 6):
    ax_band.text(xi, Ec_at(xi, V) - 0.22, "+", fontsize=16, ha="center")

for xi in np.linspace(-4.5, -W / 2 - 0.2, 6):
    ax_band.text(xi, Ev_at(xi, V) + 0.22, "−", fontsize=16, ha="center")

# 少数キャリア密度
if V > 0:
    minor_h = minor_h0 * (1 + 2.5 * V)
elif V < Vz:
    minor_h = minor_h0 * 0.15
else:
    minor_h = minor_h0 / (1 + 2.5 * abs(V))

draw_sail(ax_band, -3.45, Ec_at(-3.45, V), minor_h, minor_w, r"$n_p$", "up")

draw_sail(
    ax_band,
    2.65,
    Ec_at(2.65, V),
    major_h,
    major_w,
    r"$n_n$",
    "up",
    white_cut_y=Ec_p,
    white_region="below"
)

draw_sail(
    ax_band,
    -2.30,
    Ev_at(-2.30, V),
    major_h,
    major_w,
    r"$p_p$",
    "down",
    white_cut_y=Ev_n,
    white_region="above"
)

draw_sail(ax_band, 3.90, Ev_at(3.90, V), minor_h, minor_w, r"$p_n$", "down")

# =========================
# 説明
# =========================
if V > 0:
    ax_band.text(1.0, 0.35, "順方向：障壁低下・空乏層縮小", fontsize=13)
    ax_band.text(1.0, 0.17, "電流が急激に増加", fontsize=13)
elif V > Vz:
    ax_band.text(1.0, 0.35, "逆方向：障壁増加・空乏層拡大", fontsize=13)
    ax_band.text(1.0, 0.17, "逆方向電流はほぼ飽和", fontsize=13)
else:
    ax_band.text(0.5, 0.35, "ツェナー降伏：強電界トンネル", fontsize=13)
    ax_band.text(0.5, 0.17, "逆方向電流が急激に増加", fontsize=13)

    ax_band.annotate(
        "Zener tunneling",
        xy=(0.0, Ev_at(0.0, V) + 0.08),
        xytext=(0.0, Ec_at(0.0, V) - 0.08),
        arrowprops=dict(arrowstyle="->", lw=2),
        fontsize=12,
        ha="center"
    )

# 内部電界
ax_band.annotate(
    "",
    xy=(-1.1, -1.48),
    xytext=(1.1, -1.48),
    arrowprops=dict(arrowstyle="->", lw=2)
)
ax_band.text(-0.15, -1.38, "内部電界 E", fontsize=13)

# 障壁高さ
ax_band.annotate(
    "",
    xy=(0.65, Ec_at(0.65, V)),
    xytext=(0.65, Ec_at(0.65, V) + VD),
    arrowprops=dict(arrowstyle="<->", lw=2)
)

ax_band.text(
    0.85,
    Ec_at(0.65, V) + VD / 2,
    f"VD - V = {VD:.2f} V",
    fontsize=13
)

# ラベル
ax_band.text(-4.8, 0.67, "p型", fontsize=14)
ax_band.text(3.7, 0.67, "n型", fontsize=14)

ax_band.text(-5.05, Ec[0] + 0.02, "Ec", fontsize=14)
ax_band.text(5.05, Ec[-1] + 0.02, "Ec", fontsize=14)

ax_band.text(-5.05, Ev[0] - 0.10, "Ev", fontsize=14)
ax_band.text(5.05, Ev[-1] - 0.10, "Ev", fontsize=14)

ax_band.set_xlim(-5.2, 7.3)
ax_band.set_ylim(-1.95, 1.35)
ax_band.set_xlabel("位置 x")
ax_band.set_ylabel("電子のエネルギー")
ax_band.set_title(
    f"pn接合：バイアス印加時のバンド図　V = {V:.2f} V（{title}）",
    fontsize=14
)
ax_band.grid(alpha=0.2)

st.pyplot(fig, clear_figure=True)
