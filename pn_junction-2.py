import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# PN Junction Formation App
# Streamlit version
# ============================================================

st.set_page_config(
    page_title="PN Junction Formation",
    layout="centered"
)

# ============================================================
# 1. Title and slider
# ============================================================

st.title("PN Junction Formation")
st.caption("Band bending, carrier diffusion, depletion region, and built-in potential")

t = st.slider(
    "Time",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.01
)

# ============================================================
# 2. Fixed random coordinates
# ============================================================

np.random.seed(42)
R = np.random.uniform

# Majority holes in p-region
hpx = R(-4.8, 0.0, 150)
hpy = R(-0.3, -0.05, 150)

# Minority electrons in p-region
epx = R(-4.8, 0.0, 15)
epy = R(0.05, 0.2, 15)

# Majority electrons in n-region
enx = R(0.0, 4.8, 150)
eny = R(0.05, 0.3, 150)

# Minority holes in n-region
hnx = R(0.0, 4.8, 15)
hny = R(-0.3, -0.05, 15)

# Ionized acceptors and donors
ipx = R(-0.8, 0.0, 50)
ipy = R(1.5, 2.5, 50)

inx = R(0.0, 0.8, 50)
iny = R(1.5, 2.5, 50)

# ============================================================
# 3. Physical parameters
# ============================================================

Vbi = 1.4
Vd = Vbi * t
w = 0.8 * np.sqrt(t) + 1e-3

x = np.linspace(-5, 5, 300)

# Band bending
S = 0.5 * Vd * np.tanh(x / w)
Ec = 3.0 - S
Ev = 1.0 - S

Ecp = Ec[0]
Ecn = Ec[-1]
Evp = Ev[0]
Evn = Ev[-1]

Eth_e = Ecp + 0.6 * (1 - t)
Eth_h = Evn - 0.6 * (1 - t)

Xp = -2.5
Xn = 2.5

# ============================================================
# 4. Figure
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=120)

ax.axis("off")
ax.set_xlim(-5, 5)
ax.set_ylim(-2.5, 6.0)

# Depletion region
if w > 0.05:
    ax.axvspan(-w, w, color="lightskyblue", alpha=0.15)

# Band lines
ax.plot(x, Ec, "k-", lw=1.5)
ax.plot(x, Ev, "k-", lw=1.5)

# Fermi levels
if t < 0.99:
    ax.plot([-5, 0], [1.3 + 0.5 * Vd] * 2, "k:", lw=2)
    ax.plot([0, 5], [2.7 - 0.5 * Vd] * 2, "k:", lw=2)
    ax.text(-4.8, 1.3 + 0.5 * Vd + 0.08, r"$E_{Fp}$", fontsize=11)
    ax.text(3.8, 2.7 - 0.5 * Vd + 0.08, r"$E_{Fn}$", fontsize=11)
else:
    ax.plot([-5, 5], [2.0] * 2, "k-.", lw=1.5)
    ax.text(4.0, 2.08, r"$E_F$", fontsize=11)

# Threshold guide lines
ax.plot([-2.5, 5], [Eth_e, Eth_e], "k--", lw=1.2)
ax.plot([-5, 2.5], [Eth_h, Eth_h], "k--", lw=1.2)


# ============================================================
# 5. Distribution drawing function
# ============================================================

def draw_distribution(X, Eb, sx, sy, W, H, Eth, c1, c2, kT):
    yy = np.linspace(Eb, Eb + sy * H, 300)
    dE = np.abs(yy - Eb)

    shape = np.sqrt(dE) * np.exp(-dE / kT)
    shape_max = np.max(shape)

    if shape_max > 1e-6:
        shape = (shape / shape_max) * W
    else:
        shape = np.zeros_like(shape)

    xx = X + sx * shape

    if sy > 0:
        mask = yy > Eth
    else:
        mask = yy < Eth

    ax.fill_betweenx(yy, X, xx, where=~mask, facecolor=c1, alpha=1.0)
    ax.fill_betweenx(yy, X, xx, where=mask, facecolor=c2, alpha=1.0)
    ax.plot(xx, yy, "k-", lw=1.0)
    ax.plot([X, X], [Eb, Eb + sy * H], "k-", lw=1.0)

    return yy, mask


# Minority carriers
draw_distribution(
    Xp, Ecp,
    sx=-1, sy=1,
    W=0.15, H=0.8,
    Eth=Ecp,
    c1="lightgray", c2="lightgray",
    kT=0.15
)

draw_distribution(
    Xn, Evn,
    sx=1, sy=-1,
    W=0.15, H=0.8,
    Eth=Evn,
    c1="skyblue", c2="skyblue",
    kT=0.15
)

# Majority carriers
yy_n, mask_n = draw_distribution(
    Xn, Ecn,
    sx=1, sy=1,
    W=0.7, H=3.0,
    Eth=Eth_e,
    c1="white", c2="darkgray",
    kT=0.5
)

yy_p, mask_p = draw_distribution(
    Xp, Evp,
    sx=-1, sy=-1,
    W=0.7, H=3.0,
    Eth=Eth_h,
    c1="white", c2="skyblue",
    kT=0.5
)

# ============================================================
# 6. Labels
# ============================================================

ax.text(Xp - 0.4, Ecp + 0.1, r"$n_{p0}$", fontsize=13)
ax.text(Xp - 1.1, Evp - 0.2, r"$p_{p0}$", fontsize=13)
ax.text(Xn + 1.0, Ecn + 0.1, r"$n_{n0}$", fontsize=13)
ax.text(Xn + 0.3, Evn - 0.2, r"$p_{n0}$", fontsize=13)

if np.any(mask_n):
    ax.text(Xn + 0.2, Eth_e + 0.2, "a", fontsize=13)

if np.any(mask_p):
    ax.text(Xp - 0.4, Eth_h - 0.4, "b", fontsize=13)

if Ecp > Ecn + 0.05:
    ax.text(
        1.9,
        (Ecp + Ecn) / 2,
        rf"$eV_D$={Vd:.2f}",
        fontsize=12
    )

ax.text(-4.8, 3.25, r"$E_c$", fontsize=12)
ax.text(-4.8, 1.25, r"$E_v$", fontsize=12)

ax.text(-4.7, -2.15, "p-region", fontsize=12)
ax.text(3.4, -2.15, "n-region", fontsize=12)

if w > 0.05:
    ax.text(-0.95, -2.15, "depletion region", fontsize=11)

# ============================================================
# 7. Carrier movement
# ============================================================

hp = hpx + 2.8 * t * np.exp(-np.abs(hpx) * 0.6) + 0.05 * np.sin(t * 50 + hpx * 20)
en = enx - 2.8 * t * np.exp(-np.abs(enx) * 0.6) + 0.05 * np.sin(t * 50 + enx * 20)

ep = epx + 0.05 * np.sin(t * 50 + epx * 20)
hn = hnx + 0.05 * np.sin(t * 50 + hnx * 20)

vp = hp < -w
vn = en > w
vep = ep < -w
vhn = hn > w

# Majority holes
ax.scatter(
    hp[vp],
    np.interp(hp[vp], x, Ev) + hpy[vp],
    s=20,
    facecolors="none",
    edgecolors="red"
)

# Majority electrons
ax.scatter(
    en[vn],
    np.interp(en[vn], x, Ec) + eny[vn],
    s=20,
    color="blue"
)

# Minority electrons in p-region
ax.scatter(
    ep[vep],
    np.interp(ep[vep], x, Ec) + epy[vep],
    s=20,
    color="blue"
)

# Minority holes in n-region
ax.scatter(
    hn[vhn],
    np.interp(hn[vhn], x, Ev) + hny[vhn],
    s=20,
    facecolors="none",
    edgecolors="red"
)

# Ionized acceptors
mask_acc = (ipx > -w) & (ipx < 0)
ax.scatter(
    ipx[mask_acc],
    ipy[mask_acc],
    marker="_",
    s=50,
    color="red"
)

# Ionized donors
mask_don = (inx < w) & (inx > 0)
ax.scatter(
    inx[mask_don],
    iny[mask_don],
    marker="+",
    s=50,
    color="blue"
)

# ============================================================
# 8. Display
# ============================================================

fig.tight_layout()

st.pyplot(fig, use_container_width=False)
plt.close(fig)

# ============================================================
# 9. Explanation
# ============================================================

st.markdown(
    f"""
**Current parameters**

- Time parameter: **t = {t:.2f}**
- Built-in potential: **V_D = {Vd:.2f} V**
- Depletion width parameter: **w = {w:.2f}**

**Symbols**

- Blue dots: electrons
- Red open circles: holes
- Blue plus signs: ionized donors
- Red minus signs: ionized acceptors
"""
)
