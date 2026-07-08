import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st

# =========================================================
# Streamlit settings
# =========================================================
st.set_page_config(
    page_title="pn Junction Simulator",
    layout="wide"
)

st.title("pn Junction Simulator")
st.caption("Impurity distribution, space-charge distribution, and potential distribution")

# Avoid garbled minus signs in matplotlib
mpl.rcParams["axes.unicode_minus"] = False

# =========================================================
# Spatial coordinate
# =========================================================
x = np.linspace(-5, 5, 2000)

# =========================================================
# Sidebar controls
# =========================================================
st.sidebar.header("Parameters")

log_Na = st.sidebar.slider(
    "log10(NA): acceptor density",
    min_value=14.0,
    max_value=18.0,
    value=16.0,
    step=0.1
)

log_Nd = st.sidebar.slider(
    "log10(ND): donor density",
    min_value=14.0,
    max_value=18.0,
    value=16.0,
    step=0.1
)

Na = 10 ** log_Na
Nd = 10 ** log_Nd

# =========================================================
# Calculation function
# =========================================================
def calc(Na: float, Nd: float):
    """Calculate normalized impurity, charge, and potential profiles."""

    W = 3.5  # total depletion width in arbitrary units

    # Charge neutrality condition: NA*xp = ND*xn
    xp = W * Nd / (Na + Nd)   # depletion width on p-side
    xn = W * Na / (Na + Nd)   # depletion width on n-side

    # Impurity distribution: acceptor is negative, donor is positive
    impurity = np.where(x < 0, -Na, Nd)

    # Space-charge distribution in the depletion region
    charge = np.zeros_like(x)
    charge[(x >= -xp) & (x < 0)] = -Na
    charge[(x >= 0) & (x <= xn)] = Nd

    # Potential profile obtained by integrating the charge distribution twice
    V = np.zeros_like(x)

    mask_p = (x >= -xp) & (x < 0)
    mask_n = (x >= 0) & (x <= xn)
    mask_r = x > xn

    V[mask_p] = 0.5 * Na * (x[mask_p] + xp) ** 2

    V0 = 0.5 * Na * xp ** 2
    V[mask_n] = V0 + Na * xp * x[mask_n] - 0.5 * Nd * x[mask_n] ** 2

    Vmax = V0 + Na * xp * xn - 0.5 * Nd * xn ** 2
    V[mask_r] = Vmax

    # Normalize for visualization
    scale = max(Na, Nd)
    impurity = impurity / scale
    charge = charge / scale

    if np.max(V) != 0:
        V = V / np.max(V)

    return impurity, charge, V, xp, xn

# =========================================================
# Run calculation
# =========================================================
impurity, charge, V, xp, xn = calc(Na, Nd)

# =========================================================
# Numerical information
# =========================================================
col1, col2, col3 = st.columns(3)

col1.metric("Acceptor Density NA", f"{Na:.2e} cm^-3")
col2.metric("Donor Density ND", f"{Nd:.2e} cm^-3")

if Nd < Na:
    relation = r"$N_D < N_A$"
elif Nd > Na:
    relation = r"$N_D > N_A$"
else:
    relation = r"$N_D = N_A$"

col3.markdown(f"### {relation}")

st.markdown(
    rf"""
Charge neutrality condition:

$$
N_A x_p = N_D x_n
$$

Current depletion widths:

$$
x_p = {xp:.2f}, \qquad x_n = {xn:.2f}
$$
"""
)

# =========================================================
# Figure
# =========================================================
fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
fig.subplots_adjust(hspace=0.55)

ax1, ax2, ax3 = axes

# Impurity distribution
ax1.step(x, impurity, where="post", color="deepskyblue", lw=3)
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

# Space-charge distribution
ax2.step(x, charge, where="post", color="blue", lw=3)
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

# Potential distribution
ax3.plot(x, V, color="deepskyblue", lw=3)

# Depletion region boundaries
for ax in axes:
    ax.axvline(-xp, color="gray", ls="--", lw=1)
    ax.axvline(0, color="black", lw=1.5)
    ax.axvline(xn, color="gray", ls="--", lw=1)
    ax.axhline(0, color="black", lw=1)
    ax.set_xlim(-5, 5)
    ax.set_yticks([])
    ax.grid(False)

# Axis settings
ax1.set_title("(a) Impurity Distribution", fontsize=14)
ax2.set_title("(b) Space-Charge Distribution", fontsize=14)
ax3.set_title("(c) Potential Distribution", fontsize=14)

ax1.set_ylabel("Impurity", fontsize=13)
ax2.set_ylabel("Charge", fontsize=13)
ax3.set_ylabel("Potential", fontsize=13)
ax3.set_xlabel("Position x", fontsize=13)

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

st.pyplot(fig)

# Close figure to avoid memory accumulation during reruns
plt.close(fig)

# =========================================================
# Explanation
# =========================================================
st.markdown(
    r"""
### Description

**Top panel: Impurity distribution**

The p-side is represented by the acceptor density, while the n-side is represented by the donor density.
The acceptor side is shown as negative and the donor side as positive for visualization.

**Middle panel: Space-charge distribution**

Inside the depletion region, mobile carriers are removed. Therefore, only ionized acceptors and ionized donors remain.
This fixed ion charge forms the space-charge region.

**Bottom panel: Potential distribution**

The potential profile is determined by Poisson's equation. The curvature of the potential reflects the charge density in the depletion region.

### Key relation

$$
N_A x_p = N_D x_n
$$

A higher doping concentration results in a narrower depletion width on that side.
"""
)
