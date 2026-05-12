import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------
# Page setting
# -----------------------------
st.set_page_config(layout="wide")

# -----------------------------
# Constants
# -----------------------------
k_B = 8.617e-5   # eV/K
Eg = 1.1         # eV

Nc = 2.8e19      # cm^-3
Nv = 1.04e19     # cm^-3

Ec = Eg / 2
Ev = -Eg / 2


# -----------------------------
# Basic functions
# -----------------------------
def intrinsic_density(T):
    if T <= 0:
        return 0.0
    return np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * k_B * T))


def acceptor_ionized_fraction(T, Ea_depth):
    if T <= 0:
        return 0.0

    prefactor = 5.0
    frac = prefactor * np.exp(-Ea_depth / (k_B * T))

    return float(np.clip(frac, 0.0, 1.0))


def intrinsic_fraction(T, NA):
    if T <= 0:
        return 0.0

    ni = intrinsic_density(T)
    if ni <= 0:
        return 0.0

    r = ni / NA
    x = np.log10(max(r, 1e-30))

    x0 = -0.5
    dx = 0.35

    frac = 1.0 / (1.0 + np.exp(-(x - x0) / dx))
    return float(np.clip(frac, 0.0, 1.0))


def carrier_density_p_type(T, NA, Ea_depth):
    ni = intrinsic_density(T)

    frac_acceptor = acceptor_ionized_fraction(T, Ea_depth)

    p_from_acceptor = NA * frac_acceptor

    frac_intrinsic = intrinsic_fraction(T, NA)
    n_from_intrinsic = ni * frac_intrinsic
    p_from_intrinsic = ni * frac_intrinsic

    p_total = p_from_acceptor + p_from_intrinsic
    n_total = n_from_intrinsic

    return (
        ni,
        n_total,
        p_total,
        p_from_acceptor,
        p_from_intrinsic,
        n_from_intrinsic,
        frac_acceptor,
        frac_intrinsic
    )


def fermi_level_p_type(T, frac_acceptor, frac_intrinsic, E_A):
    if T <= 0:
        return E_A

    Ef_extrinsic = E_A - 0.9 * (E_A - Ev) * frac_acceptor
    Ef = (1.0 - 0.85 * frac_intrinsic) * Ef_extrinsic + (0.85 * frac_intrinsic) * 0.0

    return Ef


def density_to_points(density, max_points=160, log_min=12, log_max=19):
    if density <= 0:
        return 0

    log_n = np.log10(density)
    points = (log_n - log_min) / (log_max - log_min) * max_points

    return int(np.clip(points, 0, max_points))


# -----------------------------
# Sampling functions
# -----------------------------
def sample_conduction(T, n_points):
    if n_points <= 0:
        return np.array([])

    scale = max(k_B * T, 1e-6)
    dE = np.random.exponential(scale=scale, size=n_points)

    return Ec + dE


def sample_valence(T, n_points):
    if n_points <= 0:
        return np.array([])

    scale = max(k_B * T, 1e-6)
    dE = np.random.exponential(scale=scale, size=n_points)

    return Ev - dE


def sample_acceptor_level(T, n_points, E_A):
    if n_points <= 0:
        return np.array([])

    width = min(0.004, 0.12 * k_B * T)

    return E_A + np.random.normal(0.0, width, size=n_points)


# -----------------------------
# Plot
# -----------------------------
def plot_band(T_C, NA, Ea_depth):
    T = T_C + 273.15
    E_A = Ev + Ea_depth

    (
        ni,
        n_total,
        p_total,
        p_from_acceptor,
        p_from_intrinsic,
        n_from_intrinsic,
        frac_acceptor,
        frac_intrinsic
    ) = carrier_density_p_type(T, NA, Ea_depth)

    Ef = fermi_level_p_type(T, frac_acceptor, frac_intrinsic, E_A)

    p_acceptor_valence_display = density_to_points(
        p_from_acceptor,
        max_points=120,
        log_min=12,
        log_max=19
    )

    n_acceptor_bound_display = p_acceptor_valence_display

    p_intrinsic_display = density_to_points(
        p_from_intrinsic,
        max_points=160,
        log_min=8,
        log_max=19
    )

    n_intrinsic_display = p_intrinsic_display

    fig, ax = plt.subplots(figsize=(4.8, 5.0), dpi=120)

    ax.plot([0, 1], [Ec, Ec], 'k', linewidth=2)
    ax.plot([0, 1], [Ev, Ev], 'k', linewidth=2)
    ax.plot([0, 1], [E_A, E_A], '--', color='green', linewidth=1.5)
    ax.plot([0, 1], [Ef, Ef], 'r--', linewidth=1.5)

    # Band and level labels
    ax.text(1.02, Ec + 0.02, "Ec", va="bottom")
    ax.text(1.02, Ev - 0.02, "Ev", va="top")
    ax.text(1.02, E_A + 0.020, "Ea", va="bottom", color="green")
    # Ef label: 少し右にずらして、Ea/Evと重なりにくくする
    ax.text(1.02, Ef - 0.001, "Ef", va="top", color="r")

    if n_acceptor_bound_display > 0:
        y_a = sample_acceptor_level(T, n_acceptor_bound_display, E_A)
        x_a = np.random.uniform(0.18, 0.82, size=n_acceptor_bound_display)

        ax.scatter(
            x_a, y_a,
            s=12,
            color='blue',
            label="Acceptor-captured electrons"
        )

    if p_acceptor_valence_display > 0:
        y_h_acc = sample_valence(T, p_acceptor_valence_display)
        x_h_acc = np.random.uniform(0.18, 0.82, size=p_acceptor_valence_display)

        ax.scatter(
            x_h_acc, y_h_acc,
            s=14,
            facecolors='white',
            edgecolors='red',
            linewidths=1.0,
            label="Acceptor-generated holes"
        )

    if n_intrinsic_display > 0:
        y_e_int = sample_conduction(T, n_intrinsic_display)
        x_e_int = np.random.uniform(0.18, 0.82, size=n_intrinsic_display)

        ax.scatter(
            x_e_int, y_e_int,
            s=12,
            color='purple',
            label="Intrinsic electrons"
        )

    if p_intrinsic_display > 0:
        y_h_int = sample_valence(T, p_intrinsic_display)
        x_h_int = np.random.uniform(0.18, 0.82, size=p_intrinsic_display)

        ax.scatter(
            x_h_int, y_h_int,
            s=12,
            facecolors='white',
            edgecolors='purple',
            linewidths=1.0,
            label="Intrinsic holes"
        )

    if p_acceptor_valence_display > 0:
        ax.annotate(
            "",
            xy=(0.10, E_A - 0.01),
            xytext=(0.10, Ev + 0.01),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="gray")
        )

    if n_intrinsic_display > 0:
        ax.annotate(
            "",
            xy=(0.90, Ec - 0.01),
            xytext=(0.90, Ev + 0.01),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="gray")
        )

    ax.set_xlim(0, 1.15)
    ax.set_ylim(-0.9, 0.75)
    ax.set_xticks([])
    ax.set_ylabel("Energy (eV)")
    ax.tick_params(axis='both', direction='in')
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()

    return (
        fig,
        ni,
        n_total,
        p_total,
        frac_acceptor,
        frac_intrinsic
    )


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("p型半導体　パラメータ：T, Na, Ea")

col1, col2 = st.columns([1.0, 2.4])

with col1:
    st.subheader("Controls")

    T_C = st.slider(
        "Temperature (°C)",
        -273,
        1000,
        25,
        1
    )

    log_NA = st.slider(
        "log10(NA) [cm⁻³]",
        12.0,
        19.0,
        16.0,
        0.1
    )

    Ea_depth = st.slider(
        "Ea - Ev (eV)",
        0.03,
        0.30,
        0.045,
        0.005
    )

with col2:
    NA = 10 ** log_NA

    (
        fig,
        ni,
        n_total,
        p_total,
        frac_acceptor,
        frac_intrinsic
    ) = plot_band(T_C, NA, Ea_depth)

    graph_col, info_col = st.columns([1.5, 1.0])

    with graph_col:
        st.pyplot(fig, use_container_width=True)

    with info_col:
        st.subheader("Parameters")

        st.markdown(
            f"""
**Temperature**  
{T_C:.0f} °C  
{T_C + 273.15:.2f} K  

**Na**  
{NA:.2e} cm⁻³  

**Ea - Ev**  
{Ea_depth:.3f} eV  

---

**Carrier density**

p = {p_total:.2e} cm⁻³  

n = {n_total:.2e} cm⁻³  

ni = {ni:.2e} cm⁻³  

---

**Ionization**

Acceptor ionized fraction  
{frac_acceptor:.3e}  

Intrinsic fraction  
{frac_intrinsic:.3e}  
"""
        )
