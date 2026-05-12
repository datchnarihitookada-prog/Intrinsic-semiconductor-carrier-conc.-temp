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


def donor_ionized_fraction(T, Ed_depth):
    """
    Arrhenius-type donor activation model.
    Ed_depth が大きいほど深いドナーとなり、電離しにくい。
    """
    if T <= 0:
        return 0.0

    prefactor = 5.0
    frac = prefactor * np.exp(-Ed_depth / (k_B * T))

    return float(np.clip(frac, 0.0, 1.0))


def intrinsic_fraction(T, ND):
    if T <= 0:
        return 0.0

    ni = intrinsic_density(T)
    if ni <= 0:
        return 0.0

    r = ni / ND
    x = np.log10(max(r, 1e-30))

    x0 = -0.5
    dx = 0.35

    frac = 1.0 / (1.0 + np.exp(-(x - x0) / dx))
    return float(np.clip(frac, 0.0, 1.0))


def carrier_density_n_type(T, ND, Ed_depth):
    ni = intrinsic_density(T)

    frac_donor = donor_ionized_fraction(T, Ed_depth)

    n_from_donor = ND * frac_donor

    frac_intrinsic = intrinsic_fraction(T, ND)
    n_from_intrinsic = ni * frac_intrinsic
    p_from_intrinsic = ni * frac_intrinsic

    n_total = n_from_donor + n_from_intrinsic
    p_total = p_from_intrinsic

    return (
        ni,
        n_total,
        p_total,
        n_from_donor,
        n_from_intrinsic,
        p_from_intrinsic,
        frac_donor,
        frac_intrinsic
    )


def fermi_level_n_type(T, frac_donor, frac_intrinsic, E_D):
    if T <= 0:
        return E_D

    Ef_extrinsic = E_D + 0.9 * (Ec - E_D) * frac_donor
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


def sample_donor_level(T, n_points, E_D):
    if n_points <= 0:
        return np.array([])

    width = min(0.004, 0.12 * k_B * T)

    return E_D + np.random.normal(0.0, width, size=n_points)


# -----------------------------
# Plot
# -----------------------------
def plot_band(T_C, ND, Ed_depth):
    T = T_C + 273.15
    E_D = Ec - Ed_depth

    (
        ni,
        n_total,
        p_total,
        n_from_donor,
        n_from_intrinsic,
        p_from_intrinsic,
        frac_donor,
        frac_intrinsic
    ) = carrier_density_n_type(T, ND, Ed_depth)

    Ef = fermi_level_n_type(T, frac_donor, frac_intrinsic, E_D)

    n_donor_conduction_display = density_to_points(
        n_from_donor,
        max_points=120,
        log_min=12,
        log_max=19
    )

    # 電離していないドナーに残っている電子
    n_donor_bound_density = ND * (1.0 - frac_donor)
    n_donor_bound_display = density_to_points(
        n_donor_bound_density,
        max_points=120,
        log_min=12,
        log_max=19
    )

    n_intrinsic_display = density_to_points(
        n_from_intrinsic,
        max_points=160,
        log_min=8,
        log_max=19
    )

    p_intrinsic_display = n_intrinsic_display

    fig, ax = plt.subplots(figsize=(4.8, 5.0), dpi=120)

    ax.plot([0, 1], [Ec, Ec], 'k', linewidth=2)
    ax.plot([0, 1], [Ev, Ev], 'k', linewidth=2)
    ax.plot([0, 1], [E_D, E_D], '--', color='green', linewidth=1.5)
    ax.plot([0, 1], [Ef, Ef], 'r--', linewidth=1.5)

    # Labels
    ax.text(1.03, Ec + 0.035, "Ec", va="bottom")
    ax.text(1.03, Ev - 0.035, "Ev", va="top")
    ax.text(1.03, E_D - 0.020, "Ed", va="top", color="green")
    ax.text(1.02, Ef - 0.001, "Ef", va="top", color="r")

    # Donor-bound electrons
    if n_donor_bound_display > 0:
        y_d = sample_donor_level(T, n_donor_bound_display, E_D)
        x_d = np.random.uniform(0.18, 0.82, size=n_donor_bound_display)

        ax.scatter(
            x_d, y_d,
            s=12,
            color='blue',
            label="Donor-bound electrons"
        )

    # Donor-generated conduction electrons
    if n_donor_conduction_display > 0:
        y_e_donor = sample_conduction(T, n_donor_conduction_display)
        x_e_donor = np.random.uniform(0.18, 0.82, size=n_donor_conduction_display)

        ax.scatter(
            x_e_donor, y_e_donor,
            s=14,
            color='red',
            label="Donor-generated electrons"
        )

    # Intrinsic electrons
    if n_intrinsic_display > 0:
        y_e_int = sample_conduction(T, n_intrinsic_display)
        x_e_int = np.random.uniform(0.18, 0.82, size=n_intrinsic_display)

        ax.scatter(
            x_e_int, y_e_int,
            s=12,
            color='purple',
            label="Intrinsic electrons"
        )

    # Intrinsic holes
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

    # Donor excitation arrow
    if n_donor_conduction_display > 0:
        ax.annotate(
            "",
            xy=(0.10, Ec - 0.01),
            xytext=(0.10, E_D + 0.01),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="gray")
        )

    # Intrinsic excitation arrow
    if n_intrinsic_display > 0:
        ax.annotate(
            "",
            xy=(0.90, Ec - 0.01),
            xytext=(0.90, Ev + 0.01),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="gray")
        )

    ax.set_xlim(0, 1.15)
    ax.set_ylim(-0.75, 0.75)
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
        frac_donor,
        frac_intrinsic
    )


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("n型半導体　パラメータ：T, Nd, Ed")

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

    log_ND = st.slider(
        "log10(Nd) [cm⁻³]",
        12.0,
        19.0,
        16.0,
        0.1
    )

    Ed_depth = st.slider(
        "Ec - Ed (eV)",
        0.03,
        0.30,
        0.045,
        0.005
    )

with col2:
    ND = 10 ** log_ND

    (
        fig,
        ni,
        n_total,
        p_total,
        frac_donor,
        frac_intrinsic
    ) = plot_band(T_C, ND, Ed_depth)

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

**Nd**  
{ND:.2e} cm⁻³  

**Ec - Ed**  
{Ed_depth:.3f} eV  

---

**Carrier density**

n = {n_total:.2e} cm⁻³  

p = {p_total:.2e} cm⁻³  

ni = {ni:.2e} cm⁻³  

---

**Ionization**

Donor ionized fraction  
{frac_donor:.3e}  

Intrinsic fraction  
{frac_intrinsic:.3e}  
"""
        )
