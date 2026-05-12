import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(layout="wide")

# -----------------------------
# Constants
# -----------------------------
k_B = 8.617e-5  # eV/K

# Si-like effective density of states at 300 K
Nc_300 = 2.8e19   # cm^-3
Nv_300 = 1.04e19  # cm^-3


# -----------------------------
# Functions
# -----------------------------
def fermi_dirac(E, Ef, T):
    if T <= 0:
        return np.where(E < Ef, 1.0, 0.0)

    x = (E - Ef) / (k_B * T)
    x = np.clip(x, -700, 700)

    return 1.0 / (1.0 + np.exp(x))


def Nc_T(T):
    return Nc_300 * (T / 300.0) ** 1.5


def Nv_T(T):
    return Nv_300 * (T / 300.0) ** 1.5


def intrinsic_density(T, Eg):
    if T <= 0:
        return 0.0

    Nc = Nc_T(T)
    Nv = Nv_T(T)

    return np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * k_B * T))


def carrier_densities(T, Eg, Ef):
    if T <= 0:
        return 0.0, 0.0

    Ec = Eg / 2
    Ev = -Eg / 2

    Nc = Nc_T(T)
    Nv = Nv_T(T)

    # Non-degenerate approximation
    n = Nc * np.exp(-(Ec - Ef) / (k_B * T))
    p = Nv * np.exp(-(Ef - Ev) / (k_B * T))

    return n, p


def plot_distribution(T, Eg, Ef):
    Ec = Eg / 2
    Ev = -Eg / 2

    margin = 1.2
    E_min = Ev - margin
    E_max = Ec + margin

    E = np.linspace(E_min, E_max, 5000)

    f = fermi_dirac(E, Ef, T)

    # 3D-like DOS shape
    gc = np.zeros_like(E)
    gv = np.zeros_like(E)

    gc[E >= Ec] = np.sqrt(E[E >= Ec] - Ec)
    gv[E <= Ev] = np.sqrt(Ev - E[E <= Ev])

    # Electron and hole distributions
    electron_dist = gc * f
    hole_dist = gv * (1.0 - f)

    # Normalize for visualization
    max_val = max(
        np.max(electron_dist),
        np.max(hole_dist),
        1e-30
    )

    electron_dist_norm = electron_dist / max_val
    hole_dist_norm = hole_dist / max_val

    fig, ax = plt.subplots(figsize=(5.8, 4.4), dpi=120)

    ax.plot(
        E,
        f,
        linewidth=2,
        label="Fermi-Dirac f(E)"
    )

    # Electron density distribution
    ax.plot(
    E,
    electron_dist_norm,
    linewidth=2,
    label="Electron density distribution"
    )

    ax.fill_between(
    E,
    electron_dist_norm,
    alpha=0.30
    )

    # Hole density distribution
    ax.plot(
    E,
    hole_dist_norm,
    linewidth=2,
    linestyle="--",
    label="Hole density distribution"
    )

    ax.fill_between(
    E,
    hole_dist_norm,
    alpha=0.25
    )

    ax.axvline(Ev, linestyle="--", linewidth=1.5)
    ax.axvline(Ef, linestyle="--", linewidth=1.5)
    ax.axvline(Ec, linestyle="--", linewidth=1.5)

    ax.text(Ev + 0.02, 0.92, "Ev")
    ax.text(Ef + 0.02, 0.50, "Ef")
    ax.text(Ec + 0.02, 0.92, "Ec")

    ax.set_xlabel("Energy E (eV)")
    ax.set_ylabel("Probability / normalized density")

    ax.set_xlim(E_min, E_max)
    ax.set_ylim(-0.03, 1.05)

    ax.tick_params(axis="both", direction="in")
    ax.legend(fontsize=8)

    fig.tight_layout()

    ni = intrinsic_density(T, Eg)
    n, p = carrier_densities(T, Eg, Ef)

    f_Ec = fermi_dirac(Ec, Ef, T)
    hole_Ev = 1.0 - fermi_dirac(Ev, Ef, T)

    if Ef > 0.05:
        semiconductor_type = "n-type-like"
    elif Ef < -0.05:
        semiconductor_type = "p-type-like"
    else:
        semiconductor_type = "intrinsic-like"

    return (
        fig,
        Ec,
        Ev,
        Ef,
        ni,
        n,
        p,
        f_Ec,
        hole_Ev,
        semiconductor_type
    )


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("フェルミ分布 × 状態密度による電子・正孔分布")

col1, col2 = st.columns([0.8, 1.8])

with col1:
    st.subheader("Controls")

    T_C = st.slider(
        "Temperature (°C)",
        -273,
        1000,
        25,
        1
    )

    Eg = st.slider(
        "Bandgap Eg (eV)",
        0.1,
        3.0,
        1.1,
        0.01
    )

    Ef = st.slider(
        "Fermi level Ef (eV)",
        -1.0,
        1.0,
        0.0,
        0.01
    )

with col2:
    T = T_C + 273.15

    (
        fig,
        Ec,
        Ev,
        Ef,
        ni,
        n,
        p,
        f_Ec,
        hole_Ev,
        semiconductor_type
    ) = plot_distribution(T, Eg, Ef)

    graph_col, info_col = st.columns([1.5, 0.9])

    with graph_col:
        st.pyplot(fig, use_container_width=True)

    with info_col:
        st.subheader("Parameters")

        st.markdown(
            f"""
**Temperature**  
{T_C:.0f} °C  
{T:.2f} K  

**Bandgap Eg**  
{Eg:.2f} eV  

**Fermi level Ef**  
{Ef:.3f} eV  

**Type**  
{semiconductor_type}

---

**Band positions**

Ec = +{Ec:.3f} eV  

Ev = {Ev:.3f} eV  

---

**Band-edge occupation**

f(Ec) = {f_Ec:.3e}  

1 - f(Ev) = {hole_Ev:.3e}  

---

**Carrier density**

n = {n:.3e} cm⁻³  

p = {p:.3e} cm⁻³  

ni = {ni:.3e} cm⁻³  

---

**Visualization**

Electron distribution  
= DOS × f(E)  

Hole distribution  
= DOS × [1 - f(E)]  
"""
        )
