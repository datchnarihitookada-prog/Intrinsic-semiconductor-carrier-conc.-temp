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
k_B = 8.617e-5  # eV/K

# Silicon parameters
Eg_Si = 1.12
Nc_Si_300 = 2.8e19
Nv_Si_300 = 1.04e19


# -----------------------------
# Functions
# -----------------------------
def Nc_T(Nc_300, T):
    return Nc_300 * (T / 300.0) ** 1.5


def Nv_T(Nv_300, T):
    return Nv_300 * (T / 300.0) ** 1.5


def intrinsic_density(T, Eg, Nc_300, Nv_300):

    Nc = Nc_T(Nc_300, T)
    Nv = Nv_T(Nv_300, T)

    ni = np.sqrt(Nc * Nv) * np.exp(
        -Eg / (2 * k_B * T)
    )

    return ni


def plot_intrinsic_density(Eg_other semiconductor):

    # Temperature range
    T = np.linspace(250, 2000, 2000)

    # x-axis
    inv_T = 1000 / T

    # Silicon
    ni_Si = intrinsic_density(
        T,
        Eg_Si,
        Nc_Si_300,
        Nv_Si_300
    )

    # Compared other semiconductor
    ni_other semiconductor = intrinsic_density(
        T,
        Eg_other semiconductor,
        Nc_Si_300,
        Nv_Si_300
    )

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(
        figsize=(5.6, 4.2),
        dpi=120
    )

    ax.semilogy(
        inv_T,
        ni_Si,
        linewidth=2.5,
        color="blue",
        label=f"Si : Eg = {Eg_Si:.2f} eV"
    )

    ax.semilogy(
        inv_T,
        ni_other semiconductor,
        linewidth=2.5,
        color="red",
        label=f"other semiconductor : Eg = {Eg_other semiconductor:.2f} eV"
    )

    # Axis labels
    ax.set_xlabel(
        "1000 / T (K⁻¹)",
        fontsize=11
    )

    ax.set_ylabel(
        "Intrinsic carrier density ni (cm⁻³)",
        fontsize=11
    )

    # Axis ranges
    ax.set_xlim(0.5, 4.0)
    ax.set_ylim(1e7, 1e19)

    # Tick style
    ax.tick_params(
        axis="both",
        direction="in",
        labelsize=10
    )

    # No grid
    ax.grid(False)

    # Legend
    ax.legend(
        fontsize=9,
        loc="upper right"
    )

    fig.tight_layout()

    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("真性キャリア密度の温度依存性：Siとの比較")

col1, col2, col3 = st.columns([0.75, 1.7, 0.75])

# -----------------------------
# Left controls
# -----------------------------
with col1:

    st.subheader("Controls")

    Eg_other semiconductor = st.slider(
        "Bandgap of compared other semiconductor Eg (eV)",
        0.5,
        6.0,
        3.4,
        0.01
    )

# -----------------------------
# Center graph
# -----------------------------
with col2:

    fig = plot_intrinsic_density(Eg_other semiconductor)

    st.pyplot(
        fig,
        use_container_width=True
    )

# -----------------------------
# Right parameters
# -----------------------------
with col3:

    st.subheader("Parameters")

    st.markdown(
        f"""
### Fixed other semiconductor

Si  

Eg = {Eg_Si:.2f} eV  

---

### Compared other semiconductor

Eg = {Eg_other semiconductor:.2f} eV  

---

### Graph

x-axis : 1000 / T  

0.5 → 4.0 K⁻¹  

---

y-axis : ni  

10⁷ → 10¹⁹ cm⁻³  

---

Left side : High temperature  

Right side : Low temperature
"""
    )
