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

# -----------------------------
# Fermi-Dirac function
# -----------------------------
def fermi_dirac(E, Ef, T):
    if T <= 0:
        return np.where(E < Ef, 1.0, 0.0)

    return 1.0 / (1.0 + np.exp((E - Ef) / (k_B * T)))


# -----------------------------
# Plot
# -----------------------------
def plot_fermi_dirac(T, Eg):

    # intrinsic semiconductor
    Ec = Eg / 2
    Ev = -Eg / 2
    Ef = 0.0

    E = np.linspace(-1.5, 1.5, 2000)

    f = fermi_dirac(E, Ef, T)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)

    # Fermi-Dirac distribution
    ax.plot(E, f, linewidth=2, label="Fermi-Dirac distribution")

    # Band edges
    ax.axvline(Ec, linestyle="--", linewidth=1.5, color="red")
    ax.axvline(Ev, linestyle="--", linewidth=1.5, color="blue")
    ax.axvline(Ef, linestyle="--", linewidth=1.5, color="green")

    # Labels
    ax.text(Ec + 0.02, 0.90, "Ec", color="red")
    ax.text(Ev + 0.02, 0.90, "Ev", color="blue")
    ax.text(Ef + 0.02, 0.50, "Ef", color="green")

    ax.set_xlabel("Energy E (eV)")
    ax.set_ylabel("Electron occupation probability")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.02, 1.02)

    ax.tick_params(axis='both', direction='in')

    ax.legend()

    fig.tight_layout()

    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("フェルミディラック分布関数")

col1, col2 = st.columns([1.0, 2.0])

with col1:

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
        5.0,
        1.1,
        0.01
    )

with col2:

    T = T_C + 273.15

    fig = plot_fermi_dirac(T, Eg)

    st.pyplot(fig, use_container_width=True)

    st.markdown(
        f"""
### Parameters

- Temperature = {T_C:.0f} °C
- Temperature = {T:.2f} K
- Bandgap Eg = {Eg:.2f} eV

### Band positions

- Ec = +{Eg/2:.3f} eV
- Ev = -{Eg/2:.3f} eV
- Ef = 0.000 eV
"""
    )
