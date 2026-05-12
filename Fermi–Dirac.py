import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(layout="wide")

k_B = 8.617e-5  # eV/K


def fermi_dirac(E, Ef, T):
    if T <= 0:
        return np.where(E < Ef, 1.0, 0.0)

    x = (E - Ef) / (k_B * T)
    x = np.clip(x, -700, 700)

    return 1.0 / (1.0 + np.exp(x))


def plot_fermi_dirac(T, Eg, Ef):
    Ec = Eg / 2
    Ev = -Eg / 2

    E = np.linspace(-1.5, 1.5, 3000)
    f = fermi_dirac(E, Ef, T)

    f_Ec = fermi_dirac(Ec, Ef, T)
    hole_Ev = 1.0 - fermi_dirac(Ev, Ef, T)

    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=120)

    ax.plot(E, f, linewidth=2, label="Fermi-Dirac distribution")

    ax.axvline(Ev, linestyle="--", linewidth=1.5, color="blue")
    ax.axvline(Ef, linestyle="--", linewidth=1.5, color="green")
    ax.axvline(Ec, linestyle="--", linewidth=1.5, color="red")

    ax.text(Ev + 0.02, 0.90, "Ev", color="blue")
    ax.text(Ef + 0.02, 0.50, "Ef", color="green")
    ax.text(Ec + 0.02, 0.90, "Ec", color="red")

    if Ef > 0.05:
        semiconductor_type = "n-type"
    elif Ef < -0.05:
        semiconductor_type = "p-type"
    else:
        semiconductor_type = "intrinsic"

    ax.set_title(f"{semiconductor_type} semiconductor")
    ax.set_xlabel("Energy E (eV)")
    ax.set_ylabel("Electron occupation probability")

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(axis="both", direction="in")
    ax.legend()

    fig.tight_layout()

    return fig, semiconductor_type, Ec, Ev, f_Ec, hole_Ev


st.title("フェルミディラック分布関数")

col1, col2 = st.columns([0.8, 1.6])

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

    fig, semiconductor_type, Ec, Ev, f_Ec, hole_Ev = plot_fermi_dirac(T, Eg, Ef)

    graph_col, info_col = st.columns([1.5, 0.8])

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

---

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
"""
        )
