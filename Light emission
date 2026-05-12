import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(layout="wide")

# -----------------------------
# Constants
# -----------------------------
hc = 1240.0  # eV nm


# -----------------------------
# Helper functions
# -----------------------------
def wavelength_from_energy(E):
    if E <= 0:
        return np.inf
    return hc / E


def plot_recombination(Eg, E_in):
    Ec = Eg / 2
    Ev = -Eg / 2

    absorbed = E_in >= Eg
    E_emit = Eg if absorbed else None
    lambda_emit = wavelength_from_energy(E_emit) if absorbed else None

    fig, ax = plt.subplots(figsize=(4.8, 5.0), dpi=120)

    # Band edges
    ax.plot([0, 1], [Ec, Ec], "k", linewidth=2)
    ax.plot([0, 1], [Ev, Ev], "k", linewidth=2)

    # Labels
    ax.text(1.03, Ec + 0.035, "Ec", va="bottom")
    ax.text(1.03, Ev - 0.035, "Ev", va="top")

    # Incident photon arrow
    ax.annotate(
        "",
        xy=(0.15, Ev + E_in),
        xytext=(0.15, Ev),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="orange")
    )

    ax.text(
        0.18,
        Ev + E_in / 2,
        f"incident\nhν = {E_in:.2f} eV",
        va="center",
        color="orange"
    )

    if absorbed:
        # Electron in conduction band
        ax.scatter(
            [0.55],
            [Ec + 0.08],
            s=70,
            color="red",
            label="Excited electron"
        )

        # Hole in valence band
        ax.scatter(
            [0.55],
            [Ev - 0.08],
            s=70,
            facecolors="white",
            edgecolors="red",
            linewidths=1.5,
            label="Hole"
        )

        # Recombination arrow
        ax.annotate(
            "",
            xy=(0.55, Ev + 0.02),
            xytext=(0.55, Ec - 0.02),
            arrowprops=dict(arrowstyle="->", lw=1.8, color="blue")
        )

        ax.text(
            0.60,
            0.0,
            f"emission\nhν ≈ Eg\n= {Eg:.2f} eV\nλ ≈ {lambda_emit:.0f} nm",
            va="center",
            color="blue"
        )

        result_text = "Radiative recombination occurs"

    else:
        ax.text(
            0.42,
            0.0,
            "No band-to-band excitation\nhν < Eg",
            va="center",
            color="gray"
        )

        result_text = "No band-to-band absorption"

    ax.set_xlim(0, 1.25)
    ax.set_ylim(Ev - 0.6, Ec + 0.6)
    ax.set_xticks([])
    ax.set_ylabel("Energy (eV)")
    ax.tick_params(axis="both", direction="in")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    return fig, absorbed, result_text, E_emit, lambda_emit


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("半導体の発光再結合　パラメータ：Eg, hν")

col1, col2 = st.columns([1.0, 2.4])

with col1:
    st.subheader("Controls")

    Eg = st.slider(
        "Band gap Eg (eV)",
        0.5,
        6.5,
        1.1,
        0.05
    )

    E_in = st.slider(
        "Incident photon energy hν (eV)",
        0.1,
        7.0,
        2.0,
        0.05
    )

with col2:
    fig, absorbed, result_text, E_emit, lambda_emit = plot_recombination(Eg, E_in)

    graph_col, info_col = st.columns([1.5, 1.0])

    with graph_col:
        st.pyplot(fig, use_container_width=True)

    with info_col:
        st.subheader("Result")

        st.markdown(
            f"""
**Band gap Eg**  
{Eg:.2f} eV  

**Incident photon energy**  
{E_in:.2f} eV  

---

**判定**

{result_text}

---
"""
        )

        if absorbed:
            st.markdown(
                f"""
**Emitted photon energy**  
{E_emit:.2f} eV  

**Emission wavelength**  
{lambda_emit:.0f} nm  
"""
            )
        else:
            st.markdown(
                """
**Reason**  
照射エネルギーがバンドギャップより小さいため、価電子帯から伝導帯への直接励起は起こりません。
"""
            )
