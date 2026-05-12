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

# Silicon fixed parameters
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

    ni = np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * k_B * T))

    return ni


def plot_intrinsic_density(Eg_material):
    T = np.linspace(250, 1200, 1000)  # K
    inv_T = 1000 / T

    ni_Si = intrinsic_density(
        T,
        Eg_Si,
        Nc_Si_300,
        Nv_Si_300
    )

    ni_material = intrinsic_density(
        T,
        Eg_material,
        Nc_Si_300,
        Nv_Si_300
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=120)

    ax.semilogy(
        inv_T,
        ni_Si,
        linewidth=2.2,
        label=f"Si: Eg = {Eg_Si:.2f} eV"
    )

    ax.semilogy(
        inv_T,
        ni_material,
        linewidth=2.2,
        label=f"Material: Eg = {Eg_material:.2f} eV"
    )

    ax.set_xlabel("1000 / T (K⁻¹)")
    ax.set_ylabel("Intrinsic carrier density ni (cm⁻³)")

    # 重要：反転しない
    # 左：高温、右：低温
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

# Silicon fixed parameters
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

    ni = np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * k_B * T))

    return ni


def plot_intrinsic_density(Eg_material):
    T = np.linspace(250, 1200, 1000)  # K
    inv_T = 1000 / T

    ni_Si = intrinsic_density(
        T,
        Eg_Si,
        Nc_Si_300,
        Nv_Si_300
    )

    ni_material = intrinsic_density(
        T,
        Eg_material,
        Nc_Si_300,
        Nv_Si_300
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=120)

    ax.semilogy(
        inv_T,
        ni_Si,
        linewidth=2.2,
        label=f"Si: Eg = {Eg_Si:.2f} eV"
    )

    ax.semilogy(
        inv_T,
        ni_material,
        linewidth=2.2,
        label=f"Material: Eg = {Eg_material:.2f} eV"
    )

    ax.set_xlabel("1000 / T (K⁻¹)")
    ax.set_ylabel("Intrinsic carrier density ni (cm⁻³)")

    # 重要：反転しない
    # 左：高温、右：低温
    ax.set_xlim(0.5, 4.0)
    ax.set_ylim(1e7, 1e19)

    ax.tick_params(axis="both", direction="in")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(False)

    fig.tight_layout()

    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("真性キャリア密度の温度依存性：Siとの比較")

col1, col2, col3 = st.columns([0.75, 1.6, 0.75])

with col1:
    st.subheader("Controls")

    Eg_material = st.slider(
        "Bandgap of compared material Eg (eV)",
        0.5,
        6.0,
        3.4,
        0.01
    )

with col2:
    fig = plot_intrinsic_density(Eg_material)
    st.pyplot(fig, use_container_width=True)

with col3:
    st.subheader("Parameters")

    st.markdown(
        f"""
**Fixed material**

Si  

Eg = {Eg_Si:.2f} eV  

---

**Compared material**

Eg = {Eg_material:.2f} eV  

---

**Plot**

x-axis: 1000 / T  

y-axis: ni  

---

左側：高温  

右側：低温  

右に行くほど ni は低下します。
"""
    )

    ax.tick_params(axis="both", direction="in")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(False)

    fig.tight_layout()

    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("真性キャリア密度の温度依存性：Siとの比較")

col1, col2, col3 = st.columns([0.75, 1.6, 0.75])

with col1:
    st.subheader("Controls")

    Eg_material = st.slider(
        "Bandgap of compared material Eg (eV)",
        0.5,
        6.0,
        3.4,
        0.01
    )

with col2:
    fig = plot_intrinsic_density(Eg_material)
    st.pyplot(fig, use_container_width=True)

with col3:
    st.subheader("Parameters")

    st.markdown(
        f"""
**Fixed material**

Si  

Eg = {Eg_Si:.2f} eV  

---

**Compared material**

Eg = {Eg_material:.2f} eV  

---

**Plot**

x-axis: 1000 / T  

y-axis: ni  

---

左側：高温  

右側：低温  

右に行くほど ni は低下します。
"""
    )
