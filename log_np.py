import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# =====================================
# 定数
# =====================================
Nv300 = 1.04e19
Nc300 = 2.80e19
kB = 8.617333262e-5  # eV/K

# =====================================
# 計算
# =====================================
def calculate(Eg, Ea, Na_user):
    Na = max(Na_user, 1.0)

    T = np.linspace(1, 2273.15, 1500)

    Nv = Nv300 * (T / 300) ** 1.5
    Nc = Nc300 * (T / 300) ** 1.5

    Ef = np.zeros_like(T)

    for i, temp in enumerate(T):
        nv = Nv[i]
        nc = Nc[i]

        E = np.linspace(0, Eg, 3000)

        lhs = nv / (1 + np.exp(E / (kB * temp)))

        rhs = (
            Na / (1 + np.exp((Ea - E) / (kB * temp)))
            + nc / (1 + np.exp((Eg - E) / (kB * temp)))
        )

        Ef[i] = E[np.argmin(np.abs(lhs - rhs))]

    p = Nv * np.exp(-Ef / (kB * T))

    return T, p


# =====================================
# Streamlit UI
# =====================================
st.title("p-type carrier concentration")

st.sidebar.header("Parameters")

Eg = st.sidebar.slider(
    "Eg (eV)",
    min_value=0.5,
    max_value=6.5,
    value=2.0,
    step=0.01
)

Ea = st.sidebar.slider(
    "Ea (eV)",
    min_value=0.01,
    max_value=1.0,
    value=0.2,
    step=0.01
)

logNa = st.sidebar.slider(
    "log10(Na)",
    min_value=13.0,
    max_value=20.0,
    value=14.0,
    step=0.1
)

Na = 10 ** logNa

T, p = calculate(Eg, Ea, Na)

# =====================================
# Plot
# =====================================
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(
    1 / T,
    np.log10(p),
    lw=3
)

ax.set_xlabel("1/T (K$^{-1}$)", fontsize=14)
ax.set_ylabel("log p (cm$^{-3}$)", fontsize=14)

ax.set_xlim(0, 0.012)
ax.set_ylim(0, 23) 
ax.tick_params(direction="in", labelsize=12)

st.pyplot(fig)

# =====================================
# 数値表示
# =====================================
st.write("### Parameters")
st.write(f"Eg = {Eg:.2f} eV")
st.write(f"Ea = {Ea:.2f} eV")
st.write(f"Na = {Na:.2e} cm⁻³")
