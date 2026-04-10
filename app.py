```python
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

k_B = 8.617e-5
Eg = 1.1
Ef = 0.0
Ec = Eg / 2
Ev = -Eg / 2

Nc = 2.8e19
Nv = 1.04e19

def intrinsic_density(T):
    return np.sqrt(Nc * Nv) * np.exp(-Eg / (2 * k_B * T))

def density_to_points(n):
    log_n = np.log10(n)
    return int(np.clip((log_n - 8) * 20, 0, 300))

def sample_conduction(T, n_points):
    scale = k_B * T
    dE = np.random.exponential(scale=scale, size=n_points)
    return Ec + dE

def sample_valence(T, n_points):
    scale = k_B * T
    dE = np.random.exponential(scale=scale, size=n_points)
    return Ev - dE

st.title("Intrinsic Semiconductor Visualization")

T = st.slider("Temperature (K)", 50, 1000, 300, step=50)

ni = intrinsic_density(T)
n_points = density_to_points(ni)

fig, ax = plt.subplots(figsize=(4, 6))

ax.plot([0, 1], [Ec, Ec])
ax.plot([0, 1], [Ev, Ev])
ax.plot([0, 1], [Ef, Ef], linestyle="--")

if n_points > 0:
    y_e = sample_conduction(T, n_points)
    x_e = np.random.uniform(0.2, 0.8, size=n_points)
    ax.scatter(x_e, y_e, s=10, label="Electrons")

if n_points > 0:
    y_h = sample_valence(T, n_points)
    x_h = np.random.uniform(0.2, 0.8, size=n_points)
    ax.scatter(x_h, y_h, s=10, label="Holes")

ax.set_xlim(0, 1)
ax.set_ylim(-1.5, 1.5)
ax.set_xticks([])
ax.set_ylabel("Energy (eV)")
ax.set_title(f"T = {T} K, ni ≈ {ni:.2e} cm^-3")
ax.legend()

st.pyplot(fig)

