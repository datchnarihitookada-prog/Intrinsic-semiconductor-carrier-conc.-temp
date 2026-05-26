import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --- 定数 ---
KBOLTZ = 8.617e-5  # eV/K

# Si parameters
NC_VAL = 280.0 * 1e17   # 2.8e19 cm^-3
NV_VAL = 10.4 * 1e18    # 1.04e19 cm^-3


def run_simulation(celsius, gap, fermi):
    kelvin = max(celsius + 273.15, 0.01)

    ec = gap / 2.0
    ev = -gap / 2.0

    v_axis = np.linspace(ev - 1.2, ec + 1.2, 1500)

    f_stat = 1.0 / (
        1.0 + np.exp(np.clip((v_axis - fermi) / (KBOLTZ * kelvin), -700, 700))
    )

    carrier_e = np.where(v_axis >= ec, np.sqrt(v_axis - ec), 0.0) * f_stat
    carrier_h = np.where(v_axis <= ev, np.sqrt(ev - v_axis), 0.0) * (1.0 - f_stat)

    scale = max(np.max(carrier_e), np.max(carrier_h), 1e-30)

    f_ec = 1.0 / (
        1.0 + np.exp(np.clip((ec - fermi) / (KBOLTZ * kelvin), -700, 700))
    )

    f_ev = 1.0 - 1.0 / (
        1.0 + np.exp(np.clip((ev - fermi) / (KBOLTZ * kelvin), -700, 700))
    )

    nc_t = NC_VAL * (kelvin / 300.0) ** 1.5
    nv_t = NV_VAL * (kelvin / 300.0) ** 1.5

    n_density = nc_t * np.exp(-(ec - fermi) / (KBOLTZ * kelvin))
    p_density = nv_t * np.exp(-(fermi - ev) / (KBOLTZ * kelvin))
    ni_density = np.sqrt(nc_t * nv_t) * np.exp(-gap / (2 * KBOLTZ * kelvin))

    return (
        v_axis,
        f_stat,
        carrier_e / scale,
        carrier_h / scale,
        ec,
        ev,
        kelvin,
        f_ec,
        f_ev,
        n_density,
        p_density,
        ni_density,
    )


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Fermi-Dirac Distribution and Carrier Density Simulator")

left, right = st.columns([1, 2])

with left:
    st.subheader("Parameters")

    temp_c = st.slider("Temperature (°C)", -273.0, 1000.0, 25.0, 1.0)
    eg = st.slider("Band gap Eg (eV)", 0.10, 3.00, 1.10, 0.01)
    ef = st.slider("Fermi level Ef (eV)", -1.00, 1.00, 0.00, 0.01)


(
    v_x,
    f_d,
    en,
    hn,
    ec,
    ev,
    kv,
    fc,
    hv,
    n,
    p,
    ni,
) = run_simulation(temp_c, eg, ef)


semi_class = (
    "n-type-like" if ef > 0.05 else
    "p-type-like" if ef < -0.05 else
    "intrinsic-like"
)


with left:
    st.subheader("Results")

    st.text(
        f"""
[ Parameters ]
Temp : {temp_c:.0f} °C ({kv:.2f} K)
Eg   : {eg:.2f} eV
Ef   : {ef:.3f} eV
Type : {semi_class}

[ Band positions ]
Ec = +{ec:.3f} eV
Ev = {ev:.3f} eV

[ Occupation ]
f(Ec)     = {fc:.3e}
1 - f(Ev) = {hv:.3e}

[ Carrier density ]
n  = {n:.3e} cm^-3
p  = {p:.3e} cm^-3
ni = {ni:.3e} cm^-3
"""
    )


with right:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(v_x, f_d, linewidth=2, label="Fermi-Dirac f(E)", color="#4a4a4a")
    ax.plot(v_x, en, linewidth=2, color="blue", label="Electron density")
    ax.plot(v_x, hn, linewidth=2, linestyle="--", color="red", label="Hole density")

    ax.fill_between(v_x, en, color="blue", alpha=0.15)
    ax.fill_between(v_x, hn, color="red", alpha=0.15)

    ax.axvline(ev, linestyle=":", color="#7f8c8d")
    ax.axvline(ef, linestyle="-.", color="#2980b9")
    ax.axvline(ec, linestyle=":", color="#7f8c8d")

    ax.text(ev + 0.02, 0.92, "Ev")
    ax.text(ef + 0.02, 0.50, "Ef")
    ax.text(ec + 0.02, 0.92, "Ec")

    ax.set_xlabel("Energy E (eV)")
    ax.set_ylabel("Probability / normalized density")
    ax.set_xlim(ev - 1.2, ec + 1.2)
    ax.set_ylim(-0.03, 1.05)

    ax.legend(loc="upper left", fontsize=8)

    # ユーザー設定に合わせて、目盛りは内向き・グリッドなし
    ax.tick_params(direction="in")

    st.pyplot(fig)
