import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="PN Junction Dynamic Simulation",
    layout="centered"
)

# ==========================================
# 1. パラメータ設定
# ==========================================
Nx = 400
x = np.linspace(-1.0, 1.0, Nx)
Nt = 201

Eg = 1.2
V_delta = 0.7

# ==========================================
# 2. バンド構造・フェルミ準位の事前計算
# ==========================================
Ec_history = np.zeros((Nt, Nx))
Ev_history = np.zeros((Nt, Nx))
Ef_history = np.zeros((Nt, Nx))
W_dep_history = np.zeros(Nt)

Ec_p_side = 0.5

for step_i in range(Nt):
    progress_i = step_i / (Nt - 1)

    if progress_i < 0.1:
        W_dep = 0.0
        curr_v = 0.0
        ef_weight = 0.0
    else:
        p = (progress_i - 0.1) / 0.9
        W_dep = 0.25 * p
        curr_v = V_delta * p
        ef_weight = p

    W_dep_history[step_i] = W_dep
    Ec = np.zeros(Nx)

    for i, pos in enumerate(x):
        if pos < -W_dep:
            Ec[i] = Ec_p_side
        elif pos > W_dep:
            Ec[i] = Ec_p_side - curr_v
        else:
            if W_dep > 0:
                if pos < 0:
                    Ec[i] = Ec_p_side - 0.5 * curr_v * ((pos + W_dep) / W_dep) ** 2
                else:
                    Ec[i] = (Ec_p_side - curr_v) + 0.5 * curr_v * ((W_dep - pos) / W_dep) ** 2
            else:
                Ec[i] = Ec_p_side

    Ev = Ec - Eg

    Ef_p = Ec_p_side - 0.95
    Ef_n = (Ec_p_side - V_delta) - 0.25
    Ef = np.where(x < 0, Ef_p, Ef_n * (1 - ef_weight) + Ef_p * ef_weight)

    Ec_history[step_i] = Ec
    Ev_history[step_i] = Ev
    Ef_history[step_i] = Ef

# ==========================================
# 3. キャリア配置
# ==========================================
num_maj = 35
num_min = 2
np.random.seed(42)

x_holes_maj_init = np.random.uniform(-0.95, -0.05, num_maj)
hole_speeds = np.random.uniform(0.5, 0.8, num_maj)

u_h = np.random.uniform(0, 1, num_maj)
x_holes_maj_eq = -0.95 + 0.70 * (u_h ** 0.5)
x_holes_maj_eq[0] = 0.4
x_holes_maj_eq[1] = 0.6

x_elecs_maj_init = np.random.uniform(0.05, 0.95, num_maj)
elec_speeds = np.random.uniform(0.5, 0.8, num_maj)

u_e = np.random.uniform(0, 1, num_maj)
x_elecs_maj_eq = 0.95 - 0.70 * (u_e ** 0.5)
x_elecs_maj_eq[0] = -0.4
x_elecs_maj_eq[1] = -0.6

x_holes_min_fixed = np.random.uniform(0.3, 0.95, num_min)
x_elecs_min_fixed = np.random.uniform(-0.95, -0.3, num_min)

y_offsets_h_maj = np.random.uniform(0.04, 0.12, num_maj)
y_offsets_h_min = np.random.uniform(0.04, 0.12, num_min)
y_offsets_e_maj = np.random.uniform(0.04, 0.12, num_maj)
y_offsets_e_min = np.random.uniform(0.04, 0.12, num_min)

# ==========================================
# 4. Streamlit UI
# ==========================================
st.title("PN Junction Dynamic Simulation")

step = st.slider(
    "Time Step",
    min_value=0,
    max_value=Nt - 1,
    value=0,
    step=1
)

progress = step / (Nt - 1)
p_weight = max(0.0, (progress - 0.1) / 0.9)

if progress < 0.1:
    status_str = "1. Before Junction (Flat Bands)"
elif progress < 0.90:
    status_str = "2. Transient State (Diffusion & Drift U-Turn)"
else:
    status_str = "3. Thermal Equilibrium (Carrier-Free Depletion Region)"

maj_p = 1.5e17 * (1.0 - 0.05 * p_weight)
min_n = 1.5e3 * (1.0 + 20 * p_weight)
Vbi_curr = V_delta * p_weight

st.markdown(f"### Status: {status_str}")

st.code(
    f"[p-region (Left)]  Majority(Hole): {maj_p:.2e} cm^-3 | Minority(Electron): {min_n:.2e} cm^-3\n"
    f"[n-region (Right)] Majority(Elec): {maj_p:.2e} cm^-3 | Minority(Hole):     {min_n:.2e} cm^-3\n"
    f"[Potential]         Built-in Potential (Vbi): {Vbi_curr:.4f} V"
)

# ==========================================
# 5. 描画用計算
# ==========================================
W_dep = W_dep_history[step]
Ec_curr = Ec_history[step]
Ev_curr = Ev_history[step]
Ef_curr = Ef_history[step]

if p_weight < 0.4:
    blend = 0.0
else:
    t = (p_weight - 0.4) / 0.6
    blend = 3 * t**2 - 2 * t**3

part_x_hole, part_y_hole = [], []
part_x_elec, part_y_elec = [], []

# ホール
for i in range(num_maj):
    if progress < 0.1:
        curr_x = x_holes_maj_init[i]
    else:
        x_diff = x_holes_maj_init[i] + (p_weight * hole_speeds[i] * 0.7)
        curr_x = (1.0 - blend) * x_diff + blend * x_holes_maj_eq[i]

    curr_x = max(-0.99, min(0.99, curr_x))
    part_x_hole.append(curr_x)

    y_val = np.interp(curr_x, x, Ev_curr)
    part_y_hole.append(y_val - y_offsets_h_maj[i])

for i in range(num_min):
    curr_x = x_holes_min_fixed[i]
    part_x_hole.append(curr_x)

    y_val = np.interp(curr_x, x, Ev_curr)
    part_y_hole.append(y_val - y_offsets_h_min[i])

# 電子
for i in range(num_maj):
    if progress < 0.1:
        curr_x = x_elecs_maj_init[i]
    else:
        x_diff = x_elecs_maj_init[i] - (p_weight * elec_speeds[i] * 0.7)
        curr_x = (1.0 - blend) * x_diff + blend * x_elecs_maj_eq[i]

    curr_x = max(-0.99, min(0.99, curr_x))
    part_x_elec.append(curr_x)

    y_val = np.interp(curr_x, x, Ec_curr)
    part_y_elec.append(y_val + y_offsets_e_maj[i])

for i in range(num_min):
    curr_x = x_elecs_min_fixed[i]
    part_x_elec.append(curr_x)

    y_val = np.interp(curr_x, x, Ec_curr)
    part_y_elec.append(y_val + y_offsets_e_min[i])

# ==========================================
# 6. Matplotlib描画
# ==========================================
fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=120)

ax.plot(x, Ec_curr, color="darkblue", lw=3, label="Conduction Band (Ec)")
ax.plot(x, Ev_curr, color="darkred", lw=3, label="Valence Band (Ev)")
ax.plot(x, Ef_curr, color="green", linestyle="--", lw=2, label="Fermi Level (Ef)")

if W_dep > 0:
    ax.axvspan(-W_dep, W_dep, color="lightgray", alpha=0.25, label="Depletion Region")
    ax.axvline(-W_dep, color="gray", linestyle=":", lw=1.2)
    ax.axvline(W_dep, color="gray", linestyle=":", lw=1.2)

ax.scatter(
    part_x_elec,
    part_y_elec,
    color="blue",
    s=35,
    alpha=0.8,
    zorder=5,
    label="Electron"
)

ax.scatter(
    part_x_hole,
    part_y_hole,
    color="red",
    s=35,
    alpha=0.8,
    zorder=5,
    label="Hole"
)

ax.set_xlim(-1.0, 1.0)
ax.set_ylim(-2.2, 1.3)

ax.set_xlabel("Position", fontsize=10)
ax.set_ylabel("Energy [eV]", fontsize=10)

ax.tick_params(labelsize=9)
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(loc="lower left", framealpha=0.9, fontsize=8)

fig.tight_layout()

st.pyplot(fig, use_container_width=False)
plt.close(fig)
