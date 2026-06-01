import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# =====================================
# 初期値
# =====================================

Eg0 = 2.0
Ea0 = 0.2
Na0 = 1e14

Nv300 = 1.04e19
Nc300 = 2.80e19

kB = 8.617333262e-5

# =====================================
# 計算
# =====================================

def calculate(Eg, Ea, Na_user):

    Na = max(Na_user, 1.0)

    T = np.linspace(50, 1500, 1000)

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
# 初回計算
# =====================================

T, p = calculate(Eg0, Ea0, Na0)

# =====================================
# Figure
# =====================================

fig = plt.figure(figsize=(8, 6))

ax = plt.axes([0.12, 0.25, 0.80, 0.60])

line, = ax.plot(
    1 / T,
    np.log10(p),
    lw=3
)

ax.set_xlabel("1/T (K$^{-1}$)", fontsize=12)
ax.set_ylabel("log p (cm$^{-3}$)", fontsize=12)

ax.set_xlim(0, 0.012)
ax.grid(True)

# =====================================
# スライダー
# =====================================

axEg = plt.axes([0.18, 0.15, 0.65, 0.03])
axEa = plt.axes([0.18, 0.10, 0.65, 0.03])
axNa = plt.axes([0.18, 0.05, 0.65, 0.03])

sEg = Slider(
    ax=axEg,
    label="Eg (eV)",
    valmin=0.5,
    valmax=6.5,
    valinit=Eg0
)

sEa = Slider(
    ax=axEa,
    label="Ea (eV)",
    valmin=0.01,
    valmax=1.0,
    valinit=Ea0
)

sNa = Slider(
    ax=axNa,
    label="log10(Na)",
    valmin=0,
    valmax=20,
    valinit=np.log10(Na0)
)

# =====================================
# 更新
# =====================================

def update(val):

    Eg = sEg.val
    Ea = sEa.val

    if sNa.val < 0.1:
        Na = 0
    else:
        Na = 10 ** sNa.val

    T, p = calculate(Eg, Ea, Na)

    line.set_xdata(1 / T)
    line.set_ydata(np.log10(p))

    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)

    fig.canvas.draw_idle()

sEg.on_changed(update)
sEa.on_changed(update)
sNa.on_changed(update)

plt.show()
