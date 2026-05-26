import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- 半導体材料の物理パラメータ設定 ---
KBOLTZ = 8.617e-5

# [シリコン (Si)]
NC_VAL = 280.0 * 1e17   # 2.8e19 cm^-3
NV_VAL = 10.4 * 1e18    # 1.04e19 cm^-3


# 初期パラメータ
t_start, g_start, f_start = 25.0, 1.10, 0.00

window, plot_ax = plt.subplots(figsize=(11, 5.5))
plt.subplots_adjust(bottom=0.25, right=0.6)

def run_simulation(celsius, gap, fermi):
    kelvin = max(celsius + 273.15, 0.01)
    ec, ev = gap / 2.0, -gap / 2.0
    v_axis = np.linspace(ev - 1.2, ec + 1.2, 1500)
    
    # 統計関数とDOS計算
    f_stat = 1.0 / (1.0 + np.exp(np.clip((v_axis - fermi) / (KBOLTZ * kelvin), -700, 700)))
    carrier_e = np.where(v_axis >= ec, np.sqrt(v_axis - ec), 0.0) * f_stat
    carrier_h = np.where(v_axis <= ev, np.sqrt(ev - v_axis), 0.0) * (1.0 - f_stat)
    scale = max(np.max(carrier_e), np.max(carrier_h), 1e-30)
    
    # 物理濃度の評価
    f_ec = 1.0 / (1.0 + np.exp(np.clip((ec - fermi) / (KBOLTZ * kelvin), -700, 700)))
    f_ev = 1.0 - 1.0 / (1.0 + np.exp(np.clip((ev - fermi) / (KBOLTZ * kelvin), -700, 700)))
    nc_t, nv_t = NC_VAL * (kelvin / 300.0)**1.5, NV_VAL * (kelvin / 300.0)**1.5
    
    n_density = nc_t * np.exp(-(ec - fermi) / (KBOLTZ * kelvin)) if kelvin > 0 else 0
    p_density = nv_t * np.exp(-(fermi - ev) / (KBOLTZ * kelvin)) if kelvin > 0 else 0
    ni_density = np.sqrt(nc_t * nv_t) * np.exp(-gap / (2 * KBOLTZ * kelvin)) if kelvin > 0 else 0
    
    return v_axis, f_stat, carrier_e/scale, carrier_h/scale, ec, ev, kelvin, f_ec, f_ev, n_density, p_density, ni_density

# 画面生成
v_x, f_d, en, hn, ec, ev, kv, fc, hv, n, p, ni = run_simulation(t_start, g_start, f_start)
line_f, = plot_ax.plot(v_x, f_d, linewidth=2, label="Fermi-Dirac f(E)", color="#4a4a4a")
line_e, = plot_ax.plot(v_x, en, linewidth=2, color="#00cbd6", label="Electron density")
line_h, = plot_ax.plot(v_x, hn, linewidth=2, linestyle="--", color="#ff7f0e", label="Hole density")
fill_e = plot_ax.fill_between(v_x, en, color="#00cbd6", alpha=0.2)
fill_h = plot_ax.fill_between(v_x, hn, color="#ff7f0e", alpha=0.2)

v_ev = plot_ax.axvline(ev, linestyle=":", color="#7f8c8d")
v_ef = plot_ax.axvline(f_start, linestyle="-.", color="#2980b9")
v_ec = plot_ax.axvline(ec, linestyle=":", color="#7f8c8d")
lbl_ev, lbl_ef, lbl_ec = plot_ax.text(ev+0.02, 0.92, "Ev"), plot_ax.text(f_start+0.02, 0.50, "Ef"), plot_ax.text(ec+0.02, 0.92, "Ec")
display_txt = plot_ax.text(1.1, 1.0, "", transform=plot_ax.transAxes, va='top', fontname='monospace', fontsize=10)

def draw_sidebar(tc, tk, eg, ef, ec, ev, fc, hv, n, p, ni):
    semi_class = "n-type-like" if ef > 0.05 else ("p-type-like" if ef < -0.05 else "intrinsic-like")
    display_txt.set_text(f" [ Parameters ]\n Temp : {tc:.0f} C ({tk:.2f} K)\n Eg   : {eg:.2f} eV\n Ef   : {ef:.3f} eV\n Type : {semi_class}\n\n"
                          f" [ Band positions ]\n Ec = +{ec:.3f} eV\n Ev = {ev:.3f} eV\n\n"
                          f" [ Occupation ]\n f(Ec)     = {fc:.3e}\n 1 - f(Ev) = {hv:.3e}\n\n"
                          f" [ Carrier density ]\n n  = {n:.3e} cm^-3\n p  = {p:.3e} cm^-3\n ni = {ni:.3e} cm^-3")

plot_ax.set_ylabel("Probability / normalized density")
plot_ax.set_ylim(-0.03, 1.05)
plot_ax.legend(loc="upper left", fontsize=8)
plot_ax.grid(True, alpha=0.2)

# スライダ
sld_T = Slider(plt.axes([0.15, 0.14, 0.4, 0.025]), "Temp (C)", -273.0, 1000.0, valinit=t_start, valstep=1)
sld_g = Slider(plt.axes([0.15, 0.09, 0.4, 0.025]), "Eg (eV)", 0.10, 3.00, valinit=g_start, valstep=0.01)
sld_f = Slider(plt.axes([0.15, 0.04, 0.4, 0.025]), "Ef (eV)", -1.00, 1.00, valinit=f_start, valstep=0.01)

def refresh_view(v):
    global fill_e, fill_h
    v_x, f_d, en, hn, ec, ev, kv, fc, hv, n, p, ni = run_simulation(sld_T.val, sld_g.val, sld_f.val)
    line_f.set_data(v_x, f_d); line_e.set_data(v_x, en); line_h.set_data(v_x, hn)
    v_ev.set_xdata([ev, ev]); v_ef.set_xdata([sld_f.val, sld_f.val]); v_ec.set_xdata([ec, ec])
    lbl_ev.set_x(ev+0.02); lbl_ef.set_x(sld_f.val+0.02); lbl_ec.set_x(ec+0.02)
    fill_e.remove(); fill_h.remove()
    fill_e = plot_ax.fill_between(v_x, en, color="#00cbd6", alpha=0.2)
    fill_h = plot_ax.fill_between(v_x, hn, color="#ff7f0e", alpha=0.2)
    plot_ax.set_xlim(ev - 1.2, ec + 1.2)
    draw_sidebar(sld_T.val, kv, sld_g.val, sld_f.val, ec, ev, fc, hv, n, p, ni)
    window.canvas.draw_idle()

sld_T.on_changed(refresh_view); sld_g.on_changed(refresh_view); sld_f.on_changed(refresh_view)
draw_sidebar(t_start, t_start+273.15, g_start, f_start, ec, ev, fc, hv, n, p, ni)
plt.show()
