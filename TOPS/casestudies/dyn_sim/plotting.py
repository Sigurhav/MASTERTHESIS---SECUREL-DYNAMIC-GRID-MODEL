import sys
import time
import importlib
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.widgets import RadioButtons
import src.dynamic as dps
import src.linear_ps as dps_mdl
import src.solvers as dps_sol
from utility_functions import print_init_info
importlib.reload(dps)


if __name__ == "__main__":
    # ── Load model ────────────────────────────────────────────────
    import casestudies.ps_data.maximport as model_data

    importlib.reload(model_data)
    model = model_data.load()

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    print("\n" + "=" * 80)
    print("Analyse fullført! Plot lagret som 'eigenvalue_analysis.png'")
    print("=" * 80)

    plt.show()

    print(
        f"Max initial state derivative: "
        f"{max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))):.6e}"
    )
    print_init_info(ps)

    # ── Initial bus voltages ──────────────────────────────────────
    print("\n--- Busspenninger ved initialisering ---")
    v_mag = np.abs(ps.v_0)
    v_ang = np.angle(ps.v_0)

    for i in range(len(v_mag)):
        print(
            f"  Bus {i:2d}: |V| = {v_mag[i]:.3f} p.u.  "
            f"∠ {v_ang[i]:.3f} rad / {v_ang[i] * 180 / np.pi:.2f}°"
        )

    # ── Initial generator power ───────────────────────────────────
    print("\n--- Generatoreffekt ved initialisering ---")
    for gk in ps.gen.keys():
        g = ps.gen[gk]
        scale = g.par["S_n"] / ps.s_n

        pe = np.atleast_1d(g.p_e(ps.x_0, ps.v_0).copy()) * scale
        qe = np.atleast_1d(g.q_e(ps.x_0, ps.v_0).copy()) * scale

        for i in range(len(pe)):
            tag = gk if len(pe) == 1 else f"{gk}_{i}"
            print(
                f"  {tag:>10s}: P = {pe[i]:.3f} p.u. ({pe[i] * ps.s_n:.1f} MW)  "
                f"Q = {qe[i]:.3f} p.u. ({qe[i] * ps.s_n:.1f} MVAr)"
            )

    # ── Initial line currents and power flow ──────────────────────
    print("\n--- Linjestrøm og effektflyt ved initialisering ---")
    for lk in ps.lines.keys():
        line = ps.lines[lk]
        p_f = line.p_from(ps.x_0, ps.v_0)
        q_f = line.q_from(ps.x_0, ps.v_0)
        i_f = line.i_from(ps.x_0, ps.v_0)

        for i in range(line.n_units):
            name = line.par["name"][i]
            f_bus = line.par["from_bus"][i]
            t_bus = line.par["to_bus"][i]
            i_mag = np.abs(i_f[i])
            V_n = line.par["V_n"][i]
            I_base = ps.s_n / (np.sqrt(3) * V_n) * 1e3
            i_amp = i_mag * I_base

            print(
                f"  {name:>6s} ({f_bus} → {t_bus}): "
                f"|I| = {i_mag:.3f} p.u. ({i_amp:.1f} A)  "
                f"P = {p_f[i]:.3f} p.u. ({p_f[i] * ps.s_n:.1f} MW)  "
                f"Q = {q_f[i]:.3f} p.u. ({q_f[i] * ps.s_n:.1f} MVAr)"
            )

    # ── Simulation parameters ─────────────────────────────────────
    t_end = 350
    t_event = 5.0
    x_0 = ps.x_0.copy()

    sol = dps_sol.ModifiedEulerDAE(
        ps.state_derivatives,
        ps.solve_algebraic,
        0,
        x_0,
        t_end,
        max_step=5e-3,
    )

    # ── Discover model keys ───────────────────────────────────────
    gen_keys = list(ps.gen.keys())
    battery_keys = list(ps.vsc.keys()) if hasattr(ps, "vsc") else []

    # ── Run simulation ────────────────────────────────────────────
    t = 0
    res = defaultdict(list)
    t_0 = time.time()
    event_flag = True
    event_flag2 = True
    snapshot = None

    while t < t_end:
        sys.stdout.write(f"\r{t / t_end * 100:.0f}%")

        if t_event < t < t_event + 0.01:
            ps.y_bus_red_mod[1, 1] = 1e6
        else:
            ps.y_bus_red_mod[1, 1] = 0

        # Event: disconnect line L1 at t = t_event
        if t > t_event + 0.01 and event_flag:
            event_flag = False
            ps.lines["Line"].event(ps, ps.lines["Line"].par["name"][0], "disconnect")

        # if 5 < t and event_flag:
        #     event_flag1 = False
        #     ps.gen["IB"].speedchange(x, -0.002)

        sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        res["t"].append(t)

        # ── Generator signals ─────────────────────────────────────
        for gk in gen_keys:
            g = ps.gen[gk]
            scale = g.par["S_n"] / ps.s_n

            H = g.par["H"]
            speed = np.atleast_1d(g.speed(x, v).copy())
            pm = np.atleast_1d(g.P_m(x, v).copy()) * scale
            pe = np.atleast_1d(g.p_e(x, v).copy()) * scale
            ang = np.atleast_1d(g.angle(x, v).copy())
            vt_cx = np.atleast_1d(g.v_t(x, v).copy())
            vt = np.atleast_1d(g.v_t_abs(x, v).copy())
            p_nom = np.atleast_1d(g.P_nom(x, v).copy()) / ps.s_n
            p_set = np.atleast_1d(g.P_set(x, v).copy()) / ps.s_n

            for i in range(len(speed)):
                tag = gk if len(speed) == 1 else f"{gk}_{i + 1}"
                res[f"speed__{tag}"].append(speed[i])
                res[f"P_m__{tag}"].append(pm[i])
                res[f"P_e__{tag}"].append(pe[i])
                res[f"v_t__{tag}"].append(vt[i])
                res[f"Ang__{tag}"].append(ang[i])
                res[f"Vt_ang__{tag}"].append(np.angle(vt_cx[i]))
                res[f"P_nom__{tag}"].append(p_nom[i])
                res[f"H__{tag}"].append(H[i])
                res[f"P_set__{tag}"].append(p_set[i])

        # ── Battery signals ───────────────────────────────────────
        for bk in battery_keys:
            bat = ps.vsc[bk]
            scale = bat.par["S_n"] / ps.s_n
            p = np.atleast_1d(bat.p_e(x, v).copy()) * scale

            for i in range(len(p)):
                tag = bk if len(p) == 1 else f"{bk}_{i}"
                res[f"P_bat__{tag}"].append(p[i])

    print(f"\nSimulation completed in {time.time() - t_0:.2f} seconds.")

    # ── Convert results to numpy arrays ───────────────────────────
    for k in res:
        res[k] = np.array(res[k])

    # ── Frequency metrics ─────────────────────────────────────────
    print("\n--- Frekvensmetrikker ---")
    f_nom = 50.0
    settle_tol_pct = 0.02

    t_arr = res["t"]
    speed_names = [k.replace("speed__", "") for k in res if k.startswith("speed__")]
    bat_names = [k.replace("P_bat__", "") for k in res if k.startswith("P_bat__")]

    for nam in bat_names:
        p_list = res[f"P_bat__{nam}"]

        post = t_arr > 5.05
        t_post = t_arr[post]
        p_post = p_list[post]

        p_min = min(p_post)
        p_max = max(p_post)

        p_post_mw = p_post * 100
        p_sign = np.sign(p_post_mw)
        sign_change_indices = np.where(p_sign[:-1] != p_sign[1:])[0]

        if len(sign_change_indices) == 0:
            E_first_peak = np.trapezoid(p_post_mw, t_post)
            E_reverse = 0
        else:
            first_change = sign_change_indices[0] + 1
            E_first_peak = np.trapezoid(
                p_post_mw[:first_change], t_post[:first_change]
            )
            E_reverse = np.trapezoid(
                p_post_mw[first_change:], t_post[first_change:]
            )

        if E_first_peak > 0:
            print("Discharge =", E_first_peak / 3600, "MWh")
        elif E_first_peak < 0:
            print("Charge =", -E_first_peak / 3600, "MWh")
        else:
            print("Negligible energy")

        if E_reverse > 0:
            print("Discharge =", E_reverse / 3600, "MWh")
        elif E_reverse < 0:
            print("Charge =", -E_reverse / 3600, "MWh")
        else:
            print("No reverse operation")

        print(nam, ":P_min:", p_min * 100)
        print(nam, ":P_max:", p_max * 100)

    print(
        f"{'f_final':>8s} | {'t_settle':>9s} | {'f_nadir':>8s} | "
        f"{'t_nadir':>8s} | {'f_zenith':>9s} | {'t_zenith':>9s} | "
        f"{'Peak ROCOF':>10s}"
    )
    print("-" * 100)

    # ── Parameter calculations ────────────────────────────────────
    finite_names = [name for name in speed_names if "IB" not in name]

    total_HS = sum(res[f"H__{name}"] * res[f"P_nom__{name}"] for name in finite_names)
    denominator = sum(
        res[f"H__{name}"] * res[f"P_nom__{name}"] * (res[f"speed__{name}"] * 50 + 50)
        for name in finite_names
    )
    fcoi = denominator / total_HS

    post = t_arr > (t_event + 0.01)
    t_post = t_arr[post]
    f_post = fcoi[post]

    f_final = np.mean(fcoi[t_arr > t_arr[-1] - 5])

    tol = settle_tol_pct * f_final
    outside = np.abs(f_post - f_final) > tol
    if np.any(outside):
        t_settle = t_post[outside][-1] - (t_event + 0.01)
    else:
        t_settle = 0.0

    dt = t_post[1] - t_post[0]
    rocof = np.diff(f_post) / dt
    t_rocof = t_post[:-1]

    valid_rocof = t_rocof >= 5.5
    idx_rocof_local = np.argmax(np.abs(rocof[valid_rocof]))
    idx_rocof = np.where(valid_rocof)[0][idx_rocof_local]

    max_rocof = rocof[idx_rocof]
    t_max_rocof = t_rocof[idx_rocof]
    f_at_rocof = np.interp(t_max_rocof, res["t"], fcoi)

    idx_nadir = np.argmin(f_post)
    f_nadir = f_post[idx_nadir]
    t_nadir = t_post[idx_nadir] - (t_event + 0.01)

    idx_zenith = np.argmax(f_post)
    f_zenith = f_post[idx_zenith]
    t_zenith = t_post[idx_zenith] - (t_event + 0.01)

    t_nadir_abs = t_post[idx_nadir]
    t_zenith_abs = t_post[idx_zenith]

    if abs(f_zenith - f_final) > abs(f_nadir - f_final):
        f_peak = f_zenith
    else:
        f_peak = f_nadir

    print(
        f"{f_final:>7.3f}  | {t_settle:>7.1f} s | "
        f"{f_nadir:>7.3f}  | {t_nadir:>6.1f} s | "
        f"{f_zenith:>7.3f}  | {t_zenith:>6.1f} s | "
        f"{max_rocof:>9.2f} Hz/s "
    )

    # ── Find index near target time ───────────────────────────────
    target = 300.0
    margin = 0.01

    mask = np.abs(res["t"] - target) <= margin
    indices = np.where(mask)[0]

    if len(indices) == 0:
        idx = int(np.argmin(np.abs(res["t"] - target)))
    else:
        idx = int(indices[0])

    x_idx = float(res["t"][idx])

    # ── Plot settings ─────────────────────────────────────────────
    rcParams["font.family"] = "Times New Roman"
    rcParams["font.size"] = 20

    # ── Helper: radio-button selector ─────────────────────────────
    def add_selector(fig, ax, line_groups, labels):
        n = len(labels)
        btn_h = min(0.8, n * 0.045)
        ax_r = fig.add_axes([0.02, 0.5 - btn_h / 2, 0.08, btn_h])
        radio = RadioButtons(ax_r, labels, active=0)

        for lbl in radio.labels:
            lbl.set_fontsize(max(7, min(10, int(120 / n))))

        def on_click(label):
            all_lines = [l for lines in line_groups.values() for l in lines]

            if label == "Show All":
                for l in all_lines:
                    l.set_visible(True)
            else:
                for l in all_lines:
                    l.set_visible(False)
                for l in line_groups.get(label, []):
                    l.set_visible(True)

            ax.legend(
                handles=[l for l in all_lines if l.get_visible()],
                loc="upper right",
                fontsize=20,
            )
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw_idle()

        radio.on_clicked(on_click)
        fig._radio = radio

    # ── Collect plot keys ─────────────────────────────────────────
    speed_names = [k.replace("speed__", "") for k in res if k.startswith("speed__")]
    pm_names = [k.replace("P_m__", "") for k in res if k.startswith("P_m__")]
    vt_names = [k.replace("v_t__", "") for k in res if k.startswith("v_t__")]
    bat_names = [k.replace("P_bat__", "") for k in res if k.startswith("P_bat__")]
    ang_names = [k.replace("Ang__", "") for k in res if k.startswith("Ang__")]

    # ── Plot 1: Frequency ─────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(11, 6))
    plt.subplots_adjust(left=0.12, right=0.88)

    ax1.plot(res["t"], fcoi, label="fCOI", color="tab:blue", linewidth=2)
    ax1.axhline(y=50.0, color="green", linestyle="--", linewidth=1.2, label="50 Hz")

    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_title("Centre of inertia frequency")
    ax1.grid(True)
    ax1.legend(fontsize=20)

    # ── Plot 2: Generator power ───────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    plt.subplots_adjust(left=0.28)
    groups2 = {}

    for name in pm_names:
        if name == "IB":
            continue

        l1, = ax2.plot(res["t"], res[f"P_m__{name}"], label=f"Pm: {name}")
        l2, = ax2.plot(
            res["t"],
            res[f"P_e__{name}"],
            label=f"Pe: {name}",
            linestyle="--",
        )
        y0 = res[f"P_nom__{name}"]
        l3, = ax2.plot(res["t"], np.full_like(res["t"], y0), label=f"Pnom: {name}")
        groups2[name] = [l1]

    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Power [p.u. on 100 MVA]")
    ax2.set_title("Generator Mechanical Power")
    ax2.grid(True)
    ax2.legend(fontsize=13)
    add_selector(fig2, ax2, groups2, ["Show All"] + pm_names)

    # ── Plot 3: Terminal voltage ──────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(11, 6))
    plt.subplots_adjust(left=0.28)
    groups3 = {}

    for name in vt_names:
        line, = ax3.plot(res["t"], res[f"v_t__{name}"], label=name)
        groups3[name] = [line]

    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Voltage [p.u.]")
    ax3.set_title("Terminal Voltage")
    ax3.grid(True)
    ax3.legend(fontsize=8)
    add_selector(fig3, ax3, groups3, ["Show All"] + vt_names)

    # ── Plot 4: Battery power ─────────────────────────────────────
    if bat_names:
        fig4, ax4 = plt.subplots(figsize=(11, 6))
        plt.subplots_adjust(left=0.28)
        groups4 = {}

        for name in bat_names:
            line, = ax4.plot(res["t"], res[f"P_bat__{name}"], label=name)
            groups4[name] = [line]

        ax4.set_xlabel("Time [s]")
        ax4.set_ylabel("Power [p.u. on 100 MVA]")
        ax4.set_title("Battery Power")
        ax4.grid(True)
        ax4.legend(fontsize=8)
        add_selector(fig4, ax4, groups4, ["Show All"] + bat_names)

    # ── Plot 5: Power angle ───────────────────────────────────────
    if ang_names:
        fig6, ax6 = plt.subplots(figsize=(11, 6))
        plt.subplots_adjust(left=0.28)
        groups6 = {}

        for name in ang_names:
            power_angle = np.degrees(
                np.angle(np.exp(1j * (res[f"Ang__{name}"] - res[f"Vt_ang__{name}"])))
            )
            line, = ax6.plot(res["t"], power_angle, label=name)
            groups6[name] = [line]

        ax6.axhline(y=90, color="red", linestyle="--", alpha=0.5, label="90° grense")
        ax6.axhline(y=-90, color="red", linestyle="--", alpha=0.5)
        ax6.set_xlabel("Time [s]")
        ax6.set_ylabel("Kraftvinkel [grader]")
        ax6.set_title("Power Angle δ (E vs V_t)")
        ax6.grid(True)
        ax6.legend(fontsize=8)
        add_selector(fig6, ax6, groups6, ["Show All"] + ang_names)

    # ── Generator-battery mapping ─────────────────────────────────
    gen_bat_map = {
        "GEN_1": "UIC_bat_0",
        "GEN_2": "UIC_bat_1",
        "GEN_4": "UIC_bat_2",
        "GEN_6": "UIC_bat_3",
        "GEN_7": "UIC_bat_4",
        "GEN_9": "UIC_bat_3",
    }

    # ── Plot 6: Combined generator + converter power ──────────────
    fig7, ax7 = plt.subplots(figsize=(11, 6))
    plt.subplots_adjust(left=0.28)
    groups7 = {}

    for gen_name in pm_names:
        print(gen_name)

        if gen_name == "IB":
            continue

        if gen_name in gen_bat_map and gen_bat_map[gen_name] in bat_names:
            bat_name = gen_bat_map[gen_name]

            p_combined = res[f"P_e__{gen_name}"] + res[f"P_bat__{bat_name}"]
            p_combined_idx = float(
                res[f"P_e__{gen_name}"][idx] + res[f"P_bat__{bat_name}"][idx]
            )
            p_set_idx = float(res[f"P_set__{gen_name}"][idx])
            x_idx = float(res["t"][idx])

            line1, = ax7.plot(res["t"], p_combined, label=f"{gen_name} + {bat_name}")
            line2, = ax7.plot(
                res["t"],
                res[f"P_set__{gen_name}"],
                label=f"P_set {gen_name}",
                linestyle="--",
            )

            ax7.plot(
                [x_idx, x_idx],
                [p_combined_idx, p_set_idx],
                color="black",
                linestyle=":",
                linewidth=2,
            )

            diff = p_combined_idx - p_set_idx
            y_mid = 0.5 * (p_combined_idx + p_set_idx)

            ax7.annotate(
                rf"$\Delta P = {diff:.3f}$",
                xy=(x_idx, y_mid),
                xytext=(-10, 0),
                textcoords="offset points",
                va="center",
                ha="right",
                fontsize=13,
                color="0.2",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="white",
                    ec="0.75",
                    alpha=0.95,
                ),
            )

            groups7[gen_name] = [line1, line2]

    ax7.set_xlabel("Time [s]")
    ax7.set_ylabel("Power [p.u. on 100 MVA]")
    ax7.set_title("Combined Generator and Converter Power")
    ax7.grid(True)
    ax7.legend(fontsize=13)
    add_selector(fig7, ax7, groups7, ["Show All"] + [n for n in pm_names if n != "IB"])

    plt.show()