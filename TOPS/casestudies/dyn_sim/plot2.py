import sys
from collections import defaultdict
import importlib
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.widgets import RadioButtons

import src.dynamic as dps
import src.solvers as dps_sol
from utility_functions import print_init_info
import src.linear_ps as dps_mdl

importlib.reload(dps)


if __name__ == "__main__":
    caselist = ["exportcase", "bias05", "bias2", "bias5", "bias10", "bias15"]
    all_cases = {}

    for case in caselist:
        print(case)

        model_data = importlib.import_module(f"casestudies.ps_data.{case}")
        importlib.reload(model_data)
        model = model_data.load()

        ps = dps.PowerSystemModel(model=model)
        ps.init_dyn_sim()

        # ── Simulation parameters ────────────────────────────────
        t_end = 10
        t_event = 5
        x_0 = ps.x_0.copy()

        sol = dps_sol.ModifiedEulerDAE(
            ps.state_derivatives,
            ps.solve_algebraic,
            0,
            x_0,
            t_end,
            max_step=5e-3,
        )

        # ── Discover model keys ──────────────────────────────────
        gen_keys = list(ps.gen.keys())
        battery_keys = list(ps.vsc.keys()) if hasattr(ps, "vsc") else []

        # ── Run simulation ────────────────────────────────────────
        t = 0
        res = defaultdict(list)
        t_0 = time.time()
        event_flag = True

        while t < t_end:
            sys.stdout.write(f"\r{t / t_end * 100:.0f}%")

            if t_event < t < t_event + 0.01:
                ps.y_bus_red_mod[1, 1] = 1e6
            else:
                ps.y_bus_red_mod[1, 1] = 0

            if t > t_event + 0.01 and event_flag:
                event_flag = False
                ps.lines["Line"].event(ps, ps.lines["Line"].par["name"][0], "disconnect")

            sol.step()
            x = sol.y
            v = sol.v
            t = sol.t

            res["t"].append(t)

            # ── Generators ───────────────────────────────────────
            for gk in gen_keys:
                g = ps.gen[gk]
                scale = g.par["S_n"] / ps.s_n

                H = g.par["H"]
                speed = np.atleast_1d(g.speed(x, v).copy())
                vt_cx = np.atleast_1d(g.v_t(x, v).copy())
                vt = np.atleast_1d(g.v_t_abs(x, v).copy())
                p_nom = np.atleast_1d(g.P_nom(x, v).copy()) / ps.s_n
                p_m = np.atleast_1d(g.P_m(x, v).copy()) * scale

                for i in range(len(speed)):
                    tag = gk if len(speed) == 1 else f"{gk}_{i}"
                    res[f"speed__{tag}"].append(speed[i])
                    res[f"Vt_ang__{tag}"].append(np.angle(vt_cx[i]))
                    res[f"P_nom__{tag}"].append(p_nom[i])
                    res[f"H__{tag}"].append(H[i])
                    res[f"P_m__{tag}"].append(p_m[i])

            # ── Batteries ───────────────────────────────────────
            for bk in battery_keys:
                bat = ps.vsc[bk]
                scale = bat.par["S_n"] / ps.s_n
                p = np.atleast_1d(bat.p_e(x, v).copy()) * scale

                for i in range(len(p)):
                    tag = bk if len(p) == 1 else f"{bk}_{i}"
                    res[f"P_bat__{tag}"].append(p[i])

        print(f"\nSimulation completed in {time.time() - t_0:.2f} seconds.")

        # ── Convert to numpy arrays ─────────────────────────────
        for k in res:
            res[k] = np.array(res[k])

        # ── Frequency metrics ───────────────────────────────────
        settle_tol_pct = 0.02
        t_arr = res["t"]
        speed_names = [k.replace("speed__", "") for k in res if k.startswith("speed__")]
        bat_names = [k.replace("P_bat__", "") for k in res if k.startswith("P_bat__")]

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

        pm_selected = ["G4", "G5"]

        all_cases[case] = {
            "t": res["t"].copy(),
            "fcoi": fcoi.copy(),
            "f_nadir": f_nadir,
            "t_nadir": t_nadir,
            "t_nadir_abs": t_nadir_abs,
            "max_rocof": max_rocof,
            "t_max_rocof": t_max_rocof,
            "f_at_rocof": f_at_rocof,
            "P_bat": {name: res[f"P_bat__{name}"].copy() for name in bat_names},
            "Pm": {name: res[f"P_m__{name}"].copy() for name in pm_selected if f"P_m__{name}" in res},
        }

    rcParams["font.family"] = "Times New Roman"
    rcParams["font.size"] = 15

    fig, ax = plt.subplots(figsize=(11, 6))

    y_levels = [57.8, 56.5, 55.2, 53.9, 52.6, 51.3]

    custom_labels = {
        "exportcase": "No batteries",
        "bias05": "Bias = 0.5 MW/Hz",
        "bias2": "Bias = 2 MW/Hz",
        "bias5": "Bias = 5 MW/Hz",
        "bias10": "Bias = 10 MW/Hz",
        "bias15": "Bias = 15 MW/Hz",
    }

    plot_order = ["bias15", "bias10", "bias5", "bias2", "bias05", "exportcase"]

    case_colors = {
        "bias15": "C0",
        "bias10": "C1",
        "bias5": "C2",
        "bias2": "C3",
        "bias05": "C4",
        "exportcase": "C5",
    }

    for case_name in plot_order:
        if case_name not in all_cases:
            continue

        data = all_cases[case_name]
        legend_label = custom_labels.get(case_name, case_name)

        if "P_bat" in data and len(data["P_bat"]) > 0:
            p_sum = np.sum(np.vstack(list(data["P_bat"].values())), axis=0)
            ax.plot(
                data["t"],
                p_sum,
                label=legend_label,
                linewidth=2,
                color=case_colors[case_name],
            )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_title("COI frequency for all cases")
    ax.grid(True)
    ax.legend()
    plt.show()