import control as ct
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy.integrate import cumulative_trapezoid

At = 1.1  # Turbine gain

T1_list = [0.8, 4.29, 2.57, 1.69, 0.99, 0.8, 4.29, 0.8, 0.8]
T2_list = [0.43, 2.35, 1.39, 0.93, 0.54, 0.43, 2.35, 0.43, 0.43]
Kp_list = [1.064, 0.422, 0.717, 1.855, 3.336, 1.327, 0.499, 0.796, 1.327]
Ti_list = [7.54, 26.04, 16.73, 9.7, 4.5, 7.54, 26.06, 7.52, 7.52]


def inertia_tf(K, H, Kd):
    return ct.tf([K], [2 * H, Kd])


rcParams["font.family"] = "Times New Roman"
rcParams["font.size"] = 15

s = ct.tf("s")

for n in range(len(Ti_list)):
    print(f"Now checking generator {n + 1}")

    T1 = T1_list[n]
    T2 = T2_list[n]
    Kp = Kp_list[n]
    Ti = Ti_list[n]

    Gt = At * (1 - s * T1) / (1 + s * T2)
    Tg = 0.2
    Gs = ct.tf([1], [Tg, 1])

    Td = 0
    Tf = 0.02
    R = 0.02

    Gc = Kp * (Ti * s + 1) * (1 + s * Td) / ((1 + s * Tf) * (Ti * s + R * Kp))
    Gh = Gc * Gs * Gt

    T = 100
    K = 0.12
    Gvsc = (1 / K) * (T * s / (T * s + 1))

    Gp = Gh + Gvsc

    st_gain = abs(ct.dcgain(Gp))
    k_margin = 0.95
    scale = k_margin / st_gain

    Gns = inertia_tf(50 / 23e3 * 1450 / 0.4, 2 * 5.2174, 0.01 * 50) * scale
    Gnp = inertia_tf(50 / 42e3 * 1450 / 0.4, 2 * 4.5238, 0.01 * 50)

    D = 1 / (ct.tf([1], [70.0, 1]) * Gnp)

    # ── Nyquist plot ─────────────────────────────────────────────
    ct.nyquist_plot(
        Gp * Gns,
        label="stability",
        mt_circles=[2.3],
        title=f"Nyquist plot for Generator {n + 1}",
    )
    plt.gca().set_title("")
    plt.show()

    # ── Frequency response / requirement curves ──────────────────
    mag_stab, _, omega = ct.frequency_response(1 / (1 + Gns * Gp))
    mag_perf, _, _ = ct.frequency_response(1 / (1 + Gnp * scale * Gp))
    mag_d, _, _ = ct.frequency_response(D, omega)

    plt.figure()
    plt.loglog(omega, mag_stab, label="Stability requirement curve")
    plt.loglog(omega, mag_perf, label="Performance requirement curve")
    plt.loglog(omega, mag_d, "--", color="black", label="Disturbance requirement curve")
    plt.grid(True, which="both", ls="-", color="0.65")
    plt.axhline(y=2.3, color="r", linestyle="--", linewidth=2)
    plt.ylim(0.04, 3)
    plt.xlabel("Frequency (rad/s)")
    plt.ylabel("Magnitude")
    plt.title(f"Bode plot for Generator {n + 1}")
    plt.legend()
    plt.show()

    # ── Requirement 2 + 3 ────────────────────────────────────────
    t = np.linspace(0, 10, 1000)
    m = -0.0048
    u5 = m * np.minimum(t, 3.8)

    t_out, y = ct.forced_response(Gp, T=t, U=-u5)

    lim = (t_out >= 0) & (t_out <= 7.5)
    P_75 = y[lim][-1]
    P_th = 0.008 * 1 / R
    E = np.trapezoid(y[lim], t_out[lim])

    if P_75 >= 0.86 * P_th:
        print(f"Requirement 2 is passed since {P_75} p.u is larger than {0.86 * P_th} p.u")
    else:
        print(f"Requirement 2 is failed since {P_75} p.u is smaller than {0.86 * P_th} p.u")

    if E >= 3.2 * P_th:
        print(f"Requirement 3 is passed since {E} p.u is larger than {3.2 * P_th} p.u")
    else:
        print(f"Requirement 3 is failed since {E} p.u is smaller than {3.2 * P_th} p.u")

    # ── Requirement 4 ────────────────────────────────────────────
    t2 = np.linspace(0, 50, 1000)
    m1 = -0.0028
    m2 = 0.0018

    u = m1 * np.minimum(t2, 3.1) + m2 * np.maximum(np.minimum(t2, 9.9) - 4.9, 0)

    t12, y2 = ct.forced_response(Gp, T=t2, U=-u)

    lim = (t12 >= 4.4) & (t12 <= 44.4)
    Pnadir = y2[lim][0]

    baseline = min(Pnadir, 0.5 * P_th)
    vals = np.maximum(y2[lim] - baseline, 0)
    cum_energy = cumulative_trapezoid(vals, t12[lim], initial=0.0)

    success = True
    for i in cum_energy:
        if i >= 2.5 * P_th:
            success = False
            break

    if success:
        print(
            f"Requirement 4 was passed as accumulated energy {cum_energy[-1]} "
            f"was below threshold of {2.5 * P_th}"
        )
    else:
        print(
            f"Requirement 4 failed. Accumulated energy {cum_energy[-1]} "
            f"is larger than {2.5 * P_th}"
        )

    # ── Plot for requirement 2 + 3 ───────────────────────────────
    fig, ax1 = plt.subplots()

    ax1.plot(t, u5, "--", label="ramp input", color="tab:blue")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Input u(t)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(-0.02, 0.004)
    ax1.axvline(x=7.5, color="red", linestyle=":", linewidth=1.5, label="t = 7.5 s")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(t_out, y, label="Power response", color="tab:red")
    ax2.set_ylabel("Output y(t)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.axhline(y=0, color="black", linewidth=1.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title(f"Plot for Generator {n + 1}")
    plt.show()

    # ── Plot for requirement 4 ───────────────────────────────────
    fig2, ax1 = plt.subplots()

    ax1.plot(t12, u, "--", label="ramp input", color="tab:blue")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Input u(t)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(-0.01, 0.008)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(t12, y2, label="system output", color="tab:red")
    ax2.set_ylabel("Power response", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.axvline(x=4.4, label="t = 4.4 s", color="purple", linestyle="--", linewidth=1.5)
    ax2.axhline(y=Pnadir, label="NADIR power", color="purple", linestyle="--", linewidth=1.5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title(f"Plot for Generator {n + 1}")
    plt.show()

    fig3, ax1 = plt.subplots()
    ax1.grid(True)
    ax1.plot(t12[lim], cum_energy, label="Cumulative energy", color="tab:red")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Cumulative energy [p.u]", color="tab:red")
    ax1.axhline(y=2.5 * P_th, label="Energy limit", color="purple", linestyle="--", linewidth=1.5)
    plt.title(f"Plot for Generator {n + 1}")
    plt.show()