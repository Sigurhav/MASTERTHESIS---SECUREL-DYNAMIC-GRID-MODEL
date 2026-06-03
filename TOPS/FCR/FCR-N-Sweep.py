import control as ct
import numpy as np
from scipy.integrate import cumulative_trapezoid

At = 1.1  # Turbine gain

# Average case
T1_list = [0.8, 4.29, 2.57, 1.69, 0.99, 0.8, 4.29, 0.8, 0.8]
T2_list = [0.43, 2.35, 1.39, 0.93, 0.54, 0.43, 2.35, 0.43, 0.43]
Kp_list = [1.064, 0.422, 0.717, 1.855, 3.336, 1.327, 0.499, 0.796, 1.327]
Ti_list = [7.54, 26.04, 16.73, 9.7, 4.5, 7.54, 26.06, 7.52, 7.52]
count_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def inertia_tf(K, H, Kd):
    return ct.tf([K], [2 * H, Kd])


s = ct.tf("s")

for i in range(len(Ti_list)):
    T1 = T1_list[i]
    T2 = T2_list[i]

    Gt = At * (1 - s * T1) / (1 + s * T2)
    Tg = 0.2
    Gs = ct.tf([1], [Tg, 1])

    Kp = Kp_list[i]
    Ti = Ti_list[i]
    Td = 0
    Tf = 0.02
    R = 0.08

    Gc = Kp * (Ti * s + 1) * (1 + s * Td) / ((1 + s * Tf) * (Ti * s + R * Kp))
    Gh = Gc * Gs * Gt

    passed = False
    T = 5
    K = 0.12

    while not passed:
        Gvsc = (1 / K) * (T * s / (T * s + 1))
        Gp = Gh + Gvsc

        st_gain = abs(ct.dcgain(Gp))
        k_margin = 0.95
        scale = k_margin / st_gain

        Gns = inertia_tf(50 / 23e3 * 600 / 0.1, 2 * 5.2174, 0.01 * 50) * scale
        Gnp = inertia_tf(50 / 42e3 * 600 / 0.1, 2 * 4.5238, 0.01 * 50)
        D = 1 / (ct.tf([1], [70.0, 1]) * Gnp)

        mag_stab, _, omega = ct.frequency_response(1 / (1 + Gns * Gp))
        mag_perf, _, _ = ct.frequency_response(1 / (1 + Gnp * scale * Gp))
        mag_dist, _, _ = ct.frequency_response(D, omega)

        perf = np.all(mag_perf < mag_dist)
        stab = np.max(mag_stab) < 2.3

        # ── Requirement 2 + 3 ─────────────────────────────────────
        t = np.linspace(0, 10, 1000)
        m = -0.0048
        u5 = m * np.minimum(t, 3.8)

        t_resp, y = ct.forced_response(Gp, T=t, U=-u5)

        lim = (t_resp >= 0) & (t_resp <= 7.5)
        P_75 = y[lim][-1]
        P_th = 0.008 * 1 / R
        E = np.trapezoid(y[lim], t_resp[lim])

        req2 = P_75 >= 0.86 * P_th
        req3 = E >= 3.2 * P_th

        # ── Requirement 4 ─────────────────────────────────────────
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

        success = np.all(cum_energy < 2.5 * P_th)

        if stab and perf and success and req2 and req3:
            print(f"success for Tf = {T} for generator {count_list[i]}")
            passed = True

        T += 5

        if T > 300:
            print(f"Not possible for generator {count_list[i]}")
            break