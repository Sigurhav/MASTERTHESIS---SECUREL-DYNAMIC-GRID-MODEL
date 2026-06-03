import control as ct
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np




At = 1.1 # Turbine gain

T1_list = [0.8, 4.29, 2.57, 1.69, 0.99, 0.8, 4.29, 0.8, 0.8]
T2_list = [0.43, 2.35, 1.39, 0.93, 0.54, 0.43, 2.35, 0.43, 0.43]
Kp_list = [1.064, 0.422, 0.717, 1.855, 3.336, 1.327, 0.499, 0.796, 1.327]
Ti_list = [7.54, 26.04, 16.73, 9.7, 4.5, 7.54, 26.06, 7.52, 7.52]



for i in range(len(Ti_list)):
    T1 = T1_list[i]  # (q0-qnl)*Tw

    T2 = T2_list[i]  # q0*Tw/2

    s = ct.tf('s')
    Gt = At * (1 - s * T1) / (1 + s * T2)

    Tg = 0.2  # Servo time constant
    Gs = ct.tf([1], [Tg, 1])

    Kp = Kp_list[i]
    Ti = Ti_list[i]  # We set this quite high so that we don't manage the performance requirement. Try to make it smaller later to see that it will let the plant pass.
    Td = 0
    Tf = 0.02
    R = 0.04

    Gc = Kp * (Ti * s + 1) * (1 + s * Td) / ((1 + s * Tf) * (Ti * s + R * Kp))

    Gh = Gc * Gs * Gt  # Plant transfer function

    T = 235


    K = 0.04


    Gvsc = 1 / K * (T * s / (T * s + 1))
    Gp = Gh + Gvsc

    t = np.linspace(0, 500, 1000)
    df = np.zeros_like(t)
    df[t >= 1] = 0.002
    t, y = ct.forced_response(Gvsc, T=t, U =df)
    t, y2 = ct.forced_response(Gh, T=t, U=df)
    t, y3 = ct.forced_response(Gp, T=t, U=df)

    fig, ax1 = plt.subplots()
    #E = np.trapezoid(y, t)
    #Assuming S=100 MVA and that converter is large enough to achive full response
    # #E_MWh = (E)/3600
    # print('Required energy is:',E_MWh, 'MWh')
    # size = max(y)
    # size_mw = (size)*95*8
    # print('Required vsc is:',size_mw, 'MW')
    rcParams["font.family"] = "Times New Roman"  # or "Times New Roman", etc.
    rcParams["font.size"] = 15
    plt.title("Step response HHPP")
    ax1.plot(t, y, label='Power response VSC', color='tab:red')
    ax1.plot(t, y2, label='Power response generator 5', color='tab:green')
    ax1.plot(t, y3, label='Combined response', color='tab:blue')
    #ax1.plot(t, df, '--', label='ramp input', color='tab:blue')
    ax1.set_ylabel('Injected Power [pu]')
    ax1.set_xlabel('Time [s]')
    # ax1.tick_params(axis='y', labelcolor='tab:red')
    # baseline = 0
    # plt.fill_between(t, y, baseline, where=(y >= baseline), alpha=0.3, color='green')
    # plt.annotate(
    #     f'Injected Energy = {E_MWh:.4f} p.u h',
    #     xy=(t[len(t) // 2], y.max() / 2),
    #     xytext=(0, 0),
    #     textcoords='offset points',
    #     bbox=dict(boxstyle='round', fc='white', alpha=0.9),
    #     #arrowprops=dict(arrowstyle='->')
    # )
    #ax1.axhline(y=0, color='black', linewidth=1.5)
    plt.grid(True)
    plt.legend()
    plt.show()


    def inertia_tf(K, H, Kd):
        return ct.tf([K], [2 * H, Kd])


    st_gain = abs(ct.dcgain(Gp))
    k_margin = 0.95  # Margin defined in requirements
    scale = k_margin / st_gain  # Scale our transfer function with this value to convert it to system base
    # System inertia transfer function for the stability requirement of FCR-N
    Gns = inertia_tf(50 / 23e3 * 600 / 0.1, 2 * 5.2174, 0.01 * 50) * scale

    # System inertia transfer function for the performance requirement for FCR-N
    Gnp = inertia_tf(50 / 42e3 * 600 / 0.1, 2 * 4.5238, 0.01 * 50)
    D = 1 / (ct.tf([1], [70.0, 1]) * Gnp)  # Reformulated



    ct.nyquist_plot(Gp * Gns, label='No VSC', mt_circles=[2.3], title =("Nyquist for Generator", (i+1)) )

    plt.gca().set_title("")  # remove the automatically added title
    plt.show()

    # mag, _, omega = ct.frequency_response(Gp)
    # mag_p, _, omega = ct.frequency_response(Gvsc)

    mag, _, omega = ct.frequency_response(1 / (1 + Gns * Gp))

    magp, _, omega = ct.frequency_response(1 / (1 + Gnp * scale * Gp))
    mag_p, _, omega = ct.frequency_response(1 / (1 + Gnp * scale * Gp))
    mag_d, _, _ = ct.frequency_response(D, omega)
    plt.loglog(omega, mag)

    plt.loglog(omega, mag_p)
    plt.loglog(omega, mag_d, color='black', linestyle='--')

    plt.grid(True, which="both", ls="-", color='0.65')
    plt.axhline(y=2.3, color='r', linestyle='--', linewidth=2)
    plt.ylim(0.04, 3)
    # plt.legend(["No VSC", "Ki=0.02", "Ki=0.04", "Ki = 0.06", 'Ki = 0.1', 'Ki = 0.2', "Disturbance requirement curve"])
    plt.xlabel("Frequency (rad/s)")
    plt.ylabel("Magnitude")
    plt.legend(['Stability requirement curve', "Performance requirement curve", 'Stability requirement curve'])
    plt.title(("Bode plot for Generator", i+1))
    plt.show()

    # "Performance requirement curve" 'Stability requirement curve'