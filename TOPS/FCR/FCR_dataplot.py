import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# -----------------------------
# Data
# -----------------------------

# R = 0.02
gen_1_fcrn_2 = [355, 3000, 3000, 3000, 3000, 3000]
gen_1_fcrd_2 = [320, 3000, 3000, 3000, 3000, 3000]
gen_2_fcrn_2 = [3000, 3000, 3000, 3000, 3000, 3000]
gen_2_fcrd_2 = [3360, 3000, 3000, 3000, 3000, 3000]
gen_3_fcrn_2 = [1580, 3000, 3000, 3000, 3000, 3000]
gen_3_fcrd_2 = [1120, 3000, 3000, 3000, 3000, 3000]
gen_4_fcrn_2 = [250, 3000, 3000, 3000, 3000, 3000]
gen_4_fcrd_2 = [225, 3000, 3000, 3000, 3000, 3000]
gen_5_fcrn_2 = [30, 35, 35, 40, 40, 40]
gen_5_fcrd_2 = [20, 25, 465, 3000, 3000, 3000]
gen_6_fcrn_2 = [275, 3000, 3000, 3000, 3000, 3000]
gen_6_fcrd_2 = [250, 3000, 3000, 3000, 3000, 3000]
gen_7_fcrn_2 = [3000, 3000, 3000, 3000, 3000, 3000]
gen_7_fcrd_2 = [2700, 3000, 3000, 3000, 3000, 3000]
gen_8_fcrn_2 = [485, 3000, 3000, 3000, 3000, 3000]
gen_8_fcrd_2 = [435, 3000, 3000, 3000, 3000, 3000]
gen_9_fcrn_2 = [275, 3000, 3000, 3000, 3000, 3000]
gen_9_fcrd_2 = [250, 3000, 3000, 3000, 3000, 3000]

# R = 0.04
gen_1_fcrn_4 = [105, 160, 235, 2910, 3000, 3000]
gen_1_fcrd_4 = [3000, 140, 200, 450, 3000, 3000]
gen_2_fcrn_4 = [990, 2915, 3000, 3000, 3000, 3000]
gen_2_fcrd_4 = [3000, 1490, 3000, 3000, 3000, 3000]
gen_3_fcrn_4 = [370, 600, 3000, 3000, 3000, 3000]
gen_3_fcrd_4 = [3000, 535, 3000, 3000, 3000, 3000]
gen_4_fcrn_4 = [75, 110, 140, 185, 345, 3000]
gen_4_fcrd_4 = [3000, 95, 120, 150, 215, 525]
gen_5_fcrn_4 = [0, 0, 0, 0, 0, 0]
gen_5_fcrd_4 = [5, 5, 5, 5, 5, 10]
gen_6_fcrn_4 = [85, 125, 160, 245, 2115, 3000]
gen_6_fcrd_4 = [3000, 105, 140, 190, 365, 3000]
gen_7_fcrn_4 = [835, 1835, 3000, 3000, 3000, 3000]
gen_7_fcrd_4 = [3000, 1245, 3000, 3000, 3000, 3000]
gen_8_fcrn_4 = [145, 225, 565, 3000, 3000, 3000]
gen_8_fcrd_4 = [3000, 200, 360, 3000, 3000, 3000]
gen_9_fcrn_4 = [85, 125, 160, 245, 1820, 3000]
gen_9_fcrd_4 = [3000, 105, 140, 185, 355, 3000]

# R = 0.06
gen_1_fcrn_6 = [50, 80, 95, 115, 130, 150]
gen_1_fcrd_6 = [3000, 65, 80, 95, 110, 125]
gen_2_fcrn_6 = [505, 785, 1200, 3000, 3000, 3000]
gen_2_fcrd_6 = [3000, 700, 970, 3000, 3000, 3000]
gen_3_fcrn_6 = [190, 290, 385, 3000, 3000, 3000]
gen_3_fcrd_6 = [3000, 3000, 345, 3000, 3000, 3000]
gen_4_fcrn_6 = [35, 50, 60, 65, 70, 80]
gen_4_fcrd_6 = [3000, 35, 45, 50, 55, 60]
gen_5_fcrn_6 = [0, 0, 0, 0, 0, 0]
gen_5_fcrd_6 = [5, 5, 5, 5, 5, 5]
gen_6_fcrn_6 = [40, 55, 70, 80, 85, 95]
gen_6_fcrd_6 = [3000, 45, 55, 60, 70, 75]
gen_7_fcrn_6 = [425, 665, 950, 3000, 3000, 3000]
gen_7_fcrd_6 = [3000, 590, 810, 3000, 3000, 3000]
gen_8_fcrn_6 = [75, 110, 140, 170, 220, 395]
gen_8_fcrd_6 = [3000, 95, 120, 150, 185, 255]
gen_9_fcrn_6 = [40, 55, 70, 80, 85, 95]
gen_9_fcrd_6 = [3000, 45, 55, 60, 70, 75]

# R = 0.08
gen_1_fcrn_8 = [30, 45, 55, 60, 65, 70]
gen_1_fcrd_8 = [3000, 3000, 40, 45, 50, 55]
gen_2_fcrn_8 = [305, 490, 630, 815, 3000, 3000]
gen_2_fcrd_8 = [3000, 3000, 565, 715, 3000, 3000]
gen_3_fcrn_8 = [115, 180, 230, 280, 380, 3000]
gen_3_fcrd_8 = [3000, 3000, 205, 250, 315, 635]
gen_4_fcrn_8 = [15, 20, 20, 20, 20, 20]
gen_4_fcrd_8 = [3000, 10, 10, 10, 10, 5]
gen_5_fcrn_8 = [0, 0, 0, 0, 0, 0]
gen_5_fcrd_8 = [0, 0, 0, 0, 0, 0]
gen_6_fcrn_8 = [20, 25, 30, 35, 35, 40]
gen_6_fcrd_8 = [3000, 3000, 20, 25, 25, 25]
gen_7_fcrn_8 = [260, 415, 535, 670, 3000, 3000]
gen_7_fcrd_8 = [3000, 3000, 475, 595, 3000, 3000]
gen_8_fcrn_8 = [45, 65, 85, 95, 110, 120]
gen_8_fcrd_8 = [3000, 3000, 70, 80, 90, 100]
gen_9_fcrn_8 = [20, 25, 30, 35, 35, 35]
gen_9_fcrd_8 = [3000, 3000, 20, 20, 25, 25]

# R = 0.1
gen_1_fcrn_10 = [15, 25, 30, 30, 35, 35]
gen_1_fcrd_10 = [3000, 3000, 20, 20, 20, 25]
gen_2_fcrn_10 = [210, 340, 453, 525, 625, 3000]
gen_2_fcrd_10 = [3000, 3000, 3000, 470, 560, 1275]
gen_3_fcrn_10 = [75, 125, 160, 190, 215, 250]
gen_3_fcrd_10 = [3000, 3000, 3000, 165, 190, 220]
gen_4_fcrn_10 = [0, 0, 0, 0, 0, 0]
gen_4_fcrd_10 = [3000, 5, 5, 5, 5, 5]
gen_5_fcrn_10 = [0, 0, 0, 0, 0, 0]
gen_5_fcrd_10 = [0, 0, 0, 0, 0, 0]
gen_6_fcrn_10 = [10, 10, 5, 5, 0, 5]
gen_6_fcrd_10 = [3000, 5, 5, 5, 5, 5]
gen_7_fcrn_10 = [175, 285, 370, 440, 520, 1870]
gen_7_fcrd_10 = [3000, 3000, 3000, 395, 480, 740]
gen_8_fcrn_10 = [30, 40, 55, 60, 70, 75]
gen_8_fcrd_10 = [3000, 3000, 3000, 50, 55, 60]
gen_9_fcrn_10 = [5, 10, 5, 5, 5, 5]
gen_9_fcrd_10 = [3000, 5, 5, 5, 5, 5]

# R = 0.12
gen_1_fcrn_12 = [10, 10, 10, 10, 10, 10]
gen_1_fcrd_12 = [3000, 5, 5, 5, 5, 5]
gen_2_fcrn_12 = [150, 250, 325, 385, 445, 510]
gen_2_fcrd_12 = [3000, 3000, 3000, 3000, 400, 460]
gen_3_fcrn_12 = [55, 90, 115, 140, 155, 175]
gen_3_fcrd_12 = [3000, 3000, 3000, 3000, 135, 155]
gen_4_fcrn_12 = [0, 0, 0, 0, 0, 0]
gen_4_fcrd_12 = [3000, 5, 5, 5, 5, 5]
gen_5_fcrn_12 = [0, 0, 0, 0, 0, 0]
gen_5_fcrd_12 = [0, 0, 0, 0, 0, 0]
gen_6_fcrn_12 = [0, 0, 0, 0, 0, 0]
gen_6_fcrd_12 = [3000, 5, 5, 5, 5, 5]
gen_7_fcrn_12 = [125, 210, 270, 325, 375, 425]
gen_7_fcrd_12 = [3000, 3000, 3000, 3000, 355, 385]
gen_8_fcrn_12 = [20, 30, 35, 40, 45, 45]
gen_8_fcrd_12 = [3000, 3000, 3000, 30, 30, 35]
gen_9_fcrn_12 = [0, 0, 0, 0, 0, 0]
gen_9_fcrd_12 = [3000, 5, 5, 5, 5, 5]

x_axis = [1, 0.5, 0.33, 0.25, 0.2, 0.167]


# -----------------------------
# Organize data
# -----------------------------

fcrn_data = {
    1: {0.02: gen_1_fcrn_2, 0.04: gen_1_fcrn_4, 0.06: gen_1_fcrn_6, 0.08: gen_1_fcrn_8, 0.1: gen_1_fcrn_10, 0.12: gen_1_fcrn_12},
    2: {0.02: gen_2_fcrn_2, 0.04: gen_2_fcrn_4, 0.06: gen_2_fcrn_6, 0.08: gen_2_fcrn_8, 0.1: gen_2_fcrn_10, 0.12: gen_2_fcrn_12},
    3: {0.02: gen_3_fcrn_2, 0.04: gen_3_fcrn_4, 0.06: gen_3_fcrn_6, 0.08: gen_3_fcrn_8, 0.1: gen_3_fcrn_10, 0.12: gen_3_fcrn_12},
    4: {0.02: gen_4_fcrn_2, 0.04: gen_4_fcrn_4, 0.06: gen_4_fcrn_6, 0.08: gen_4_fcrn_8, 0.1: gen_4_fcrn_10, 0.12: gen_4_fcrn_12},
    5: {0.02: gen_5_fcrn_2, 0.04: gen_5_fcrn_4, 0.06: gen_5_fcrn_6, 0.08: gen_5_fcrn_8, 0.1: gen_5_fcrn_10, 0.12: gen_5_fcrn_12},
    6: {0.02: gen_6_fcrn_2, 0.04: gen_6_fcrn_4, 0.06: gen_6_fcrn_6, 0.08: gen_6_fcrn_8, 0.1: gen_6_fcrn_10, 0.12: gen_6_fcrn_12},
    7: {0.02: gen_7_fcrn_2, 0.04: gen_7_fcrn_4, 0.06: gen_7_fcrn_6, 0.08: gen_7_fcrn_8, 0.1: gen_7_fcrn_10, 0.12: gen_7_fcrn_12},
    8: {0.02: gen_8_fcrn_2, 0.04: gen_8_fcrn_4, 0.06: gen_8_fcrn_6, 0.08: gen_8_fcrn_8, 0.1: gen_8_fcrn_10, 0.12: gen_8_fcrn_12},
    9: {0.02: gen_9_fcrn_2, 0.04: gen_9_fcrn_4, 0.06: gen_9_fcrn_6, 0.08: gen_9_fcrn_8, 0.1: gen_9_fcrn_10, 0.12: gen_9_fcrn_12},
}

fcrd_data = {
    1: {0.02: gen_1_fcrd_2, 0.04: gen_1_fcrd_4, 0.06: gen_1_fcrd_6, 0.08: gen_1_fcrd_8, 0.1: gen_1_fcrd_10, 0.12: gen_1_fcrd_12},
    2: {0.02: gen_2_fcrd_2, 0.04: gen_2_fcrd_4, 0.06: gen_2_fcrd_6, 0.08: gen_2_fcrd_8, 0.1: gen_2_fcrd_10, 0.12: gen_2_fcrd_12},
    3: {0.02: gen_3_fcrd_2, 0.04: gen_3_fcrd_4, 0.06: gen_3_fcrd_6, 0.08: gen_3_fcrd_8, 0.1: gen_3_fcrd_10, 0.12: gen_3_fcrd_12},
    4: {0.02: gen_4_fcrd_2, 0.04: gen_4_fcrd_4, 0.06: gen_4_fcrd_6, 0.08: gen_4_fcrd_8, 0.1: gen_4_fcrd_10, 0.12: gen_4_fcrd_12},
    5: {0.02: gen_5_fcrd_2, 0.04: gen_5_fcrd_4, 0.06: gen_5_fcrd_6, 0.08: gen_5_fcrd_8, 0.1: gen_5_fcrd_10, 0.12: gen_5_fcrd_12},
    6: {0.02: gen_6_fcrd_2, 0.04: gen_6_fcrd_4, 0.06: gen_6_fcrd_6, 0.08: gen_6_fcrd_8, 0.1: gen_6_fcrd_10, 0.12: gen_6_fcrd_12},
    7: {0.02: gen_7_fcrd_2, 0.04: gen_7_fcrd_4, 0.06: gen_7_fcrd_6, 0.08: gen_7_fcrd_8, 0.1: gen_7_fcrd_10, 0.12: gen_7_fcrd_12},
    8: {0.02: gen_8_fcrd_2, 0.04: gen_8_fcrd_4, 0.06: gen_8_fcrd_6, 0.08: gen_8_fcrd_8, 0.1: gen_8_fcrd_10, 0.12: gen_8_fcrd_12},
    9: {0.02: gen_9_fcrd_2, 0.04: gen_9_fcrd_4, 0.06: gen_9_fcrd_6, 0.08: gen_9_fcrd_8, 0.1: gen_9_fcrd_10, 0.12: gen_9_fcrd_12},
}

def plot_generator(generator_number, data_dict, title_prefix):
    plt.figure(figsize=(11, 6))

    used_x_values = set()

    for R_value, y_values in data_dict[generator_number].items():
        x_values = x_axis[:len(y_values)]
        y_plot = np.array(y_values, dtype=float)
        y_plot[y_plot >= 300] = np.nan

        valid_mask = ~np.isnan(y_plot)
        x_valid = np.array(x_values)[valid_mask]
        y_valid = y_plot[valid_mask]

        if len(x_valid) > 0:
            used_x_values.update(x_valid.tolist())
            plt.plot(x_valid, y_valid, marker='o', linewidth=2, label=f'R = {R_value}')


    used_x_values = sorted(used_x_values)
    plt.xticks(used_x_values, rotation=30, ha='right')
    plt.tick_params(axis='x', pad=0.5, labelsize = 20)
    plt.tick_params(axis='y', pad=0.5, labelsize = 20)


    plt.xlabel('VSC frequency bias [p.u/Hz]', fontsize = 25)
    plt.ylabel('Min Tf to pass', fontsize = 25)
    plt.title(f'Generator {generator_number} - {title_prefix}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

for gen in range(1, 10):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 20
    plot_generator(gen, fcrn_data, 'FCR-N')
    plot_generator(gen, fcrd_data, 'FCR-D')