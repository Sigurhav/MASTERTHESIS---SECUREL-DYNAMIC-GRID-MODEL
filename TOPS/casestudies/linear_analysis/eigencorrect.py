import src.dynamic as dps
import src.linear_ps as dps_mdl
import src.solvers as dps_sol
from eigenanalysis import analyze_eigenvalues
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the model
import casestudies.ps_data.Sixtyfiveimport as model_data
import importlib

importlib.reload(model_data)
model = model_data.load()
ps = dps.PowerSystemModel(model=model)
ps.init_dyn_sim()

t_end = 100

# 2. Run dynamic sim
sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, ps.x_0.copy(), t_end, max_step=5e-3)
t = 0
event_flag = True
t_event = 10.1

while t < t_end:

    if 10 < t < 10.01:
        ps.y_bus_red_mod[0, 0] = 1e6
    else:
        ps.y_bus_red_mod[0, 0] = 0

        # Event: disconnect line L1 at t = t_event
    if t > t_event + 0.01 and event_flag:
            event_flag = False
            ps.lines['Line'].event(ps, ps.lines['Line'].par['name'][0], 'disconnect')
    sol.step()
    x = sol.y
    t = sol.t

# 3. Retrive last state variables
x_island = sol.y.copy()
v_island = sol.v.copy()


# 5. Lineraize around island operating point.
ps_lin = dps_mdl.PowerSystemModelLinearization(ps)

ps_lin.linearize(x0=x_island)
ps_lin.eigenvalue_decomposition()


# 6. Analyze eigenvalues
results, fig = analyze_eigenvalues(ps_lin, ps)
plt.show()