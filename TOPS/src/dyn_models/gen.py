import numpy as np
from src.dyn_models.utils import DAEModel
import src.utility_functions as dps_uf


class GEN(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for req_attr, default in zip(['PF_n', 'N_par', 'R', 'X_l'], [1, 1, 0, 0]):
            if req_attr not in self.par.dtype.names:
                new_field = np.ones(len(self.par), dtype=[(req_attr, float)])
                new_field[req_attr] *= default
                self.par = dps_uf.combine_recarrays(self.par, new_field)

        gen_bus_idx = dps_uf.lookup_strings(self.par['bus'], self.sys_par['bus_names'])

        fix_idx = self.par['V_n'] == 0
        self.par['V_n'][fix_idx] = self.sys_par['bus_v_n'][gen_bus_idx][fix_idx]

        fix_idx = self.par['S_n'] == 0
        self.par['S_n'][fix_idx] = self.sys_par['s_n']

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

        self.tripped = np.zeros(self.n_units, dtype=bool)

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def load_flow_pv(self):
        return self.bus_idx['terminal'], -self.par['P'] * self.par['N_par'], self.par['V']

    def init_from_load_flow(self, x_0, v_0, S):
        X_0 = self.local_view(x_0)
        p = self.par

        fix_idx = p['V_n'] == 0
        p['V_n'][fix_idx] = self.sys_par['bus_v_n'][self.bus_idx['terminal']][fix_idx]

        fix_idx = p['S_n'] == 0
        p['S_n'][fix_idx] = self.sys_par['s_n']

        s_pu = S / p['S_n'] / p['N_par']
        v_g = v_0[self.bus_idx['terminal']]
        I_g = np.conj(s_pu / v_g)

        e_q_tmp = v_g + 1j * p['X_q'] * I_g
        angle = np.angle(e_q_tmp)

        d = np.exp(1j * (angle - np.pi / 2))
        q = np.exp(1j * angle)

        I_g_dq = I_g * np.exp(1j * (np.pi / 2 - angle))
        I_d = I_g_dq.real
        I_q = I_g_dq.imag

        v_g_dq = v_g * np.exp(1j * (np.pi / 2 - angle))
        v_d = v_g_dq.real
        v_q = v_g_dq.imag

        e_q_t = v_q + p['X_d_t'] * I_d
        e_d_t = v_d - p['X_q_t'] * I_q

        e_q_st = v_q + p['X_d_st'] * I_d
        e_d_st = v_d - p['X_q_st'] * I_q

        e_q = e_q_t + I_d * (p['X_d'] - p['X_d_t'])
        e_q_0 = e_q.copy()

        PF_n = p['PF_n'] if 'PF_n' in p.dtype.names else 1
        self._input_values['P_m'] = s_pu.real / PF_n
        self._input_values['E_f'] = e_q_0

        X_0['speed'][:] = np.zeros_like(angle)
        X_0['angle'][:] = angle
        X_0['e_q_t'][:] = e_q_t
        X_0['e_d_t'][:] = e_d_t
        X_0['e_q_st'][:] = e_q_st
        X_0['e_d_st'][:] = e_d_st

    def dyn_const_adm(self):
        idx_bus = self.bus_idx['terminal']
        bus_v_n = self.sys_par['bus_v_n'][idx_bus]
        z_n = bus_v_n ** 2 / self.sys_par['s_n']
        impedance = 1j * self.par['X_d_st'] * self.par['V_n'] ** 2 / self.par['S_n'] / z_n
        Y = self.par['N_par'] / impedance
        return Y, (idx_bus,) * 2

    def state_list(self):
        return ['speed', 'angle', 'e_q_t', 'e_d_t', 'e_q_st', 'e_d_st']

    def input_list(self):
        return ['V_t_abs', 'V_t_angle', 'P_m', 'E_f', 'v_aux', 'v_pss', 'f_n']

    def int_par_list(self):
        return ['f']

    def reduced_system(self):
        return self.par['bus']

    def current_injections(self, x, v):
        p = self.par
        X = self.local_view(x)

        i_inj = (X['e_q_st'] / (1j * p['X_d_st']) * self.q(x, v)
                 + X['e_d_st'] / (1j * p['X_q_st']) * self.d(x, v)) * p['N_par']

        I_n = p['S_n'] / (np.sqrt(3) * p['V_n'])
        i_n = self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'])
        I_inj = i_inj * I_n / i_n[self.bus_idx_red['terminal']]

        return self.bus_idx_red['terminal'], I_inj

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        p = self.par

        PF_n = p['PF_n'] if 'PF_n' in p.dtype.names else 1
        H = p['H'] / PF_n
        dTau = (self.P_m(x, v) - self.p_e(x, v) / PF_n) / (1 + X['speed'])


        dX['speed'][:] = 1 / (2 * H) * (dTau - p['D'] * X['speed'])
        dX['angle'][:] = X['speed'] * 2 * np.pi * p['f_n']
        dX['e_q_t'][:] = 1 / p['T_d0_t'] * (self.E_f(x, v) + self.v_aux(x, v) - X['e_q_t'] - self.i_d(x, v) * (p['X_d'] - p['X_d_t']))
        dX['e_d_t'][:] = 1 / p['T_q0_t'] * (-X['e_d_t'] + self.i_q(x, v) * (p['X_q'] - p['X_q_t']))
        dX['e_q_st'][:] = 1 / p['T_d0_st'] * (X['e_q_t'] - X['e_q_st'] - self.i_d(x, v) * (p['X_d_t'] - p['X_d_st']))
        dX['e_d_st'][:] = 1 / p['T_q0_st'] * (X['e_d_t'] - X['e_d_st'] + self.i_q(x, v) * (p['X_q_t'] - p['X_q_st']))



    # --- Rotor reference frames ---

    def d(self, x, v):
        return np.exp(1j * (self.local_view(x)['angle'] - np.pi / 2))

    def q(self, x, v):
        return np.exp(1j * self.local_view(x)['angle'])

    # --- Voltage ---

    def v_t(self, x, v):
        return v[self.bus_idx_red['terminal']]

    def v_t_abs(self, x, v):
        return np.abs(self.v_t(x, v))

    def v_setp(self, x, v):
        return self.par['V']

    # --- Internal states ---

    def angle(self, x, v):
        return self.local_view(x)['angle']

    def speed(self, x, v):
        return self.local_view(x)['speed']

    def e_q_t(self, x, v):
        return self.local_view(x)['e_q_t']

    def e_d_t(self, x, v):
        return self.local_view(x)['e_d_t']

    def e_q_st(self, x, v):
        return self.local_view(x)['e_q_st']

    def e_d_st(self, x, v):
        return self.local_view(x)['e_d_st']

    def e_st(self, x, v):
        return self.e_q_st(x, v) * self.q(x, v) + self.e_d_st(x, v) * self.d(x, v)

    def e_t(self, x, v):
        return self.e_q_t(x, v) * self.q(x, v) + self.e_d_t(x, v) * self.d(x, v)

    # --- Current ---

    def i(self, x, v):
        return (self.e_st(x, v) - self.v_t(x, v)) / (1j * self.par['X_d_st'])

    def i_d(self, x, v):
        return (self.i(x, v) * np.exp(1j * (np.pi / 2 - self.angle(x, v)))).real

    def i_q(self, x, v):
        return (self.i(x, v) * np.exp(1j * (np.pi / 2 - self.angle(x, v)))).imag

    # --- Power (p.u. on generator base) ---

    def s_e(self, x, v):
        return self.v_t(x, v) * np.conj(self.i(x, v))

    def p_e(self, x, v):
        return self.s_e(x, v).real

    def q_e(self, x, v):
        return self.s_e(x, v).imag

    # --- Power (physical units) ---

    def S_e(self, x, v):
        return self.s_e(x, v) * self.par['S_n']

    def P_e(self, x, v):
        return self.p_e(x, v) * self.par['S_n']

    def Q_e(self, x, v):
        return self.q_e(x, v) * self.par['S_n']

    def P_nom(self, x, v):
        PF_n = self.par['cosphi_n'] if 'cosphi_n' in self.par.dtype.names else 1
        return self.par['S_n'] * PF_n

    def P_set(self, x, v):
        return self.par['P']


# -------------------------------------------------------------------------------------- #


class IB(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for req_attr, default in zip(['PF_n', 'N_par', 'R', 'X_l'], [1, 1, 0, 0]):
            if req_attr not in self.par.dtype.names:
                new_field = np.ones(len(self.par), dtype=[(req_attr, float)])
                new_field[req_attr] *= default
                self.par = dps_uf.combine_recarrays(self.par, new_field)

        gen_bus_idx = dps_uf.lookup_strings(self.par['bus'], self.sys_par['bus_names'])

        fix_idx = self.par['V_n'] == 0
        self.par['V_n'][fix_idx] = self.sys_par['bus_v_n'][gen_bus_idx][fix_idx]

        fix_idx = self.par['S_n'] == 0
        self.par['S_n'][fix_idx] = self.sys_par['s_n']

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    def speedchange(self, x, f):
        self.local_view(x)['speed'] = f


    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def load_flow_pv(self):
        return self.bus_idx['terminal'], -self.par['P'] * self.par['N_par'], self.par['V']

    def init_from_load_flow(self, x_0, v_0, S):
        X_0 = self.local_view(x_0)
        p = self.par

        fix_idx = p['V_n'] == 0
        p['V_n'][fix_idx] = self.sys_par['bus_v_n'][self.bus_idx['terminal']][fix_idx]

        fix_idx = p['S_n'] == 0
        p['S_n'][fix_idx] = self.sys_par['s_n']

        s_pu = S / p['S_n'] / p['N_par']
        v_g = v_0[self.bus_idx['terminal']]
        I_g = np.conj(s_pu / v_g)

        e_q_tmp = v_g + 1j * p['X_q'] * I_g
        angle = np.angle(e_q_tmp)

        I_g_dq = I_g * np.exp(1j * (np.pi / 2 - angle))
        I_d = I_g_dq.real
        I_q = I_g_dq.imag

        v_g_dq = v_g * np.exp(1j * (np.pi / 2 - angle))
        v_d = v_g_dq.real
        v_q = v_g_dq.imag

        e_q_t = v_q + p['X_d_t'] * I_d
        e_d_t = v_d - p['X_q_t'] * I_q

        e_q_st = v_q + p['X_d_st'] * I_d
        e_d_st = v_d - p['X_q_st'] * I_q

        e_q = e_q_t + I_d * (p['X_d'] - p['X_d_t'])
        e_q_0 = e_q.copy()

        PF_n = p['PF_n'] if 'PF_n' in p.dtype.names else 1
        self._input_values['P_m'] = s_pu.real / PF_n
        self._input_values['E_f'] = e_q_0

        X_0['speed'][:] = 0.00
        X_0['angle'][:] = angle
        X_0['e_q_t'][:] = e_q_t
        X_0['e_d_t'][:] = e_d_t
        X_0['e_q_st'][:] = e_q_st
        X_0['e_d_st'][:] = e_d_st

    def dyn_const_adm(self):
        idx_bus = self.bus_idx['terminal']
        bus_v_n = self.sys_par['bus_v_n'][idx_bus]
        z_n = bus_v_n ** 2 / self.sys_par['s_n']
        impedance = 1j * self.par['X_d_st'] * self.par['V_n'] ** 2 / self.par['S_n'] / z_n
        Y = self.par['N_par'] / impedance
        return Y, (idx_bus,) * 2

    def state_list(self):
        return ['speed', 'angle', 'e_q_t', 'e_d_t', 'e_q_st', 'e_d_st']

    def input_list(self):
        return ['V_t_abs', 'V_t_angle', 'P_m', 'E_f', 'v_aux', 'v_pss', 'f_n']

    def int_par_list(self):
        return ['f']

    def reduced_system(self):
        return self.par['bus']

    def current_injections(self, x, v):
        p = self.par
        X = self.local_view(x)

        i_inj = (X['e_q_st'] / (1j * p['X_d_st']) * self.q(x, v)
                 + X['e_d_st'] / (1j * p['X_q_st']) * self.d(x, v)) * p['N_par']

        I_n = p['S_n'] / (np.sqrt(3) * p['V_n'])
        i_n = self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'])
        I_inj = i_inj * I_n / i_n[self.bus_idx_red['terminal']]

        return self.bus_idx_red['terminal'], I_inj

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        p = self.par

        dX['speed'] = 0
        dX['angle'][:] = X['speed'] * 2 * np.pi * p['f_n']
        dX['e_q_t'][:] = 1 / p['T_d0_t'] * (self.E_f(x, v) + self.v_aux(x, v) - X['e_q_t'] - self.i_d(x, v) * (p['X_d'] - p['X_d_t']))
        dX['e_d_t'][:] = 1 / p['T_q0_t'] * (-X['e_d_t'] + self.i_q(x, v) * (p['X_q'] - p['X_q_t']))
        dX['e_q_st'][:] = 1 / p['T_d0_st'] * (X['e_q_t'] - X['e_q_st'] - self.i_d(x, v) * (p['X_d_t'] - p['X_d_st']))
        dX['e_d_st'][:] = 1 / p['T_q0_st'] * (X['e_d_t'] - X['e_d_st'] + self.i_q(x, v) * (p['X_q_t'] - p['X_q_st']))

    # --- Rotor reference frames ---

    def d(self, x, v):
        return np.exp(1j * (self.local_view(x)['angle'] - np.pi / 2))

    def q(self, x, v):
        return np.exp(1j * self.local_view(x)['angle'])

    # --- Voltage ---

    def v_t(self, x, v):
        return v[self.bus_idx_red['terminal']]

    def v_t_abs(self, x, v):
        return np.abs(self.v_t(x, v))

    def v_setp(self, x, v):
        return self.par['V']

    # --- Internal states ---

    def angle(self, x, v):
        return self.local_view(x)['angle']

    def speed(self, x, v):
        return self.local_view(x)['speed']

    def e_q_t(self, x, v):
        return self.local_view(x)['e_q_t']

    def e_d_t(self, x, v):
        return self.local_view(x)['e_d_t']

    def e_q_st(self, x, v):
        return self.local_view(x)['e_q_st']

    def e_d_st(self, x, v):
        return self.local_view(x)['e_d_st']

    def e_st(self, x, v):
        return self.e_q_st(x, v) * self.q(x, v) + self.e_d_st(x, v) * self.d(x, v)

    def e_t(self, x, v):
        return self.e_q_t(x, v) * self.q(x, v) + self.e_d_t(x, v) * self.d(x, v)

    # --- Current ---

    def i(self, x, v):
        return (self.e_st(x, v) - self.v_t(x, v)) / (1j * self.par['X_d_st'])

    def i_d(self, x, v):
        return (self.i(x, v) * np.exp(1j * (np.pi / 2 - self.angle(x, v)))).real

    def i_q(self, x, v):
        return (self.i(x, v) * np.exp(1j * (np.pi / 2 - self.angle(x, v)))).imag

    # --- Power (p.u. on generator base) ---

    def s_e(self, x, v):
        return self.v_t(x, v) * np.conj(self.i(x, v))

    def p_e(self, x, v):
        return self.s_e(x, v).real

    def q_e(self, x, v):
        return self.s_e(x, v).imag

    # --- Power (physical units) ---

    def S_e(self, x, v):
        return self.s_e(x, v) * self.par['S_n']

    def P_e(self, x, v):
        return self.p_e(x, v) * self.par['S_n']

    def Q_e(self, x, v):
        return self.q_e(x, v) * self.par['S_n']

    def P_nom(self, x, v):
        PF_n = self.par['cosphi_n'] if 'cosphi_n' in self.par.dtype.names else 1
        return self.par['S_n'] * PF_n
    def P_set(self, x, v):
        return self.par['P']