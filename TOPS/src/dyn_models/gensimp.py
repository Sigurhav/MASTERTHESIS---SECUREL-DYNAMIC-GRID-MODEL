

import numpy as np
from src.dyn_models.utils import DAEModel
import src.utility_functions as dps_uf

class GEN(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for req_attr, default in zip(['PF_n', 'N_par'], [1, 1]):
            if not req_attr in self.par.dtype.names:
                new_field = np.ones(len(self.par), dtype=[(req_attr, float)])
                new_field[req_attr] *= default
                self.par = dps_uf.combine_recarrays(self.par, new_field)

        fix_idx = self.par['V_n'] == 0
        gen_bus_idx = dps_uf.lookup_strings(self.par['bus'], self.sys_par['bus_names'])
        self.par['V_n'][fix_idx] = self.sys_par['bus_v_n'][gen_bus_idx][fix_idx]

        fix_idx = self.par['S_n'] == 0
        self.par['S_n'][fix_idx] = self.sys_par['s_n']

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])


    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def load_flow_pv(self):
        return self.bus_idx['terminal'], -self.par['P'] * self.par['N_par'], self.par['V']

    def init_from_load_flow(self, x_0, v_0, S):
        """Initialize generator states from load flow solution."""
        X_0 = self.local_view(x_0)
        p = self.par

        # Fill in missing nominal values if needed
        fix_idx = self.par['V_n'] == 0
        self.par['V_n'][fix_idx] = self.sys_par['bus_v_n'][self.bus_idx['terminal']][fix_idx]

        fix_idx = self.par['S_n'] == 0
        self.par['S_n'][fix_idx] = self.sys_par['s_n']

        # Convert to per-unit on generator base
        s_pu = S / p['S_n'] / p['N_par']
        v_g = v_0[self.bus_idx['terminal']]
        I_g = np.conj(s_pu / v_g)  # Complex current in p.u.

        # Calculate internal voltage behind synchronous reactance
        E = v_g + 1j * p['Xs'] * I_g
        angle = np.angle(E)
        speed = np.zeros_like(angle)  # Zero deviation = synchronous speed

        # Initialize inputs for steady-state equilibrium
        PF_n = p['PF_n'] if 'PF_n' in p.dtype.names else 1
        self._input_values['P_m'] = s_pu.real / PF_n  # Mechanical power = electrical power
        self._input_values['E_f'] = np.abs(E)  # Internal voltage magnitude

        # Set initial states
        X_0['speed'][:] = speed
        X_0['angle'][:] = angle

    def state_list(self):
        return ['speed', 'angle']

    def input_list(self):
        return ['P_m', 'E_f', 'v_pss']

    def reduced_system(self):
        return self.par['bus']

    def dyn_const_adm(self):
        """Return constant admittance representing generator internal impedance."""
        idx_bus = self.bus_idx['terminal']
        bus_v_n = self.sys_par['bus_v_n'][idx_bus]
        z_n = bus_v_n ** 2 / self.sys_par['s_n']

        # Generator impedance in p.u. (generator base)
        impedance_pu_gen = 1j * self.par['Xs']

        # Convert to system p.u. base
        impedance = impedance_pu_gen * self.par['V_n'] ** 2 / self.par['S_n'] / z_n

        # Admittance, accounting for parallel units
        Y = self.par['N_par'] / impedance

        return Y, (idx_bus,) * 2

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        p = self.par

        omega_s = 2 * np.pi * p['f_n']
        P_e = self.p_e(x, v)

        #dX['speed'][:] = (omega_s / (2.0 * p['H'])) * (self.P_m(x, v) - P_e - p['Kd'] * X['speed'])
        dX['speed'][:] = (self.P_m(x, v) - P_e - p['D'] * X['speed']) / (2.0 * p['H'])
        dX['angle'][:] = X['speed'] * omega_s

    def current_injections(self, x, v):
        """Calculate current injections into the network."""
        p = self.par
        X = self.local_view(x)

        # Internal voltage source: E∠δ
        E_internal = self.E_f(x, v) * np.exp(1j * X['angle'])

        # Current from voltage source through impedance (defined in dyn_const_adm)
        # This is the Norton equivalent current: I = E / (jXs)
        I_inj = E_internal / (1j * p['Xs'])

        # Convert to system base
        I_n = p['S_n'] / (np.sqrt(3) * p['V_n'])
        i_n = self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'])
        I_inj = I_inj * I_n / i_n[self.bus_idx_red['terminal']]

        # Account for parallel units
        I_inj = I_inj * p['N_par']

        return self.bus_idx_red['terminal'], I_inj

    def angle(self, x, v):
        return self.local_view(x)['angle']

    def speed(self, x, v):
        return self.local_view(x)['speed']

    def v_t(self, x, v):
        return v[self.bus_idx_red['terminal']]

    def i(self, x, v):
        # Update to use Xs instead of X_d_st
        return (self.E_f(x, v) * np.exp(1j * self.angle(x, v)) - self.v_t(x, v)) / (1j * self.par['Xs'])

    def s_e(self, x, v):
        return self.v_t(x, v) * np.conj(self.i(x, v))

    def p_e(self, x, v):
        return self.s_e(x, v).real

    def q_e(self, x, v):
        return self.s_e(x, v).imag

    def v_t_abs(self, x, v):
        return np.abs(v[self.bus_idx_red['terminal']])

    def S_e(self, x, v):
        return self.s_e(x, v) * self.par['S_n']

    def P_e(self, x, v):
        return self.p_e(x, v) * self.par['S_n']

    def Q_e(self, x, v):
        return self.q_e(x, v) * self.par['S_n']

    def v_setp(self, x, v):
        return self.par['V']




class IB(DAEModel):
    """
    Infinite bus model - second order generator with controllable frequency.
    Speed (frequency) can be set externally and remains constant.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for req_attr, default in zip(['PF_n', 'N_par'], [1, 1]):
            if not req_attr in self.par.dtype.names:
                new_field = np.ones(len(self.par), dtype=[(req_attr, float)])
                new_field[req_attr] *= default
                self.par = dps_uf.combine_recarrays(self.par, new_field)

        fix_idx = self.par['V_n'] == 0
        gen_bus_idx = dps_uf.lookup_strings(self.par['bus'], self.sys_par['bus_names'])
        self.par['V_n'][fix_idx] = self.sys_par['bus_v_n'][gen_bus_idx][fix_idx]

        fix_idx = self.par['S_n'] == 0
        self.par['S_n'][fix_idx] = self.sys_par['s_n']

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    def set_frequency(self, x, freq_deviation):
        """
        Set the frequency deviation for the infinite bus.

        Parameters:
        -----------
        x : state vector
        freq_deviation : float or array
            Frequency deviation in per-unit (e.g., 0.01 for 1% deviation)
        """
        X = self.local_view(x)
        X['speed'][:] = freq_deviation

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def load_flow_pv(self):
        return self.bus_idx['terminal'], -self.par['P'] * self.par['N_par'], self.par['V']

    def init_from_load_flow(self, x_0, v_0, S):
        """Initialize infinite bus states from load flow solution."""
        X_0 = self.local_view(x_0)
        p = self.par

        # Fill in missing nominal values if needed
        fix_idx = self.par['V_n'] == 0
        self.par['V_n'][fix_idx] = self.sys_par['bus_v_n'][self.bus_idx['terminal']][fix_idx]

        fix_idx = self.par['S_n'] == 0
        self.par['S_n'][fix_idx] = self.sys_par['s_n']

        # Convert to per-unit on generator base
        s_pu = S / p['S_n'] / p['N_par']
        v_g = v_0[self.bus_idx['terminal']]
        I_g = np.conj(s_pu / v_g)  # Complex current in p.u.

        # Calculate internal voltage behind synchronous reactance
        E = v_g + 1j * p['Xs'] * I_g
        angle = np.angle(E)
        speed = np.zeros_like(angle)  # Zero deviation = synchronous speed

        # Initialize inputs for steady-state equilibrium
        PF_n = p['PF_n'] if 'PF_n' in p.dtype.names else 1
        self._input_values['P_m'] = s_pu.real / PF_n  # Mechanical power = electrical power
        self._input_values['E_f'] = np.abs(E)  # Internal voltage magnitude

        # Set initial states
        X_0['speed'][:] = speed
        X_0['angle'][:] = angle

    def state_list(self):
        return ['speed', 'angle']

    def input_list(self):
        return ['P_m', 'E_f', 'v_pss']

    def reduced_system(self):
        return self.par['bus']

    def dyn_const_adm(self):
        """Return constant admittance representing generator internal impedance."""
        idx_bus = self.bus_idx['terminal']
        bus_v_n = self.sys_par['bus_v_n'][idx_bus]
        z_n = bus_v_n ** 2 / self.sys_par['s_n']

        # Generator impedance in p.u. (generator base)
        impedance_pu_gen = 1j * self.par['Xs']

        # Convert to system p.u. base
        impedance = impedance_pu_gen * self.par['V_n'] ** 2 / self.par['S_n'] / z_n

        # Admittance, accounting for parallel units
        Y = self.par['N_par'] / impedance

        return Y, (idx_bus,) * 2

    def state_derivatives(self, dx, x, v):
        """
        State derivatives for infinite bus.
        Speed is fixed (dω/dt = 0) - represents infinite inertia.
        Angle still rotates based on current speed.
        """
        dX = self.local_view(dx)
        X = self.local_view(x)
        p = self.par

        omega_s = 2 * np.pi * p['f_n']

        # Infinite bus: speed doesn't change (infinite inertia)
        dX['speed'][:] = 0

        # Angle still rotates at current speed
        dX['angle'][:] = X['speed'] * omega_s

    def current_injections(self, x, v):
        """Calculate current injections into the network."""
        p = self.par
        X = self.local_view(x)

        # Internal voltage source: E∠δ
        E_internal = self.E_f(x, v) * np.exp(1j * X['angle'])

        # Current from voltage source through impedance (defined in dyn_const_adm)
        # This is the Norton equivalent current: I = E / (jXs)
        I_inj = E_internal / (1j * p['Xs'])

        # Convert to system base
        I_n = p['S_n'] / (np.sqrt(3) * p['V_n'])
        i_n = self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'])
        I_inj = I_inj * I_n / i_n[self.bus_idx_red['terminal']]

        # Account for parallel units
        I_inj = I_inj * p['N_par']

        return self.bus_idx_red['terminal'], I_inj

    def angle(self, x, v):
        return self.local_view(x)['angle']

    def speed(self, x, v):
        return self.local_view(x)['speed']

    def v_t(self, x, v):
        return v[self.bus_idx_red['terminal']]

    def i(self, x, v):
        return (self.E_f(x, v) * np.exp(1j * self.angle(x, v)) - self.v_t(x, v)) / (1j * self.par['Xs'])

    def s_e(self, x, v):
        return self.v_t(x, v) * np.conj(self.i(x, v))

    def p_e(self, x, v):
        return self.s_e(x, v).real

    def q_e(self, x, v):
        return self.s_e(x, v).imag

    def v_t_abs(self, x, v):
        return np.abs(v[self.bus_idx_red['terminal']])

    def S_e(self, x, v):
        return self.s_e(x, v) * self.par['S_n']

    def P_e(self, x, v):
        return self.p_e(x, v) * self.par['S_n']

    def Q_e(self, x, v):
        return self.q_e(x, v) * self.par['S_n']

    def v_setp(self, x, v):
        return self.par['V']