from .blocks import *
import src.utility_functions as dps_uf


class VSC(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def add_blocks(self):
        p = self.par
        self.pll = PLL1(T_filter=self.par['T_pll'], bus=p['bus'])

        self.pi_p = PIRegulator(K_p=p['P_K_p'], K_i=p['P_K_i'])
        self.pi_p.input = lambda x, v: self.P_setp(x, v) - self.P(x, v)

        self.pi_q = PIRegulator(K_p=p['Q_K_p'], K_i=p['Q_K_i'])
        self.pi_q.input = lambda x, v: self.Q_setp(x, v) - self.Q(x, v)

        self.lag_p = TimeConstant(T=p['T_i'])
        self.lag_p.input = self.pi_p.output
        self.lag_q = TimeConstant(T=p['T_i'])
        self.lag_q.input = self.pi_q.output

        self.I_d = self.lag_p.output
        self.I_q = self.lag_q.output

    def I_inj(self, x, v):
        return (self.I_d(x, v) - 1j*self.I_q(x, v))*np.exp(1j*self.pll.output(x, v))

    def input_list(self):
        return ['P_setp', 'Q_setp']

    def P(self, x, v):
        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
        V = abs(v[self.bus_idx_red['terminal']])*v_n
        return np.sqrt(3)*V*self.I_d(x, v)

    def Q(self, x, v):
        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
        V = abs(v[self.bus_idx_red['terminal']])*v_n
        return np.sqrt(3)*V*self.I_q(x, v)

    def p_e(self, x, v):
        """Active power in per unit on S_n base."""
        return self.P(x, v) / self.par['S_n']

    def q_e(self, x, v):
        """Reactive power in per unit on S_n base."""
        return self.Q(x, v) / self.par['S_n']

    def load_flow_pq(self):
        return self.bus_idx['terminal'], -self.par['P_setp'], -self.par['Q_setp']

    def init_from_load_flow(self, x_0, v_0, S):
        self._input_values['P_setp'] = self.par['P_setp']
        self._input_values['Q_setp'] = self.par['Q_setp']

        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]

        V_0 = v_0[self.bus_idx_red['terminal']]*v_n

        I_d_0 = self.par['P_setp']/(abs(V_0)*np.sqrt(3))
        I_q_0 = self.par['Q_setp']/(abs(V_0)*np.sqrt(3))

        self.pi_p.initialize(
            x_0, v_0, self.lag_p.initialize(x_0, v_0, I_d_0)
        )

        self.pi_q.initialize(
            x_0, v_0, self.lag_q.initialize(x_0, v_0, I_q_0)
        )

    def current_injections(self, x, v):
        i_n = self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'])
        # self.P(x, v)
        return self.bus_idx_red['terminal'], self.I_inj(x, v)/i_n[self.bus_idx_red['terminal']]



class UIC_load(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Use system bus voltage if V_n is not specified
        fix_idx = self.par['V_n'] == 0
        bus_idx = dps_uf.lookup_strings(self.par['bus'], self.sys_par['bus_names'])
        self.par['V_n'][fix_idx] = self.sys_par['bus_v_n'][bus_idx][fix_idx]

        # Initialize bus index arrays
        self.bus_idx = np.array(
            np.zeros(self.n_units),
            dtype=[(key, int) for key in self.bus_ref_spec().keys()]
        )
        self.bus_idx_red = np.array(
            np.zeros(self.n_units),
            dtype=[(key, int) for key in self.bus_ref_spec().keys()]
        )

    # -------------------------------------------------------------------------
    # Model specification methods
    # -------------------------------------------------------------------------

    def load_flow_pq(self):
        return self.bus_idx['terminal'], -self.par['p_ref']*self.par['S_n'], -self.par['q_ref']*self.par['S_n']

    def int_par_list(self):
        """Integer parameters."""
        return ['f']

    def bus_ref_spec(self):
        """Map the terminal connection to the bus name."""
        return {'terminal': self.par['bus']}

    def state_list(self):
        """State variables of the UIC model."""
        return ['vi_x', 'vi_y', 'x_filter']

    def input_list(self):
        """Input signals to the UIC model."""
        return ['p_ref', 'q_ref', 'v_ref']

    # -------------------------------------------------------------------------
    # Core dynamics
    # -------------------------------------------------------------------------

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        par = self.par

        # --- Reconstruct complex internal voltage ---
        vi = X['vi_x'] + 1j * X['vi_y']
        # --- Reference signals ---
        s_ref = self.p_ref(x, v) + 1j * self.q_ref(x, v)

        # --- Current reference from power reference (in vi dq-frame) ---
        i_ref = np.conj(s_ref / vi)

        # --- Measured line current ---
        i_a = self.i_a(x, v)

        # --- Error signals ---
        i_error = i_ref - i_a

        v_error = 0  # Voltage control disabled (handled by generator AVR)

        omega_0 = 100 * np.pi  # Nominal angular frequency [rad/s]
        dvi = (1j * omega_0 * par['Ki'] * i_error + omega_0 * par['Kv'] * v_error + 1j * vi * X['x_filter'] * par['perfect_tracking'])
        delta_omega = (dvi / vi).imag
        dX['x_filter'] = (1 / par['T_filter']) * (delta_omega - X['x_filter'])
        dX['vi_x'] = np.real(dvi)
        dX['vi_y'] = np.imag(dvi)



    # -------------------------------------------------------------------------
    # Current injection (Norton equivalent)
    # -------------------------------------------------------------------------

    def i_inj(self, x, v):

        X = self.local_view(x)
        vi = X['vi_x'] + 1j * X['vi_y']
        xf = self.par['xf']
        return vi / (1j * xf)

    def current_injections(self, x, v):
        """Return bus index and scaled current injection for the network solver."""
        i_n_r = self.par['S_n'] / self.sys_par['s_n']
        return self.bus_idx_red['terminal'], self.i_inj(x, v) * i_n_r

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def init_from_load_flow(self, x_0, v_0, S):

        par = self.par
        X = self.local_view(x_0)

        v_t = v_0[self.bus_idx_red['terminal']]

        S_local = (self.par['p_ref'] + 1j * self.par['q_ref'])#S / par['S_n']
        current = np.conj(S_local / v_t)

        # Internal voltage behind filter reactance
        vi = v_t + 1j * current * par['xf']
        S_internal = vi * np.conj(current)
        # Set initial input references
        #self._input_values['v_ref'] = abs(vi)
        self._input_values['p_ref'] = S_internal.real
        self._input_values['q_ref'] = S_internal.imag

        # Set initial states
        X['vi_x'] = np.real(vi)
        X['vi_y'] = np.imag(vi)
        X['x_filter'] = 0.0  # No frequency deviation at t=0

    # -------------------------------------------------------------------------
    # Dynamic admittance
    # -------------------------------------------------------------------------

    def dyn_const_adm(self):

        par = self.par
        idx_bus = self.bus_idx['terminal']
        bus_v_n = self.sys_par['bus_v_n'][idx_bus]

        # Convert impedance from local to global per-unit base
        z_base_global = bus_v_n ** 2 / self.sys_par['s_n']
        z_base_local = par['V_n'] ** 2 / par['S_n']
        z_local = 1j * par['xf']
        z_global = z_local * z_base_local / z_base_global

        Y = 1 / z_global
        return Y, (idx_bus,) * 2

    # -------------------------------------------------------------------------
    # Output / measurement methods
    # -------------------------------------------------------------------------

    def v_t(self, x, v):
        """Terminal bus voltage [p.u., complex]."""
        return v[self.bus_idx_red['terminal']]

    def v_q(self, x, v):
        """Q-axis component of terminal voltage in the internal dq-frame."""
        return (self.v_t(x, v) * np.exp(-1j * self.local_view(x)['angle'])).imag

    def v_i(self, x, v):
        """Internal voltage vi [p.u., complex]."""
        X = self.local_view(x)
        return X['vi_x'] + 1j * X['vi_y']

    def i_a(self, x, v):
        """
        Line current from terminal to internal voltage node [p.u., complex].

            i_a = -(v_t - vi) / (j·xf)
        """
        par = self.par
        X = self.local_view(x)
        v_t = v[self.bus_idx_red['terminal']]
        vi = X['vi_x'] + 1j * X['vi_y']
        return -(v_t - vi) / (1j * par['xf'])

    def s_e(self, x, v):
        """Apparent power at the terminal bus [p.u., complex]."""
        return self.v_t(x, v) * np.conj(self.i_a(x, v))

    def p_e(self, x, v):
        """Active power at the terminal bus [p.u.]."""
        return self.s_e(x, v).real

    def q_e(self, x, v):
        """Reactive power at the terminal bus [p.u.]."""
        return self.s_e(x, v).imag



class UIC_bat(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Use system bus voltage if V_n is not specified
        fix_idx = self.par['V_n'] == 0
        bus_idx = dps_uf.lookup_strings(self.par['bus'], self.sys_par['bus_names'])
        self.par['V_n'][fix_idx] = self.sys_par['bus_v_n'][bus_idx][fix_idx]

        # Initialize bus index arrays
        self.bus_idx = np.array(
            np.zeros(self.n_units),
            dtype=[(key, int) for key in self.bus_ref_spec().keys()]
        )
        self.bus_idx_red = np.array(
            np.zeros(self.n_units),
            dtype=[(key, int) for key in self.bus_ref_spec().keys()]
        )

    # -------------------------------------------------------------------------
    # Model specification methods
    # -------------------------------------------------------------------------

    def load_flow_pv(self):
        """Define the converter as a PV bus for load flow initialization."""
        return self.bus_idx['terminal'], -self.par['p_ref'] * self.par['S_n'], self.par['v_ref']

    def int_par_list(self):
        """Integer parameters."""
        return ['f']

    def bus_ref_spec(self):
        """Map the terminal connection to the bus name."""
        return {'terminal': self.par['bus']}

    def state_list(self):
        """State variables of the UIC model."""
        return ['vi_x', 'vi_y', 'x_filter']

    def input_list(self):
        """Input signals to the UIC model."""
        return ['p_ref', 'q_ref', 'v_ref']

    # -------------------------------------------------------------------------
    # Core dynamics
    # -------------------------------------------------------------------------

    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        par = self.par

        # --- Reconstruct complex internal voltage ---
        vi = X['vi_x'] + 1j * X['vi_y']

        # --- Reference signals ---
        s_ref = self.p_ref(x, v) + 1j * self.q_ref(x, v)
        v_ref = self.v_ref(x, v)

        i_ref = np.conj(s_ref / vi)

        # --- Measured line current ---
        i_a = self.i_a(x, v)

        # --- Error signals ---
        i_error = i_ref - i_a

        v_error = 0  # Voltage control disabled (handled by generator AVR)

        omega_0 = 100 * np.pi  # Nominal angular frequency [rad/s]

        dvi = (1j * omega_0 * par['Ki'] * i_error + omega_0 * par['Kv'] * v_error + 1j * vi * X['x_filter'] * par[
            'perfect_tracking'])

        delta_omega = (dvi / vi).imag

        dX['x_filter'] = (1 / par['T_filter']) * (delta_omega - X['x_filter'])
        dX['vi_x'] = np.real(dvi)
        dX['vi_y'] = np.imag(dvi)

    # -------------------------------------------------------------------------
    # Current injection (Norton equivalent)
    # -------------------------------------------------------------------------

    def i_inj(self, x, v):
        X = self.local_view(x)
        vi = X['vi_x'] + 1j * X['vi_y']
        xf = self.par['xf']
        return vi / (1j * xf)

    def current_injections(self, x, v):
        """Return bus index and scaled current injection for the network solver."""
        i_n_r = self.par['S_n'] / self.sys_par['s_n']
        return self.bus_idx_red['terminal'], self.i_inj(x, v) * i_n_r

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def init_from_load_flow(self, x_0, v_0, S):
        par = self.par
        X = self.local_view(x_0)

        v_t = v_0[self.bus_idx_red['terminal']]

        # Convert apparent power to local per-unit and compute current
        S_local = S / par['S_n']
        current = np.conj(S_local / v_t)

        # Internal voltage behind filter reactance
        vi = v_t + 1j * current * par['xf']
        S_internal = vi * np.conj(current)

        # Set initial input references
        self._input_values['v_ref'] = abs(vi)
        self._input_values['p_ref'] = S_internal.real
        self._input_values['q_ref'] = S_internal.imag

        # Set initial states
        X['vi_x'] = np.real(vi)
        X['vi_y'] = np.imag(vi)
        X['x_filter'] = 0.0  # No frequency deviation at t=0

    # -------------------------------------------------------------------------
    # Dynamic admittance
    # -------------------------------------------------------------------------

    def dyn_const_adm(self):
        par = self.par
        idx_bus = self.bus_idx['terminal']
        bus_v_n = self.sys_par['bus_v_n'][idx_bus]

        # Convert impedance from local to global per-unit base
        z_base_global = bus_v_n ** 2 / self.sys_par['s_n']
        z_base_local = par['V_n'] ** 2 / par['S_n']
        z_local = 1j * par['xf']
        z_global = z_local * z_base_local / z_base_global

        Y = 1 / z_global
        return Y, (idx_bus,) * 2

    # -------------------------------------------------------------------------
    # Output / measurement methods
    # -------------------------------------------------------------------------

    def v_t(self, x, v):
        """Terminal bus voltage [p.u., complex]."""
        return v[self.bus_idx_red['terminal']]

    def v_q(self, x, v):
        """Q-axis component of terminal voltage in the internal dq-frame."""
        return (self.v_t(x, v) * np.exp(-1j * self.local_view(x)['angle'])).imag

    def v_i(self, x, v):
        """Internal voltage vi [p.u., complex]."""
        X = self.local_view(x)
        return X['vi_x'] + 1j * X['vi_y']

    def i_a(self, x, v):
        """
        Line current from terminal to internal voltage node [p.u., complex].

            i_a = -(v_t - vi) / (j·xf)
        """
        par = self.par
        X = self.local_view(x)
        v_t = v[self.bus_idx_red['terminal']]
        vi = X['vi_x'] + 1j * X['vi_y']
        return -(v_t - vi) / (1j * par['xf'])

    def s_e(self, x, v):
        """Apparent power at the terminal bus [p.u., complex]."""
        return self.v_t(x, v) * np.conj(self.i_a(x, v))

    def p_e(self, x, v):
        """Active power at the terminal bus [p.u.]."""
        return self.s_e(x, v).real

    def q_e(self, x, v):
        """Reactive power at the terminal bus [p.u.]."""
        return self.s_e(x, v).imag