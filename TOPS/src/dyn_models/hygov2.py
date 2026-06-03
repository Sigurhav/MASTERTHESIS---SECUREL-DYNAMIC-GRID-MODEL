from src.dyn_models.blocks import *
from src.dyn_models.utils import auto_init

class GOV:
    def connections(self):
        return [
            {
                'input': 'speed',
                'source': {
                    'container': 'gen',
                    'mdl': '*',
                    'id': self.par['gen'],
                },
                'output': 'speed',
            },
            {
                'output': 'output',
                'destination': {
                    'container': 'gen',
                    'mdl': '*',
                    'id': self.par['gen'],
                },
                'input': 'P_m',
            },
        ]


class Hydro_gov(GOV, DAEModel):
    """Hydro turbine governor and turbine model.

    Implements the block diagram:
        Δω → lead-lag → PI (Kp + Kp/(sTi)) → servo 1/(1+sTy) → turbine (1-sTw)/(1+sTw/2) → ΔP_m
.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output = lambda x, v: self.P_m(x, v)
        self.dG_max = np.inf #Create internal parameter. Infinite if not defined
        self.dG_min = -np.inf

    def P_m(self, x, v):
        X = self.local_view(x)
        p = self.par
        Pm = self.int_par['bias'] + ((p['T1'] + p['T2']) / p['T2']) * p['At'] * X['x_t'] - (p['T1'] / p['T2']) * X['G'] * p['At']
        return np.maximum(Pm, 0) #Power can not go below Pmin

    def input_list(self):
        return ['speed']

    def state_list(self):
        return ['x_ll','x_int', 'G', 'x_t']

    def int_par_list(self):
        # Collect initial mechanical power output
        return ['bias']

    def init_from_connections(self, x0, v0, output_0):
        #Defines P_m to be equal to the initial output value
        auto_init(self, x0, v0, output_0['output'])

        Pm_init = output_0['output'] #Collect all mechanical powers on machine base
        G_init = Pm_init / 1.1 # Find actual per unit position of gates in steady-state
        self.dG_max = 1.0 - G_init #Define maximum and minimum possible change in gate position.
        self.dG_min = 0.0 - G_init




    def state_derivatives(self, dx, x, v):
        d = self.local_view(dx)
        X = self.local_view(x)
        p = self.par

        speed = self.speed(x, v)

        # Lead-lag block
        #if p['Tf'].any == 0:   # or p['Td'] == p['Tf']: #Cannot devide by zero
        #    lead_lag_out = -speed
        #else:
        d['x_ll'][:] = (-1 / p['Tf']) * X['x_ll'] + (1 / p['Tf']) * (-speed)
        lead_lag_out = (1 - p['Td']/p['Tf']) * X['x_ll'] + (p['Td']/p['Tf']) * (-speed)

        integrator_error = lead_lag_out - X['G'] * p['rho']

        prop_term = lead_lag_out * p['Kp'] #Proportional term

        d['x_int'][:] = (integrator_error * p['Kp']) / p['Ti']

        servo_input = X['x_int'] + prop_term


        # Servo block
        d['G'][:] = (servo_input - X['G']) / p['Ty']

        #Find index where gate position has exceeded its limit
        lower_lim_idx = (X['G'] <= self.dG_min) & (d['G'] < 0) #Comment out code after & to ensure the generator remains at 0 Pm once reached
        d['G'][lower_lim_idx] *= 0 #Set change to zero if limit is exceeded

        upper_lim_idx = (X['G'] >= self.dG_max) & (d['G'] > 0)
        d['G'][upper_lim_idx] *= 0

        # Anti-windup
        #Integrator must shut off when gate exceeds its maximum
        int_upper_idx = (X['G'] >= self.dG_max) & (integrator_error > 0)
        int_lower_idx = (X['G'] <= self.dG_min) & (integrator_error < 0) #Comment out code after & to ensure the generator remains at 0 Pm once reached
        d['x_int'][int_upper_idx] *= 0
        d['x_int'][int_lower_idx] *= 0


        # Turbine block
        d['x_t'][:] = (-1/p['T2']) * X['x_t'] + (1/p['T2']) * X['G']