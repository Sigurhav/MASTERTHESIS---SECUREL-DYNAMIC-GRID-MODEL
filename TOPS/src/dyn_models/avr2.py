from src.dyn_models.blocks import *
from src.dyn_models.utils import auto_init

class AVR:
    def connections(self):
        return [
            {
                'input': 'v_t',
                'source': {
                    'container': 'gen',
                    'mdl': '*',
                    'id': self.par['gen'],
                },
                'output': 'v_t_abs',
            },
            {
                'input': 'v_setp',
                'source': {
                    'container': 'gen',
                    'mdl': '*',
                    'id': self.par['gen'],
                },
                'output': 'v_setp',
            },
            {
                'input': 'v_pss',
                'source': {
                    'container': 'gen',
                    'mdl': '*',
                    'id': self.par['gen'],
                },
                'output': 'v_pss',
            },
            {
                'output': 'output',
                'destination': {
                    'container': 'gen',
                    'mdl': '*',
                    'id': self.par['gen'],
                },
                'input': 'E_f',
            }
        ]


class SEXS(DAEModel, AVR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output = lambda x, v: self.E_f(x, v)

    def E_f(self, x, v):
        X = self.local_view(x)
        p = self.par
        return X['x_tcl'] #self.int_par['bias'] + ((p['T1'] + p['T2']) / p['T2']) * p['At'] * X['x_t'] - (p['T1'] / p['T2']) * X['G'] * p['At']

    def input_list(self):
        return ['v_setp', 'v_t', 'v_pss'] #Speed deviation input

    def state_list(self):
        return ['x_ll', 'x_tcl']

    def int_par_list(self):
        return ['bias']

    def init_from_connections(self, x0, v0, output_0):
        auto_init(self, x0, v0, output_0['output'])


    def state_derivatives(self, dx, x, v):
        d = self.local_view(dx)
        X = self.local_view(x)
        p = self.par

        input_signal = self.v_setp(x, v) - self.v_t(x, v) + self.v_pss(x, v) + self.int_par['bias']

        # Lead-lag: G(s) = (1 + s*T_a) / (1 + s*T_b)
        d['x_ll'][:] = (-1 / p['T_b']) * X['x_ll'] + (1 / p['T_b']) * input_signal
        lead_lag_out = (1 - p['T_a'] / p['T_b']) * X['x_ll'] + (p['T_a'] / p['T_b']) * input_signal

        #Gain bl

        gain_out = p['K'] * lead_lag_out
        d['x_tcl'][:] = 1 / p['T_e'] * (gain_out - X['x_tcl'])

        # Lims on state variable (clamping)
        lower_lim_idx = (X['x_tcl'] <= self.par['E_min']) & (d['x_tcl'] < 0)
        d['x_tcl'][lower_lim_idx] *= 0

        upper_lim_idx = (X['x_tcl'] >= self.par['E_max']) & (d['x_tcl'] > 0)
        d['x_tcl'][upper_lim_idx] *= 0