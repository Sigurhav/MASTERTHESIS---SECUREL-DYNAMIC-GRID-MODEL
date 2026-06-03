
import numpy as np
from src.dyn_models.utils import DAEModel
from src.dyn_models.utils import auto_init
import src.utility_functions as dps_uf

class COM:
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
                'output': 'power',
                'destination': {
                    'container': 'gen',
                    'mdl': '*',
                    'id': self.par['gen'],
                },
                'input': 'P_m',
            },


        ]

class hygov(DAEModel, COM):


    def input_list(self):
        return ['speed']

    def state_list(self):

        return ['Pm', 'c']


    def state_derivatives(self, dx, x, v):
        dX = self.local_view(dx)
        X = self.local_view(x)
        par = self.par
        #dX['c'] = (-self.speed(x, v) / 0.04 - X['c']) / 10

        dX['Pm'] = (par['A_t'] * X['c'] - par['A_t'] * par['T1']* dX['c'] - X['Pm']) / par['T1']
        #print(dX['Pm'])
        return

    def init_from_load_flow(self, x_0, v_0, S):
        X = self.local_view(x_0)
        X['c'] = self.par['c_ref']
        print(self.par['c_ref'])

    def power(self, x, v):
        X = self.local_view(x)
        return X['Pm']

