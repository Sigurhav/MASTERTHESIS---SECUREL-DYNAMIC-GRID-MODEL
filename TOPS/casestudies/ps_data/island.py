def load():
    return {
        'base_mva': 1,
        'f': 50,
        'slack_bus': 'B2',

        'buses': [
            ['name',    'V_n'],
            ['B1',      1],
            ['B2',      1],
            ['B3',      1],
            #['B4',      1],

        ],

        'lines': [
            ['name',    'from_bus', 'to_bus',   'length',   'S_n',  'V_n',  'unit',     'R',    'X',   'B'],
            ['L1',        'B1',           'B2',     1,         1,      1,    'p.u.',     0.01,   0.1,    0],
            ['L2',        'B1',           'B3',     1,         1,      1,    'p.u.',     0.01,   0.1,    0],
            #['L3',        'B4',           'B1',     1,         1,      1,     'p.u.',    0.01,   0.1,    0],
        ],

        'generators': {
            'GEN': [
                ['name', 'bus', 'f_n', 'S_n', 'V_n', 'P', 'V', 'H', 'D', 'Xs'],
                ['G1',   'B1',    50,    1,    1,     1,  1.0, 1.5, 10, 0.3],
            ],
            'IB': [
                ['name', 'bus', 'f_n', 'S_n', 'V_n', 'P', 'V', 'H', 'D', 'Xs'],
                ['IB',    'B2',   50,   1,      1,  -0.2, 1.0, 999999, 0, 0.2],  # Infinite bus on B1
            ],
        },

        'loads': [
            ['name', 'bus', 'P', 'Q', 'model'],
            ['L1', 'B3',    0.8,  0, 'Z'],

        ],


        #'gov': {
          #  'TGOV1': [
          #      ['name', 'gen', 'R', 'D_t', 'V_min', 'V_max', 'T_1', 'T_2', 'T_3'],
          #      ['GOV1', 'G1', 0.1,  0.1,    0,        1,     5,    2,    2],
       #     ]
      #   },
        'hygov2': {
            'Hydro_gov': [
                ['name', 'gen', 'Td', 'Tf', 'Kp', 'Ti', 'Ty', 'T1', 'T2', 'At', 'rho'],
                ['GOV1', 'G1',   0,    0.05,   2,    10,   0.5, 2.33, 1.27,  1.1,   0.06],
            ],
        },


        'avr': {
                'SEXS': [
                    ['name', 'gen', 'K', 'T_a', 'T_b', 'T_e', 'E_min', 'E_max'],
                    ['AVR1', 'G1',   100, 2.0,   10.0, 0.1, -10, 10],
                ]

        },
    }
