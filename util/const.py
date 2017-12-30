import os

import numpy as np

# t_value1: chilled water supply temperature (cold)
# t_value2: chilled water return temperature (warm)
# t_value3: cooling water supply temperature (hot)
# t_value4: cooling water return temperature (cool)

COLS_TO_USE = [
    'c_flowRate', 'e_flowRate',
    'p_ch1Watt', 'p_ch2Watt', 'p_ch3Watt',
    't_value1', 't_value2', 't_value3', 't_value4'
]
CONFIGURABLE_COLS = [
    'c_flowRate', 'e_flowRate',
    't_value1', 't_value2', 't_value3', 't_value4'
]

# when power consumption is too low, it's likely an outlier
MIN_PC_CUTOFF = 3e4

CHILLERS = ['chiller' + str(i) for i in range(1,5)]
DATE_FOLDERS = os.listdir('data')

TRAIN_FOLDERS = [DATE_FOLDERS[i] for i in range(len(DATE_FOLDERS)) if i % 5 != 0]
TEST_FOLDERS = [DATE_FOLDERS[i] for i in range(len(DATE_FOLDERS)) if i % 5 == 0]
