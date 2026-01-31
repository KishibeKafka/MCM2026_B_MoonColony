import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Q1.load_data import load_data

class Q3:
    def __init__(self):
        self.P0 = 1e5
        self.P_max = 2e5 # max population
        self.pir = 0.02
        self.w = 186.374 * 365 # USA 2024
        self.r = 0.7

    def population_growth(self):
