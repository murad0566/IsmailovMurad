import pandas as pd
import numpy as np

def replace(data, column, to_replace):
    '''data - DataFrame, column - str (sample: column), to_replace - dict (sample {"Man":"M"}'''
    data.replace(''+ : to_replace, inplace = True)