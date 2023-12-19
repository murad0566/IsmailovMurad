import pandas as pd 
import sklearn as sk
import numpy as np
import copy
import os
import datetime
import time
import warnings
import hashlib
import pickle

warnings.filterwarnings("ignore")
pd.set_option('max_columns', 200) 
from datetime import datetime, timedelta
from adlib import LoadMaster as lm

hash_f = hashlib.blake2b(digest_size = 10)
path_hist = r'D:\Обмен\_Скрипты\Резервные копии рабочих папок\Папка Директора по автоматизации/hist.json'

def write_user_hist(f):
    try:
        path_hist = r'D:\Обмен\_Скрипты\Резервные копии рабочих папок\Папка Директора по автоматизации/hist/' + str(datetime.now().date()) +'_'+ os.getlogin() + '.pkl'
        try:
            with open(path_hist, 'rb') as fp:
                t = pickle.load(fp)
        except:
            t = {}

        hash_f.update(bytes(str(time.time()), 'utf-8'))
        t.update({hash_f.hexdigest():{'user':os.getlogin(),'time':str(datetime.now()),'modul':'Counter','func':f}})
        with open(path_hist, 'wb') as fp:
            pickle.dump(t, fp)
    except:
        pass


def help():
    write_user_hist('help')
    print('week_spliter() - split a variable into time intervals')
    print('query - agg_funcs')
    print("levenshtein - counts Levenshtein's distance")
    
   
def week_spliter(data, time_name = None, date_range = 7 ):
    '''data - DataFrame or Series; time_name - str, var's name to split; date_range - int, range to split (sample: 7, 'month')'''
    write_user_hist('week_spliter')
    data1  = data.copy()
    data = data1
    var_name = 'Week'
    if time_name == None and type(data) == pd.core.series.Series:
        data = data.to_frame()
        time_name = data.columns[0]

    data[time_name] = data[time_name].dt.normalize()
    last_day = data[time_name].max()
    min_day= data[time_name].min()
    while last_day > min_day:
        if (date_range == 7) or (date_range == 14):
            check = last_day-timedelta(days = date_range - 1)
            data.loc[((data[time_name] >= check) & (data[time_name]<=last_day)), var_name] = check.strftime("%Y-%m-%d") + ' - ' + last_day.strftime("%Y-%m-%d")
            if last_day.weekday() != 6:
                while last_day.weekday() != 6:
                    last_day+=timedelta(days = 1)
                    check = last_day-timedelta(days = date_range - 1)
                data.loc[((data[time_name] >= check) & (data[time_name]<=last_day)), var_name] = check.strftime("%Y-%m-%d") + ' - ' + last_day.strftime("%Y-%m-%d")
                last_day-=timedelta(days = date_range)
            else:   
                last_day-=timedelta(days = date_range)
        else:
            data[var_name] = data[time_name].dt.strftime('%Y-%m')
            break
    return data[var_name]

def get_ratio(series, ratio_type = 'opened'):
    '''series - data pd.Series; ratio_type - type of used data (all - for all data, opened - for only notna() data)'''
    write_user_hist('get_ratio')
    if ratio_type == 'opened':    
        return (series.sum() * 100/ series.count())
    else:
        return (series.sum() * 100/ len(series))

def query(data, x, y, group_by, agg_func = 'div'):
    '''data - pd.DataFrame; x and y - , opened - for only notna() data)'''
    write_user_hist('query')
    aggx = x.split('(')[0]
    x = x.split('(')[1][:-1]
    aggy = y.split('(')[0]
    y = y.split('(')[1][:-1]
    
    if x == y:
        d = data.groupby(group_by).agg({x:[aggx,aggy]})
    else:
        d = data.groupby(group_by).agg({x:aggx,y:aggy})
        
    if agg_func == 'div':
        return d.iloc[:,0] / d.iloc[:,1]
    elif agg_func == 'mult':
        return d.iloc[:,0] * d.iloc[:,1]
    elif agg_func == 'sub':
        return d.iloc[:,0] - d.iloc[:,1]
    elif agg_func == 'plus':
        return d.iloc[:,0] + d.iloc[:,1]
              
def levenshtein(word_1,word_2):
    write_user_hist('levenshtein')
    size_x = len(word_1) 
    size_y = len(word_2)
    if size_x > size_y:
        word_1, word_2 = word_2, word_1
        size_x, size_y = size_y, size_x  
        
    current_row = range(size_x +1)
    for i in range(1, size_y+1):
        prev_row, current_row = current_row, [i] +[0] * size_x
        for j in range(1, size_x+1):
            add,delete,change = prev_row[j] + 1, current_row[j-1]+1, prev_row[j-1]
            if word_1[j-1] != word_2[i-1]:
                change +=1
            current_row[j] = min(add,delete,change)
    return current_row[size_x]