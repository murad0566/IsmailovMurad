import numpy as np
import pandas as pd
import math as mt
import os
import datetime
import pickle

from datetime import datetime as dt
from datetime import timedelta
import warnings
import shutil
import time
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 100)

from adlib import PrettyWriter as pw
from adlib.BruteFormater import BruteData as bf
from adlib import LoadMaster as lm
from adlib import Counter as cnt

from tqdm import tqdm  


import hashlib
hash_f = hashlib.blake2b(digest_size = 10)

def write_user_hist(f):
    try:
        path_hist = r'D:\Обмен\_Скрипты\Резервные копии рабочих папок\Папка Директора по автоматизации/hist/' + str(datetime.datetime.now().date()) +'_'+ os.getlogin() + '.pkl'
        try:
            with open(path_hist, 'rb') as fp:
                t = pickle.load(fp)
        except:
            t = {}

        hash_f.update(bytes(str(time.time()), 'utf-8'))
        t.update({hash_f.hexdigest():{'user':os.getlogin(),'time':str(datetime.datetime.now()),'modul':'EazyWorker','func':f}})
        with open(path_hist, 'wb') as fp:
            pickle.dump(t, fp)
    except:
        pass


def help():
    write_user_hist('help')
    print('''
    ew.dd() - удаление дубликатов
    ew.vc() - подсчёт колличества уникальных значений
    ew.nt() - создание столбца день/месяц/год
    ew.tot() - загрузка актуального тотала 
    ew.anti_f() - вычитание списков или фрэймов с сохранением порядка эллементов
    ew.get_stat() - даёт основную статистику по скорингу
    ew.gr() - функция get_ratio()
    ''')


def vc(data, col = 'ApplicationID', si = False, inplace=False):
    write_user_hist('vc')
    '''data(pd.Dataframe), col(str) - column to count values, si(bool) - sort indexs'''
    if inplace:
        x = data
    else:
        x = data.copy()
        
    if si:
        x = data.value_counts(col).sort_index().to_frame().reset_index()
    else:
        x = data.value_counts(col).to_frame().reset_index()
    x.columns = [col, 'Count']
    return x


def dd(df, time = False, on = 'ApplicationID', keep = 'last', inplace = False):
    
    '''data(pd.Dataframe), time(str) - column to sort values, on(str) - column to find duplicates'''
    write_user_hist('dd')
    #df = bf(df)
    
    def dds(df, time, on):
        y = df.sort_values(time).drop_duplicates(on, keep = keep)
        return y
    
    if inplace == False:
        df = df.copy()
    
    s_s = [2, 3, 4]
    dup_sum = df.duplicated(on).sum()
    if ((dup_sum%10 == 1)&(int((dup_sum%100)/10) == 1)):
        print(f'{dup_sum} дубликат')
    elif ((dup_sum%10) in s_s):
        print(f'{dup_sum} дубликата')
    else:
        print(f'{dup_sum} дубликатов')
    
    if time:
        return df.sort_values(time).drop_duplicates(on, keep = keep, inplace=inplace) #dds(df, time, on) 
    else:
        try:
            return df.sort_values('IssueTime').drop_duplicates(on, keep = keep, inplace=inplace) #dds(df, 'IssueTime', on)
        except:
            try:
                return df.sort_values('ApplicationDate').drop_duplicates(on, keep = keep, inplace=inplace) #dds(df, 'ApplicationDate', on)
            except:        
                return df.drop_duplicates(on, keep = keep, inplace=inplace)
    #return df
    


def anti_f(a, b):
    write_user_hist('anti_f')
    '''a(list or pd.DataFrame), b(list or pd.DataFrame) - list or frame to substract'''
    def mmset(a, b):
    
        list_new = [i for i in a if i not in b]
    
    
#        c = list(set(a) - set(b))
#        list_i = []
#        list_j = []
#       for i,j in enumerate(a):
#            list_i.append(i)
#            list_j.append(j)
#        d = pd.DataFrame({'Number':list_i, 'Value':list_j})
#        list_new = list(d[d.Value.isin(c)].Value)
        return(list_new)

    if (type(a) == pd.DataFrame) & (type(b) == pd.DataFrame):
        list_a = list(a)
        list_b = list(b)
        new = mmset(list_a, list_b)
        return(a[new])

    if (type(a) == pd.DataFrame) & (type(list(b)) == list):  
        list_a = list(a)
        new = mmset(list_a, b)
        return(a[new])

    if (type(list(a)) == list) & (type(list(b)) == list):     
        new = mmset(a, b)
        return(new)


def prev_month(x):
    write_user_hist('prev_month')
    if (int(x[5:]) == 1):
        date_start = x + '-01'
        date_stop = str(int(x[:4])-1) + '-12-01'    
    elif (int(x[5:]) <11):
        date_start = x + '-01'
        date_stop = x[:4] + '-0' + str(int(x[5:])-1) + '-01'
    else:
        date_start = x + '-01'
        date_stop = x[:4] + '-' + str(int(x[5:])-1) + '-01'
    return date_stop[:7]

def tot(dropd = False):
    write_user_hist('tot')
    '''dd(bool) - drop duplicates'''
    
    date = prev_month(str(dt.now())[:7])
    path = 'D:\Обмен\Для ежемесячных' + '/' + date + '/' + 'TOTAL ' + date + '.frt'
    base = lm.load(path)
    print(path)
    if dropd == False:
        return base
    else:
        return dd(base)

def buuid(dropd = False):
    write_user_hist('buuid')
    '''dd(bool) - drop duplicates'''

    date = str(dt.now())[:7]
    path = 'D:\Обмен\_Glued Reports' + '/' + date + ' Borrower_uuid.frt'
    base = lm.load(path)
    print(path)
    if dropd == False:
        return base
    else:
        return dd(base)

def get_stat(sc, com, gb=False):
    write_user_hist('get_stat')
    '''sc(pd.Dataframe) - scoring, com(pd.Dataframe) - common or total'''
    sc = dd(sc)
    com = dd(com)
    com['Exist_com'] = 1
    base = lm.merge(sc, com, on = 'ApplicationID')
    base.ApplicationDate = bf(base.ApplicationDate)
    a = len(base)
    b = len(base[base.Issue == 1])
    c = len(base[base.Exist_com == 1])
    d = base.LoanSum.sum()
    e = base.ApplicationDate.min().strftime('%Y-%m-%d')
    f = base.ApplicationDate.max().strftime('%Y-%m-%d')
    g = len(base[base.LenCH < 180])
    ddf = pd.DataFrame({'Параметр':['Дата первой заявки', 'Дата последней заявки', 'Заявок в СПР', 'Заявок с молодой КИ', 'Одобрено', 'Выдано', 'Конверсия в одобрение', 'Конверсия в выдачу', 'Выдано к одобрено', 'Средний чек'],
                        'Значение':[e, f, a, g, b, c, f'{round(100*b/a, 1)}%', f'{round(100*c/a, 1)}%', f'{round(100*c/b, 1)}%', int(d/c)]})
    if gb:
        display(ddf)
        return(base, ddf)
    else:
        return(ddf)


def nt(df, time=None, period='day', stat=True, si = False):
    write_user_hist('nt')
    '''df(pd.Dataframe) - data, col(str) - date column, period(str) - period to split'''
    if time == None:
        if 'IssueTime' in df.columns:
            time = 'IssueTime'
        elif 'ApplicationDate' in df.columns:
            time = 'ApplicationDate'
        else:
            print('No Date column')
    df[time] = bf(df[time])
    if period == 'year':
        df[period] = df[time].apply(lambda x: x.strftime('%Y'))
    if period == 'month':
        df[period] = df[time].apply(lambda x: x.strftime('%Y-%m'))
    if period == 'day':
        df[period] = df[time].apply(lambda x: x.strftime('%Y-%m-%d'))
        
    if stat == True:
        return vc(df, col = period, si = si)
    return df
    
def gr(series):
    write_user_hist('gr')
    '''series(pd.Series)'''
    return round(series.sum() / series.count()* 100, 1)

