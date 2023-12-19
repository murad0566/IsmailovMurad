import os
import numpy as np
import re
import pyreadstat
import datetime
import time
import concurrent.futures

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import xml.etree.ElementTree as ET
import json
import pickle

from tqdm import tqdm
from adlib import BruteFormater as bf
from datetime import timedelta
from pyarrow import csv



'''
load_files()
reduce_mem_usage()
save()
find_files()
'''


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
        t.update({hash_f.hexdigest():{'user':os.getlogin(),'time':str(datetime.datetime.now()),'modul':'LoadMaster','func':f}})
        with open(path_hist, 'wb') as fp:
            pickle.dump(t, fp)
    except:
        pass



def help():
    write_user_hist('help')
    print('load()- file (or files) load')
    print('reduce_mem_usage() - data frame memory reduction')
    print('save() - saving a file in any format ')
    print('find() - search files in folders')
    print('merge() - fast merge')
    print('filter() - pd.filter(like = )')


def find_and_fill(start_dir, to_find, since = 'min', for_d = 'max', time_stop = 60):
    write_user_hist('find_and_fill')
    def file_filler(f, since = 'min', for_d = 'max', dp1 = 1, dp2 = 2):
        def find_bigger(f, since, for_d):
            imax = None
            if for_d == 'max':
                day_max = pd.to_datetime('2017-01-01', format = '%Y-%m-%d')
                for i in range(len(f)):
                    t = f[i].split('/')[-1]
                    t = t.split('.')[0]
                    t1 = t.split('_')
                    try:
                        if dp1 == 3:
                            t1[3] = t1[3][4:]
                            t1[4] = t1[4][2:]
                        elif dp1 == 4:
                            t1[4] = t1[4][4:]
                            t1[5] = t1[5][2:]
                        to_test = pd.to_datetime(t1[dp2], format = '%Y-%m-%d')                       
                        pd.to_datetime(t1[dp1], format = '%Y-%m-%d')
                    except:
                        continue
                    if  to_test >= day_max:
                        day_max = pd.to_datetime(t1[dp2], format = '%Y-%m-%d')
                        day_min = pd.to_datetime(t1[dp1], format = '%Y-%m-%d')
                        imax = i
            else:
                day_max = pd.to_datetime(for_d, format = '%Y-%m-%d')
                day_min = pd.to_datetime(for_d, format = '%Y-%m-%d')
                for i in range(len(f)):
                    t = f[i].split('/')[-1]
                    t = t.split('.')[0]
                    t1 = t.split('_')
                    try:
                        if dp1 == 3:
                            t1[3] = t1[3][4:]
                            t1[4] = t1[4][2:]
                        elif dp1 == 4:
                            t1[4] = t1[4][4:]
                            t1[5] = t1[5][2:]
                        to_test = pd.to_datetime(t1[dp2], format = '%Y-%m-%d')
                        pd.to_datetime(t1[dp1], format = '%Y-%m-%d')
                    except:
                        continue
                        
                    if to_test == day_max:
                        if t1[dp1] == since:
                            imax = i
                            return f[imax], pd.to_datetime(t1[dp1], format = '%Y-%m-%d')

                        if (pd.to_datetime(t1[dp1], format = '%Y-%m-%d') <= day_min)  and (pd.to_datetime(t1[dp1], format = '%Y-%m-%d') >= pd.to_datetime(since, format = '%Y-%m-%d')):
                            day_max = pd.to_datetime(t1[dp2], format = '%Y-%m-%d')
                            day_min = pd.to_datetime(t1[dp1], format = '%Y-%m-%d')
                            imax = i
            if imax == None:
                return None, for_d
            return f[imax], day_min-timedelta(days =1)
        
        r = []
        real_since = since
        st = time.time()
        while True:
            if time.time()- st >= time_stop:
                break
            if len(r)!= 0:
                if dp1 == 3 or dp1 == 4:
                    if pd.to_datetime(r[-1].split('/')[-1].split('_')[dp1][4:], format = '%Y-%m-%d') == pd.to_datetime(since, format = '%Y-%m-%d'):
                        break
                else:
                    if pd.to_datetime(r[-1].split('/')[-1].split('_')[dp1], format = '%Y-%m-%d' )== pd.to_datetime(since, format = '%Y-%m-%d'):
                        break
            
            if for_d == 'max':      
                bigger, for_d = find_bigger(f,since ,'max')
            else:
                bigger, for_d = find_bigger(f,since, for_d)
                if len(r) != 0 and bigger == r[-1]:
                    if dp1 == 3 or dp1 == 4:
                        if gs > pd.to_datetime(since, format = '%Y-%m-%d'):
                            if gs !=  pd.to_datetime(r[-1].split('/')[-1].split('_')[dp1][4:]):
                                 r.append(min_date)                    
                            break
                        if pd.to_datetime(real_since, format = '%Y-%m-%d') < pd.to_datetime(r[-1].split('/')[-1].split('_')[dp1][4:], format = '%Y-%m-%d'):
                            since = (pd.to_datetime(since, format = '%Y-%m-%d') - timedelta(days = 1)).strftime(format = '%Y-%m-%d')
                            continue
                        if pd.to_datetime(bigger.split('/')[-1].split('_')[dp1][4:], format = '%Y-%m-%d') >= pd.to_datetime(r[-1].split('/')[-1].split('_')[dp1][4:], format = '%Y-%m-%d'):
#                             r = r[:-1]
                            break
                        else:
                            break    
                    else:
                        if gs > pd.to_datetime(since, format = '%Y-%m-%d'):
                            if gs !=  pd.to_datetime(r[-1].split('/')[-1].split('_')[dp1]):
                                 r.append(min_date)                    
                            break
                        if pd.to_datetime(real_since, format = '%Y-%m-%d') < pd.to_datetime(r[-1].split('/')[-1].split('_')[dp1], format = '%Y-%m-%d'):
                            since = (pd.to_datetime(since, format = '%Y-%m-%d') - timedelta(days = 1)).strftime(format = '%Y-%m-%d')
                            continue
                        if pd.to_datetime(bigger.split('/')[-1].split('_')[dp1], format = '%Y-%m-%d') >= pd.to_datetime(r[-1].split('/')[-1].split('_')[dp1], format = '%Y-%m-%d'):
#                             r = r[:-1]
#                              print(bigger, since, for_d)
                            
                            break
                        else:
                            break
            
            if bigger == None:
                if for_d == 'max':
                    for_d = pd.to_datetime(str(datetime.datetime.now().date()), format = '%Y-%m-%d')
                if type(for_d) == str:
                    for_d = pd.to_datetime(for_d, format = '%Y-%m-%d')
                    
                if for_d >= pd.to_datetime(since, format = '%Y-%m-%d'):
                    for_d = for_d+timedelta(days =1)
                    continue
                else:
                    break
            r.append(bigger)
        if dp1 == 3 or dp1 == 4:
            if pd.to_datetime(r[-2].split('/')[-1].split('_')[dp1][4:], format = '%Y-%m-%d') <= pd.to_datetime(r[-1].split('/')[-1].split('_')[dp1][4:], format = '%Y-%m-%d'):
                r = r[:-1]
        else:
            if pd.to_datetime(r[-2].split('/')[-1].split('_')[dp1], format = '%Y-%m-%d') <= pd.to_datetime(r[-1].split('/')[-1].split('_')[dp1], format = '%Y-%m-%d'):
                r = r[:-1]
        return r


    def find_min_date(f, dp1, dp2):
        day_min = pd.to_datetime(str(datetime.datetime.now().date()), format = '%Y-%m-%d')
        for i in range(len(f)):
            t = f[i].split('/')[-1]
            t = t.split('.')[0]
            t1 = t.split('_')
            try:
                if dp1 == 3:
                    t1[3] = t1[3][4:]
                    t1[4] = t1[4][2:]
                elif dp1 == 4:
                    t1[4] = t1[4][4:]
                    t1[5] = t1[5][2:]
            except:
                continue
            try:
                to_test = pd.to_datetime(t1[dp1], format = '%Y-%m-%d')                       
                pd.to_datetime(t1[dp1], format = '%Y-%m-%d')
            except:
                continue
            if  to_test <= day_min:
                day_min = pd.to_datetime(t1[dp1], )
                imin = i
        return f[imin], day_min.strftime(format = '%Y-%m-%d')

    if type(to_find) == list:
        file_name = to_find[0]
    else: 
        file_name = to_find
    f = find_files(start_dir, to_find)
    fn = []
    for file in f:
        file = file.replace('\\','/')
        fn.append(file)
    f = fn
    if file_name == 'short':
        dp1 = 3
        dp2 = 4
    elif 'scoring' in file_name:
        dp1 = -2
        dp2 = -1
    else:
        dp1 = 1
        dp2 = 2
        
    gs = pd.to_datetime(find_min_date(f, dp1, dp2)[1], format = '%Y-%m-%d')
    if since == 'min':
        since = find_min_date(f, dp1, dp2)[1]
        min_date = find_min_date(f, dp1, dp2)[0]
        gs = pd.to_datetime(since, format = '%Y-%m-%d')
    
    res = file_filler(f , since = since, for_d = for_d, dp1 = dp1, dp2 = dp2)
    return res


def load_files(names, usecols = None, log=False, append = False, mem_reduce = False,  tqdm_info = False):

    return load(names, usecols, log, append, mem_reduce,  tqdm_info)

   
def load(names, usecols = None, log = False, append = False, mem_reduce = False,  tqdm_info = False):
    write_user_hist('load')
    ''' names - list of filenames, cols - list of columns to load, log - boolean, write log load, mem_reduce - boolean, reduce memory usage, append - boolean, append loaded DFs, tqdm_info - boolean, add progress bar '''
    def xmlParser(path_x, usecols = None):
        f = open(path_x)
        xml_data = f.read()
        xml_data = xml_data.replace('\n','').replace('\t','')
        root = ET.XML(xml_data)
        
        def parse_okb(root):
            def parse_element(element, parsed = None):
                if parsed is None:
                    parsed = dict()
                
                for key in element.keys():
                    if ((usecols != None) and (key not in usecols)):
                        continue
                    if key not in parsed:
                        parsed[key] = element.attrib.get(key)
                    if element.text:
                        parsed[element.attrib.get(key)] = element.text
                    else:
                        pass
                    
                for child in list(element):
                    parse_element(child, parsed)
                return parsed
            
            def parse_root(root):
                return [parse_element(child) for child in list(root)]
            
            def process_data(root):
                structure_data = parse_root(root)
                return pd.DataFrame(structure_data)
            
            return process_data(root)
        
        def parse_equ(root): 
            def parse_element(element, parsed = None):
                if parsed is None:
                    parsed = dict()

                for key in element.keys():
                    if ((usecols != None) and (key not in usecols)):
                        continue
                    if key not in parsed:
                        parsed[key] = element.attrib.get(key)
                    if element.text:
                        parsed[key + '_text'] = element.text
                    else:
                        pass

                for child in list(element):
                    parse_element(child, parsed)
                return parsed

            def parse_root(root):
                return [parse_element(child) for child in list(root)]

            def process_data(root):
                structure_data = parse_root(root)
                return pd.DataFrame(structure_data)

            return process_data(root)
        
        def parse_nbki(root): 
            def parse_element(element, parsed = None):
                if parsed is None:
                    parsed = dict()

                key = element.tag
                    
                if (key not in parsed) :
                    if (usecols == None):
                        parsed[key] = element.text
                    elif (key in usecols):
                        parsed[key] = element.text
                    

                for child in list(element):
                    parse_element(child, parsed)
                return parsed

            def parse_root(root):
                return [parse_element(child) for child in list(root)]

            def process_data(root):
                structure_data = parse_root(root)
                return pd.DataFrame(structure_data)

            return process_data(root)
        
        if 'equ' in path_x:
            return parse_equ(root)
        if 'nbki'in path_x:
            return parse_nbki(root)
        else: 
            return parse_okb(root)

    def load_data(name):
        name = name.replace('\\', '//')
        d = None
        if log:
            print(name + ' is loading!')
        if ('.xlsx' in name) or ('.xls' in name) :
            d = pd.read_excel(name)
        elif '.csv' in name:
            try:
                d = csv.read_csv(name, read_options = pa.csv.ReadOptions(use_threads = True, column_names = usecols),
                    parse_options = pa.csv.ParseOptions(delimiter = ';', ignore_empty_lines = True))
                if len(d.column_names) == 1:
                    d = csv.read_csv(name, read_options = pa.csv.ReadOptions(use_threads = True, column_names = usecols),
                        parse_options = pa.csv.ParseOptions(delimiter = ',', ignore_empty_lines = True))
                        
                    if len(d.column_names) == 1:
                        d = csv.read_csv(name, read_options = pa.csv.ReadOptions(use_threads = True, column_names = usecols),
                            parse_options = pa.csv.ParseOptions(delimiter = '\t', ignore_empty_lines = True))
                            
                d = d.to_pandas(use_threads = True)
                    
            except:
                try:
                    d = pd.read_csv(name, sep=';', usecols = usecols, encoding = 'utf-8_sig')
                    if len(d.columns) == 1:
                        d = pd.read_csv(name, sep=',', usecols = usecols, encoding = 'utf-8_sig')
                except:
                    try:
                        d = pd.read_csv(name, sep=';', usecols = usecols, encoding = 'cp1251')
                        if len(d.columns) == 1:
                            d = pd.read_csv(name, sep=',', usecols = usecols, encoding = 'cp1251')
                    except:
                        print('ERROR!!! ',name)
        elif '.frt' in name or '.feather' in name:
            d = pd.read_feather(name, use_threads = True, columns = usecols)
        elif '.sav' in name or '.zsav' in name:
            d = pyreadstat.read_file_multiprocessing(pyreadstat.read_sav, name, 24)[0]
        elif '.json.gz' in name:
            d = pd.read_json(name, lines = True, compression = 'gzip')
        elif '.json' in name:
            with open(name, 'r') as fp:
                d = json.load(fp)
        elif '.pkl' in name:
            with open(name, 'rb') as fp:
                d = pickle.load(fp)
        elif '.xml' in name:
            d = xmlParser(name, usecols)
        if log:
            print(name + ' is loaded!')
        return d

    if type(names) == str:
        d = load_data(names)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if tqdm_info:
                futures = [executor.submit(load_data, n) for n in tqdm(names)]
            else:
                futures = [executor.submit(load_data, n) for n in names]
            
        d = [future.result() for future in futures]
    if append:
        if type(d) == list:
            if type(d[0]) == dict:
                    r = {}
                    for t in d:
                        r.update(t)
                    d = r
            else:
                d = [x for x in d if type(x) == pd.DataFrame]
                d = d[0].append(d[1:]).reset_index().drop(columns = 'index')
        
    
    if mem_reduce:
        if append:
            return reduce_mem_usage(d, tqdm_info)
        else:
            for i in range(len(d)):
                if type(d) != list:
                    return reduce_mem_usage(d, tqdm_info)
                d[i] = reduce_mem_usage(d[i], tqdm_info)
            return d
    else: 
        if type(d) == list and len(d) == 1:
            return d[0]
        else:
            return d


def reduce_mem_usage(df, tqdm_info = False, inplace = True):
    '''df - pd.dataframe, tqdm_info - boolean, add progress bar, inplace = boolean, changes the original DataFrame'''
    write_user_hist('reduce_mem_usage')
    if inplace == False:
        df = df.copy()
       
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    if tqdm_info:
        for_func = tqdm(df.columns)
    else:
        for_func = df.columns

    for col in for_func:
        col_type = df[col].dtype
        
        if col_type != object and col_type != '<M8[ns]':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int' or (df[col] % 1).sum() == 0:
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    try:
                        df[col] = df[col].astype(np.int8)
                    except:
                        df[col] = df[col].astype(pd.Int8Dtype())
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    try:
                        df[col] = df[col].astype(np.int16)
                    except:
                        df[col] = df[col].astype(pd.Int16Dtype())
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    try:
                        df[col] = df[col].astype(np.int32)
                    except:
                        df[col] = df[col].astype(pd.Int32Dtype())
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    try:
                        df[col] = df[col].astype(np.int64)
                    except:
                        df[col] = df[col].astype(pd.Int64Dtype())
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
#        else:
#            if df[col].unique().shape[0] < df[df[col].notna()].shape[0] / 2:
#                df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by: {:.1f} %'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def find_files(start_dir = r'D:\Обмен' , to_find = None, since = None, to = None , time_stop = 60):
    return find(start_dir, to_find, since, to , time_stop)

      
def find(start_dir = r'D:\Обмен' , to_find = None, since = None, to = None , time_stop = 60):
    '''start_dir - str, start directory to find files; to_find - str or list; substring to must be found in the file's path (sample: '.xml' or ['csv', 'NU']; since - str, start date to start search (sample: '2021-01-01' or 'min'); to - str, end date to finish search (sample: '2021-01-01' or 'max'); time_stop - int, max allowed time to search (sample: 60 (sec))'''
    write_user_hist('find')   
    start_dir = start_dir.replace('\\', '//')
    if since or to:
        return find_and_fill(start_dir, to_find, since = since, for_d = to, time_stop = time_stop)
    path_f = []
    if to_find:
        if type(to_find) == str:
            to_find = [to_find]
        find_strs = '|'.join(to_find)
        for d, dirs, files in os.walk(start_dir):
            for f in files:
                path = os.path.join(d, f)
                if re.findall(find_strs, path) == to_find:
                    path_f.append(path)
    else:        
        for d, dirs, files in os.walk(start_dir):
            for f in files:
                path = os.path.join(d, f)
                path_f.append(path)
                
    dr = []
    for d, dirs, files in os.walk(start_dir):
        dr.append(dirs)            
    
    if (len(path_f) == 0) and (len(dr) == 0):
        if to_find:
            for f in os.listdir(start_dir):
                path = os.path.join(start_dir, f)
                if re.findall(find_strs, path) == to_find:
                    path_f.append(path)
        else:
            path_f = os.listdir(start_dir)

    return path_f     

          
def save(df, save_path, file_name = None, format = 'csv', mem_reduce = False):
    '''df - DataFrame; save_path - str, directory to save; file_name - str; format - str, supported formats: csv, xlsx, frt, mem_reduce - boolean, reduce memory usage '''
    write_user_hist('save')
    save_path = save_path.replace('\\', '/')
    if (save_path[-4]  == '.' ) or (save_path[-5] == '.'):
        file_name = save_path.split('/')[-1]
        format = file_name.split('.')[1]
        file_name = file_name.split('.')[0]
        save_path = '/'.join(save_path.split('/')[:-1])
    if (save_path[-1] != '/'):
        save_path+='/'
    if (file_name != None) and (len(file_name.split('.')) == 2):
        format = file_name.split('.')[1]
        file_name = file_name.split('.')[0]
    if mem_reduce:
        df = reduce_mem_usage(df)
        
    if format == 'csv':
        df.to_csv(save_path + file_name + '.' + format, sep = ';', encoding = 'utf-8_sig', index = False)
    elif format == 'xlsx':
        df.to_excel(save_path + file_name + '.' + format, index = False)
    elif format == 'json':
        with open(save_path + file_name + '.' + format, 'w') as fp:
            json.dump(df, fp)
    elif format == 'pkl':
        with open(save_path + file_name + '.' + format, 'wb') as fp:
            pickle.dump(df, fp)
    elif (format == 'zsav') or (format == 'sav'):
        if format == 'zsav':
            pyreadstat.write_sav(df, save_path + file_name + '.' + format, compress = True)
        else:
            pyreadstat.write_sav(df, save_path + file_name + '.' + format, compress = False)
    elif (format == 'frt') or (format == 'feather'):
        while True:
            try:
                df.to_feather(save_path + file_name + '.' + format)
                break
            except Exception as e:
                print(str(e))
                if 'feather does not support serializing a non-default index for the index;' in str(e):
                    df = df.reset_index().drop(columns = ['index'])
                elif 'ApplicationID' in str(e):
                    df.ApplicationID = df.ApplicationID.astype(str)
                elif 'Conversion failed for column' in str(e): 
                    splited = str(e).split(' ') 
                    try:
                        df[splited[-4]] = bf.BruteFloat(df[splited[-4]])
                    except:
                        df[splited[-4]] = df[splited[-4]].astype(str)
                else:
                    return str(e)

    else:
        print(format, ' is unsupported format!')

       
def merge(x, y, on = None, lsuffix = '_x', rsuffix = '_y', how = 'left'):
    write_user_hist('merge')
    if (on == None) and ('LoanID' in x.columns) and ('LoanID' in y.columns):
        on = 'LoanID'
    elif (on == None) and ('ApplicationID' in x.columns) and ('ApplicationID' in y.columns):
        on = 'ApplicationID'
        
    if (type(y) == pd.core.frame.DataFrame) | ((type(y) == list) & (len(y) == 1)):
        return x.set_index(on).join(y.set_index(on), how = how, lsuffix = lsuffix, rsuffix = rsuffix).reset_index()
    elif type(y) == list:
        d = x.set_index(on).join(y[0].set_index(on), how = how, lsuffix = lsuffix, rsuffix = rsuffix).reset_index()
        for i in range(1, len(y)):
            d = d.set_index(on).join(y[i].set_index(on), how = how, lsuffix = lsuffix, rsuffix = rsuffix).reset_index()
        return d

def filter(data, like):
    '''data - pd.DataFrame, like - list or str. Set '^' before find columns if you want the columns to start with the given word (sample '^savpleCols'); use (?i) for any register *sample: (?i)('cat')) '''
    write_user_hist('filter')
    to_regex = ''
    if type(like) == list:
        for l in like:
            if type(l) == list:
                to_add = '('+''.join([f'(?=.*{i})' for i in l]) + '.*)'
                if to_regex == '':
                    to_regex += to_add
                else:
                    to_regex += '|'+to_add
            elif type(l) == str:
                if to_regex == '':
                    to_regex += f'({l})'
                else:
                    to_regex += '|'+f'({l})'
            else:
                print('Unkown type!')
    elif type(like) == str:
        to_regex += like
    else:
        print('Unkown type!')
    return data.filter(regex = to_regex)