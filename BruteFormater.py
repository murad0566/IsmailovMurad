import pandas as pd
import numpy as np
from datetime import datetime, timedelta 
import time
import os
import warnings
from tqdm import tqdm
from adlib import LoadMaster as lm
warnings.filterwarnings('ignore')
import pickle

import hashlib
hash_f = hashlib.blake2b(digest_size = 10)

def write_user_hist(f):
    try:
        path_hist = r'D:\Обмен\_Скрипты\Резервные копии рабочих папок\Папка Директора по автоматизации/hist/' + str(datetime.now().date()) +'_'+ os.getlogin() + '.pkl'
        try:
            with open(path_hist, 'rb') as fp:
                t = pickle.load(fp)
        except:
            t = {}

        hash_f.update(bytes(str(time.time()), 'utf-8'))
        t.update({hash_f.hexdigest():{'user':os.getlogin(),'time':str(datetime.now()),'modul':'BruteFormater','func':f}})
        with open(path_hist, 'wb') as fp:
            pickle.dump(t, fp)
    except:
        pass

class BruteController(Exception):
    def __init__(self, message):
        super().__init__(message)
        
     
def help():
    write_user_hist('help')
    print('BruteFloat() - conver object to flaot')
    print('BruteInt() - conver object to int')
    print('BruteDate() - conver object to datetime')
    print('BruteData() - convert all coluns to therir real format')
    print('BruteReason() - convert reason from marketing_v2 to normal format reason')
    print('pretty_date() - convert datetime variable to formated date string variable')


def BruteFloat(d, err = 'raise'):
    try:
        d = d.astype(str).str.replace(',','.').replace('-', np.nan).replace('', np.nan).replace(' ', np.nan).replace('nan', np.nan).replace('<NA>', np.nan).replace('None', np.nan)
    except:
        d = str(d)
        if (len(d) == 1) and ( (d == '-') or (d == ' ') or (d == '')):
            d = np.nan
        elif (d == 'nan') or (d == '<NA>') or (d == 'None'):
            d = np.nan    
        d = d.replace(',','.').replace(' ','')
    try:
        d1 = pd.to_numeric(d)
    except:
        try:
            d1 = pd.to_numeric(d, downcast = 'integer')
        except:
            d1 = pd.to_numeric(d, errors = 'coerce')
    if type(d1) != pd.core.series.Series:
        if (err == 'raise') and ((d1 == np.nan) and (d != np.nan)):
            raise BruteController(d, 'can not conver to numeric')
        else:
            return d1
        
    if (d1.isna().sum() * 0.99) <= d.isna().sum():
        return d1
    
    else:
        if err == 'raise':
            raise BruteController('after numerisation DataFrame contains too many NAs')
        else:
            return pd.to_numeric(d, errors = 'coerce')
 

def BruteInt(x): 
    write_user_hist('BruteInt')
    try:
        d = d.astype(str).str.replace(',','.').replace('-', np.nan).replace('', np.nan).replace(' ', np.nan).replace('nan', np.nan).replace('<NA>', np.nan).replace('None', np.nan)
    except:
        d = str(d)
        if (len(d) == 1) and ( (d == '-') or (d == ' ') or (d == '')):
            d = np.nan
        elif (d == 'nan') or (d == '<NA>') or (d == 'None'):
            d = np.nan    
        d = d.replace(',','.').replace(' ','')
    try:
        d1 = pd.to_numeric(d)
    except:
        try:
            d1 = pd.to_numeric(d, downcast = 'integer')
        except:
            d1 = pd.to_numeric(d, errors = 'coerce', downcast = 'integer')
    if type(d1) != pd.core.series.Series:
        if (err == 'raise') and ((dl == np.nan) and (d != np.nan)):
            raise BruteController(d, 'can not conver to numeric')
        else:
            return d1
        
    if (d1.isna().sum() * 0.99) < d.isna().sum():
        return d1
    
    else:
        if err == 'raise':
            raise BruteController('after numerisation DataFrame contains too many NAs')
        else:
            return pd.to_numeric(d, errors = 'coerce', downcast = 'integer')
   
  
def BruteReason(data_o, inplace = False):
    write_user_hist('BruteReason')
    di_ri = {'nbki-passport' : 'Несоответствие по номеру паспорта (НБКИ)',
        'nbki-mfo-overdue' : 'Просрочка по МФО (НБКИ)',
        'nbki-bank-overdue' : 'Просрочка в банке (НБКИ)',
        'merge-passport' : 'Несоответствие по номеру паспорта (Объединенная КИ)',
        'merge-mfo-overdue' : 'Просрочка по МФО (Объединенная КИ)',
        'merge-bank-overdue' : 'Просрочка в банке (Объединенная КИ)',
        'no-credit-history': 'Нет КИ',
        'no-nbki-credit-history-length' : 'Длина КИ меньше определенной (При отсутствии КИ в НБКИ)',
        'no-nbki-overdue-days' : 'Просрочка больше определенного кол-ва дней (При отсутствии КИ в НБКИ)',
        'no-nbki-overdue-sum' : 'Просрочка больше определенной суммы (При отсутствии КИ в НБКИ)',
        'first-strategy-api-internal-scoring' : 'Первая стратегия: клиент с АПИ и внутренний скоринг больше определенного значения',
        'first-strategy-ch-length-internal-scoring' : 'Первая стратегия: длина КИ меньше определенного значения и внутренний скоринг больше определенного значения',
        'first-strategy-nbki-scoring' : 'Первая стратегия: скоринг НБКИ в определенном промежутке значений',
        'first-strategy-limits-decline' : 'Первая стратегия: отказ по скорингу после перехода на определение доступных лимитов',
        'second-strategy-internal-scoring' : 'Вторая стратегия: внутренний скоринг больше определенного значения',
        'second-strategy-api-internal-scoring' : 'Вторая стратегия: клиент пришел с АПИ и внутренний скоринг больше определенного значения',
        'second-strategy-ch-length-internal-scoring' : 'Вторая стратегия: длина КИ меньше определенного значения и внутренний скоринг больше определенного значения',
        'second-strategy-nbki-scoring' : 'Вторая стратегия: скоринг НБКИ в определенном промежутке значений',
        'second-strategy-limits-decline' : 'Вторая стратегия: отказ по скорингу после перехода на определение лимитов',
        'third-strategy-nbki-scoring' : 'Третья стратегия: скоринг в определенном промежутке значений',
        'third-strategy-no-nbki-scoring-internal-scoring' : 'Третья стратегия: нет скоринга НБКИ и внутренний скоринг больше определенного значения',
        'third-strategy-api-internal-scoring' : 'Третья стратегия: клиент пришел с АПИ и внутренний скоринг больше определенного значения',
        'third-strategy-ch-length-internal-scoring' : 'Третья стратегия: длина КИ меньше определенного значения и внутренний скоринг больше определенного значения',
        'third-strategy-limits-decline' : 'Третья стратегия: отказ по внутреннему скорингу после перехода на определение лимитов',
        'nbki-mfo-history-overdue' : 'Историческая просрочка по МФО (НБКИ)',
        'nbki-mfo-history-enter-overdue' : 'Историческое вхождение в просрочку по МФО (НБКИ)',
        'nbki-bank-history-overdue' : 'Историческая просрочка в банке (НБКИ)',
        'nkbi-bank-history-enter-overdue' : 'Историческое вхождение в просрочку в банке (НБКИ)',
        'merge-mfo-history-overdue' : 'Историческая просрочка по МФО (Объединенная КИ)',
        'merge-mfo-history-enter-overdue' : 'Историческое вхождение в просрочку по МФО (Объединенная КИ)',
        'merge-bank-history-overdue' : 'Историческая просрочка в банке (Объединенная КИ)',
        'merge-bank-history-enter-overdue' : 'Историческое вхождение в просрочку в банке (Объединенная КИ)',
        'third-strategy-overdue-sum-and-days' : 'Третья стратегия: просрочка больше 10 дней и суммарная задолженность больше 2к',
        'third-strategy-ch-length' : 'Третья стратегия: длина КИ меньше определенного значения',
        'second-strategy-ch-length' : 'Вторая стратегия: длина КИ меньше определенного значения',
        'first-strategy-ch-length' : 'Первая стратегия: длина КИ меньше определенного значения',
        'model-old-user-stop-factor-1' : 'Стоп-фактор 1. Просрочка у нас больше 31 дня',
        'model-old-user-stop-factor-2-1' : 'Стоп-фактор 2. Наличие текущей просрочки 1 стратегия',
        'model-old-user-stop-factor-2-2' : 'Стоп-фактор 2. Наличие текущей просрочки 2 стратегия',
        'model-old-user-stop-factor-2-random-variable' : 'Стоп-фактор 2. Наличие текущей просрочки. Рандомная переменная',
        'model-old-user-stop-factor-2-after-check-scoring' : 'Стоп-фактор 2. Наличие текущей просрочки. После проверки скоринга',
        'model-old-user-stop-factor-3-our-overdue-exists' : 'Проверка просрочки до 31 дня в последнем нашем займе (3 стратегия). Есть просрочка.',
        'model-old-user-stop-factor-3-our-overdue-not-exists' : 'Проверка просрочки до 31 дня в последнем нашем займе (3 стратегия). Нет просрочки.',
        'model-old-user-less-overdue-bank' : 'Наличие текущей просрочки в банке от 1 до 30 дней',
        'model-old-user-less-overdue-microloan' : 'Наличие текущей просрочки в микрозаймах от 1 до 30 дней',
        'model-old-user-scoring-after-less-overdue-exists' : 'Проверка скоринга при наличии просрочки от 1 до 30 (90) дней',
        'model-old-user-stop-factor-3-1' : 'Просрочка у нас больше 10 дней и сумма просрочки в банке больше 5000',
        'model-old-user-stop-factor-3-2' : 'Просрочка у нас больше 10 дней и сумма просрочки в МФО больше 5000',
        'merge-length-ki-overdue-sum' : 'Отказ по смерженной КИ при длине меньше определенного значения и суммарной просрочке больше граничных условий по МФО и банкам',
        'nbki-length-ki-overdue-sum' : 'Отказ по НБКИ КИ при длине меньше определенного значения и суммарной просрочке больше граничных условий по МФО и банкам',
        'identification-megafon-decline' : 'Отказ по длине ки или скор баллу и времени жизни абонента в мегафон',
        'ch-length-lt-then' : 'Длина КИ меньше определенного значения',
        'overdue-sum-mfo-and-bank' : 'Сумма просрочки МФО до 30 и в банке до 90 более 5000',
        'passport-number' : 'Найден такой же паспорт из бки у другого заёмщика',
        'passport-year' : 'Несоотвествия года паспорта при несоответствии по номеру',
        'megafon-no-data': 'Мегафон нет данных',
        'megafon-lifetime' : 'Мегафон не прошла проверка LIFETIME',
        'megafon-block-cnt' : 'Мегафон не прошла проверка BLOCK_CNT',
        'megafon-block-dur' : 'Мегафон не прошла проверка BLOCK_DUR',
        'megafon-all-clc' : 'Мегафон не прошла проверка ALL_CLC',
        'megafon-score' : 'Мегафон не прошла проверка SCORE или LIFETIME',
        'megafon-circle' : 'Мегафон не прошла проверка CIRCLE',
        'model-old-user-scoring-more-10-decile-after-random-variable' : 'Отказ по скорингу (>= 10 дециль)',
        'model-old-user-scoring-less-7-decile-after-random-variable' : 'Отказ по скорингу (< 7 дециль) после генерации рандомной переменной',
        'model_1_0919-score-decline' : 'Отказ по скорингу',
        'identification-tele2-decline' : 'Отказ по скорингу tele2',
        'model-old-user-stop-factor-4-1-common-overdue' : 'Просрочка в последнем займе больше 31 дня',
        'model-old-user-stop-factor-4-1-our-overdue-and-mfo-plus-bank-sum' : 'Просрочка по последнему займу у нас более 10 дней и текущая просрочка в МФО и/или не МФО более 5000 руб',
        'model-old-user-stop-factor-4-1-strategy-1_decile-more-9' : 'Первая стратегия. Отказ, если дециль > 9 после генерации рандомной переменной',
        'model-old-user-stop-factor-4-1-strategy-2-decile-more-6' : 'Вторая стратегия. Отказ, если дециль > 6 после текущей просрочки',
        'model-old-user-stop-factor-4-1-strategy-2_decile-more-9' : 'Вторая стратегия. Отказ, если дециль > 9 после текущей просрочки',
        'model-old-user-stop-factor-4-1-strategy-2_decile-more-9_after_limit_by_rules' : 'Вторая стратегия. Отказ, если дециль > 9 после лимита по правлам',
        'model-old-user-stop-factor-1' : 'Стоп-фактор 1. Просрочка у нас больше 10 дней',
        'model-old-user-stop-factor-5-1-common-overdue' : 'Просрочка более 30 дней',
        'model-old-user-stop-factor-5-2-scoring-point-more' : 'Отказ по скорингу',
        'model_new_user_scoring_more_than_1_1220_1_1' : 'Отказ по скорингу',
        'experiment_overdue_mfo_bank_flag' : 'Отказ по экспериментальному флагу просрочки в МФО/Банке',
        'fin-card-not-found': 'Не найдена финкарта','fin-karta-rate-more-than-8': 'FinKartaRate < 8',
        'fin-karta-score-v3-more-than-037381' : 'FinKartaScoreV3 < 0.37381',
        'overdue-mfo-bank-flag' : 'Просрочка в банке или МФО',
        'scoring.phone.not-found' : 'Телефон заемщика не найден в qiwi',
        'qiwi-score' : 'Скорбалл Qiwi больше порогового значения',
        'closed-mfo-share-90' : 'Отношение активных займов к закрытым займам за последние 90 дней равно нулю',
        'unproblem_mfo_amount_and_closed_mfo_amount_ratio' : 'Отношение активных займов к закрытым займам за последние 90 дней равно нулю',
        'total_overdue_bank_mfo_flag' : 'Суммарный долг в МФО с просрочкой 1-1095 дней + Суммарный долг в Банках с просрочкой 1-1825 дней > 0',
        'new_user_left_branch_scoring_more_than' : 'Отказ по скорингу',
        'mfo-overdue-sum-days' : 'Есть просрочка в МФО до 30 дней > 20000 руб',
        'mfo-overdue-sum-days-and-current-overdue' : 'Есть просрочка в МФО до 30 дней > 20000 руб и текущая просрочка > 5000 руб и займов, закрытых у нас, < 5'}
    def get_description(reas):
        try:
            return di_ri[reas]
        except:
            return 'Not found!'
    if inplace:
        data = data_o
    else:
        data = data_o.copy()
    for reason in data['Причина отказа'].unique():
        data.loc[data['Причина отказа'] == reason, 'Причина отказа'] = get_description(reason)
    return data


def BruteDate(x, rec = 1):  
    
    formats = ['%d-%b-%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%d.%m.%Y %H:%M', '%d-%m-%Y %H:%M:%S', '%d.%m.%y %H:%M:%S', '%d.%m.%Y', '%Y%m%d', '%b %Y', '%m/%d/%Y %H:%M:%S' , '%m/%d/%Y', '%d/%m/%Y']
    try_cnt = 0
    try:
        x = x.replace(' ', pd.NaT).replace('', pd.NaT).replace('0', pd.NaT).replace('-', pd.NaT)
    except:
        if x == ' ':
            return pd.NaT
    def date_convert(x, format):
        return pd.to_datetime(x, format = format)
    
    while True:
        try:
            return date_convert(x, formats[try_cnt])
        except:
            try_cnt+=1
            if  try_cnt -1 == len(formats):
                if (type(x) == pd.core.series.Series) and rec <= 2:
                    x = x.to_frame()
                    feature_name = x.columns[0]
                    filter_date = (x[feature_name].str.contains(r'[a-z]', regex = True))
                    filter_date = filter_date.fillna(False)
                    x.loc[filter_date, feature_name] = BruteDate(x.loc[filter_date][feature_name], rec+1)
                    x.loc[~filter_date, feature_name] = BruteDate(x.loc[~filter_date][feature_name], rec+1)
                    x = x[feature_name]
                   
                    return pd.to_datetime(x)
                    
                x = x.astype(str).str.replace('', '').replace('-', '').replace('', '').replace(' ','').replace('nan', '').replace('<NA>','').replace('None', '').replace('NaT', '')
                x = x.apply(BruteDate)
                return x

      
def BruteData(df, mem_reduce = False, tqdm_info = False, inplace = True):
    '''df - DataFrame, mem_reduce - bool, reduce memory usage, tqdm_info - boolean, add progress bar, inplace - boolean'''
    write_user_hist('BruteData')
    if inplace == False:
        df = df.copy()
    if type(df) == str:
        try:
            return BruteFloat(df)
        except:
            return BruteDate(df)



    try:
        for_func = df.columns    
        if tqdm_info:
            for_func = tqdm(df.columns)
            
        for col in for_func:
            if df[col].dtype == 'object':
                try:
                    df[col] = BruteFloat(df[col])
                except:
                    try:
                        df[col] = BruteDate(df[col])
                    except:
                        pass
    except:
        if df.dtype == 'object':
            try:
                df = BruteFloat(df)
            except:
                try:
                    df = BruteDate(df)
                except:
                    pass
    

    if mem_reduce:
        df = lm.reduce_mem_usage(df)
    return df


def pretty_date(x, format = '%B %Y'):
    write_user_hist('pretty_date')
    try:
        return x.dt.strftime(format) 
    except:
        return x.strftime(format) 