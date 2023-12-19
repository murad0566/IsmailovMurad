import os
from datetime import datetime, timedelta 
import time
import pickle
import hashlib

users = {'n.dimitriev' : 'Nikita',
        'r.marinov' : 'Rodion',
        'r.kudryashev' : 'Roman',
        'm.ismailov' : 'Murad',
        'e.milyukova' : 'Katya',
        'e.kornilova' : 'Katrin',
        'a.shauklis' : 'Sasha',
        'a.plotnikov' : 'Artem'}

name = users[os.getlogin()]

'''
'''



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
        t.update({hash_f.hexdigest():{'user':os.getlogin(),'time':str(datetime.now()),'modul':'adlib','func':f}})
        with open(path_hist, 'wb') as fp:
            pickle.dump(t, fp)
    except:
        pass


if '__init__':
    write_user_hist('init')
    if name == 'Nikita':
         print('Ave, Creator! Have a nice day! :)')
    elif name == 'Artem':
         print('Aloha, Borchanin! Have a nice day! :)')
    elif name == 'Rodion':
         print('Lord Optimization Director, greetings from Graph Automation Director! Have a nice day! :)')
    else:
        print(name + ', hello from Director! Have a nice day! :)')
       
    print('Last start: '+ str(datetime.now()))