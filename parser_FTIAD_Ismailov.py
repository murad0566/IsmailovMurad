#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup

while True:
    url_ex = input('')
    if not url_ex.isnumeric():       
        break
    else:
        print('Enter only string, please')
        
def get_links(url):
    responce = urlopen(url)
    path = responce.read().decode('utf-8')
    tree = BeautifulSoup(path)
    
    links = set()
    for link in tree.find_all('a'):
        if link.has_attr('href'):
            s = link.get('href')
            if (s.startswith('https')):
                links.add(link.get('href'))
    return links


level = int(input())
start = 0


all_links = get_links(url_ex)
while start < level:
    for i in all_links:
        try:
            nowlinks = get_links(i)
            all_links = all_links | nowlinks 
        except:
            print('Not Found')

    start += 1
    
links_num = []

for elem in all_links:
    for i in range(len(all_links)):
        att = f'{i} {elem}'
        links_num.append(att)
        
with open('urls.txt', 'w') as f:
    for s in links_num:
        f.write(s + '\n')
        

f = open('urls.txt', 'w')

def get_page(k,url): 
    response = requests.get(url)
    f = open(f'data/{k + 1}.html', 'w')
    f.write(response.text)
    f.close()


# In[43]:





# In[ ]:




