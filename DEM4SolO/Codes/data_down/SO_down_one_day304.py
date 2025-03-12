
# modules
import requests
import wget
from bs4 import BeautifulSoup as bs
import os
from tqdm import tqdm
import time
from datetime import date
from dateutil.relativedelta import relativedelta


link_parent = "https://www.sidc.be/EUI/data/L1/"
out_path = '/userhome/youn_j/Dataset/SO_FSI_2022/304_L5'

download_list = []
start_date = date(2023, 10, 16)
end_date = date(2023, 10, 16)

print("START")

while start_date <= end_date:
#     print (start_date.strftime("%Y/%m/%d"))
    
    _day = str(start_date.strftime("%Y/%m/%d"))
    link = link_parent + _day
    print(link)
    j = 1
    for i in range(5):
        try:
            page = requests.get(link, timeout=10) 
        except:
            print('retry',j,'/ 3')
            if j == 4:
                print(_day)
                raise SystemError
            j += 1
            time.sleep(10)
            continue
            
            
    html = bs(page.text, "html.parser")
    print(page.status_code) #200: successful connection , 404: Fail    
    
    if page.status_code == 404:
        start_date = start_date + relativedelta(days=1)
        continue
    _list = html.find_all('a')
    for i in range(len(_list)):
        file = _list[i]
        _name, extension = os.path.splitext(file["href"])
        if 'fsi304-image_' in _name and extension == '.fits':
            print(str(file['href']))
            download_list.append(link + "/" +  str(file['href']))
     
    start_date = start_date + relativedelta(days=1)

# download
print("{0} files will be downloaded...".format(len(download_list)))
for _fits in tqdm(download_list):
    try:
        wget.download(_fits, out=out_path)
    except:
        print(_fits, "is not available") 

# for j in range(len(download_list)):
