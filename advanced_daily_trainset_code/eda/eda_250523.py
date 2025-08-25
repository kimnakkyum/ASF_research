import pandas as pd
import numpy as np
import geopandas as gpd
import os
from tqdm import tqdm
from glob import glob
from advanced_daily_trainset_code.data_cleansing.config import config
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB
import sqlalchemy as sa
import warnings
warnings.filterwarnings("ignore")
from itertools import product
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
import matplotlib.ticker as mtick

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

db = GetDB()
path = fr'{config.dropbox_path}\19. 2025 업무\250101 농림축산검역본부\250535_기상청데이터_API'

# ++ ===================================================================================================================

# 강수량 데이터 확인
rainfall_count = db.query('''
    select time, count(*) from dm.climate_rainfall
    group by time
    order by time
''')
rainfall_count['time'] = rainfall_count['time'].dt.strftime('%Y-%m-%d')
date_range = pd.date_range(start = '2020-01-01', end = '2025-04-30', freq = 'D').strftime('%Y-%m-%d')
dif_date = pd.DataFrame(list(set(date_range) - set(rainfall_count['time'])), columns = ['date'])
dif_date.sort_values(by='date', inplace=True)

api_date_list = [i.split('_')[-1].split('.')[0] for i in glob(fr'{path}/rainfall/*.csv')]
print(set(dif_date['date']) - set(api_date_list), set(api_date_list) - set(dif_date['date']))

# 결측이 제거 안된 날짜가 있는지 확인. 해당 날짜에 대해서 다시 api 조회해볼 필요가 있을 듯
rainfall_full_count_date = rainfall_count[rainfall_count['count'] == 4198401]
rainfall_full_count_date.to_excel(fr'{config.dropbox_path}\19. 2025 업무\250101 농림축산검역본부\250428_ASF 분석 데이터 구축\eda\강수량\결측 없는 일자.xlsx', index=False)

rainfall_1 = db.query('''
    select * from dm.climate_rainfall
    where time = '2022-10-31'
''')

rainfall_2 = db.query('''
    select * from dm.climate_rainfall
    where time = '2025-03-31'
''')

# ++ ===================================================================================================================

# 기온 6h 데이터 확인
temperature_6h_count = db.query('''
    select time, count(*) from dm.climate_temperature_6h
    group by time
    order by time
''')
temperature_6h_count['time'] = temperature_6h_count['time'].dt.strftime('%Y-%m-%d')
date_range = pd.date_range(start = '2020-01-01', end = '2025-04-30', freq = 'D').strftime('%Y-%m-%d')
dif_date = pd.DataFrame(list(set(date_range) - set(temperature_6h_count['time'])), columns = ['date'])
dif_date.sort_values(by='date', inplace=True)

api_date_list = [i.split('_')[-1].split('.')[0] for i in glob(fr'{path}/temperature_6h/*.csv')]
print(set(dif_date['date']) - set(api_date_list), set(api_date_list) - set(dif_date['date']))

# 결측이 제거 안된 날짜가 있는지 확인. 해당 날짜에 대해서 다시 api 조회해볼 필요가 있을 듯
temperature_6h_full_count_date = temperature_6h_count[temperature_6h_count['count'] == 4198401]

# ++ ===================================================================================================================

# 기온 15h 데이터 확인
temperature_15h_count = db.query('''
    select time, count(*) from dm.climate_temperature_15h
    group by time
    order by time
''')
temperature_15h_count['time'] = temperature_15h_count['time'].dt.strftime('%Y-%m-%d')
date_range = pd.date_range(start = '2020-01-01', end = '2025-04-30', freq = 'D').strftime('%Y-%m-%d')
dif_date = pd.DataFrame(list(set(date_range) - set(temperature_15h_count['time'])), columns = ['date'])
dif_date.sort_values(by='date', inplace=True)

api_date_list = [i.split('_')[-1].split('.')[0] for i in glob(fr'{path}/temperature_15h/*.csv')]
print(set(dif_date['date']) - set(api_date_list), set(api_date_list) - set(dif_date['date']))

# 결측이 제거 안된 날짜가 있는지 확인. 해당 날짜에 대해서 다시 api 조회해볼 필요가 있을 듯
temperature_15h_full_count_date = temperature_15h_count[temperature_15h_count['count'] == 4198401]

# 아래의 날짜는 데이터가 두번씩 들어간 것 같음
temperature_15h_dup_date = temperature_15h_count[temperature_15h_count['count'] > 1_000_000]

# ++ ===================================================================================================================

# 습도 데이터 확인
humidity_count = db.query('''
    select time, count(*) from dm.climate_humidity
    group by time
    order by time
''')
humidity_count['time'] = humidity_count['time'].dt.strftime('%Y-%m-%d')
date_range = pd.date_range(start = '2020-01-01', end = '2025-04-30', freq = 'D').strftime('%Y-%m-%d')
dif_date = pd.DataFrame(list(set(date_range) - set(humidity_count['time'])), columns = ['date'])
dif_date.sort_values(by='date', inplace=True)

api_date_list = [i.split('_')[-1].split('.')[0] for i in glob(fr'{path}/humidity/*.csv')]
print(sorted(list(set(dif_date['date']) - set(api_date_list))), set(api_date_list) - set(dif_date['date']))