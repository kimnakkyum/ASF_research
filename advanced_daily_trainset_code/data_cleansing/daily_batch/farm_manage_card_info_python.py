import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from advanced_daily_trainset_code.data_cleansing.config import config
import os
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

db = GetDB()


###### 박승규 주무관의 전달 자료  (방역실태조사자료 기반 매칭 데이터) 불러오기
path = fr'{config.dropbox_path}\19. 2025 업무\250101 농림축산검역본부\250428_ASF 분석 데이터 구축\양돈방역관리카드_방역실태조사_매칭_결과'
file_name = 'ASF 점검실적(19.1.1~25.5.25)활용 방역관리카드 정보 현행화 250609.xlsm'
file_path = os.path.join(path, file_name)
df = pd.read_excel(file_path, sheet_name='2. 양돈방역관리카드 원천 데이터', dtype=str)



###### 유창/무창/하우스 여부 칼럼

# + 621개 존재하나 우선은 skip하고 추후에 이슈 논의 필요
df[['brd_stle_1', 'brd_stle_2', 'brd_stle_3']] = df[['brd_stle_1', 'brd_stle_2', 'brd_stle_3']].astype('Int64')

df[df[['brd_stle_1', 'brd_stle_2', 'brd_stle_3']].sum(axis=1) == 2][['brd_stle_1', 'brd_stle_2', 'brd_stle_3']] # 621개 존재

# + 3개의 칼럼 모두 값이 없는 경우 최빈 칼럼에 1로 결측치 대체
cols_1 = ['brd_stle_1', 'brd_stle_2', 'brd_stle_3']
most_common_col = df[cols_1].sum().idxmax()

def fix_brd_style(row):
    if row[cols_1].sum() == 0:
        row[most_common_col] = 1
    return row[cols_1]

df[cols_1] = df.apply(fix_brd_style, axis=1)

# + 결측치 0으로 채우기
df[cols_1] = df[cols_1].fillna(0)



######  울타리/농장입구사람소독시설/농장입구차량소독시설/퇴비장방조망등차단막설치/전실/돈사구멍메우기조밀망등설치/농장 둘레 생석회 도포/퇴비사·사료빈 주변 청결유지 및 소독 칼럼

# + 칼럼 별 방역실태조사 값으로 대체 후 결측인 농장에 대해 디폴트 Y로 결측치 대체
cols_2 = ['울타리', '농장입구사람소독시설', '농장입구차량소독시설', '퇴비장방조망등차단막설치', '전실',
          '돈사구멍메우기조밀망등설치', '농장 둘레 생석회 도포', '퇴비사·사료빈 주변 청결유지 및 소독']

cols_3 = ['intrcp_dsnfc_fclty_01_at', 'intrcp_dsnfc_fclty_02_at', 'intrcp_dsnfc_fclty_03_at', 'intrcp_dsnfc_fclty_04_at', 'intrcp_dsnfc_fclty_06_at',
          'intrcp_dsnfc_fclty_09_at', 'flfl_02_at', 'flfl_05_at']

# + 1. 현행화 칼럼 대상으로 방역실태조사 값으로 대체
for c2, c3 in zip(cols_2, cols_3):
    df[c3] = np.where(df[c2].notna(), df[c2], df[c3])

# + 2. 결측 농장 대상으로 디폴트 Y로 대체
df[cols_3] = df[cols_3].fillna('Y')

# + 3. 분석 대상 칼럼이나 현행화 대상이 아닌 칼럼 결측치 디폴트 (Y)로 대체
col_4 = [
    'intrcp_dsnfc_fclty_05_at',
    'intrcp_dsnfc_fclty_07_at',
    'intrcp_dsnfc_fclty_08_at',
    'flfl_01_at',
    'flfl_03_at',
    'flfl_04_at',
    'flfl_07_at',
    'flfl_09_at',
    'flfl_10_at'
]

df[col_4] = df[col_4].fillna('Y')



###### 양돈방역관리카드 스키마와 동일한 칼럼명 추출

# + 양돈방역관리카드 원천 정보 불러오기
farm_manage_card_info = db.query(f'''
    select *
    from m2msys.tn_aph_dsnfc_manage_frmhs_info_raw
    limit 0
''')

df_final = df[farm_manage_card_info.columns]

# + 기존 값과 자릿수 맞추기
zfill_info = {
    'frmhs_no': 8,
    'mnnm': 4,
    'slno': 4,
    'bldg_mnnm': 5,
    'bldg_slno': 5,
    'moblphon_no_1': 3,
    'tlphon_no_1': 3
}

for col, width in zfill_info.items():
    df_final[col] = df_final[col].apply(
        lambda x: str(x).zfill(width) if pd.notna(x) else pd.NA
    )



###### DB 180에 데이터 적재
password = quote_plus('Bigvalue1@3$')
DATABASE_URL = f'postgresql://postgres:{password}@10.10.12.180:5432/geoai'
db = create_engine(DATABASE_URL)
conn = db.connect()
df_final.to_sql('tn_aph_dsnfc_manage_frmhs_info_filled_clean',
             con=conn,
             if_exists='append',
             chunksize=1000,
             schema='asf',
             index=False,
             method='multi')

conn.close()