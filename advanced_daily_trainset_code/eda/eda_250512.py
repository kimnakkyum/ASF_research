import pandas as pd
import numpy as np
import geopandas as gpd
import os
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

db = GetDB()
path = fr'{config.dropbox_path}\19. 2025 업무\250101 농림축산검역본부\250428_ASF 분석 데이터 구축'

# ++ ===================================================================================================================

# 농장 별로 처음 mother_pig_co, porker_co가 notnull이 된 시점
mother_porker_first_notnull_date = db.query('''
    WITH 
    base AS (
        SELECT 
            std_dt,
            frmhs_no AS farm_serial_no,
            mother_pig_co,
            porker_co
        FROM geoai_mt.tn_mobile_frmhs_blvstck_avg_hist_clean
        WHERE frmhs_no IN (
            SELECT pf.farm_serial_no 
            FROM asf.tb_pig_farm_serial_no_clean_asf pf
            JOIN asf.tb_asf_analysis_target_farm_clean_asf at
              ON pf.farm_serial_no = at.farm_serial_no
            WHERE at.occurrence_yn = 1
        )
        AND lstksp_cl LIKE '413%'
        AND bsns_sttus_se IN ('1', '2') 
        AND mgr_code IN ('104001', '104002') 
        AND master_sttus_se IN ('1', '2', '3', 'Z')
        ORDER BY farm_serial_no, std_dt
    ),
    
    mother_first_filled AS (
        SELECT DISTINCT ON (farm_serial_no)
            farm_serial_no,
            std_dt AS mother_first_filled_date
        FROM base
        WHERE mother_pig_co IS NOT NULL
        ORDER BY farm_serial_no, std_dt
    ),
    
    porker_first_filled AS (
        SELECT DISTINCT ON (farm_serial_no)
            farm_serial_no,
            std_dt AS porker_first_filled_date
        FROM base
        WHERE porker_co IS NOT NULL
        ORDER BY farm_serial_no, std_dt
    ),
    
    base_first_date AS (
        SELECT DISTINCT ON (farm_serial_no)
            farm_serial_no,
            std_dt AS base_first_date
        FROM base
        ORDER BY farm_serial_no, std_dt
    ),
    
    base_last_date AS (
        SELECT DISTINCT ON (farm_serial_no)
            farm_serial_no,
            std_dt AS base_last_date
        FROM base
        ORDER BY farm_serial_no, std_dt desc
    )
    
    SELECT 
        COALESCE(mf.farm_serial_no, pf.farm_serial_no, bfd.farm_serial_no) AS farm_serial_no,
        mother_first_filled_date,
        EXTRACT(YEAR FROM mother_first_filled_date) AS mother_first_filled_year,
        porker_first_filled_date,
        EXTRACT(YEAR FROM porker_first_filled_date) AS porker_first_filled_year,
        base_first_date,
        base_last_date
    FROM mother_first_filled mf
    FULL OUTER JOIN porker_first_filled pf
        ON mf.farm_serial_no = pf.farm_serial_no
    LEFT JOIN base_first_date bfd
        ON COALESCE(mf.farm_serial_no, pf.farm_serial_no) = bfd.farm_serial_no
    LEFT JOIN base_last_date bld
        ON COALESCE(mf.farm_serial_no, pf.farm_serial_no) = bld.farm_serial_no
    ORDER BY farm_serial_no;
''')

mother_porker_first_notnull_date[['mother_first_filled_year', 'porker_first_filled_year']] = mother_porker_first_notnull_date[['mother_first_filled_year', 'porker_first_filled_year']].astype('Int64')

mother_first_notnull_year = mother_porker_first_notnull_date.groupby(['mother_first_filled_year']).size().reset_index(name = 'mother_first_filled_year_count')
mother_first_notnull_year.rename(columns = {'mother_first_filled_year' : 'year'}, inplace=True)

porker_first_notnull_year = mother_porker_first_notnull_date.groupby(['porker_first_filled_year']).size().reset_index(name = 'porker_first_filled_year_count')
porker_first_notnull_year.rename(columns = {'porker_first_filled_year' : 'year'}, inplace=True)

match = pd.merge(mother_first_notnull_year,
                 porker_first_notnull_year,
                 on = 'year')
match = pd.melt(match, id_vars = 'year')

# 원하는 레이블 매핑 딕셔너리
label_map = {
    'mother_first_filled_year_count': '모돈사육두수',
    'porker_first_filled_year_count': '비육돈사육두수'
}
match['variable'] = match['variable'].map(label_map)
match.rename(columns = {'variable' : '구분'}, inplace=True)

plt.figure(figsize = (8, 6))
sns.barplot(data = match,
            x = 'year',
            y = 'value',
            hue = '구분'
            )
plt.title('연도별 모돈, 비육돈 사육두수 정보 최초 입력 농장 수')
plt.xlabel('연도')
plt.ylabel('농장 개수')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(fr'{path}/eda/모돈사육두수, 비육돈사육두수/연도별 모돈, 비육돈 사육두수 정보 최초 입력 농장 수.png')
plt.show()

# 날짜 변환
sub = mother_porker_first_notnull_date.copy()
sub['mother_first_filled_date'] = pd.to_datetime(sub['mother_first_filled_date'], errors='coerce')
sub['porker_first_filled_date'] = pd.to_datetime(sub['porker_first_filled_date'], errors='coerce')
sub['base_first_date'] = pd.to_datetime(sub['base_first_date'], errors='coerce')
sub['base_last_date'] = pd.to_datetime(sub['base_last_date'], errors='coerce')

# 연도 범위
years = range(2020, 2026)

results = []

for year in years:

    year_start_threshold = pd.to_datetime(f"{year}-01-01")
    year_end_threshold = pd.to_datetime(f"{year}-12-31")

    sub2 = sub[(sub['base_last_date'] >= year_start_threshold) & (sub['base_first_date'] <= year_end_threshold)]

    mother_known = sub2['mother_first_filled_date'] <= year_end_threshold
    porker_known = sub2['porker_first_filled_date'] <= year_end_threshold

    mother_null = (~mother_known)
    porker_null = (~porker_known)
    both_null = (~mother_known) & (~porker_known)

    total = len(sub2)
    print(total)
    result = {
        '연도': year,
        '모돈사육두수 결측 개수': mother_null.sum(),
        '비육돈사육두수 결측 개수': porker_null.sum(),
        'both_null 수': both_null.sum(),
        '모돈사육두수 결측률': round(mother_null.sum() / total, 4),
        '비육돈사육두수 결측률': round(porker_null.sum() / total, 4),
        'both_null 비율': round(both_null.sum() / total, 4),
    }
    results.append(result)

# 결과 데이터프레임
result_df = pd.DataFrame(results)
print(result_df)

melt = pd.melt(result_df, id_vars = '연도', value_vars = ['모돈사육두수 결측률',
                                                        '비육돈사육두수 결측률'],
               var_name = '구분', value_name = '비율')
melt['구분'] = melt['구분'].str.split(' ').str[0]
melt['비율'] = melt['비율'] * 100

plt.figure(figsize = (8, 6))
sns.barplot(data = melt,
            x = '연도',
            y = '비율',
            hue = '구분'
            )
plt.title('연도별 모돈, 비육돈 사육두수 결측률')
plt.xlabel('연도')
plt.ylabel('결측률(%)')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(fr'{path}/eda/모돈사육두수, 비육돈사육두수/연도별 모돈, 비육돈 사육두수 결측률.png')
plt.show()

# ++ ===================================================================================================================

# 일자 별 모돈, 비육돈 결측 개수 및 비율
mother_porker_notnull_count = db.query('''with 
    step01 as (
        select * from geoai_mt.tn_mobile_frmhs_blvstck_avg_hist_clean
        where frmhs_no in (
            select pf.farm_serial_no 
            from asf.tb_pig_farm_serial_no_clean_asf pf
            join asf.tb_asf_analysis_target_farm_clean_asf at
            on pf.farm_serial_no = at.farm_serial_no
            where at.occurrence_yn = 1
        )
    ),
    step02 as (
        select 
            std_dt, 
            count(mother_pig_co) as mother_pig_co_notnull_cnt, 
            count(porker_co) as porker_co_notnull_cnt,
            count(*) as cnt
        from step01 
        where bsns_sttus_se in ('1', '2') 
            and mgr_code in ('104001', '104002') 
            and master_sttus_se IN ('1', '2', '3', 'Z')
            and lstksp_cl like '413%'
        group by std_dt
        order by std_dt
    )

    select *, 
    (mother_pig_co_notnull_cnt::float / cnt) as mother_pig_co_notnull_ratio, 
    (porker_co_notnull_cnt::float / cnt) as porker_co_notnull_ratio
    from step02
''')

# 연도별 결측 비율
mother_porker_notnull_count['year'] = pd.to_datetime(mother_porker_notnull_count['std_dt']).dt.year
mother_porker_notnull_count_groupby = mother_porker_notnull_count.groupby(['year'])[
    ['mother_pig_co_notnull_ratio', 'porker_co_notnull_ratio']].mean().reset_index()

melt = pd.melt(mother_porker_notnull_count_groupby, id_vars = 'year')

# 원하는 레이블 매핑 딕셔너리
label_map = {
    'mother_pig_co_notnull_ratio': '모돈사육두수',
    'porker_co_notnull_ratio': '비육돈사육두수'
}
melt['variable'] = melt['variable'].map(label_map)
melt.rename(columns = {'variable' : '구분'}, inplace=True)

plt.figure(figsize = (8, 6))
sns.barplot(data = melt,
            x = 'year',
            y = 'value',
            hue = '구분'
            )
plt.title('연도별 모돈사육두수, 비육돈사육두수 유효값 비율')
plt.xlabel('연도')
plt.ylabel('유효값 비율')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(fr'{path}/eda/모돈사육두수, 비육돈사육두수/연도별 모돈사육두수, 비육돈사육두수 유효값 비율.png')
plt.show()

# ++ ===================================================================================================================

# 농장/야생 발생 시군 내 양돈 농장
asf_analysis_target_farm = db.query('''
    select farm_serial_no from asf.tb_asf_analysis_target_farm_clean_asf
    where (occurrence_yn = 1) -- or (occurrence_near_yn = 1)
''')

# ++ ===================================================================================================================

# 독립변수 데이터셋 로드
all_files = glob(os.path.join(path, '*.parquet'))
base_df = pd.concat([gpd.read_parquet(file) for file in all_files], ignore_index=True)
base_df['standard_date'] = pd.to_datetime(base_df['standard_date'])
base_df = base_df[base_df['standard_date'] >= '2020-01-01'] # 2020년 이후 데이터만 활용
base_df['year'] = base_df['standard_date'].dt.year

# 농장/야생 발생 시군 내 양돈 농장 필터링
base_df = base_df[base_df['farm_serial_no'].isin(asf_analysis_target_farm['farm_serial_no'])]
base_df = base_df.fillna(0)

# 환경 비율 합산 1 넘어가는 케이스 확인. 원천이 서로 달라서 생기는 문제
# 정확하게 하려면 중첩되는 면적에 대한 처리가 필요하나, 현재는 중첩 면적을 그대로 활용 중
for radius in [1000, 3000, 6000, 9000, 12000] :
    ratio_sum = (base_df[f'산림지역_{radius}'] + base_df[f'하천지역_{radius}'] + base_df[f'시가화지역_{radius}'] + base_df[f'농지지역_{radius}'])
    forest_river_sum = (base_df[f'산림지역_{radius}'] + base_df[f'하천지역_{radius}'])
    landcover_sum = (base_df[f'시가화지역_{radius}'] + base_df[f'농지지역_{radius}'])
    print(radius,
          (ratio_sum >= 1).sum() / ratio_sum.shape[0],
          (forest_river_sum >= 1).sum() / ratio_sum.shape[0],
          (landcover_sum >= 1).sum() / ratio_sum.shape[0]
          )

    # 1000 0.025247420014574862 0.0 0.0
    # 3000 0.0029149721432097605 0.0 0.0
    # 6000 0.0009873292743129835 0.0 0.0
    # 9000 0.0007757587155316298 0.0 0.0
    # 12000 0.0003291097581043278 0.0 0.0

# ++ ===================================================================================================================

# 시도 폴리곤 정보 불러오기
sido_shp_paths = glob(fr'{config.dropbox_path}\데이터연구\1. 데이터관리\3. 데이터 수집방안(개발팀 전달)\00 데이터수집\202502\도로명주소_전자지도\*\TL_SCCO_CTPRVN.shp')
sido_shp = pd.concat([gpd.read_file(sido_shp_path, encoding='cp949') for sido_shp_path in sido_shp_paths],
                        ignore_index=True)
sido_shp.crs = 'epsg:5179'

# 분석 대상 농장과 시도 폴리곤 매칭
match = gpd.sjoin(sido_shp, base_df.drop_duplicates(subset='farm_serial_no'), predicate='contains')
sido_count = match.groupby(['CTP_KOR_NM']).size().sort_values(ascending=False).reset_index(name = 'count')
sido_count['ratio'] = sido_count['count'] / sido_count['count'].sum()
sido_count = pd.merge(sido_shp, sido_count, on = 'CTP_KOR_NM')
sido_count.sort_values(by='ratio', ascending=False, inplace=True)
sido_count['sido_count_str'] = sido_count['CTP_KOR_NM'] + '\n' + '(' + sido_count['count'].astype(str) + '개)'
sido_count['sido_ratio_str'] = sido_count['CTP_KOR_NM'] + '\n' + '(' + round(sido_count['ratio'] * 100, 2).astype(str) + '%)'
sido_count.to_parquet(fr'{path}/eda/시도, 시군 내 개수/시도 별 운영, 휴업 양돈농장 수.parquet')

for year in list(range(2020, 2025 + 1)) :

    # 해당 연도의 분석 데이터
    base_df_year = base_df[base_df['year'] == year]
    base_df_year = base_df_year.drop_duplicates(subset='farm_serial_no')

    # 분석 대상 농장과 시도 폴리곤 매칭
    match_year = gpd.sjoin(sido_shp, base_df_year, predicate='contains')
    sido_count_year = match_year.groupby(['CTP_KOR_NM']).size().reset_index(name='count')
    sido_count_year['ratio'] = sido_count_year['count'] / sido_count_year['count'].sum()
    sido_count_year = pd.merge(sido_shp, sido_count_year, on='CTP_KOR_NM')
    sido_count_year.sort_values(by='ratio', ascending=False, inplace=True)
    sido_count_year['sido_count_str'] = sido_count_year['CTP_KOR_NM'] + '\n' + '(' + sido_count_year['count'].astype(str) + '개)'
    sido_count_year['sido_ratio_str'] = sido_count['CTP_KOR_NM'] + '\n' + '(' + round(sido_count['ratio'] * 100, 2).astype(str) + '%)'
    sido_count_year.to_parquet(fr'{path}/eda/시도, 시군 내 개수/{year} 시도 별 운영, 휴업 양돈농장 수.parquet')
    print(sido_count_year)

# ++ ===================================================================================================================

# 시도코드
sido_df = pd.read_csv(fr'{config.dropbox_path}\데이터연구\6. Tool\시도코드.txt', sep='|', dtype=str)

# 시군구 폴리곤 정보 불러오기
sigungu_shp_paths = glob(fr'{config.dropbox_path}\데이터연구\1. 데이터관리\3. 데이터 수집방안(개발팀 전달)\00 데이터수집\202502\도로명주소_전자지도\*\TL_SCCO_SIG.shp')
sigungu_shp = pd.concat([gpd.read_file(sigungu_shp_path, encoding='cp949') for sigungu_shp_path in sigungu_shp_paths],
                        ignore_index=True)
sigungu_shp.crs = 'epsg:5179'
sigungu_shp['sido_cd'] = sigungu_shp['SIG_CD'].str[:2]
sigungu_shp['sigun_nm'] = sigungu_shp['SIG_KOR_NM'].str.split(' ').str[0]

# 시도명 매칭
sigungu_shp = pd.merge(sigungu_shp, sido_df[['sido_cd', 'sido_nm']], on = 'sido_cd')
sigungu_shp['sido_sigun_nm'] = sigungu_shp['sido_nm'] + ' ' + sigungu_shp['sigun_nm']

# 인천 강화, 대구 군위, 경기, 강원, 충북, 경북 필터링
sigungu_shp_filtered = sigungu_shp[
    (sigungu_shp['SIG_CD'].str[:2].isin(['41', '51', '43', '47'])) |
    (sigungu_shp['SIG_CD'].isin(['27720', '28710']))
    ]

# 시군명으로 dissolve
sigun_shp_filtered_dissolve = sigungu_shp_filtered.dissolve(by='sigun_nm').reset_index().drop(['SIG_ENG_NM', 'SIG_KOR_NM'], axis=1)

# 분석 대상 농장과 시군 폴리곤 매칭
match = gpd.sjoin(sigun_shp_filtered_dissolve, base_df.drop_duplicates(subset='farm_serial_no'), predicate='contains')
sigun_count = match.groupby(['sido_sigun_nm']).size().reset_index(name = 'count')
sigun_count['ratio'] = sigun_count['count'] / sigun_count['count'].sum()
sigun_count = pd.merge(sigun_shp_filtered_dissolve, sigun_count, on = 'sido_sigun_nm')
sigun_count.sort_values(by='ratio', ascending=False, inplace=True)
sigun_count['sigun_count_str'] = sigun_count['sigun_nm'] + '\n' + '(' + sigun_count['count'].astype(str) + '개)'
sigun_count['sigun_ratio_str'] = sigun_count['sigun_nm'] + '\n' + '(' + round(sigun_count['ratio'] * 100, 2).astype(str) + '%)'
sigun_count.to_parquet(fr'{path}/eda/시도, 시군 내 개수/시군 별 운영, 휴업 양돈농장 수.parquet')

for year in list(range(2020, 2025 + 1)) :

    # 해당 연도의 분석 데이터
    base_df_year = base_df[base_df['year'] == year]
    base_df_year = base_df_year.drop_duplicates(subset='farm_serial_no')
    print(base_df_year.shape[0])

    # 분석 대상 농장과 시군 폴리곤 매칭
    match_year = gpd.sjoin(sigun_shp_filtered_dissolve, base_df_year, predicate='contains')
    sigun_count_year = match_year.groupby(['sido_sigun_nm']).size().reset_index(name='count')
    sigun_count_year['ratio'] = sigun_count_year['count'] / sigun_count_year['count'].sum()
    sigun_count_year = pd.merge(sigun_shp_filtered_dissolve, sigun_count_year, on='sido_sigun_nm')
    sigun_count_year.sort_values(by='ratio', ascending=False, inplace=True)
    sigun_count_year['sigun_count_str'] = sigun_count_year['sigun_nm'] + '\n' + '(' + sigun_count_year['count'].astype(str) + '개)'
    sigun_count_year['sigun_ratio_str'] = sigun_count_year['sigun_nm'] + '\n' + '(' + round(sigun_count_year['ratio'] * 100, 2).astype(str) + '%)'
    sigun_count_year.to_parquet(fr'{path}/eda/시도, 시군 내 개수/{year} 시군 별 운영, 휴업 양돈농장 수.parquet')
    print(sigun_count_year)

# ++ ===================================================================================================================