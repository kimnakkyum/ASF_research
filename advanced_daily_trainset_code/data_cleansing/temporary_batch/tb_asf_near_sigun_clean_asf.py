import pandas as pd
import geopandas as gpd
import numpy as np
from glob import glob
import warnings
warnings.filterwarnings('ignore')
import sqlalchemy as sa
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB
from advanced_daily_trainset_code.data_cleansing.config import config

db = GetDB()

# ++ ===================================================================================================================

# 농장&야생발생 인근 시군 추출하기

# 야생 발생
query = """
    SELECT distinct on (odr)
    TO_DATE(sttemnt_de, 'YYYYMMDD') AS asf_org_day,
    ST_Transform(ST_SetSRID(ST_Point(lo, la), 4326), 5179) AS geometry
    FROM m2msys.tn_diss_occ_ntcn_info_raw
    order by odr, last_change_dt desc
"""
wild_occ = gpd.read_postgis(query, db.engine, geom_col='geometry')

# 20년 이후 야생발생 추출
wild_occ['asf_org_day'] = pd.to_datetime(wild_occ['asf_org_day'])
wild_occ = wild_occ[wild_occ['asf_org_day'] >= '2020-01-01']

# ++ ===================================================================================================================

# 농장 발생
query = sa.text("""
SELECT DISTINCT ON (frmhs_no, sttemnt_de) 
    LPAD(frmhs_no, 8, '0') AS farm_serial_no,
    TO_DATE(sttemnt_de, 'YYYYMMDD') AS asf_org_day,
    fg.geometry AS geometry
FROM asf.tn_diss_occrrnc_frmhs_raw_asf oc
join asf.tb_farm_geometry_clean_asf fg
on LPAD(oc.frmhs_no, 8, '0') = fg.farm_serial_no
and fg.geom_type = 'Point'
WHERE diss_cl = '8' 
    AND delete_at = 'N' 
    AND SLAU_AT = 'N' 
    AND CNTRL_DT IS NOT NULL   
    AND ESTBS_NM NOT LIKE '%예살%'   
    AND ESTBS_NM NOT LIKE '%출하%'
    AND ESTBS_NM NOT LIKE '%차량%'   
    AND ESTBS_NM LIKE '%차%'
""")
frmhs_occ = gpd.read_postgis(query, db.engine, geom_col='geometry')

# 20년 이후 농장발생 추출
frmhs_occ['asf_org_day'] = pd.to_datetime(frmhs_occ['asf_org_day'])
frmhs_occ = frmhs_occ[frmhs_occ['asf_org_day'] >= '2020-01-01']

# 농장&야생 발생 결합
tot = pd.concat([wild_occ[['geometry']], frmhs_occ[['geometry']]], ignore_index=True)

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

# ++ ===================================================================================================================

# 인천 강화, 대구 군위, 경기, 강원, 충북, 경북 필터링
sigungu_shp_filtered = sigungu_shp[
    (sigungu_shp['SIG_CD'].str[:2].isin(['41', '51', '43', '47'])) |
    (sigungu_shp['SIG_CD'].isin(['27720', '28710']))
    ]

# 시군명으로 dissolve
sigun_shp_filtered_dissolve = sigungu_shp_filtered.dissolve(by='sigun_nm').reset_index().drop(['SIG_ENG_NM', 'SIG_KOR_NM'], axis=1)

# 필터링 된 지역 내 농장/야생 발생 시군 파악
sigun_shp_filtered_dissolve_occ = sigun_shp_filtered_dissolve[sigun_shp_filtered_dissolve.geometry.intersects(tot.unary_union)]

# 필터링 된 지역 내 농장/야생 발생
tot = gpd.sjoin(tot, sigun_shp_filtered_dissolve_occ[['geometry']], how='inner', predicate='intersects').drop(['index_right'], axis=1)

# 필터링 된 지역 1km 버퍼, buffer는 인접 시군 간 이격이 큰 시군이 있어 넉넉하게 1km로 부여
sigun_shp_filtered_dissolve_occ['geometry'] = sigun_shp_filtered_dissolve_occ['geometry'].buffer(1000)

# 전체 시군 중, 야생 + 농장 발생 시군과 겹치는 시군
sigun_shp_filtered_dissolve_occ_intersects = sigun_shp_filtered_dissolve[sigun_shp_filtered_dissolve.geometry.intersects(sigun_shp_filtered_dissolve_occ.unary_union)]
sigun_shp_filtered_dissolve_occ_intersects['occurrence_yn'] = np.where(sigun_shp_filtered_dissolve_occ_intersects['SIG_CD'].isin(sigun_shp_filtered_dissolve_occ['SIG_CD']), 1, 0)
sigun_shp_filtered_dissolve_occ_intersects['occurrence_near_yn'] = np.where(sigun_shp_filtered_dissolve_occ_intersects['occurrence_yn'] == 0, 1, 0)

sigun_shp_filtered_dissolve_occ_intersects['farm_occurrence_max_date'] = frmhs_occ['asf_org_day'].max().strftime('%Y-%m-%d')
sigun_shp_filtered_dissolve_occ_intersects['wild_occurrence_max_date'] = wild_occ['asf_org_day'].max().strftime('%Y-%m-%d')
sigun_shp_filtered_dissolve_occ_intersects = sigun_shp_filtered_dissolve_occ_intersects[['farm_occurrence_max_date',
                                                                                         'wild_occurrence_max_date',
                                                                                         'sido_nm',
                                                                                         'sigun_nm',
                                                                                         'occurrence_yn',
                                                                                         'occurrence_near_yn',
                                                                                         'geometry']]

# DB INSERT
sigun_shp_filtered_dissolve_occ_intersects.to_postgis(
    'tb_asf_near_sigun_clean_asf',
    con=db.engine,
    if_exists='append',
    chunksize=1000,
    schema='asf',
    index=False
)