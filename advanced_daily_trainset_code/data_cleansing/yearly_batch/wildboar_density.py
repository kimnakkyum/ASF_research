import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import pandas as pd
from glob import glob
from advanced_daily_trainset_code.data_cleansing.config import config
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import urllib
import sqlalchemy as sa
from tqdm import tqdm

#parameter base
host="10.10.12.180"
port = '5432' # PostgreSQL 서버 호스트 (예: 'localhost')
dbname="geoai"     # 데이터베이스 이름
user="postgres"      # 사용자 이름
password="Bigvalue1@3$" # 비밀번호
password = urllib.parse.quote_plus(password)

engine = sa.create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}')

# ++ ===================================================================================================================

# 시도코드
sido_df = pd.read_csv(fr'{config.dropbox_path}\데이터연구\6. Tool\시도코드.txt', sep='|', dtype=str)

# 시군구 폴리곤 경로
sigungu_shp_paths = glob(
    fr'{config.dropbox_path}\데이터연구\1. 데이터관리\3. 데이터 수집방안(개발팀 전달)\00 데이터수집\202501\도로명주소_전자지도\*\TL_SCCO_SIG.shp')

# 시군구 폴리곤 정보 불러오기
sigungu_shp = pd.concat(
    [gpd.read_file(sigungu_shp_path, encoding='cp949') for sigungu_shp_path in sigungu_shp_paths],
    ignore_index=True)
sigungu_shp.crs = 'epsg:5179'
sigungu_shp['sido_cd'] = sigungu_shp['SIG_CD'].str[:2]
sigungu_shp['시군명'] = sigungu_shp['SIG_KOR_NM'].str.split(' ').str[0]

sigungu_shp = pd.merge(sigungu_shp, sido_df[['sido_cd', 'sido_nm']], on='sido_cd')
sigungu_shp['sigungu_nm'] = sigungu_shp['sido_nm'] + ' ' + sigungu_shp['시군명']

# 시군 단위로 dissolve
sigun_shp_dissolve = sigungu_shp.dissolve(by='sigungu_nm').reset_index()

def remove_holes(geom):
    if geom.type == 'Polygon':
        return Polygon(geom.exterior)
    elif geom.type == 'MultiPolygon':
        return MultiPolygon([Polygon(poly.exterior) for poly in geom.geoms])
    else:
        return geom

sigun_shp_dissolve['geometry'] = sigun_shp_dissolve['geometry'].apply(remove_holes)

# ++ ===================================================================================================================

# 2019 ~ 2023년도까지 데이터 생성
for year in list(map(str, range(2019, 2023 + 1))) :

    # 서식밀도 정보
    wildboar_density = pd.read_csv(fr"{config.dropbox_path}\17 2024 업무\240101 농림축산검역본부\ASF\데이터\서식밀도\서식밀도_원천\멧돼지 서식밀도_{year}.csv")

    # 마산 -> 창원 : geometry의 시군구 값과 일치를 위해
    # 청원 -> 청주 : geometry의 시군구 값과 일치를 위해
    # 연기 -> 세종 : geometry의 시군구 값과 일치를 위해
    # 북제주 -> 제주 / 남제주 -> 서귀포 : 19년 기준 전 후로 제공하는 서식밀도 지역명이 변경되어 19년 이후 기준으로 맞추는 작업 + geometry의 시군구 값과 일치를 위해
    wildboar_density['시군구'] = wildboar_density['시군구'].replace({'마산': '창원', '청원': '청주', '연기': '세종', '북제주':'제주', '남제주':'서귀포'})

    # '-' 값 null로 대체
    wildboar_density['밀도'] = np.where(wildboar_density['밀도'] == '-', np.nan, wildboar_density['밀도'])
    wildboar_density['밀도'] = wildboar_density['밀도'].astype(float)

    print('원천 데이터 내 값 존재 비율 : ', wildboar_density['밀도'].notnull().sum() / wildboar_density.shape[0]) # 원천 데이터 내 값 존재 비율 68%
    print('전체 시군 내 값 존재 비율 : ', wildboar_density['밀도'].notnull().sum() / sigun_shp_dissolve.shape[0]) # 전체 시군 내 값 존재 비율 36%

    # 시군구 명 변경으로 인해 중복이 존재할 경우, 밀도 높은 값 선택
    wildboar_density = wildboar_density.sort_values(by='밀도', ascending=False)
    wildboar_density = wildboar_density.drop_duplicates(subset=['시도', '시군구'], keep='first')

    # 서식밀도 정보와 시군 shp 매칭을 위한 매핑 테이블 생성
    match_df = pd.DataFrame()

    for idx, row in wildboar_density.iterrows():
        # 시군구 앞단어는 같고 뒷단어들이 다를 때만 매칭하도록 ('양주', '남양주' 때문에 잘못 매칭됨 ) -> 예시. 양주시 - 양주 매칭하도록
        matches = sigun_shp_dissolve[sigun_shp_dissolve['시군명'].apply(lambda x: x.startswith(row['시군구']) if x else False)]
        if not matches.empty:
            matches['시군구'] = row['시군구']
            match_df = pd.concat([match_df, matches])

    # 경상남도 고성군, 강원도 고성군으로 인해 중복 행 발생 -> 중복 제거하여 활용
    match_df = match_df.drop_duplicates(subset='sigungu_nm').reset_index(drop=True)
    print('시군구 명 중복 존재 행 : ', match_df[match_df['sigungu_nm'].duplicated(keep=False)].shape[0])

    match_df = sigun_shp_dissolve[['sido_nm', '시군명', 'sigungu_nm', 'geometry']].merge(match_df[['sigungu_nm', '시군구']], on='sigungu_nm', how='left')

    # 경상남도 고성군, 강원도 고성군으로 인해 중복 행 발생. 현재까지는 둘다 밀도가 결측이므로, 중복 제거하여 활용하면 문제 없음
    boar_density = match_df.merge(wildboar_density[['시군구', '밀도']], on='시군구', how='left')
    boar_density = boar_density.drop_duplicates(subset='sigungu_nm').reset_index(drop=True)
    boar_density.rename(columns = {'시군명' : 'sigun_nm', '밀도' : 'density'}, inplace=True)
    boar_density['year'] = year
    boar_density = boar_density[['year', 'sido_nm', 'sigun_nm', 'density', 'geometry']]
    print('원천 정제 후 전체 시군 내 값 존재 비율 : ', boar_density['density'].notnull().sum() / boar_density.shape[0])

    # 원천 데이터 db insert
    boar_density.to_postgis(
        name = 'tb_wildboar_density_raw',
        con = engine,
        schema='geoai_polygon', if_exists='append', index=False
    )

    # ++ ===================================================================================================================

    # 서식밀도를 조사하지 않은 지역에 대해서 보간 작업 수행
    boar_density_notnull = boar_density[boar_density['density'].notnull()]
    boar_density_isnull = boar_density[boar_density['density'].isnull()].drop(['density'], axis=1)

    # 인근 시군 조회를 위해 버퍼 1km. 공간이 끊겨져 있을 가능성을 대비해 넉넉하게 1km 부여
    boar_density_isnull['geometry'] = boar_density_isnull.geometry.buffer(1000)

    epoch = 1
    while boar_density_isnull.shape[0] > 0 :

        print(epoch)

        # 밀도 결측인 시군 인근의 밀도가 결측이 아닌 시군 집계
        match = gpd.sjoin(boar_density_isnull,
                          boar_density_notnull[['density', 'geometry']],
                          how = 'left',
                          predicate = 'intersects'
                          ).drop(['index_right'], axis=1)

        # 결측 보완된 건이 1개라도 있을 경우
        if match['density'].notnull().sum() > 0 :
            match_groupby = match.groupby(['year', 'sido_nm', 'sigun_nm', 'geometry'])['density'].mean().reset_index()

            # 결측 보완 성공
            match_groupby_notnull = match_groupby[match_groupby['density'].notnull()].drop(['geometry'], axis=1)
            match_groupby_notnull = pd.merge(match_groupby_notnull,
                                             boar_density[['year', 'sido_nm', 'sigun_nm', 'geometry']],
                                             on = ['year', 'sido_nm', 'sigun_nm'])
            match_groupby_notnull = gpd.GeoDataFrame(match_groupby_notnull, geometry = 'geometry', crs = 'epsg:5179')
            boar_density_notnull = pd.concat([boar_density_notnull, match_groupby_notnull], ignore_index=True)

            # 결측 보완 실패
            boar_density_isnull = match_groupby[match_groupby['density'].isnull()].drop(['density'], axis=1)
            boar_density_isnull = gpd.GeoDataFrame(boar_density_isnull, geometry = 'geometry', crs = 'epsg:5179')
            epoch+=1

        # 인근에 결측 보완 대상인 시군이 없을 경우
        else :
            boar_density_isnull = boar_density_isnull.drop(['geometry'], axis=1)
            boar_density_isnull = pd.merge(boar_density_isnull,
                                             boar_density[['year', 'sido_nm', 'sigun_nm', 'geometry']],
                                             on = ['year', 'sido_nm', 'sigun_nm'])
            boar_density_isnull = gpd.GeoDataFrame(boar_density_isnull, geometry = 'geometry', crs = 'epsg:5179')
            break

    boar_density_interp = pd.concat([boar_density_notnull, boar_density_isnull], ignore_index=True)
    boar_density_interp['density'].fillna(0, inplace=True) # 인근 시군이 존재하지 않는 경우, 0으로 대체
    print('보간 후 행 수 : ', boar_density_interp.shape[0])

    boar_density_interp.to_postgis(
        name = 'tb_wildboar_density_interp_clean',
        con = engine,
        schema='geoai_polygon', if_exists='append', index=False
    )