import os
from glob import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB
from advanced_daily_trainset_code.data_cleansing.config import config
from advanced_daily_trainset_code.dl_model.model_test.integrated_model.v18_base_model.v18_model_config import model_config

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu

# ++ ===================================================================================================================

def ostu_threshold(env_df, env_col):
    threshold = {}
    # threshold's thresholding
    for col in env_col:
        values = env_df[col].dropna().values.astype(float)
        threshold[col.replace('_rate', '')] = threshold_otsu(values)
    return threshold

def label_by_otsu_threshold(df, column, new_col_name=None):
    df = df.copy()

    # numpy 배열로 변환 (중요)
    values = df[column].dropna().astype(float).values

    # Otsu 임계값 계산
    thresh = threshold_otsu(values)

    if new_col_name is None:
        new_col_name = column.replace('_rate', '') + '_center'

    df[new_col_name] = (df[column] >= thresh).astype(int)

    return df, thresh

def label_dryfield_vs_urban(df, dryfield_col, urban_col, prefix='center_'):
    df = df.copy()

    # Otsu threshold 계산
    dryfield_values = df[dryfield_col].dropna().astype(float).values
    urban_values = df[urban_col].dropna().astype(float).values

    dryfield_thresh = threshold_otsu(dryfield_values)
    urban_thresh = threshold_otsu(urban_values)

    # 라벨 초기화
    df[prefix + 'dryfield'] = 0
    df[prefix + 'urban'] = 0

    # 행마다 비교 및 라벨 부여
    for index, row in df.iterrows():
        farm_val = row[dryfield_col]
        urban_val = row[urban_col]

        farm_valid = farm_val >= dryfield_thresh
        urban_valid = urban_val >= urban_thresh

        if farm_valid and (farm_val > urban_val):
            df.loc[index, prefix + 'dryfield'] = 1
        elif urban_valid and (urban_val > farm_val):
            df.loc[index, prefix + 'urban'] = 1
        # else: 둘 다 threshold 못 넘기거나 같으면 0

    return df, dryfield_thresh, urban_thresh

def label_none_dryfield(df, column, new_col_name=None):
    df = df.copy()

    # 조건: 아직 어떤 center도 지정되지 않은 행
    condition = (df['center_forest'] + df['center_dryfield'] + df['center_urban'] + df['center_habitat'] == 0)

    # Otsu 적용할 대상만 선택
    values = df.loc[condition, column].dropna().astype(float).values

    # 임계값 계산
    thresh = threshold_otsu(values)

    # 새로운 컬럼 이름 생성
    if new_col_name is None:
        new_col_name = column.replace('_rate', '') + '_center'

    # 기본값 0으로 초기화
    df[new_col_name] = 0

    # 조건 만족 + 값이 존재하는 경우에만 Otsu 적용
    df.loc[condition & df[column].notna(), new_col_name] = (
                df.loc[condition & df[column].notna(), column] >= thresh).astype(int)

    return df, thresh



def env_generate_name_tag(row):
    tags = []

    if row.get('center_forest', 0) == 1:
        tags.append('산림')

    if row.get('center_dryfield', 0) == 1:
        tags.append('밭농지')

    elif row.get('center_non_dryfield', 0) == 1:
        tags.append('밭이외농지')

    elif row.get('center_urban', 0) == 1:
        tags.append('시가화')

    if row.get('center_habitat', 0) == 1:
        tags.append('서식지')

    if tags:
        return '_'.join(tags)
    else:
        return '비특이환경'



# 학습/검증/테스트 데이터셋 load
def make_env_cluster(df) :

    # 환경변수 클러스터링 변수 선정
    env_df = df[['standard_date', 'farm_serial_no', 'asf_occurrence_yn'] + model_config.cluster_env_col]

    env_df.fillna(0, inplace=True)

    # 룰 베이스 적용

    # + 0. otsu's threshold 산출
    threshold = ostu_threshold(env_df, model_config.cluster_env_col)

    # + 1. 산림 중심 판정
    env_df_labeled, forest_thresh = label_by_otsu_threshold(
        env_df,
        column='radius_1km_forest_area_rate',
        new_col_name='center_forest'
    )

    # + 2. 밭/시가화 중심 판정
    env_df_labeled, dry_t, urban_t = label_dryfield_vs_urban(
        df=env_df_labeled,
        dryfield_col='radius_1km_farmland_dryfield_rate',
        urban_col='radius_1km_urban_area_rate',
        prefix='center_'
    )

    # + 3. 서식지 중심 판정
    env_df_labeled, habitat_thresh = label_by_otsu_threshold(
        env_df_labeled,
        column='radius_9km_wildboar_habitat_area_rate',
        new_col_name='center_habitat'
    )

    # + 4. 밭 이외의 농지 중심 판정 (비특이환경 대상으로만)
    env_df_labeled, none_dryfield = label_none_dryfield(
        env_df_labeled,
        column='radius_1km_farmland_none_dryfield_rate',
        new_col_name='center_non_dryfield'
    )

    env_df_labeled = env_df_labeled[['standard_date', 'farm_serial_no', 'asf_occurrence_yn'] + model_config.env_col]

    # # + 4. name tage 생성
    # env_df_labeled['name_tag'] = env_df_labeled.apply(generate_name_tag, axis=1)

    # # +_5. 원핫 인코딩
    # cluster_onehot = pd.get_dummies(env_df_labeled['cluster'], prefix='cluster')
    # cluster_onehot = cluster_onehot.astype(int)
    # env_df_labeled = pd.concat([env_df_labeled[['standard_date', 'farm_serial_no', 'asf_occurrence_yn']], cluster_onehot], axis=1)

    return env_df_labeled