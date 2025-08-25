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


# k-means 결과 불러와서 one hot encoding 형태로 변수 변환 (clustering.eda_250808.py)


def visit_generate_name_tag(row):
    """
    row: 원핫 인코딩된 컬럼들이 포함된 한 행(Series)
    예: visit_cluster_0, visit_cluster_1, ..., visit_cluster_9
    """
    # 1인 컬럼 찾기
    onehot_cols = [c for c in row.index if c.startswith('visit_cluster_')]
    one_indices = [int(c.split('_')[-1]) for c in onehot_cols if row[c] == 1]

    if not one_indices:  # 값이 1인 컬럼이 없는 경우
        cid = 9
    else:
        cid = one_indices[0]  # 첫 번째 1 위치 사용 (여러 개면 첫 번째만)

    return model_config.cluster_name_map.get(cid, f'클러스터 {cid}')



# 학습/검증/테스트 데이터셋 load
def make_visit_cluster(df) :

    db = GetDB()

    # k-means 결과 데이터 불러오기
    cluster_df = pd.read_sql(
        '''
        SELECT * FROM asf.tb_visit_cluster_result
        ''', con=db.engine
    )

    cluster_df = df.merge(cluster_df, on='farm_serial_no', how='left')

    # '20-'23 기간에 차량방문이 존재하지 않는 데이터 새로운 클러스터 지정 (1686 농장 중 133호)
    cluster_df['cluster'] = cluster_df['cluster'].fillna(9).astype('int64')

    # # 클러스터 name tag
    # cluster_df['name_tag'] = cluster_df['cluster'].apply(generate_name_tag)

    # 클러스터 값이 0~9인 경우 원핫 인코딩
    cluster_ohe = pd.get_dummies(
        cluster_df['cluster'],
        prefix='visit_cluster',
        prefix_sep='_',
        dtype='int'
    )
    cluster_df = pd.concat([cluster_df, cluster_ohe], axis=1)

    cluster_df = cluster_df[['standard_date', 'farm_serial_no', 'asf_occurrence_yn'] + model_config.visit_col]

    return cluster_df