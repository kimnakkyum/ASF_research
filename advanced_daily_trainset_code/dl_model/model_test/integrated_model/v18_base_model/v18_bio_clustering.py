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

def add_scale_columns(df, value_col, bins, col_names):
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')  # 숫자 변환

    for i in range(len(col_names)):
        lower = bins[i]
        upper = bins[i + 1]

        # 일반적인 구간 처리
        if upper == float('inf'):
            mask = df[value_col] >= lower
        else:
            mask = (df[value_col] >= lower) & (df[value_col] < upper)

        df[col_names[i]] = 0
        df.loc[mask, col_names[i]] = 1

    return df



# 간단한 포맷을 위한 매핑 (원하는 형태로 조절 가능)
def clean_col(col_name):
    return col_name.replace('_scale_yn', '').replace('_scale_mother_pig_yn', '').replace('_mother_pig', '')

# 클러스터명 생성 함수
def bio_generate_name_tag(row):
    general = [name for col, name in model_config.general_scale_map.items() if col in row and row[col] == 1]
    mother = [name for col, name in model_config.mother_pig_scale_map.items() if col in row and row[col] == 1]

    if general and mother:
        return f"{'·'.join(general)}+{'·'.join(mother)}"
    elif general:
        return f"{'·'.join(general)}"
    elif mother:
        return f"{'·'.join(mother)}"
    else:
        return "이상치"



# 학습/검증/테스트 데이터셋 load
def make_bio_cluster(df) :

    # 평균사육두수, 모돈 평균사육두수 불러오기
    bio_df = df[['standard_date', 'farm_serial_no', 'asf_occurrence_yn'] + model_config.cluster_bio_col]

    bio_df = add_scale_columns(bio_df, 'average_breeding_livestock_count', model_config.bio_bins, model_config.bio_col)
    bio_df = add_scale_columns(bio_df, 'average_mother_pig_breeding_livestock_count', model_config.bio_mother_pig_bins, model_config.bio_mother_pig_col)

    bio_df = bio_df[['standard_date', 'farm_serial_no', 'asf_occurrence_yn'] + model_config.bio_col + model_config.bio_mother_pig_col]

    # # + 4. name tage 생성
    # bio_df['name_tag'] = bio_df.apply(generate_name_tag, axis=1)

    return bio_df