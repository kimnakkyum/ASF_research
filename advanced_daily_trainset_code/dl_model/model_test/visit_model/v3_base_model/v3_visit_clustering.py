import os
from glob import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB
from advanced_daily_trainset_code.data_cleansing.config import config
from advanced_daily_trainset_code.dl_model.model_test.visit_model.v3_base_model.v3_model_config import model_config
from advanced_daily_trainset_code.dl_model.model_test.visit_model.v3_base_model.v3_make_feature import build_features_rolling_by_dates

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu
import joblib



# ++ ===================================================================================================================


# k-means 결과 불러와서 one hot encoding 형태로 변수 변환 (clustering.eda_250808.py)


def generate_name_tag(row):
    onehot_cols = [c for c in row.index if c.startswith('visit_cluster_')]
    # 값이 문자열일 수도 있으니 float 변환 후 비교
    one_indices = [
        int(c.split('_')[-1])
        for c in onehot_cols
        if float(row[c]) == 1
    ]

    if not one_indices:
        cid = 9
    else:
        cid = one_indices[0]

    return model_config.cluster_name_map.get(cid, f'클러스터 {cid}')




# 학습/검증/테스트 데이터셋 load
def make_cluster(df) :

    db = GetDB()

    # visit_df = db.query('''
    # WITH
    # target_farm AS (
    #     SELECT distinct a.farm_serial_no FROM (SELECT * FROM asf.tb_trainingset_raw_asf where standard_date >= '2020-01-01' and standard_date <= '2025-03-31') a
    #     JOIN (select farm_serial_no from asf.tb_asf_analysis_target_farm_clean_asf where (occurrence_yn = 1)) b
    #     ON a.farm_serial_no = b.farm_serial_no
    # ),
    # tot AS (
    #     SELECT
    #         v.frmhs_no,
    #         DATE(v.visit_de) AS visit_date,
    #         v.visit_ty,
    #         v.regist_no
    #     FROM m2msys.tn_visit_info_clean v
    #     JOIN target_farm tf
    #         ON v.frmhs_no = tf.farm_serial_no
    #     WHERE v.visit_de BETWEEN '2021-01-01' AND '2025-03-31'
    # )
    #
    # SELECT
    #     frmhs_no,
    #     visit_date,
    #     regist_no,
    #     -- 목적 코드별 방문 횟수
    #     SUM(CASE WHEN visit_ty = '01' THEN 1 ELSE 0 END) AS livestock_transport_car_visit_count,
    #     SUM(CASE WHEN visit_ty = '03' THEN 1 ELSE 0 END) AS animal_medicine_transport_car_visit_count,
    #     SUM(CASE WHEN visit_ty = '04' THEN 1 ELSE 0 END) AS feed_transport_car_visit_count,
    #     SUM(CASE WHEN visit_ty = '05' THEN 1 ELSE 0 END) AS livestock_excreta_transport_car_visit_count,
    #     SUM(CASE WHEN visit_ty = '06' THEN 1 ELSE 0 END) AS beddingstraw_transport_car_visit_count,
    #     SUM(CASE WHEN visit_ty = '07' THEN 1 ELSE 0 END) AS compost_transport_car_visit_count,
    #     SUM(CASE WHEN visit_ty = '08' THEN 1 ELSE 0 END) AS animal_medical_car_visit_count,
    #     SUM(CASE WHEN visit_ty = '09' THEN 1 ELSE 0 END) AS artificial_insemination_car_visit_count,
    #     SUM(CASE WHEN visit_ty = '10' THEN 1 ELSE 0 END) AS consulting_car_visit_count,
    #     SUM(CASE WHEN visit_ty = '14' THEN 1 ELSE 0 END) AS roughage_transport_car_visit_count,
    #     SUM(CASE WHEN visit_ty = '18' THEN 1 ELSE 0 END) AS livestock_breeding_facilities_management_car_visit_count,
    #     SUM(CASE WHEN visit_ty = '19' THEN 1 ELSE 0 END) AS livestock_carcass_transport_car_visit_count
    #
    # FROM tot
    # GROUP BY frmhs_no, visit_date, regist_no
    # ORDER BY frmhs_no, visit_date, regist_no;
    #     ''')
    #
    # visit_df.to_parquet(r"D:\주식회사빅밸류 Dropbox\김낙겸\19. 2025 업무\250101 농림축산검역본부\250428_ASF 분석 데이터 구축\eda\군집분석\전파변수_클러스터링\visit_val_df.parquet")


    # # 차량 데이터 불러오기
    # visit_df = pd.read_parquet(r"D:\주식회사빅밸류 Dropbox\김낙겸\19. 2025 업무\250101 농림축산검역본부\250428_ASF 분석 데이터 구축\eda\군집분석\전파변수_클러스터링\visit_val_df.parquet")


    # 학습셋 변수 불러오기
    train_df = df[(df['standard_date'] >= '2020-01-01') & (df['standard_date'] <= '2023-12-31')]

    cluster_df = pd.read_sql(
        '''
        SELECT * FROM asf.tb_visit_cluster_result 
        ''', con=db.engine
    )

    train_df = train_df.merge(cluster_df, on='farm_serial_no', how='left')

    train_df = train_df[['standard_date', 'farm_serial_no', 'asf_occurrence_yn', 'cluster']]



    # 학습셋 변수 생성하기
    val_df = df[(df['standard_date'] >= '2024-01-01') & (df['standard_date'] <= '2025-03-31')]
    dates = val_df['standard_date'].unique().copy()

    # # 4년 윈도우에 따른 검증셋 변수 생성하기
    # val_df = build_features_rolling_by_dates(visit_df, val_df, dates)
    #
    # val_df.to_parquet(r"D:\주식회사빅밸류 Dropbox\김낙겸\19. 2025 업무\250101 농림축산검역본부\250428_ASF 분석 데이터 구축\eda\군집분석\전파변수_클러스터링\val_cluster_df.parquet")


    val_df_cluster = pd.read_parquet(r"D:\주식회사빅밸류 Dropbox\김낙겸\19. 2025 업무\250101 농림축산검역본부\250428_ASF 분석 데이터 구축\eda\군집분석\전파변수_클러스터링\val_cluster_df.parquet")
    val_df_cluster['frmhs_no'] = val_df_cluster['frmhs_no'].astype(str).str.zfill(8)
    val_df_cluster['standard_date'] = val_df_cluster['standard_date'].astype(str)

    val_df = val_df.merge(val_df_cluster, left_on = ['farm_serial_no', 'standard_date'], right_on = ['frmhs_no', 'standard_date'], how='left')

    feature = ['daily_avg_livestock_transport_car_visit_count',
               'daily_avg_animal_medicine_transport_car_visit_count',
               'daily_avg_feed_transport_car_visit_count',
               'daily_avg_livestock_excreta_transport_car_visit_count',
               'daily_avg_compost_transport_car_visit_count',
               'daily_avg_other',
               'cars_per_day',
               'livestock_transport_car_visit_count_diff_days_clipped_diff_days_mean',
               'animal_medicine_transport_car_visit_count_diff_days_clipped_diff_days_mean',
               'feed_transport_car_visit_count_diff_days_clipped_diff_days_mean',
               'livestock_excreta_transport_car_visit_count_diff_days_clipped_diff_days_mean',
               'compost_transport_car_visit_count_diff_days_clipped_diff_days_mean'
               ]

    val_df = val_df[['standard_date', 'farm_serial_no', 'asf_occurrence_yn'] + feature]


    # k-means 모델 불러오기
    loaded_model = joblib.load(r"D:\주식회사빅밸류 Dropbox\김낙겸\19. 2025 업무\250101 농림축산검역본부\250428_ASF 분석 데이터 구축\eda\군집분석\전파변수_클러스터링\kmeans_model.pkl")

    # 검증셋에 predict 적용
    # NaN 여부 확인
    mask_no_nan = val_df[feature].notna().all(axis=1)  # feature 전부 NaN이 아닌 행
    mask_nan = ~mask_no_nan  # 하나라도 NaN이 있는 행

    # NaN 없는 데이터에만 predict 적용
    val_df.loc[mask_no_nan, 'cluster'] = loaded_model.predict(val_df.loc[mask_no_nan, feature])

    val_df = val_df[['standard_date', 'farm_serial_no', 'asf_occurrence_yn', 'cluster']]
    train_df = train_df[['standard_date', 'farm_serial_no', 'asf_occurrence_yn', 'cluster']]

    cluster_df = pd.concat([train_df, val_df])


    # 분석 기간에 차량방문이 존재하지 않는 데이터 새로운 클러스터 지정
    cluster_df['cluster'] = cluster_df['cluster'].fillna(9).astype('int64')

    # 클러스터 값이 0~9인 경우 원핫 인코딩
    cluster_ohe = pd.get_dummies(
        cluster_df['cluster'],
        prefix='visit_cluster',
        prefix_sep='_',
        dtype='int'
    )
    cluster_df = pd.concat([cluster_df, cluster_ohe], axis=1)

    cluster_df = cluster_df[['standard_date', 'farm_serial_no', 'asf_occurrence_yn'] + model_config.visit_col]

    # # 클러스터 name tag
    # cluster_df['name_tag'] = cluster_df.apply(generate_name_tag, axis=1)

    return cluster_df