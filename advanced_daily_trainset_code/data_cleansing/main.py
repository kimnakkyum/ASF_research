import pandas as pd
from tqdm import tqdm
from functools import reduce
from datetime import datetime
import geopandas as gpd
import os
from glob import glob
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB
from advanced_daily_trainset_code.data_cleansing.config import config

from advanced_daily_trainset_code.data_cleansing.fix_batch.fix_batch_operator import FixBatchOperator
from advanced_daily_trainset_code.data_cleansing.yearly_batch.yearly_batch_operator import YearlyBatchOperator
from advanced_daily_trainset_code.data_cleansing.daily_batch.daily_batch_operator import DailyBatchOperator

if __name__ == '__main__' :

    db = GetDB()

    # ++ ===============================================================================================================

    # 발생일자 전일을 기준일자로 설정
    farm_occurrence_date = db.query('''
    SELECT previous_farm_occurrence_date 
    FROM asf.tb_farm_occurrence_date_clean_asf
    ''')

    # ++ ===============================================================================================================

    # 고정, 년, 월, 일 배치 돌도록 설정 필요

    # 고정 및 연 단위 데이터는 생성 완료 시 가공 테이블에 insert
    # 그리고 생성된 적이 있을 경우, load 해서 활용

    # 고정 데이터 생성
    fix_batch_operator = FixBatchOperator(db)

    # 연 데이터 생성
    yearly_batch_operator = YearlyBatchOperator(db)

    # 반경 범위 list
    radius_list = [1000, 3000, 6000, 9000, 12000]

    # ++ ===============================================================================================================

    # 고정 데이터 미 생성 시, 일회성으로 생성

    # fix_batch_operator.create_fix_dataset(radius_list) # 모든 고정 데이터 한번에 생성

    # fix_batch_operator._create_farm_elevation()
    # fix_batch_operator._create_farm_habitat_possibility()
    # fix_batch_operator._create_farm_environment_fix_batch(radius_list)
    # fix_batch_operator._create_farm_habitat_ratio_fix_batch(radius_list)

    # ++ ===============================================================================================================

    # 데이터 별 원천 확보 기간, 집계 반경이 상이하여 각각 생성

    # 농장 인근 토지 용도별 면적
    # farm_environment_year_start, farm_environment_year_end = '2019', '2025'
    # yearly_batch_operator._create_farm_environment(
    #     farm_environment_year_start,
    #     farm_environment_year_end,
    #     radius_list
    # )

    # 산림 내 멧돼지 수
    # forest_wildboar_year_start, forest_wildboar_year_end = '2019', '2023'
    # yearly_batch_operator._create_forest_wildboar_count(
    #     forest_wildboar_year_start,
    #     forest_wildboar_year_end
    # )

    # 농장 인근 멧돼지 수
    # farm_wildboar_year_start, farm_wildboar_year_end = '2019', '2025'
    # yearly_batch_operator._create_farm_wildboar_count(
    #     farm_wildboar_year_start,
    #     farm_wildboar_year_end,
    #     radius_list
    # )

    # ++ ===============================================================================================================

    # 기준일자 설정
    for standard_date in tqdm(farm_occurrence_date['previous_farm_occurrence_date']) :
        standard_date = datetime.strftime(standard_date, '%Y-%m-%d')

        # 일 데이터 생성 및 load
        daily_batch_operator = DailyBatchOperator(db, standard_date)
        daily_dataset = daily_batch_operator.load_daily_dataset(radius_list) # radius_list : 주변 (양돈, 모든) 농장 수 집계 범위

        # ++ ===========================================================================================================

        # 연 데이터 load
        year = standard_date.split('-')[0]
        yearly_dataset = yearly_batch_operator.load_yearly_dataset(year, radius_list)  # radius_list : 환경 비율(시가화, 농촌) 및 멧돼지 수 집계 범위

        # ++ ===========================================================================================================

        # 고정 데이터 load
        fix_dataset = fix_batch_operator.load_fix_dataset(radius_list) # radius_list : 환경 비율 (산림, 하천, 서식지), 최근접 서식지 거리 집계 범위

        # ++ ===========================================================================================================

        # 일/년/고정 데이터 매칭
        full_dataset = reduce(lambda x, y: pd.merge(x, y, on='farm_serial_no', how="left"),
                                [daily_dataset,
                                 yearly_dataset,
                                 fix_dataset
                                 ]
                                )

        # ++ ===========================================================================================================

        # dropbox에 저장
        full_dataset.to_csv(fr'{config.dropbox_path}\19. 2025 업무\250101 농림축산검역본부\250428_ASF 분석 데이터 구축\full_dataset_{standard_date}.csv',
                            index=False,
                            encoding='cp949')

        full_dataset.to_parquet(fr'{config.dropbox_path}\19. 2025 업무\250101 농림축산검역본부\250428_ASF 분석 데이터 구축\full_dataset_{standard_date}.parquet',
                                index=False)

        # ++ ===========================================================================================================

        # 환경 비율 관련 칼럼명 변경 필요
        full_dataset.rename(columns=config.full_dataset_column_translation, inplace=True)

        # 학습셋 칼럼 추출
        full_dataset = full_dataset[config.training_set_col]

        # 학습셋 upload
        full_dataset.to_postgis(
            'tb_trainingset_raw_asf',
            con=db.engine,
            if_exists='append',
            chunksize=1000,
            schema='asf',
            index=False
        )