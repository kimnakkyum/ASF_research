import os
from glob import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB
from advanced_daily_trainset_code.data_cleansing.config import config
from advanced_daily_trainset_code.dl_model.model_test.bio_model.v14_base_model.v14_model_config import model_config

from sklearn.preprocessing import MinMaxScaler

# ++ ===================================================================================================================

# 학습/검증/테스트 데이터셋 load
def get_input_data(train_col_cat = None) :

    db = GetDB()
    path = fr'{config.dropbox_path}\19. 2025 업무\250101 농림축산검역본부\250428_ASF 분석 데이터 구축'

    # train_col_cat = '환경' # 활용 변수 카테고리

    # ++ ===================================================================================================================

    # ASF 학습 데이터셋 로드
    base_df = gpd.read_postgis(sql='''select * from asf.tb_trainingset_raw_asf''',
                               con=db.engine,
                               geom_col='farm_coordinate')

    base_df['standard_date'] = pd.to_datetime(base_df['standard_date'])
    base_df = base_df[base_df['standard_date'] >= '2020-01-01']  # 2020년 이후 데이터만 활용
    base_df['standard_date'] = base_df['standard_date'].dt.strftime('%Y-%m-%d')

    # ++ ===================================================================================================================

    # 농장/야생 발생 시군 내 양돈 농장
    asf_analysis_target_farm = db.query('''
        select farm_serial_no from asf.tb_asf_analysis_target_farm_clean_asf
        where (occurrence_yn = 1) -- or (occurrence_near_yn = 1)
    ''')

    # 농장/야생 발생 시군 내 양돈 농장 필터링
    base_df = base_df[base_df['farm_serial_no'].isin(asf_analysis_target_farm['farm_serial_no'])]

    # 20년 이후 발생 농장 38개 매칭 완료
    print(base_df['asf_occurrence_yn'].sum())  # 38

    # ++ ===================================================================================================================

    # 컬럼 정보 생성
    mc = model_config()

    # 타입 미 지정 시, 전체 컬럼으로 학습
    if train_col_cat is None:
        train_col = (mc.wildboar_col + mc.around_env_col + mc.around_farm_col + mc.around_farm_occ_col + mc.around_wild_occ_col +
                     mc.car_visit_count_col + mc.infection_car_visit_count_col +
                     mc.breeding_yn_col + mc.breeding_cnt_col + mc.biosec_col)

    elif train_col_cat == '환경' :
        train_col = (mc.wildboar_col + mc.around_env_col + mc.around_farm_col + mc.around_farm_occ_col + mc.around_wild_occ_col)

    elif train_col_cat == '전파' :
        train_col = (mc.car_visit_count_col + mc.infection_car_visit_count_col)

    elif train_col_cat == '방역' :
        train_col = (mc.breeding_yn_col + mc.breeding_cnt_col + mc.biosec_col)

    else :
        raise Exception('정의된 카테고리 외 입력됨')

    # 타겟 컬럼 추가
    train_col += ['asf_occurrence_yn']

    # ++ ===================================================================================================================

    # 기준일자, 농장번호, 학습 변수
    train_df = base_df[['standard_date', 'farm_serial_no'] + train_col]
    train_df_null_cnt = train_df.isnull().sum().reset_index()

    # asf 농장, 야생 발생 관련 변수는 6km로 clip하고, 결측치도 6km로 대체
    asf_dist_col = ['infection_farm_top_near_distance', 'specimen_picking_top_near_distance_6month']
    for col in asf_dist_col :
        train_df[col] = np.where(train_df[col] > 6000,
                                 6000,
                                 train_df[col].fillna(6000)
                                 )

    # 기타 컬럼은 결측치를 0으로 대체
    train_df = train_df.fillna(0)

    # 기준일자, 농장번호를 제외하고 수치형 변수 타입으로 변환
    train_df[train_df.columns[2:]] = train_df[train_df.columns[2:]].astype(float)

    # 모든 값이 0인 컬럼 제외
    float_col_sum = train_df.select_dtypes('float64').sum()
    all_zero = float_col_sum[float_col_sum == 0].index
    train_df.drop(all_zero, axis=1, inplace=True)

    # 모돈 평균사육두수 재산출
    # + 데이터 이슈 존재 : 모돈 사육두수 원천 데이터 이슈로 인해 일반 평균사육두수보다 더 큰 경우 존재함
    # + 즉, 오류 데이터에 대해서 '모돈 평균사육두수 = 평균사육두수 - 비육돈 사육두수'로 재산출

    mask = train_df['average_mother_pig_breeding_livestock_count'] > train_df['average_breeding_livestock_count']

    # 계산 후 음수 방지
    updated_values = (
            train_df.loc[mask, 'average_breeding_livestock_count'] -
            train_df.loc[mask, 'average_porker_breeding_livestock_count']
    ).clip(lower=0)

    train_df.loc[mask, 'average_mother_pig_breeding_livestock_count'] = updated_values

    return train_df



    # # ++ ===============================================================================================================
    #
    # # 학습/검증/테스트(6:2:2)
    # date_list = sorted(train_df['standard_date'].unique())
    #
    # # 전체 개수
    # num_dates = len(date_list)
    #
    # # 6:2:2 비율 계산
    # train_size = round(num_dates * 0.6)
    # val_size = round(num_dates * 0.2)
    #
    # # 날짜 분할
    # train_dates = date_list[:train_size]
    # val_dates = date_list[train_size:train_size + val_size]
    # test_dates = date_list[train_size + val_size:]
    #
    # # 학습/검증/테스트 셋 구분
    # val_df, test_df = train_df[train_df['standard_date'].isin(val_dates)], train_df[train_df['standard_date'].isin(test_dates)]
    # train_df = train_df[train_df['standard_date'].isin(train_dates)]
    #
    # # ++ ===============================================================================================================
    #
    # # 스케일링
    # scaler = MinMaxScaler()
    # train_df.iloc[:, 2:-1] = scaler.fit_transform(train_df.iloc[:, 2:-1])
    # val_df.iloc[:, 2:-1] = scaler.transform(val_df.iloc[:, 2:-1])
    # test_df.iloc[:, 2:-1] = scaler.transform(test_df.iloc[:, 2:-1])
    #
    # # print(train_df['asf_org'].value_counts(),
    # #       val_df['asf_org'].value_counts(),
    # #       test_df['asf_org'].value_counts())
    #
    # return train_df, val_df, test_df