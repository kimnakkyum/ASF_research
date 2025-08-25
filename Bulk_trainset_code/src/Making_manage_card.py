import pandas as pd
import psycopg2
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from Bulk_trainset_code.config import config

from Bulk_trainset_code.GetData.Dataloader import GetData

from Bulk_trainset_code.src.base_farm_info import FarmFilter

def manage_card(manage, base_farm_info, pig_breeding_yn, avg_brd_had_co):
    # 데이터베이스 연결 및 쿼리 실행

    pig_breeding_yn['farms_no'] = pig_breeding_yn['farms_no'].astype('str').str.zfill(8)
    avg_brd_had_co['farms_no'] = avg_brd_had_co['farms_no'].astype('str').str.zfill(8)

    # pig_breeding_yn['std_dt'] = pd.to_datetime(pig_breeding_yn['std_dt'])
    # avg_brd_had_co['std_dt'] = pd.to_datetime(avg_brd_had_co['std_dt'])

    avg_brd_had_co['avg_brd_had_co'].fillna(0, inplace=True)

    farms = base_farm_info.merge(avg_brd_had_co, how='left').merge(pig_breeding_yn, how='left')

    # farms = pd.concat([pig_breeding_yn, avg_brd_had_co[['xmin_', 'ymin', 'avg_brd_had_co']]], axis=1)

    manage = manage[config.manange_col + ["FRMHS_NO", "LAST_CHANGER_DT"]]

    manage = manage.drop(columns="LAST_CHANGER_DT")

    manage.replace({'Y': '1', 'N': '0'}, inplace=True)

    df = farms.copy()

    # 'std_dt'를 날짜형으로 변환
    df['std_dt'] = pd.to_datetime(df['std_dt'])

    # farms_no별로 가장 최신의 std_dt 값을 가진 행 추출
    latest_df = df.loc[df.groupby('farms_no')['std_dt'].idxmax()]

    # 농장정보에 card 병합
    manage.rename(columns={'FRMHS_NO': 'farms_no'}, inplace=True)

    manage['farms_no'] = manage['farms_no'].astype('str').str.zfill(8)

    latest_df = latest_df.merge(manage, how='left', on='farms_no')

    data = latest_df[['common_at', 'black_at', 'breeding_at', 'boar_at', 'xmin_', 'ymin', 'avg_brd_had_co']]

    # 정규화 진행
    scaler = MinMaxScaler()
    data_scale = scaler.fit_transform(data)
    # scaler.fit_transform

    # 시각화 후 지역별로 5개로 분화되는 k로 결정
    k = 25

    # 그룹 수, random_state 설정
    model = KMeans(n_clusters=k, random_state=42)

    # 정규화된 데이터에 학습
    model.fit(data_scale)

    # 클러스터링 결과 각 데이터가 몇 번째 그룹에 속하는지 저장
    latest_df['cluster'] = model.fit_predict(data_scale)

    # 클러스터링 별 최빈값으로 결측치 대체
    group_col = 'cluster'  # 그룹화할 컬럼 지정

    def mode_without_minus_one(x):
        x_filtered = x[x != -1]  # -1을 제외한 값만 선택
        return x_filtered.mode().iloc[0]

    modes = latest_df.groupby(group_col)[config.manange_col].apply(mode_without_minus_one)

    # 각 컬럼의 결측치를 그룹별 최빈값으로 채움
    for col in config.manange_col:
        for group_value, row in modes.iterrows():
            mask = (latest_df[group_col] == group_value) & (latest_df[col] == -1)
            latest_df.loc[mask, col] = row[col]

    latest_df = latest_df.fillna(0)  # 클러스터에 모든 값이 결측치인 경우
    final_card = latest_df[['farms_no'] + config.manange_col]


    return final_card