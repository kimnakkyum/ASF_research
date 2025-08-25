import pandas as pd
import os
from pyproj import Transformer
from scipy.spatial import cKDTree

from Bulk_trainset_code.config import config


file_path  = os.path.join(config.FEATURE_PATH, "base_df_재산출.csv")
df = pd.read_csv(file_path, encoding='cp949')

# UTM 변환 함수 정의 (docstring 추가)
def convert_lat_lon_to_utm(latitudes, longitudes):
    """
    위도와 경도를 UTM 좌표로 변환합니다.

    Args:
        latitudes (pd.Series): 위도 값
        longitudes (pd.Series): 경도 값

    Returns:
        tuple: 변환된 x, y 좌표
    """
    transformer = Transformer.from_crs("epsg:4326", "epsg:5179", always_xy=True)
    x, y = transformer.transform(longitudes, latitudes)
    return x, y

# 좌표 변환 (모든 사용자가 이해할 수 있도록 변수명 변경)
df['longitude'], df['latitude'] = convert_lat_lon_to_utm(df['ymin'], df['xmin_'])

# 거리 계산 데이터 처리 효율성 개선
# 활용 가능하게 거리 계산 함수(process_date) 생성 후 날짜별 거리 계산 가능하도록 수정
# 거리 설정에 따른 변수명 자동 변경 가능하도록 f-string 적용
# 거리계산 함수 별도 생성(활용성 증대)
def calculate_farm_counts(df, radius):
    def process_date(date, df, radius):
        tmp = df[df['std_dt'] == date]
        coords = tmp[['longitude', 'latitude']].values
        tree = cKDTree(coords)
        counts = tree.query_ball_point(coords, r=radius)
        # 자기 자신 제외한 반경 내 개수 계산
        column_name = f'count_within_{radius // 1000}km'
        tmp[column_name] = [len(c) - 1 for c in counts]
        return tmp

    # 날짜별로 반복 처리
    unique_dates = df['std_dt'].unique()
    results = [process_date(date, df, radius) for date in unique_dates]
    # 결과 병합
    final_df = pd.concat(results)
    return final_df

# 반경 12km를 m 단위로 변환
radius = 12000

# 모든 시점에 대해 농가의 수를 계산
df_with_counts = calculate_farm_counts(df, radius)

base_farm_path  = os.path.join(config.FEATURE_PATH, "base_df_양돈농장수_12km.csv")
df_with_counts.to_csv(base_farm_path, encoding='cp949', index=False)