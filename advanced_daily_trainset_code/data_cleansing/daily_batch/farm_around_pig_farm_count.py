from pyproj import Transformer
from scipy.spatial import cKDTree
import geopandas as gpd
import pandas as pd

# +++ [이슈사항 정리] +++ ================================================================================================
# 1.
# 이슈 코드 : 운영, 휴업 중인 양돈 농장 불러오기
# 이슈 내용 : 쿼리 내에서 운영, 휴업 중인 농장 정보를 불러오는데, 각 함수마다 운영/휴업 중인 농장을 산출하는 코드가 다름. 결과가 같은지 확인 필요
# 개선 방안 : 동일한 농장 대상을 바라보도록 모든 코드 통일 필요

# ++ ===================================================================================================================


######## 운영, 휴업 중인 양돈 농장 반경 이내 양돈 농장 수 생성

def calculate_pig_farm_counts(df, radius_list):
    # 좌표 배열 생성
    coords = df[['farm_longitude', 'farm_latitude']].values

    # KDTree 생성
    tree = cKDTree(coords)

    # 반경 리스트 순회하면서 결과 추가
    for radius in radius_list:
        counts = tree.query_ball_point(coords, r=radius)
        column_name = f'radius_{radius // 1000}km_pig_farm_count'
        df[column_name] = [len(c) - 1 for c in counts]  # 자기 자신 제외

    # 결과 컬럼 정리
    result_columns = ['farm_serial_no'] + [f'radius_{radius // 1000}km_pig_farm_count' for radius in radius_list]
    return df[result_columns]

# 농장 반경 내 양돈 농가 수 count
def create_farm_around_pig_farm_count(farm_base_info, radius_list):

    # copy
    farm = farm_base_info.copy()

    # 농장 반경 내 양돈 농가 수를 계산
    pig_farm_with_counts = calculate_pig_farm_counts(farm, radius_list)
    return pig_farm_with_counts