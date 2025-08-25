from pyproj import Transformer
from scipy.spatial import cKDTree
import geopandas as gpd
import pandas as pd


######## 운영, 휴업 중인 양돈 농장 반경 이내 모든 농장 수 생성

def calculate_farm_counts(target_farms, comparison_farms, radius_list):

    # 운영, 휴업 중인 모든 농장 정보의 KDTree 생성
    tree = cKDTree(comparison_farms[['farm_longitude', 'farm_latitude']].values)

    # 운영, 휴업 중인 양돈 농장을 대상 농장으로 지정
    target_coords = target_farms[['farm_longitude', 'farm_latitude']].values

    # 결과 복사본
    df = target_farms[['farm_serial_no']].copy()

    # 반경 리스트 순회하면서 결과 추가
    for radius in radius_list:
        counts = tree.query_ball_point(target_coords, r=radius)
        column_name = f'radius_{radius // 1000}km_whole_farm_count'
        df[column_name] = [len(c) - 1 for c in counts]  # 자기 자신 제외

    # 결과 컬럼 정리
    result_columns = ['farm_serial_no'] + [f'radius_{radius // 1000}km_whole_farm_count' for radius in radius_list]
    return df[result_columns]

# 농장 반경 내 농가 수 count
def create_farm_around_farm_count(farm_base_info ,all_farms, radius_list):

    # copy
    pig_farm = farm_base_info.copy()

    # 농장 반경 내 농가 수를 계산
    farm_with_counts = calculate_farm_counts(pig_farm, all_farms, radius_list)
    return farm_with_counts