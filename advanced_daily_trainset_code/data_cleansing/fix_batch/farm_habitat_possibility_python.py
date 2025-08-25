import pandas as pd
from scipy.spatial import cKDTree
import numpy as np

# ++ ===================================================================================================================

# 농장 야생멧돼지 서식지가능성도 생성
def create_farm_habitat_possibility(db):

    # 농장 포인트
    farm_point = pd.read_sql(
        '''
        SELECT farm_serial_no, ST_X(geometry) as x, ST_Y(geometry) as y
        FROM asf.tb_farm_geometry_clean_asf
        WHERE geom_type = 'Point'
        ''', con=db.engine
    )

    # 농장 Point x, y
    farm_coords = farm_point[['x', 'y']].values

    # ++ ===============================================================================================================

    # 서식지 가능성도 포인트
    habitat_data = pd.read_sql(
        '''
        SELECT habitat_possibility, ST_X(geom) as x, ST_Y(geom) as y
        FROM geoai_polygon.tb_wildboar_habitat_possibility_raw
        ''', con=db.engine
    )

    # 서식지 포인트 KDTree 구축
    habitat_coords = habitat_data[['x', 'y']].values
    habitat_tree = cKDTree(habitat_coords)

    # ++ ===============================================================================================================

    # 농장별 거리 검색
    # 1. 가장 가까운 habitat
    dist_nearest, idx_nearest = habitat_tree.query(farm_coords, k=1)

    # 2. 500m 이내 habitat
    idxs_500m = habitat_tree.query_ball_point(farm_coords, r=500)

    # 3. 1km 이내 habitat
    idxs_1km = habitat_tree.query_ball_point(farm_coords, r=1000)

    # ++ ===================================================================================================================

    # 결과 저장용
    nearest_habitat = []
    median_500m = []
    median_1km = []

    for nearest_d, nearest_idx, idxs_500, idxs_1k in zip(dist_nearest, idx_nearest, idxs_500m, idxs_1km):

        # (1) 가장 가까운 habitat_possibility. 1km를 벗어나면 None 부여
        if nearest_d <= 1000:
            nearest_habitat.append(habitat_data.iloc[nearest_idx]['habitat_possibility'])
        else:
            nearest_habitat.append(None)

        # (2) 500m 이내 중위값
        if idxs_500:
            median_500m.append(np.median(habitat_data.iloc[idxs_500]['habitat_possibility']))
        else:
            median_500m.append(None)

        # (3) 1km 이내 중위값
        if idxs_1k:
            median_1km.append(np.median(habitat_data.iloc[idxs_1k]['habitat_possibility']))
        else:
            median_1km.append(None)

    farm_point['habitat_possibility_nearest'] = nearest_habitat
    farm_point['habitat_possibility_median_500m'] = median_500m
    farm_point['habitat_possibility_median_1km'] = median_1km

    farm_point = farm_point[['farm_serial_no', 'habitat_possibility_nearest', 'habitat_possibility_median_500m',
                             'habitat_possibility_median_1km']]

    # DB 저장
    farm_point.to_sql(
        'tb_farm_habitat_possibility_clean_asf',
        con=db.engine,
        if_exists='append',
        chunksize=1000,
        schema='asf',
        index=False
    )