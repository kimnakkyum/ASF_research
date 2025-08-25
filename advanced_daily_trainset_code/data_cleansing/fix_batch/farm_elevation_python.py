import pandas as pd
from scipy.spatial import cKDTree
import numpy as np

# ++ ===================================================================================================================

# 농장 고도 생성
def create_farm_elevation(db):

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

    # ++ ===================================================================================================================

    # 고도 포인트
    elevation_data = pd.read_sql(
        '''
        SELECT elevation, ST_X(geom) as x, ST_Y(geom) as y
        FROM geoai_polygon.tb_elevation_geom_raw
        ''', con=db.engine
    )

    # 고도 포인트 KDTree 구축
    elevation_coords = elevation_data[['x', 'y']].values
    elevation_tree = cKDTree(elevation_coords)

    # ++ ===================================================================================================================

    # 농장별 거리 검색
    # 1. 가장 가까운 elevation
    dist_nearest, idx_nearest = elevation_tree.query(farm_coords, k=1)

    # 2. 500m 이내 elevation
    idxs_500m = elevation_tree.query_ball_point(farm_coords, r=500)

    # 3. 1km 이내 elevation
    idxs_1km = elevation_tree.query_ball_point(farm_coords, r=1000)

    # ++ ===================================================================================================================

    # 결과 저장용
    nearest_elevation = []
    median_500m = []
    median_1km = []

    for nearest_d, nearest_idx, idxs_500, idxs_1k in zip(dist_nearest, idx_nearest, idxs_500m, idxs_1km):

        # (1) 가장 가까운 elevation. 1km를 벗어나면 None 부여
        if nearest_d <= 1000:
            nearest_elevation.append(elevation_data.iloc[nearest_idx]['elevation'])
        else:
            nearest_elevation.append(None)

        # (2) 500m 이내 중위값
        if idxs_500:
            median_500m.append(np.median(elevation_data.iloc[idxs_500]['elevation']))
        else:
            median_500m.append(None)

        # (3) 1km 이내 중위값
        if idxs_1k:
            median_1km.append(np.median(elevation_data.iloc[idxs_1k]['elevation']))
        else:
            median_1km.append(None)

    farm_point['elevation_nearest'] = nearest_elevation
    farm_point['elevation_median_500m'] = median_500m
    farm_point['elevation_median_1km'] = median_1km

    farm_point = farm_point[['farm_serial_no', 'elevation_nearest', 'elevation_median_500m', 'elevation_median_1km']]

    # DB 저장
    farm_point.to_sql(
        'tb_farm_elevation_clean_asf',
        con=db.engine,
        if_exists='append',
        chunksize=1000,
        schema='asf',
        index=False
    )