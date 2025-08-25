import pandas as pd
import geopandas as gpd
import sqlalchemy as sa
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

# +++ [이슈사항 정리] +++ ================================================================================================
# 1.
# 이슈 코드 : FROM 검역본부2.고도
# 이슈 내용 : 원천 테이블 미 변경
# 개선 방안 : 180 서버 내 geoai_polygon.tb_elevation_geom_raw로 변경 필요
# 반영 완료

# 2.
# 이슈 코드 : gdf_farms['elevation'] = gdf_farms['고도']
# 이슈 내용 : DB 이관을 통해 고도 -> elevation으로 컬럼명 변경이 이루어졌기 때문에 제외 대상
# 개선 방안 : 해당 코드 제외
# 반영 완료

# 3.*****
# 이슈 코드 : def farm_elevation(farm):
# 이슈 내용 : farm을 인자로 받는데, 각 함수마다 운영/휴업 중인 농장을 산출하는 코드가 다름. 결과가 같은지 확인 필요
# 개선 방안 : 동일한 농장 대상을 바라보도록 모든 코드 통일 필요

# ++ ===================================================================================================================

# 농장 고도 생성
def create_farm_elevation(db):

    query = sa.text(f'''
    
        INSERT INTO asf.tb_farm_elevation_clean_asf (
            farm_serial_no,
            elevation_nearest,
            elevation_median_500m,
            elevation_median_1km
        )
        
        WITH         
        farm_point_5179 AS (
            SELECT 
                farm_serial_no,
                geometry AS farm_coordinate
            FROM asf.tb_farm_geometry_clean_asf
            WHERE geom_type = 'Point'
        ),
        
        elevation_data AS (
            SELECT 
                elevation, 
                geom AS geom_elevation
            FROM geoai_polygon.tb_elevation_geom_raw
        ),
        
        nearest_elevation AS (
            -- 농장별 가장 가까운 고도
            SELECT DISTINCT ON (f.farm_serial_no)
                f.farm_serial_no,
                h.elevation AS nearest_elevation
            FROM farm_point_5179 f
            LEFT JOIN elevation_data h 
                ON ST_DWithin(f.farm_coordinate, h.geom_elevation, 1000)
            ORDER BY f.farm_serial_no, ST_Distance(f.farm_coordinate, h.geom_elevation)
        ),
        
        elevation_500m AS (
            -- 농장별 500m 이내 고도들의 중위값
            SELECT 
                f.farm_serial_no,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY h.elevation) AS median_500m
            FROM farm_point_5179 f
            JOIN elevation_data h 
                ON ST_DWithin(f.farm_coordinate, h.geom_elevation, 500)
            GROUP BY f.farm_serial_no
        ),
        
        elevation_1km AS (
            -- 농장별 1km 이내 고도들의 중위값
            SELECT 
                f.farm_serial_no,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY h.elevation) AS median_1km
            FROM farm_point_5179 f
            JOIN elevation_data h 
                ON ST_DWithin(f.farm_coordinate, h.geom_elevation, 1000)
            GROUP BY f.farm_serial_no
        )
        
        SELECT 
            f.farm_serial_no,
            n.nearest_elevation as elevation_nearest,
            e500.median_500m as elevation_median_500m,
            e1k.median_1km as elevation_median_1km
        FROM farm_point_5179 f
        LEFT JOIN nearest_elevation n ON f.farm_serial_no = n.farm_serial_no
        LEFT JOIN elevation_500m e500 ON f.farm_serial_no = e500.farm_serial_no
        LEFT JOIN elevation_1km e1k ON f.farm_serial_no = e1k.farm_serial_no
        ''')

    # 쿼리 실행
    with db.engine.begin() as conn:
        conn.execute(query)