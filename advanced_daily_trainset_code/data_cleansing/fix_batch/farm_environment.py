import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')
import urllib
import sqlalchemy as sa
from tqdm import tqdm
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

# ++ ===================================================================================================================

# 시점이 고정되어 있는 산림, 하천 데이터 기반으로 농장 인근 토지 용도별 면적을 산출
def create_farm_environment_fix_batch(db, radius) :

    query = sa.text(f'''
            
            INSERT INTO asf.tb_farm_environment_clean_asf (
                year,
                farm_serial_no,
                code,
                radius, 
                total_intersection_area_m2,
                total_intersection_area_m2_ratio
            )
        
            WITH 
            
            -- 농장 포인트가 주어졌을때, 인근에 존재하는 토지 용도별 면적을 산출하는 쿼리
            -- 산출결과는 1)농장별, 2) 분류 코드 3) 3km 내 분류별 면적, 4)3km 내 분류별 면적 비율
            -- 이걸 필요에 따라, 세분류를 합산하거나 하나로 합쳐서 최종적으로 비율을 구함
            -- 비율은 28093006.37를 나누어서 산출(3000 * 3000 * 3.14....)
        
            -- 운영/휴업 중인 양돈 농가 추출 & 좌표 매칭
            farm_buffer_5179 AS (
                SELECT 
                    farm_serial_no,
                    geometry
                FROM asf.tb_farm_geometry_clean_asf
                WHERE geom_type = 'Polygon'
                    AND radius = {radius}
            ),
            
            farm_buffer_max_area AS (
                SELECT MAX(ST_Area(geometry)) AS max_area
                FROM asf.tb_farm_geometry_clean_asf
                WHERE geom_type = 'Polygon'
                    AND radius = {radius}
            )
        
            SELECT
                '9999' AS year,
                i1.farm_serial_no,
                '산림지역_10000_고도' AS code,
                {radius} AS radius,
                SUM(i1.intersection_area) AS total_intersection_area_m2,
                (SUM(i1.intersection_area) / (SELECT max_area FROM farm_buffer_max_area)) AS total_intersection_area_m2_ratio
            FROM (
                SELECT
                    f.farm_serial_no,
                    ST_Area(ST_Intersection(t.geom, f.geometry)) AS intersection_area
                FROM farm_buffer_5179 AS f
                JOIN asf.tb_forest_geom_clean_asf AS t -- 국토교통부 고도 기준 & 산림청 면적 기준 소규모 산림을 제외한 산림 정보 불러오기
                  ON ST_Intersects(t.geom, f.geometry)
            ) i1
            GROUP BY i1.farm_serial_no;
    ''')

    # 쿼리 실행
    with db.engine.begin() as conn:
        conn.execute(query)