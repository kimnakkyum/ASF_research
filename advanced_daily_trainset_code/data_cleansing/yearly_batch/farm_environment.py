import warnings
warnings.filterwarnings('ignore')
import sqlalchemy as sa
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

# ++ ===================================================================================================================

# 연 단위로 갱신되는 토지 피복도 데이터 기반으로 농장 인근 토지 용도별 면적을 산출
def create_farm_environment(db, year, radius) :

    land_cover_year = str(min(max(int(year), 2020), 2024)) # 토지피복도 원천 확보 기간이 2020 ~ 2024년
    land_cover_year = '2022' if land_cover_year == '2023' else land_cover_year # 현재 2023 토지피복도 이슈 존재하여, 2022로 대체해서 활용

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
        
            farm_buffer_max_area AS(
                SELECT MAX(ST_Area(geometry)) AS max_area
                FROM asf.tb_farm_geometry_clean_asf
                WHERE geom_type = 'Polygon' 
                    AND radius = {radius}
            )
                        
            -- 1. 도시·건조지역
            SELECT
                '{year}' AS year,
                i1.farm_serial_no,
                '시가화지역' AS code,
                {radius} AS radius,
                SUM(i1.intersection_area) AS total_intersection_area_m2,
                (SUM(i1.intersection_area) / (SELECT max_area FROM farm_buffer_max_area)) AS total_intersection_area_m2_ratio
            FROM (
                SELECT
                    f.farm_serial_no,
                    ST_Area(ST_Intersection(t1.geometry, f.geometry)) AS intersection_area
                FROM farm_buffer_5179 AS f
                JOIN geoai_polygon.tb_land_cover_urban AS t1 -- 원천 경로 수정
                  ON t1.base_year = '{land_cover_year}'
                 AND ST_Intersects(t1.geometry, f.geometry)
            ) i1
            GROUP BY i1.farm_serial_no
            
            UNION ALL

            -- 2. 농지지역
            SELECT
                '{year}' AS year,
                i2.farm_serial_no,
                '농지지역_밭' AS code,
                {radius} AS radius,
                SUM(i2.intersection_area) AS total_intersection_area_m2,
                (SUM(i2.intersection_area) / (SELECT max_area FROM farm_buffer_max_area)) AS total_intersection_area_m2_ratio
            FROM (
                SELECT
                    f.farm_serial_no,
                    ST_Area(ST_Intersection(t2.geometry, f.geometry)) AS intersection_area
                FROM farm_buffer_5179 AS f
                 JOIN geoai_polygon.tb_land_cover_agriculture AS t2 -- 원천 경로 수정
                JOIN (select * from geoai_polygon.tb_land_cover_agriculture where l2_code in ('220')) AS t2 -- 중분류 코드 기준 '밭'만 불러오기
                  ON t2.base_year = '{land_cover_year}'
                 AND ST_Intersects(t2.geometry, f.geometry)
            ) i2
            GROUP BY i2.farm_serial_no
            
            ORDER BY farm_serial_no, total_intersection_area_m2 DESC
    ''')

    # 쿼리 실행
    with db.engine.begin() as conn:
        conn.execute(query)