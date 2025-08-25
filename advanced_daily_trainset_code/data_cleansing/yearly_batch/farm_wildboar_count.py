import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')
import urllib
import sqlalchemy as sa
from tqdm import tqdm
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

# ++ ===================================================================================================================

# 연 단위로 갱신되는 산림 내 멧돼지 수 데이터 기반으로 농장 인근 멧돼지 수를 산출
def create_farm_wildboar_count(db, year, radius) :

    wildboar_year = '2023' if int(year) > 2023 else year # 야생멧돼지 서식밀도 원천 데이터 확보 기간이 2019 ~ 2023년
    query = sa.text(f'''
    
        INSERT INTO asf.tb_farm_wildboar_count_clean_asf (
            year, 
            farm_serial_no, 
            radius, 
            wildboar_count
            )
    
        WITH        
        -- 운영/휴업 중인 양돈 농가 추출 & 좌표 매칭
        farm_buffer_5179 AS (
            SELECT 
                farm_serial_no,
                geometry
            FROM asf.tb_farm_geometry_clean_asf
            WHERE geom_type = 'Polygon'
                AND radius = {radius}
        ),
        
        -- 농장 반경 내 야생멧돼지 수
        farm_wildboar_count as (
            SELECT
                f.farm_serial_no,
                ST_Area(ST_Intersection(w.geometry, f.geometry)) / w.forest_area * w.forest_wildboar_count as wildboar_count
            FROM geoai_polygon.tb_forest_wildboar_count_clean w
            join farm_buffer_5179 f
            on w.year = '{wildboar_year}'
            AND ST_Intersects(w.geometry, f.geometry)
        )
    
        -- 농장 단위 집계
        select 
            '{year}' as year, 
            farm_serial_no, 
            {radius} as radius, 
            sum(wildboar_count) as wildboar_count
        from farm_wildboar_count
        group by farm_serial_no
    ''')

    # 쿼리 실행
    with db.engine.begin() as conn:
        conn.execute(query)