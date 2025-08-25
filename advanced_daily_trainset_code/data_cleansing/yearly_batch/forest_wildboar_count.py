import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')
import urllib
import sqlalchemy as sa
from tqdm import tqdm
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

# 일단 한 연도에 대해서 중첩면적만 생성. 어차피 모든 시점의 시군 polygon 정보는 동일하므로 여러번 활용 가능
# 그리고 연도 별로 sido_nm, sigun_nm으로 매칭해서 density를 붙여 최종 테이블 생성

# db = GetDB()
# conn = db.engine.raw_connection()
# with conn.cursor() as cursor:
#     with open(
#             fr"D:\DB_migration\data\geoai_mt.tb_forest_wildboar_intersection_clean\tb_forest_wildboar_intersection_clean.csv",
#             "w", encoding="utf-8") as f:
#         cursor.copy_expert(f"""
#         COPY (
#         SELECT
#             w.sido_nm,
#             w.sigun_nm,
#             f.id AS forest_id,
#             ST_Area(ST_Intersection(f.geom, ST_MakeValid(w.geometry))) / 1000000 AS intersected_area_km2 -- 면적을 m²에서 km²로 변환
#         FROM
#             geoai_polygon.tb_forest_geom_clean f
#         JOIN
#             geoai_polygon.tb_wildboar_density_interp_clean w
#         ON
#             w.year = '2023'
#             and ST_Intersects(f.geom, ST_MakeValid(w.geometry))
#         )
#         TO STDOUT WITH (DELIMITER ',', FORMAT CSV, HEADER)
#         """, f)

# 위 데이터 추출하여 geoai_mt.tb_forest_sigun_intersection_clean에 insert 함

# ++ ===================================================================================================================

# 연 단위로 갱신되는 야생멧돼지 서식밀도 데이터 기반으로 산림 내 멧돼지 수를 산출
def create_forest_wildboar_count(db, year) :

    query = sa.text(f'''
            INSERT INTO geoai_polygon.tb_forest_wildboar_count_clean (
                year,
                forest_id,
                forest_wildboar_count,
                forest_area,
                geometry
            )
            -- 산림 polygon 별 면적 및 야생멧돼지 수 산출
            WITH forest_wildboar_count AS(
                SELECT
                    d.year,
                    i.forest_id,
                    (i.intersected_area_km2 * d.density) AS forest_wildboar_count
                FROM geoai_mt.tb_forest_sigun_intersection_clean i
                JOIN geoai_polygon.tb_wildboar_density_interp_clean d
                ON d.year = '{year}'
                AND i.sido_nm = d.sido_nm
                AND i.sigun_nm = d.sigun_nm
            ),
            
            forest_wildboar_count_sum AS(
                SELECT
                    year,
                    forest_id,
                    SUM(forest_wildboar_count) AS forest_wildboar_count
                FROM forest_wildboar_count
                group by year, forest_id
            ),
            
            forest_wildboar_count_geom AS (
                SELECT
                    s.*, ST_Area(f.geom) as forest_area, f.geom as geometry
                FROM
                    forest_wildboar_count_sum s
                JOIN
                    geoai_polygon.tb_forest_geom_clean f
                ON s.forest_id = f.id
            )
            
            select * 
            from forest_wildboar_count_geom
            where forest_area != 0 -- linestring 제외
        ''')
    
    # 쿼리 실행
    with db.engine.begin() as conn:
        conn.execute(query)