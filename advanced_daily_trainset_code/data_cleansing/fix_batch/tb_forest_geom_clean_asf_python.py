import pandas as pd
import geopandas as gpd
import psycopg2

# 국토교통부 고도 & 산림청 산림 면적 기준 산림 재정의
def create_forest(db):

    # 산림 정보
    forest_df = gpd.read_postgis("""with 
    
                        -- 산림-고도 매칭 정보(최대고도, 기복량) 생성 
                        forest_stats AS (
                            SELECT 
                                f.id,
                                f.geom,
                                ST_Area(f.geom) as area,
                                MAX(e.elevation) AS max_elevation,
                                MAX(e.elevation) - MIN(e.elevation) AS elevation_range
                            FROM geoai_polygon.tb_forest_geom_clean f
                            JOIN geoai_polygon.tb_elevation_geom_raw e
                              ON ST_Intersects(f.geom, e.geom)
                            GROUP BY f.id
                        ),
                        
                        -- 국토교통부 고도 기준 & 산림청 면적 기준 소규모 산림을 제외한 산림 추출
                        filtered_forest AS (
                            SELECT id, geom
                            FROM forest_stats
                            WHERE NOT(area < 10000 AND elevation_range < 100 AND max_elevation < 200)
                        )
                        
                        select * from filtered_forest""", con=db.engine)

    # PostGIS용 테이블에 저장할 때는 to_postgis 사용
    forest_df.to_postgis(
        'tb_forest_geom_clean_asf',
        con=db.engine,
        if_exists='append',
        chunksize=1000,
        schema='asf',
        index=False
    )

