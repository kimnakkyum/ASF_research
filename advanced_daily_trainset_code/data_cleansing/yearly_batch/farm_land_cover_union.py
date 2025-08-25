import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')
import urllib
import sqlalchemy as sa
from tqdm import tqdm
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

# 토지피복도를 도엽 단위로 unary_union 하여 연산량을 줄이기 위함
db = GetDB()
conn = db.engine.raw_connection()
with conn.cursor() as cursor:
    for year in tqdm(list(map(str, range(2020, 2024 + 1)))) :
        with open(
                fr"D:\DB_migration\data\geoai_polygon.tb_land_cover_agriculture_union\tb_land_cover_agriculture_union_{year}.csv",
                "w", encoding="utf-8") as f:
            cursor.copy_expert(f"""
            COPY (
                SELECT
                    '{year}' as base_year,
                    inx_num,
                    ST_AsText(ST_UnaryUnion(ST_Collect(geometry))) AS geometry
                FROM geoai_polygon.tb_land_cover_agriculture
                WHERE base_year = '{year}'
                GROUP BY inx_num
            )
            TO STDOUT WITH (DELIMITER ',', FORMAT CSV, HEADER)
            """, f)

        with open(
                fr"D:\DB_migration\data\geoai_polygon.tb_land_cover_urban_union\tb_land_cover_urban_union_{year}.csv",
                "w", encoding="utf-8") as f:
            cursor.copy_expert(f"""
            COPY (
                SELECT
                    '{year}' as base_year,
                    inx_num,
                    ST_AsText(ST_UnaryUnion(ST_Collect(geometry))) AS geometry
                FROM geoai_polygon.tb_land_cover_urban
                WHERE base_year = '{year}'
                GROUP BY inx_num
            )
            TO STDOUT WITH (DELIMITER ',', FORMAT CSV, HEADER)
            """, f)