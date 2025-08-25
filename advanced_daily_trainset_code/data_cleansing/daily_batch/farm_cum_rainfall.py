import pandas as pd
import geopandas as gpd
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB
from pyproj import Transformer


# 과거 누적 기간에 따른 강수량 정보 생성
def create_farm_cum_rainfall(db, standard_date):
    # 강수량 정보 불러오기
    farm_rainfall_radius = db.query(f'''

                SELECT
                    farm_serial_no,

                    -- 1km
                    MAX(CASE WHEN radius = 1000  THEN rainfall_0d END) AS radius_1km_cum_rainfall_0day,
                    MAX(CASE WHEN radius = 1000  THEN rainfall_7d END) AS radius_1km_cum_rainfall_7day,
                    MAX(CASE WHEN radius = 1000  THEN rainfall_14d END) AS radius_1km_cum_rainfall_14day,
                    MAX(CASE WHEN radius = 1000  THEN rainfall_1m END) AS radius_1km_cum_rainfall_1month,

                    -- 3km
                    MAX(CASE WHEN radius = 3000  THEN rainfall_0d END) AS radius_3km_cum_rainfall_0day,
                    MAX(CASE WHEN radius = 3000  THEN rainfall_7d END) AS radius_3km_cum_rainfall_7day,
                    MAX(CASE WHEN radius = 3000  THEN rainfall_14d END) AS radius_3km_cum_rainfall_14day,
                    MAX(CASE WHEN radius = 3000  THEN rainfall_1m END) AS radius_3km_cum_rainfall_1month,

                    -- 6km
                    MAX(CASE WHEN radius = 6000  THEN rainfall_0d END) AS radius_6km_cum_rainfall_0day,
                    MAX(CASE WHEN radius = 6000  THEN rainfall_7d END) AS radius_6km_cum_rainfall_7day,
                    MAX(CASE WHEN radius = 6000  THEN rainfall_14d END) AS radius_6km_cum_rainfall_14day,
                    MAX(CASE WHEN radius = 6000  THEN rainfall_1m END) AS radius_6km_cum_rainfall_1month,

                    -- 9km
                    MAX(CASE WHEN radius = 9000  THEN rainfall_0d END) AS radius_9km_cum_rainfall_0day,
                    MAX(CASE WHEN radius = 9000  THEN rainfall_7d END) AS radius_9km_cum_rainfall_7day,
                    MAX(CASE WHEN radius = 9000  THEN rainfall_14d END) AS radius_9km_cum_rainfall_14day,
                    MAX(CASE WHEN radius = 9000  THEN rainfall_1m END) AS radius_9km_cum_rainfall_1month,

                    -- 12km
                    MAX(CASE WHEN radius = 12000 THEN rainfall_0d END) AS radius_12km_cum_rainfall_0day,
                    MAX(CASE WHEN radius = 12000 THEN rainfall_7d END) AS radius_12km_cum_rainfall_7day,
                    MAX(CASE WHEN radius = 12000 THEN rainfall_14d END) AS radius_12km_cum_rainfall_14day,
                    MAX(CASE WHEN radius = 12000 THEN rainfall_1m END) AS radius_12km_cum_rainfall_1month

                FROM asf.tb_farm_rainfall_clean_asf
                WHERE standard_date = '{standard_date}'
                GROUP BY farm_serial_no
                        ''')

    return farm_rainfall_radius