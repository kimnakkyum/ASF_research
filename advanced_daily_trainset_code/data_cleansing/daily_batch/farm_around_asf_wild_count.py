import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import sqlalchemy as sa
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

# ++ ===================================================================================================================

# 운영, 휴업 농장 대상으로 농장 인근 반경 내 발생 농장 개수 집계
def create_farm_around_asf_wild_count(db, standard_date):

    # 농장 발생 정보
    farm_around_asf_wild_count = pd.read_sql(sql = sa.text(f'''
    WITH 
        farm_point AS (
            SELECT '{standard_date}'::date as standard_date, farm_serial_no, geometry as farm_coordinate
            FROM asf.tb_farm_geometry_clean_asf
            WHERE geom_type = 'Point'
        ),
    
        step01 AS (
            SELECT distinct on (odr)
            TO_DATE(sttemnt_de, 'YYYYMMDD') AS asf_org_day,
            ST_Transform(ST_SetSRID(ST_Point(lo, la), 4326), 5179) AS asf_wild_point
            FROM m2msys.tn_diss_occ_ntcn_info_raw
            WHERE '{standard_date}'::date BETWEEN TO_DATE(sttemnt_de, 'YYYYMMDD') AND TO_DATE(sttemnt_de, 'YYYYMMDD') + INTERVAL '1 year'
            order by odr, last_change_dt desc
        ),
    
        -- 기준일자로부터 6개월/1년 이내 발생한 야생 발생과의 거리 계산
        distance_calculations AS (
            SELECT
                fp.farm_serial_no,
                fp.standard_date,
                s1.asf_org_day,
                ST_Distance(fp.farm_coordinate, s1.asf_wild_point) AS distance,
                CASE 
                    WHEN fp.standard_date BETWEEN s1.asf_org_day AND s1.asf_org_day + INTERVAL '6 months'
                    THEN TRUE ELSE FALSE
                END AS within_6months,
                CASE 
                    WHEN fp.standard_date BETWEEN s1.asf_org_day AND s1.asf_org_day + INTERVAL '1 year'
                    THEN TRUE ELSE FALSE
                END AS within_1year
            FROM farm_point fp
            JOIN step01 s1
            ON ST_DWithin(fp.farm_coordinate, s1.asf_wild_point, 12000)
        )
    
    SELECT
        farm_serial_no,
        -- 6개월 이내 거리 기준별 야생 발생 건수
        COUNT(*) FILTER (WHERE within_6months AND distance <= 1000)   AS radius_1km_specimen_picking_6month_count,
        COUNT(*) FILTER (WHERE within_6months AND distance <= 3000)   AS radius_3km_specimen_picking_6month_count,
        COUNT(*) FILTER (WHERE within_6months AND distance <= 6000)   AS radius_6km_specimen_picking_6month_count,
        COUNT(*) FILTER (WHERE within_6months AND distance <= 9000)   AS radius_9km_specimen_picking_6month_count,
        COUNT(*) FILTER (WHERE within_6months AND distance <= 12000)  AS radius_12km_specimen_picking_6month_count,
    
        -- 1년 이내 거리 기준별 야생 발생 건수
        COUNT(*) FILTER (WHERE within_1year AND distance <= 1000)   AS radius_1km_specimen_picking_1year_count,
        COUNT(*) FILTER (WHERE within_1year AND distance <= 3000)   AS radius_3km_specimen_picking_1year_count,
        COUNT(*) FILTER (WHERE within_1year AND distance <= 6000)   AS radius_6km_specimen_picking_1year_count,
        COUNT(*) FILTER (WHERE within_1year AND distance <= 9000)   AS radius_9km_specimen_picking_1year_count,
        COUNT(*) FILTER (WHERE within_1year AND distance <= 12000)  AS radius_12km_specimen_picking_1year_count
    FROM distance_calculations
    GROUP BY farm_serial_no
    ORDER BY farm_serial_no;
    '''), con=db.engine)
    return farm_around_asf_wild_count