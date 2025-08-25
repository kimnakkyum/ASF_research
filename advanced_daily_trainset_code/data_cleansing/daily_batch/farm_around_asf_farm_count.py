import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import sqlalchemy as sa
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

# ++ ===================================================================================================================

# 운영, 휴업 농장 대상으로 농장 인근 반경 내 발생 농장 개수 집계
def create_farm_around_asf_farm_count(db, standard_date):

    # 농장 발생 정보
    farm_around_asf_farm_count = pd.read_sql(sql = sa.text(f'''
    WITH 
        farm_point AS (
            SELECT '{standard_date}'::date as standard_date, farm_serial_no, geometry as farm_coordinate
            FROM asf.tb_farm_geometry_clean_asf
            WHERE geom_type = 'Point'
        ),
    
        step01 AS (
            SELECT DISTINCT ON (frmhs_no, sttemnt_de) 
                LPAD(frmhs_no, 8, '0') AS farm_serial_no,
                TO_DATE(sttemnt_de, 'YYYYMMDD') AS asf_org_day,
                fg.geometry AS asf_farms_point
            FROM asf.tn_diss_occrrnc_frmhs_raw_asf_bak oc
            JOIN asf.tb_farm_geometry_clean_asf fg
              ON LPAD(oc.frmhs_no, 8, '0') = fg.farm_serial_no
             AND fg.geom_type = 'Point'
            WHERE diss_cl = '8' 
              AND delete_at = 'N' 
              AND SLAU_AT = 'N' 
              AND CNTRL_DT IS NOT NULL   
              AND ESTBS_NM NOT LIKE '%예살%'   
              AND ESTBS_NM NOT LIKE '%출하%'
              AND ESTBS_NM NOT LIKE '%차량%'
              AND ESTBS_NM LIKE '%차%'
              AND '{standard_date}'::date BETWEEN TO_DATE(sttemnt_de, 'YYYYMMDD') AND TO_DATE(sttemnt_de, 'YYYYMMDD') + INTERVAL '1 months'
        ),
    
        distance_calculations AS (
            SELECT
                fp.farm_serial_no AS farm_serial_no,
                s1.farm_serial_no AS asf_farm_serial_no,
                ST_Distance(fp.farm_coordinate, s1.asf_farms_point) AS distance,
                s1.asf_org_day
            FROM farm_point fp
            JOIN step01 s1
            ON ST_DWithin(fp.farm_coordinate, s1.asf_farms_point, 12000)
            WHERE fp.standard_date BETWEEN s1.asf_org_day AND s1.asf_org_day + INTERVAL '1 month'
              AND fp.farm_serial_no != s1.farm_serial_no
        )
    
    SELECT
        farm_serial_no,
        COUNT(*) FILTER (WHERE distance <= 1000)   AS radius_1km_infection_farm_1month_count,
        COUNT(*) FILTER (WHERE distance <= 3000)   AS radius_3km_infection_farm_1month_count,
        COUNT(*) FILTER (WHERE distance <= 6000)   AS radius_6km_infection_farm_1month_count,
        COUNT(*) FILTER (WHERE distance <= 9000)   AS radius_9km_infection_farm_1month_count,
        COUNT(*) FILTER (WHERE distance <= 12000)  AS radius_12km_infection_farm_1month_count
    FROM distance_calculations
    GROUP BY farm_serial_no
    ORDER BY farm_serial_no;
    '''), con=db.engine)
    return farm_around_asf_farm_count