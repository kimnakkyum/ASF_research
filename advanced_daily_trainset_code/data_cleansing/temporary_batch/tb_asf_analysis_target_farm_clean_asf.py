import pandas as pd
import geopandas as gpd
import numpy as np
from glob import glob
import warnings
warnings.filterwarnings('ignore')
import sqlalchemy as sa
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB
from advanced_daily_trainset_code.data_cleansing.config import config

db = GetDB()
query = sa.text(f'''
        INSERT INTO asf.tb_asf_analysis_target_farm_clean_asf (
            nvrqs_max_date,
            frmhs_blvstck_hist_max_date,
            farm_occurrence_max_date,
            wild_occurrence_max_date,
            farm_serial_no,
            occurrence_yn,
            occurrence_near_yn,
            etc_yn,
            geometry
            )
        
        -- 발생 시군 & 발생 인근 시군을 산출한 원천 데이터 max date
        WITH occurrence_max_date AS (
            SELECT distinct farm_occurrence_max_date, wild_occurrence_max_date
            FROM asf.tb_asf_near_sigun_clean_asf
        )
        
        -- 전체 기간에 대해 운영/휴업 상태였던 양돈 농가의 좌표 매칭 데이터와 분석 대상 시군을 공간 매칭 
        SELECT 
            f.nvrqs_max_date,
            f.frmhs_blvstck_hist_max_date,
            (select farm_occurrence_max_date from occurrence_max_date) as farm_occurrence_max_date,
            (select wild_occurrence_max_date from occurrence_max_date) as wild_occurrence_max_date,
            f.farm_serial_no,
            coalesce(s.occurrence_yn, 0) as occurrence_yn, -- 발생 시군 여부
            coalesce(s.occurrence_near_yn, 0) as occurrence_near_yn, -- 발생 시군 인접 여부
            case
                when occurrence_yn is null then 1
                when (occurrence_yn is not null) AND (occurrence_near_yn is not null) then 0
            end as etc_yn, -- 발생 시군 & 발생 시군 인접 모두 아닌 경우
            f.geometry
        FROM asf.tb_farm_geometry_clean_asf f
        left JOIN asf.tb_asf_near_sigun_clean_asf s
            ON ST_Intersects(f.geometry, s.geometry)
        WHERE f.geom_type = 'Point'
''')

# 쿼리 실행
with db.engine.begin() as conn:
    conn.execute(query)