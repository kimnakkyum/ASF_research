import warnings
warnings.filterwarnings('ignore')
import urllib
import sqlalchemy as sa
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

# ++ ===================================================================================================================

db = GetDB()

# 공간정보 dict
geometry_dict = {'Point' : [0],
                 'Polygon' : [1000, 3000, 5000, 6000, 9000, 12000]
                 }

for geom_type in geometry_dict.keys():
    for radius in geometry_dict[geom_type]:

        if geom_type == 'Point' :
            geometry_sql = f"st_transform(('SRID=4326;POINT(' || b.xmin_ || ' ' || b.ymin || ')')::geometry, 5179) as geometry"
        else :
            geometry_sql = f"ST_Buffer(st_transform(('SRID=4326;POINT(' || b.xmin_ || ' ' || b.ymin || ')')::geometry, 5179), {radius}) as geometry"

        query = sa.text(f'''
                INSERT INTO asf.tb_farm_geometry_clean_asf (
                    nvrqs_max_date,
                    frmhs_blvstck_hist_max_date,
                    farm_serial_no,
                    geom_type,
                    radius,
                    geometry
                    )
                    
                WITH         
                -- 농장 좌표 데이터
                farm_point AS (
                    SELECT distinct on (farms_no) farms_no, xmin_, ymin
                    FROM m2msys.nvrqs_mobile_farms_raw
                    WHERE farms_no in (SELECT farm_serial_no FROM asf.tb_pig_farm_serial_no_clean_asf)
                    order by farms_no, std_dt desc
                ),
                
                nvrqs_max_date as (
                    SELECT 
                        max(std_dt) as nvrqs_max_date
                    FROM 
                        m2msys.nvrqs_mobile_farms_raw
                ),
                
                frmhs_blvstck_hist_max_date as (
                    SELECT 
                        max(std_dt) as frmhs_blvstck_hist_max_date
                    FROM 
                        geoai_mt.tn_mobile_frmhs_blvstck_avg_hist_clean
                ),
                
                farm_geometry as (
                    -- 전체 기간에 대해 운영/휴업 상태였던 양돈 농가의 좌표 매칭
                    SELECT 
                        a.frmhs_no as farm_serial_no,
                        '{geom_type}' AS geom_type,
                        {radius} AS radius,
                        {geometry_sql}
                    FROM (
                        SELECT DISTINCT frmhs_no
                        FROM geoai_mt.tn_mobile_frmhs_blvstck_avg_hist_clean
                        WHERE lstksp_cl LIKE '413%'
                            AND bsns_sttus_se IN ('1', '2')
                            AND mgr_code IN ('104001', '104002')
                            AND master_sttus_se IN ('1', '2', '3', 'Z')
                    ) a
                    JOIN farm_point b
                    ON a.frmhs_no = b.farms_no
                )
                
                select * from nvrqs_max_date
                cross join frmhs_blvstck_hist_max_date
                cross join farm_geometry
        ''')

        # 쿼리 실행
        with db.engine.begin() as conn:
            conn.execute(query)