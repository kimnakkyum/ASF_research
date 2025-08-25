import warnings
warnings.filterwarnings('ignore')
import sqlalchemy as sa
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

# ++ ===================================================================================================================


# 모든 축종을 사육하는 운영, 휴업 중인 농장 리스트 + 공간 정보 생성
db = GetDB()

geometry_sql = "st_transform(('SRID=4326;POINT(' || b.xmin_ || ' ' || b.ymin || ')')::geometry, 5179) as geometry"

query = sa.text(f'''
    INSERT INTO asf.tb_whole_farm_geometry_clean_asf (
        nvrqs_max_date,
        frmhs_blvstck_hist_max_date,
        farm_serial_no,
        geometry
    )

    WITH         
    farm_point AS (
        SELECT distinct on (farms_no) farms_no, xmin_, ymin
        FROM m2msys.nvrqs_mobile_farms_raw
        ORDER BY farms_no, std_dt DESC
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
        SELECT 
            a.frmhs_no as farm_serial_no,
            {geometry_sql}
        FROM (
            SELECT DISTINCT frmhs_no
            FROM geoai_mt.tn_mobile_frmhs_blvstck_avg_hist_clean
            WHERE bsns_sttus_se IN ('1', '2')
              AND mgr_code IN ('104001', '104002')
              AND master_sttus_se IN ('1', '2', '3', 'Z')
        ) a
        JOIN farm_point b ON a.frmhs_no = b.farms_no
    )

    SELECT * FROM nvrqs_max_date
    CROSS JOIN frmhs_blvstck_hist_max_date
    CROSS JOIN farm_geometry
''')

with db.engine.begin() as conn:
    conn.execute(query)