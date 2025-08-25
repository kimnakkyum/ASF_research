import warnings
warnings.filterwarnings('ignore')
import urllib
import sqlalchemy as sa
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

# ++ ===================================================================================================================

db = GetDB()
query = sa.text(f'''
        INSERT INTO asf.tb_pig_farm_serial_no_clean_asf (
            frmhs_blvstck_hist_max_date,
            farm_serial_no
            )
    
        -- 전체 기간에 대해 운영/휴업 상태였던 양돈 농가에 해당하는 농가 번호 추출 
        with max_date as (
        SELECT 
            max(std_dt) as frmhs_blvstck_hist_max_date
        FROM 
            m2msys.tn_mobile_blvstck_hist_raw
        ),
        
        farm_serial_no as (
            SELECT 
                distinct frmhs_no as farm_serial_no
            FROM 
                m2msys.tn_mobile_blvstck_hist_raw
            WHERE lstksp_cl LIKE '413%'
            AND master_sttus_se IN ('1', '2', '3', 'Z') 
            AND mgr_code IN ('104001', '104002')
            and TRIM(frmhs_no) != '')
            
            select * from max_date
            cross join
            farm_serial_no;
''')

# 쿼리 실행
with db.engine.begin() as conn:
    conn.execute(query)