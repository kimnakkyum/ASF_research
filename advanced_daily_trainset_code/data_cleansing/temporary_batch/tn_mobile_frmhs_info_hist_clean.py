import warnings
warnings.filterwarnings('ignore')
import urllib
import sqlalchemy as sa
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

# ++ ===================================================================================================================

db = GetDB()
query = sa.text(f'''
        -- 일별 농장 사용가능한 데이터 추리기
        INSERT INTO geoai_mt.tn_mobile_frmhs_info_hist_clean (
            std_dt,
            frmhs_no,
            max_change_dt,
            bsns_sttus_se,
            induty_cl
            )

        WITH step00 AS (
        -- 동시에 들어온 값 중 fclty_sn 이 가장 높은 것만 남기기 (+ delete_at = 'Y' 인 값은 제외)
            SELECT 
                frmhs_no, 
                change_dt, 
                fclty_sn, 
                DATE_TRUNC('day', change_dt) AS trunc_dt,
                bsns_sttus_se,
                induty_cl,
                ROW_NUMBER() OVER (
                    PARTITION BY frmhs_no, change_dt
                    ORDER BY fclty_sn DESC
                ) AS rn
            FROM m2msys.tn_mobile_frmhs_info_hist_raw 
            WHERE 
                delete_at = 'N'
        --        AND change_dt BETWEEN '2022-10-01' AND '2024-04-01' -- 기간을 정해놓는 것은 위험할 수 있음 (tn_mobille_frmhs_info 의 기간적 제한을 알 수 없음)
        ),
        filtered_fclty AS (
            SELECT *
            FROM step00
            WHERE rn = 1
        ),
        final_step AS (
        -- 날짜별 가장 최대의 change_dt 를 필터링 
        -- 기존의 tn_mobile_frmhs_no 가 당시의 가장 최신 데이터만 가져오기 때문
            SELECT 
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY frmhs_no, trunc_dt
                    ORDER BY change_dt DESC
                ) AS rn2
            FROM filtered_fclty
        )
        SELECT 
            trunc_dt AS std_dt,
            frmhs_no,
            change_dt AS max_change_dt,
            bsns_sttus_se,
            induty_cl
        FROM final_step
        WHERE rn2 = 1
''')

# 쿼리 실행
with db.engine.begin() as conn:
    conn.execute(query)