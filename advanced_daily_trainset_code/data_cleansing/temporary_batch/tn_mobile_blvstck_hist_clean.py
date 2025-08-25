import warnings
warnings.filterwarnings('ignore')
import urllib
import sqlalchemy as sa
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

# ++ ===================================================================================================================

db = GetDB()
query = sa.text(f'''
        -- 일별 축종 사용가능한 데이터 추리기
        INSERT INTO geoai_mt.tn_mobile_blvstck_hist_clean (
            std_dt, 
            frmhs_no, 
            lstksp_cl, 
            before_lstksp_cl,
            max_change_dt,  
            mgr_code,
            master_sttus_se,
            sear_se,
            brd_purps_code,
            brd_had_co, 
            mother_pig_co,
            porker_co,
            avg_cnt
            )

        WITH ranked_blvstck AS (
            SELECT 
                frmhs_no, 
                lstksp_cl, 
                change_dt,
                DATE_TRUNC('day', change_dt) AS trunc_dt,
                mgr_code, 
                master_sttus_se, 
                brd_purps_code,
                brd_had_co, 
                mother_pig_co,
                porker_co,
                avg_cnt,
                sear_se,
                ROW_NUMBER() OVER (
                    PARTITION BY frmhs_no, lstksp_cl, DATE_TRUNC('day', change_dt)
                    ORDER BY change_dt DESC
                ) AS rn
            FROM m2msys.tn_mobile_blvstck_hist_raw
            WHERE TRIM(frmhs_no) != '' --
        --    WHERE change_dt BETWEEN '2022-10-01' AND '2024-04-01' (기간 활용 안하는 것을 추천)
        --    where sear_se = 'Y' (해당 부분 추가 논의 필요) -- sear_se 는 211030 에 생성된 정보이나 필터링 가능한 시기는 검토가 필요함. 그 이전에는 null 값으로 들어가 있음
        )
        SELECT 
            r.trunc_dt AS std_dt, 
            r.frmhs_no, 
            COALESCE(c.after_code_no, r.lstksp_cl) AS lstksp_cl,  -- 변경된 값 적용
            CASE 
                WHEN c.after_code_no IS NOT NULL THEN r.lstksp_cl  -- 변경된 경우 이전 코드 저장
                ELSE ''
            END AS before_lstksp_cl,
            r.change_dt AS max_change_dt,
            r.mgr_code, 
            r.master_sttus_se,  
            r.sear_se,
            r.brd_purps_code,
            r.brd_had_co, 
            r.mother_pig_co,
            r.porker_co,
            r.avg_cnt
        FROM ranked_blvstck r
        LEFT JOIN geoai_mt.tb_code_change_mapping_clean c
        ON r.lstksp_cl = c.before_code_no
        WHERE rn = 1
''')

# 쿼리 실행
with db.engine.begin() as conn:
    conn.execute(query)