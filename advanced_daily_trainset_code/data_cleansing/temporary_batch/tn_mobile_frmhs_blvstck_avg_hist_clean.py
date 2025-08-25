import warnings

warnings.filterwarnings('ignore')
import urllib
import sqlalchemy as sa
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

# ++ ===================================================================================================================

db = GetDB()
query = sa.text(f'''
        -- 일별 농장 & 축종 사용가능한 데이터 매칭 후 평균사육수수 산출
        INSERT INTO geoai_mt.tn_mobile_frmhs_blvstck_avg_hist_clean_asf (
            std_dt,
            frmhs_no, 
            lstksp_cl, 
            before_lstksp_cl,
            max_change_dt_farm, 
            max_change_dt_blvstck,  
            mgr_code, 
            master_sttus_se,
            bsns_sttus_se,
            sear_se,
            induty_cl,
            brd_purps_code,
            brd_had_co, 
            mother_pig_co,
            porker_co,
            avg_brd_had_co,
            avg_mother_pig_co,
            avg_porker_co
            )

        -- 일별 농장 & 축종 사용가능한 데이터 매칭
        WITH matched_cleaned AS (
        	SELECT 
        	    a.std_dt, 
        	    a.frmhs_no, 
        	    a.lstksp_cl, 
        	    a.before_lstksp_cl,
        	    b.max_change_dt as max_change_dt_farm, 
        	    a.max_change_dt as max_change_dt_blvstck, 
        	    a.mgr_code, 
        	    a.master_sttus_se, 
        	    a.brd_purps_code,
        	    a.brd_had_co,
                a.mother_pig_co,
                a.porker_co,
        	    a.avg_cnt, 
        	    a.sear_se, 
        	    b.bsns_sttus_se, 
        	    b.induty_cl
            FROM geoai_mt.tn_mobile_blvstck_hist_clean a
            JOIN geoai_mt.tn_mobile_frmhs_info_hist_clean b
            ON a.frmhs_no = b.frmhs_no
            AND a.std_dt = b.std_dt 
        ),

        -- 모돈/비육돈 평균 산출
        avg_by_group AS (
          SELECT 
            frmhs_no, lstksp_cl, before_lstksp_cl,
            AVG(mother_pig_co) AS mother_pig_co_avg,
            AVG(porker_co) AS porker_co_avg
          FROM matched_cleaned
          WHERE mgr_code IN ('104001', '104002')
            AND master_sttus_se IN ('1', '2', '3', 'Z')
            AND bsns_sttus_se IN ('1', '2')
          GROUP BY frmhs_no, lstksp_cl, before_lstksp_cl
        ),
        
        -- 원본 + 평균 조인 후 평균으로 결측값 대체
        porker_mother_avg_filled AS (
          SELECT 
            m.*,
            COALESCE(m.mother_pig_co, a.mother_pig_co_avg) AS mother_pig_co_filled,
            COALESCE(m.porker_co, a.porker_co_avg) AS porker_co_filled
          FROM matched_cleaned m
          LEFT JOIN avg_by_group a
            ON m.frmhs_no = a.frmhs_no
            AND m.lstksp_cl = a.lstksp_cl
            AND m.before_lstksp_cl = a.before_lstksp_cl
        ),

        -- 1년간 평균사육수 산출
        avg_brd_had_co_calc AS (
            SELECT
                frmhs_no,
                lstksp_cl,
                before_lstksp_cl,
                std_dt,
                mother_pig_co_filled,
                porker_co_filled,
                AVG(NULLIF(brd_had_co, 0)) OVER (
                    PARTITION BY frmhs_no, lstksp_cl 
                    ORDER BY std_dt 
                    RANGE BETWEEN INTERVAL '1 year' PRECEDING AND CURRENT ROW
                ) AS avg_brd_had_co,
                AVG(NULLIF(mother_pig_co_filled, 0)) OVER (
                    PARTITION BY frmhs_no, lstksp_cl 
                    ORDER BY std_dt 
                    RANGE BETWEEN INTERVAL '1 year' PRECEDING AND CURRENT ROW
                ) AS avg_mother_pig_co,
                AVG(NULLIF(porker_co_filled, 0)) OVER (
                    PARTITION BY frmhs_no, lstksp_cl 
                    ORDER BY std_dt 
                    RANGE BETWEEN INTERVAL '1 year' PRECEDING AND CURRENT ROW
                ) AS avg_porker_co
            FROM porker_mother_avg_filled
            WHERE mgr_code IN ('104001', '104002')
                AND master_sttus_se IN ('1', '2', '3', 'Z')
                AND bsns_sttus_se IN ('1', '2')
        --        and sear_se = 'Y' (검토 필요)
        ),

        -- 기타 정보가 포함되어있는 테이블에 평균 사육수 매칭 (데이터 검토 시 필요한 경우가 있어 매칭하여 저장)
        avg_brd_had_co_calc_clean AS (
                SELECT 
                    a.std_dt,
                    a.frmhs_no, 
                    a.lstksp_cl, 
                    a.before_lstksp_cl,
                    a.max_change_dt_farm, 
                    a.max_change_dt_blvstck,  
                    a.mgr_code, 
                    a.master_sttus_se,
                    a.bsns_sttus_se,
                    a.sear_se,
                    a.induty_cl,
                    a.brd_purps_code,
                    a.brd_had_co, 
                    b.mother_pig_co_filled as mother_pig_co,
                    b.porker_co_filled as porker_co,
                    b.avg_brd_had_co,
                    b.avg_mother_pig_co,
                    b.avg_porker_co
                FROM 
                    matched_cleaned a
                LEFT JOIN 
                    avg_brd_had_co_calc b ON a.frmhs_no = b.frmhs_no
                                           AND a.lstksp_cl = b.lstksp_cl
                                           AND a.before_lstksp_cl = b.before_lstksp_cl
                                           AND a.std_dt = b.std_dt
        )

        select * FROM avg_brd_had_co_calc_clean
''')

# 쿼리 실행
with db.engine.begin() as conn:
    conn.execute(query)