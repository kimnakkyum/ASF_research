from Bulk_trainset_code.config import config
import psycopg2
import pandas as pd

# 모돈 평균사육두수 생성

def avg_mother_brd_had_co(one_year_before_date, target_date):
    conn = psycopg2.connect(config.conn_string)
    cur = conn.cursor()
    cur.execute(f""" 
                    WITH 
                        -- 2020년부터 2025년 최신정보까지 데이터 불러오기
                        tot as (
                        select * from m2msys.tn_mobile_frmhs_info_hist_24
                        union all 
                        select * from m2msys.tn_mobile_frmhs_info_hist_2324_new
                        ),
                        
                        -- 해당 시점 기준으로 과거 1년까지의 데이터 불러오기
                        step00 AS (
                                select 
                                    frmhs_no, 
                                    change_dt, 
                                    MAX(fclty_sn) AS fclty_sn, 
                                    DATE_TRUNC('day', change_dt) AS date_trunc,
                                    bsns_sttus_se
                                from tot
                                WHERE 
                                    change_dt BETWEEN '{one_year_before_date}' AND '{target_date}'
                                    AND delete_at = 'N'
                                group by
                                    frmhs_no, date_trunc, change_dt, bsns_sttus_se
                        ),
                            
                        -- 일 기준 마지막 데이터 기준 정보 생성
                        step01 AS (
                                SELECT 
                                    frmhs_no, 
                                    MAX(change_dt) AS max_change_dt, 
                                    date_trunc 
                                FROM 
                                    step00
                                GROUP BY 
                                    frmhs_no, date_trunc
                            ),
                        
                        -- 일 기준 마지막 데이터에 농장 상태 정보 join
                        step02 AS (
                                SELECT 
                                    b.frmhs_no, 
                                    b.max_change_dt AS change_dt, 
                                    b.date_trunc, 
                                    a.bsns_sttus_se 
                                FROM 
                                    step01 b
                                LEFT JOIN 
                                    step00 a ON a.frmhs_no = b.frmhs_no
                                              AND a.change_dt = b.max_change_dt
                        ),
                        
                        -- 일 기준 데이터 생성
                        step03 AS (
                                SELECT 
                                    frmhs_no, 
                                    lstksp_cl, 
                                    MAX(change_dt) AS max_blvstvk, 
                                    DATE_TRUNC('day', change_dt) AS date_blvstck 
                                FROM 
                                    asf.tn_mobile_blvstck_hist_2324
                                WHERE change_dt BETWEEN '{one_year_before_date}' AND '{target_date}'
                                GROUP BY 
                                    frmhs_no, lstksp_cl, date_blvstck
                        ),
                        
                        -- 일 기준 농장 데이터에 사육축종정보 join
                        step04 AS (
                            SELECT 
                                a.frmhs_no, 
                                a.lstksp_cl, 
                                a.max_blvstvk, 
                                a.date_blvstck, 
                                b.mgr_code, 
                                b.master_sttus_se, 
                                b.brd_had_co, 
                                b.avg_cnt,
                                b.brd_purps_code,
                                b.mother_pig_co,
                                b.porker_co,
                                b.sear_se
                            FROM 
                                step03 a
                            LEFT JOIN 
                                (SELECT frmhs_no, lstksp_cl, change_dt, mgr_code, master_sttus_se, brd_had_co, avg_cnt, brd_purps_code, mother_pig_co, porker_co, sear_se
                                 FROM asf.tn_mobile_blvstck_hist_2324) b
                                ON a.frmhs_no = b.frmhs_no
                                AND a.lstksp_cl = b.lstksp_cl
                                AND a.max_blvstvk = b.change_dt
                        ),
                        
                        -- 일 기준 농장,축종 정보에 농장 상태 정보 붙이기
                        step05 AS (
                                SELECT a.frmhs_no, a.lstksp_cl, a.max_blvstvk, a.date_blvstck, a.mgr_code, a.master_sttus_se, a.brd_had_co, a.avg_cnt, a.sear_se, b.bsns_sttus_se, a.mother_pig_co, a.porker_co, a.brd_purps_code
                                FROM step04 a
                                LEFT JOIN step02 b
                                ON a.frmhs_no = b.frmhs_no
                                AND a.date_blvstck = b.date_trunc
                        ),
                        
                        -- 운영, 휴업하는 농장 대상으로 과거1년까지의 모돈 평균사육두수 계산
                        avg_brd_had_co_calc AS (
                            SELECT
                                frmhs_no,
                                lstksp_cl,
                                max_blvstvk,
                                brd_had_co,
                                AVG(NULLIF(mother_pig_co, 0)) OVER (
                                    PARTITION BY frmhs_no, lstksp_cl 
                                    ORDER BY max_blvstvk 
                                    RANGE BETWEEN INTERVAL '1 year' PRECEDING AND CURRENT ROW
                                ) AS avg_mother_brd_had_co
                            FROM step05
                            WHERE 
                                (lstksp_cl LIKE '413%' OR lstksp_cl LIKE '419%')
                                AND mgr_code IN ('104001', '104002')
                                AND master_sttus_se IN ('1', '2', '3', 'Z')
                                AND bsns_sttus_se IN ('1', '2')
                                AND frmhs_no != '        '
                                and sear_se = 'Y'
                                and SUBSTR(BRD_PURPS_CODE,7) in ('0003', '0004')
                        ),
                        
                        -- 운영, 휴업하는 농장 대상으로 과거1년까지의 모돈 평균사육두수 join
                        step06 AS (
                                SELECT 
                                    a.frmhs_no, 
                                    a.lstksp_cl, 
                                    a.max_blvstvk AS change_dt, 
                                    a.mgr_code, 
                                    a.master_sttus_se, 
                                    a.brd_had_co, 
                                    a.bsns_sttus_se, 
                                    a.date_blvstck AS date, 
                                    a.brd_purps_code,
                                    a.porker_co,
                                    a.mother_pig_co,
                                    b.avg_mother_brd_had_co
                                FROM 
                                    step05 a
                                LEFT JOIN 
                                    avg_brd_had_co_calc b ON a.frmhs_no = b.frmhs_no
                                                           AND a.lstksp_cl = b.lstksp_cl
                                                           AND a.max_blvstvk = b.max_blvstvk
                        ),
                        
                        -- 월 말 기준 날짜 기준 데이터 생성
                        date_range AS (
                            SELECT
                                generated_date
                            FROM
                                generate_series('{target_date}'::date, '{target_date}'::date, interval '1 month') AS generated_date
                        ),
                        
                        -- 월 말 기준 날짜 기준 데이터에 모돈 평균사육두수 정보 join
                        step08 AS (
                                SELECT
                                    generated_date,
                                    frmhs_no,
                                    lstksp_cl,
                                    brd_had_co,
                                    master_sttus_se,
                                    mgr_code,
                                    bsns_sttus_se, 
                                    avg_mother_brd_had_co,
                                    MAX(change_dt) AS std_dt,
                                    brd_purps_code,
                                    porker_co,
                                    mother_pig_co
                                FROM 
                                    date_range
                                CROSS JOIN
                                    (SELECT * FROM step06  WHERE lstksp_cl LIKE '413%'
                                     AND master_sttus_se IN ('1', '2', '3', 'Z')) a
                                WHERE
                                    change_dt < date_range.generated_date + INTERVAL '1 month'
                                GROUP BY
                                    generated_date,
                                    frmhs_no,
                                    lstksp_cl,
                                    brd_purps_code,
                                    brd_had_co,
                                    master_sttus_se,
                                    mgr_code,
                                    bsns_sttus_se,
                                    avg_mother_brd_had_co,
                                    porker_co,
                                    mother_pig_co
                              ),
                        
                            -- 월 말 기준 날짜 기준 데이터에 축종 상태 데이터 join
                            step09 AS (
                                SELECT 
                                    a.generated_date, 
                                    a.frmhs_no as farm_serial_no,  
                                    a.brd_purps_code, 
                                    a.change_dt, 
                                    b.lstksp_cl,
                                    b.brd_had_co, 
                                    b.master_sttus_se, 
                                    b.mgr_code, 
                                    b.bsns_sttus_se, 
                                    b.avg_mother_brd_had_co,
                                    b.porker_co,
                                    b.mother_pig_co
                                    
                                FROM (
                                    SELECT 
                                        generated_date, 
                                        frmhs_no, 
                                        brd_purps_code, 
                                        MAX(std_dt) AS change_dt 
                                    FROM 
                                        step08
                                    GROUP BY 
                                        generated_date, frmhs_no, brd_purps_code
                                ) a
                                LEFT JOIN 
                                    step08 b ON a.generated_date = b.generated_date
                                             AND a.frmhs_no = b.frmhs_no
                                             AND a.brd_purps_code = b.brd_purps_code
                                             AND a.change_dt = b.std_dt
                            ),
                            
                            -- 운영, 휴업 중인 농장에 해당하는 데이터 필터링
                            filltered_final_step as (
                            select * from step09
                            where bsns_sttus_se IN ('1', '2')
                            AND master_sttus_se IN ('1', '2', '3', 'Z')
                            AND mgr_code IN ('104001', '104002')
                            AND (lstksp_cl LIKE '413%' OR lstksp_cl LIKE '419%')
                            )
                            
                            select * from filltered_final_step"""
                            )

    avg_motehr_brd_had_co = cur.fetchall()
    avg_motehr_brd_had_co = pd.DataFrame(avg_motehr_brd_had_co)
    avg_motehr_brd_had_co.columns = [desc[0] for desc in cur.description]
    conn.close()

    avg_motehr_brd_had_co = avg_motehr_brd_had_co[['farm_serial_no', 'change_dt', 'avg_mother_brd_had_co']]

    # 농장 데이터 기준으로 집계
    avg_motehr_brd_had_co = avg_motehr_brd_had_co.groupby('farm_serial_no')['avg_mother_brd_had_co'].sum().reset_index()

    return avg_motehr_brd_had_co