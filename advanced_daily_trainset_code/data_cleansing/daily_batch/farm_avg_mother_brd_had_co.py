import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

# +++ [이슈사항 정리] +++ ================================================================================================
# 1. *****
# 이슈 코드 : 모돈 정의 쿼리
# 이슈 내용 : 전달 받은 모돈 정의와 해당 사항 기반 생성 쿼리의 일치 여부가 확실치 않음
# 개선 방안 : sm의 모돈 정의에 맞게 로직 설계됐는지 검토 필요 (https://kncom.slack.com/archives/C0757L49WE6/p1731571014753289)

# 2.
# 이슈 코드 : group by, select alias 순서
# 이슈 내용 : group by에 select에서 alias 선언한 컬럼이 먼저 사용됨. 왜 에러가 안나지
# 개선 방안 : group by에 alias 안먹은 컬럼 그대로 기입
# 반영 완료

# 3.
# 이슈 코드 : 원천 테이블명 미 변경
# 이슈 내용 : 원천 테이블명 미 변경
# 개선 방안 : 180 서버 내 이관된 테이블로 변경 필요
# 반영 완료

# 4.
# 이슈 코드 : AND (lstksp_cl LIKE '413%' OR lstksp_cl LIKE '419%')
# 이슈 내용 : 왜 419 야생 멧돼지가 포함되는지?
# 개선 방안 :

# 5.
# 이슈 코드
# where master_sttus_se IN ('1', '2', '3', 'Z')
# AND bsns_sttus_se IN ('1', '2')
# AND mgr_code IN ('104001', '104002')
# AND (lstksp_cl LIKE '413%' OR lstksp_cl LIKE '419%')
# 이슈 내용 : 맨 뒤에 필터링이 들어가다보니, 앞에서 연산이 무거움
# 개선 방안 :




# +++ [낙겸 이슈사항 정리] +++===================================================================================================================
# 1.
# 이슈 코드
# from m2msys.tn_mobile_frmhs_info_hist_raw
# WHERE change_dt BETWEEN '{one_year_before_date}' AND '{standard_date}'
# from m2msys.tn_mobile_blvstck_hist_raw
# WHERE change_dt BETWEEN '{one_year_before_date}' AND '{standard_date}'
# 이슈 내용 : standard_date 11:59분 59초까지 포함되는 것이 아닌 00시 00:00 00초까지만 포함됨
# 개선 방안 : change_dt >= '{one_year_before_date}' and chage_dt < '{standard_date}' + 1day




# ++ ===================================================================================================================

# 모돈 평균사육두수 생성

def create_avg_mother_brd_had_co(db, standard_date):

    standard_date_dt = datetime.strptime(standard_date, '%Y-%m-%d')
    one_year_before = standard_date_dt - relativedelta(years=1)
    one_year_before_date = one_year_before.strftime('%Y-%m-%d')

    farm_avg_mother_brd_had_co = db.query(f"""WITH
    
                                    -- 해당 시점 기준으로 과거 1년까지의 농장 데이터 불러오기
                                    step00 AS (
                                            select 
                                                frmhs_no, 
                                                change_dt, 
                                                MAX(fclty_sn) AS fclty_sn, 
                                                DATE_TRUNC('day', change_dt) AS date_trunc,
                                                bsns_sttus_se
                                            from m2msys.tn_mobile_frmhs_info_hist_raw
                                            WHERE 
                                                change_dt BETWEEN '{one_year_before_date}' AND '{standard_date}'
                                                AND bsns_sttus_se IN ('1', '2')
                                                AND delete_at = 'N'
                                            group by
                                                frmhs_no, change_dt, DATE_TRUNC('day', change_dt), bsns_sttus_se
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
                                    -- 이렇게 한 이유는, 과거 일자 별로 이 농장의 운영/휴업 여부를 판별하기 위함
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
            
                                    -- 축종 일 기준 데이터 생성, change_dt 기준 마지막 시간을 가져옴
                                    step03 AS (
                                            SELECT 
                                                frmhs_no, 
                                                lstksp_cl, 
                                                MAX(change_dt) AS max_blvstvk, 
                                                DATE_TRUNC('day', change_dt) AS date_blvstck 
                                            FROM 
                                                m2msys.tn_mobile_blvstck_hist_raw
                                            WHERE change_dt BETWEEN '{one_year_before_date}' AND '{standard_date}'
                                                AND lstksp_cl LIKE '413%'
                                                AND master_sttus_se IN ('1', '2', '3', 'Z') 
                                                AND mgr_code IN ('104001', '104002')
                                                AND sear_se = 'Y'
                                                AND TRIM(frmhs_no) != ''
                                            GROUP BY 
                                                frmhs_no, lstksp_cl, DATE_TRUNC('day', change_dt)
                                    ),
            
                                    -- 축종 일, change_dt 기준 마지막 데이터에 사육축종정보 join
                                    -- mgr_code : 축종운영상태(방역본부)
                                    -- master_sttus_se : 축종운영상태
                                    -- brd_had_co : 사육두수
                                    -- sear_se : 과거 더미데이터 조회방지용 칼럼으로 해당부분이 Y가 아닌 경우는 과거 이력형식으로 남아 있는 데이터로서 현재 운용중인 농가로 파악하지 않음
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
                                        LEFT JOIN (
                                            SELECT frmhs_no, lstksp_cl, change_dt, mgr_code, master_sttus_se, brd_had_co, avg_cnt, brd_purps_code, mother_pig_co, porker_co, sear_se
                                                FROM m2msys.tn_mobile_blvstck_hist_raw
                                            WHERE lstksp_cl LIKE '413%'
                                                AND master_sttus_se IN ('1', '2', '3', 'Z') 
                                                AND mgr_code IN ('104001', '104002')
                                                AND sear_se = 'Y'
                                                AND TRIM(frmhs_no) != ''
                                            ) b
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
            
                                    -- 과거1년까지의 모돈 평균사육두수 계산
                                    -- 여기서 모돈 정의하는 부분이 sm 채널 공유된 것과 같은지 확인이 필요함
                                    -- RANGE BETWEEN INTERVAL '1 year' PRECEDING AND CURRENT ROW : max_blvstck 기준 과거 1년치 데이터부터 현재까지의 범위를 윈도우로 설정
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
                                        WHERE (SUBSTR(BRD_PURPS_CODE,7) = '0003')
                                        OR ((SUBSTR(BRD_PURPS_CODE,7) = '0004') AND (mother_pig_co >0))
                                    ),
            
                                    -- 앞서 산출한 농장 대상으로 과거1년까지의 모돈 평균사육두수 join
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
                                            generate_series('{standard_date}'::date, '{standard_date}'::date, interval '1 month') AS generated_date
                                    ),
            
                                    -- 월 말 기준 날짜 기준 데이터에 모돈 평균사육두수 정보 join
                                    step08 AS (
                                            SELECT
                                                generated_date,
                                                frmhs_no,
                                                lstksp_cl,
                                                brd_purps_code,
                                                brd_had_co,
                                                master_sttus_se,
                                                mgr_code,
                                                bsns_sttus_se, 
                                                avg_mother_brd_had_co,
                                                MAX(change_dt) AS std_dt,
                                                porker_co,
                                                mother_pig_co
                                            FROM 
                                                date_range
                                            CROSS JOIN
                                                step06
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
                                    )
        
                                    -- 농장 별 모돈평균사육수수 합산
                                    SELECT farm_serial_no, sum(avg_mother_brd_had_co)
                                    FROM step09
                                    GROUP BY farm_serial_no
                                    """
                )
    return farm_avg_mother_brd_had_co