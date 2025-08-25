import pandas as pd
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

# +++ [이슈사항 정리] +++ ================================================================================================
# 1. *****
# 이슈 코드 : 모돈 정의 쿼리
# 이슈 내용 : 전달 받은 모돈 정의와 해당 사항 기반 생성 쿼리의 일치 여부가 확실치 않음
# 개선 방안 : sm의 모돈 정의에 맞게 로직 설계됐는지 검토 필요 (https://kncom.slack.com/archives/C0757L49WE6/p1731571014753289)

# 2.
# 이슈 코드 : asf.tn_mobile_blvstck_hist_2324
# 이슈 내용 : 원천 테이블명 미 변경
# 개선 방안 : 180 서버 내 m2msys.tn_mobile_blvstck_hist_raw로 변경 필요
# 반영 완료

# ++ ===================================================================================================================

# 모돈 사육 여부 생성
def create_farm_mother_pig_breeding_yn(db, standard_date):

    farm_mother_pig_breeding_yn = db.query(f"""WITH
                                step01 AS (
                                    -- 돼지 축종이면서 BRD_PURPS_CODE의 끝 4자리가 0003인 데이터 추출
                                    SELECT *, TO_DATE(TO_CHAR(change_dt, 'YYYY-MM') || '-01', 'YYYY-MM-DD') AS std_dt
                                        FROM m2msys.tn_mobile_blvstck_raw -- 전체 기간에 대한 blvstck 마스터 테이블 적재 필요
                                     WHERE 1=1
                                        AND DATE_TRUNC('day', change_dt) = '{standard_date}'
                                        AND MASTER_STTUS_SE IN ('1','2','3','Z')
                                        AND MGR_CODE IN ('104001','104002')
                                        AND SEAR_SE = 'Y'
                                        AND lstksp_cl LIKE '413%'
                                        AND SUBSTR(BRD_PURPS_CODE,7) = '0003'
                                        AND TRIM(frmhs_no) != ''
                
                                    UNION ALL
                
                                    -- 돼지 축종이면서 BRD_PURPS_CODE의 끝 4자리가 0004이고 현재 모돈사육수수가 0 초과인 데이터 추출
                                    SELECT *, TO_DATE(TO_CHAR(change_dt, 'YYYY-MM') || '-01', 'YYYY-MM-DD') AS std_dt
                                        FROM m2msys.tn_mobile_blvstck_raw -- 전체 기간에 대한 blvstck 마스터 테이블 적재 필요
                                     WHERE 1=1
                                        AND DATE_TRUNC('day', change_dt) = '{standard_date}'
                                        AND MASTER_STTUS_SE IN ('1','2','3','Z')
                                        AND MGR_CODE IN ('104001','104002')
                                        AND SEAR_SE = 'Y'
                                        AND lstksp_cl LIKE '413%'
                                        AND SUBSTR(BRD_PURPS_CODE,7) = '0004'
                                        AND MOTHER_PIG_CO > 0
                                        AND TRIM(frmhs_no) != ''
                                )
                                
                                SELECT DISTINCT ON (frmhs_no)
                                frmhs_no as farm_serial_no,
                                1 AS mother_at
                                FROM step01
                     """)
    return farm_mother_pig_breeding_yn