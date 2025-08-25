import pandas as pd
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB
import numpy as np

# +++ [이슈사항 정리] +++ ================================================================================================
# 1. *****
# 이슈 코드 : 운영, 휴업 중인 양돈 농장 불러오기
# 이슈 내용 : 쿼리 내에서 운영, 휴업 중인 농장 정보를 불러오는데, 각 함수마다 운영/휴업 중인 농장을 산출하는 코드가 다름. 결과가 같은지 확인 필요
# 개선 방안 : 동일한 농장 대상을 바라보도록 모든 코드 통일 필요

# 2.
# 이슈 코드 : df['present_breeding_livestock_count'] = np.where(df['present_breeding_livestock_count'] >= 1, 1, 0)
# 이슈 내용 : 현재사육두수가 1보다 크면 여(1) 1보다 작으면 부(0) 로직이 맞는지 검토
# 개선 방안 : 확인 완료. 그대로 활용하면 됨

# [낙겸 이슈사항 정리] +++ ================================================================================================
# 1.
# 이슈 코드 : FROM dm.tb_farm_information_clean WHERE standard_date = '{standard_date}'
# 이슈 내용 : 농장 운영 코드 farm_operation_status_code 조건 없음
# 개선 방안 : 농장 운영 코드 farm_operation_status_code in ('1', '2') 추가 필요



# 운영 ,휴업 중인 양돈 농장 및 축종 별 사육 여부 생성

def create_farm_breeding_yn(db, standard_date):

    farm_breeding_yn = db.query(f"""WITH

                    frmhs_info AS (
                        SELECT DISTINCT ON (frmhs_no)
                            frmhs_no,
                            bsns_sttus_se
                        FROM geoai_mt.tn_mobile_frmhs_info_hist_clean
                        WHERE std_dt <= '{standard_date}'
                        ORDER BY frmhs_no, std_dt DESC
                    ),

                    base_farm AS (
                        SELECT DISTINCT ON (frmhs_no, lstksp_cl, before_lstksp_cl)
                            frmhs_no,
                            lstksp_cl,
                            brd_had_co,
                            mother_pig_co,
                            porker_co,
                            bsns_sttus_se,
                            mgr_code,
                            master_sttus_se
                        FROM geoai_mt.tn_mobile_frmhs_blvstck_avg_hist_clean_asf
                        WHERE std_dt <= '{standard_date}'
                          AND lstksp_cl LIKE '413%'
                        ORDER BY frmhs_no, lstksp_cl, before_lstksp_cl, std_dt DESC
                    ),

                    filtered_farm AS (
                        SELECT
                            a.frmhs_no AS farm_serial_no,
                            a.lstksp_cl,
                            CASE WHEN a.brd_had_co > 0 THEN 1 ELSE 0 END AS breeding_yn,
                            CASE WHEN a.mother_pig_co > 0 THEN 1 ELSE 0 END AS mother_pig_breeding_yn,
                            CASE WHEN a.porker_co > 0 THEN 1 ELSE 0 END AS porker_breeding_yn
                        FROM base_farm a
                        JOIN frmhs_info f ON a.frmhs_no = f.frmhs_no
                        WHERE a.bsns_sttus_se IN ('1', '2')
                          AND f.bsns_sttus_se IN ('1', '2')
                          AND a.mgr_code IN ('104001', '104002')
                          AND a.master_sttus_se IN ('1', '2', '3', 'Z')
                    ),

                    -- 일반 축종 피벗 처리 (breeding_yn 기준)
                    breeding_yn_pivot AS (
                        SELECT
                            farm_serial_no,
                            MAX(CASE WHEN lstksp_cl = '413016' THEN breeding_yn ELSE 0 END) AS common_pig_breeding_yn,
                            MAX(CASE WHEN lstksp_cl = '413015' THEN breeding_yn ELSE 0 END) AS black_pig_breeding_yn,
                            MAX(CASE WHEN lstksp_cl = '413013' THEN breeding_yn ELSE 0 END) AS breeding_pig_breeding_yn,
                            MAX(CASE WHEN lstksp_cl = '413014' THEN breeding_yn ELSE 0 END) AS captive_wildboar_breeding_yn
                        FROM filtered_farm
                        GROUP BY farm_serial_no
                    ),

                    -- 모돈/비육돈 정보 별도 max 집계
                    mother_porker_breeding_yn_unique AS (
                        SELECT
                            farm_serial_no,
                            MAX(mother_pig_breeding_yn) AS mother_pig_breeding_yn,
                            MAX(porker_breeding_yn) AS porker_breeding_yn
                        FROM filtered_farm
                        GROUP BY farm_serial_no
                    )

                    -- 최종 결과 조합
                    SELECT
                        b.farm_serial_no,
                        common_pig_breeding_yn,
                        black_pig_breeding_yn,
                        breeding_pig_breeding_yn,
                        captive_wildboar_breeding_yn,
                        u.mother_pig_breeding_yn,
                        u.porker_breeding_yn
                    FROM breeding_yn_pivot b
                    LEFT JOIN mother_porker_breeding_yn_unique u
                        ON b.farm_serial_no = u.farm_serial_no
    """)

    return farm_breeding_yn