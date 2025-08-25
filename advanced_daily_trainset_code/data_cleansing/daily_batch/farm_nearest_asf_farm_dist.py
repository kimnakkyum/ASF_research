import pandas as pd
import numpy as np
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

# +++ [이슈사항 정리] +++ ================================================================================================
# 1. *****
# 이슈 코드 : 운영, 휴업 중인 양돈 농장 불러오기
# 이슈 내용 : 쿼리 내에서 운영, 휴업 중인 농장 정보를 불러오는데, 각 함수마다 운영/휴업 중인 농장을 산출하는 코드가 다름. 결과가 같은지 확인 필요
# 개선 방안 : 동일한 농장 대상을 바라보도록 모든 코드 통일 필요

# 2. 
# 이슈 코드 : WHERE s2.standard_date - s1.asf_org_day BETWEEN INTERVAL '0 days' AND INTERVAL '21 days'
# 이슈 내용 : 농장 발생 데이터를 기준일자 포함 -21까지로 추출하여 거리 계산 수행함
# 개선 방안 : 기준일자-1 부터 기준일자-22까지로 변경 필요
# 반영 완료

# 3.
# 이슈 코드 : FROM m2msys.tn_diss_occrrnc_frmhs
# 이슈 내용 : 원천 테이블 미 변경
# 개선 방안 : 180 서버 내 asf.tn_diss_occrrnc_frmhs_raw_asf로 변경 필요
# 변경 완료

# 4.
# 이슈 코드 : FROM monthly_report_partition
# 이슈 내용 : 원천 테이블 미 변경
# 개선 방안 : 180 서버 내 이관된 테이블로 변경 필요
# 변경 완료

# 5.
# 이슈 코드 : where min_distance != 0
# 이슈 내용 : min_distance로 조회 시, 해당 농장과 좌표가 같은 농장 존재할 경우 문제 발생
# 개선 방안 : CROSS JOIN 시 본인은 제외하도록 WHERE 조건 수정 필요
# 반영 완료

# 6.
# 이슈 코드 : # Check if results are empty
# 이슈 내용 : 불필요한 코드로 보임
# 개선 방안 : 역학 기간 내에 발생한 농장이 없을 수 있으므로, 이 코드가 있어야 정상적으로 작동함 -> 그래서 필요함

# 7.
# 이슈 코드 : TO_DATE(sttemnt_de, 'YYYYMMDD') - INTERVAL '1 day' AS asf_org_day,
# 이슈 내용 : 왜 하루 빼는지 확인 필요
# 개선 방안 : 확인 필요
# 확인 완료

# ++ ===================================================================================================================


# 운영, 휴업 농장 대상으로 최근접 농장과의 거리 생성

def create_farm_nearest_asf_farm_dist(db, standard_date):

    farm_nearest_asf_farm_dist = db.query(f"""WITH 
    
                    -- 농장 좌표 데이터
                    farm_point AS (
                        SELECT distinct on (farms_no) farms_no, xmin_, ymin
                        FROM m2msys.nvrqs_mobile_farms_raw
                        WHERE farms_no in (SELECT farm_serial_no FROM asf.tb_pig_farm_serial_no_clean_asf)
                        order by farms_no, std_dt desc
                    ),
                    
                    frmhs_info as (
                        select distinct on (frmhs_no)
                        frmhs_no, bsns_sttus_se
                        from geoai_mt.tn_mobile_frmhs_info_hist_clean
                        where std_dt <= '{standard_date}'
                        order by frmhs_no, std_dt desc
                    ),
    
                    -- 농장 발생 정보 불러오기
                    step01 AS (
                        SELECT DISTINCT ON (frmhs_no, sttemnt_de) 
                            LPAD(frmhs_no, 8, '0') AS farm_serial_no,
                            TO_DATE(sttemnt_de, 'YYYYMMDD') AS asf_org_day,
                            fg.geometry AS asf_farms_point
                        FROM asf.tn_diss_occrrnc_frmhs_raw_asf_bak oc
                        join asf.tb_farm_geometry_clean_asf fg
                        on LPAD(oc.frmhs_no, 8, '0') = fg.farm_serial_no
                        and fg.geom_type = 'Point'
                        WHERE diss_cl = '8' 
                            AND delete_at = 'N' 
                            AND SLAU_AT = 'N' 
                            AND CNTRL_DT IS NOT NULL   
                            AND ESTBS_NM NOT LIKE '%예살%'   
                            AND ESTBS_NM NOT LIKE '%출하%'
                            AND ESTBS_NM NOT LIKE '%차량%'   
                            AND ESTBS_NM LIKE '%차%'
                            AND '{standard_date}'::date BETWEEN TO_DATE(sttemnt_de, 'YYYYMMDD') AND TO_DATE(sttemnt_de, 'YYYYMMDD') + INTERVAL '1 months'
                    ),
                                        
                    -- 농장&축종 테이블 중 기준일자 이전의 가장 최신 hist 정보를 해당 일자 정보로 활용
                    -- 운영/휴업 중인 양돈 농가 추출 & 좌표 매칭
                    step02 AS (
                        SELECT 
                            a.frmhs_no as farm_serial_no, 
                            b.xmin_, b.ymin
                        FROM (
                            SELECT DISTINCT ON (frmhs_no, lstksp_cl, before_lstksp_cl)
                                frmhs_no, bsns_sttus_se, mgr_code, master_sttus_se
                            FROM geoai_mt.tn_mobile_frmhs_blvstck_avg_hist_clean
                            WHERE std_dt <= '{standard_date}'
                              AND lstksp_cl LIKE '413%'
                            ORDER BY frmhs_no, lstksp_cl, before_lstksp_cl, std_dt DESC
                        ) a
                        JOIN farm_point b
                            ON a.frmhs_no = b.farms_no
                        JOIN frmhs_info f
                            ON a.frmhs_no = f.frmhs_no
                        WHERE a.bsns_sttus_se IN ('1', '2')
                            AND f.bsns_sttus_se IN ('1', '2')
                            AND a.mgr_code IN ('104001', '104002')
                            AND a.master_sttus_se IN ('1', '2', '3', 'Z')
                    ),
                    
                    -- 농장 단위로 중복제거하여 양돈 농가 좌표 데이터 추출
                    step03 AS (
                        SELECT distinct on (farm_serial_no)
                        '{standard_date}'::date as standard_date,
                        farm_serial_no, 
                        st_transform(('SRID=4326;POINT(' || xmin_ || ' ' || ymin || ')')::geometry, 5179) as farm_coordinate
                        FROM step02
                    ),

                    -- 기준일자로부터 1달 이내 발생한 농장과의 거리 계산
                    distance_calculations AS (
                        SELECT
                            s3.farm_serial_no AS step02_farms_no,
                            s3.standard_date,
                            s3.farm_coordinate,
                            s1.farm_serial_no AS step01_farms_no,
                            s1.asf_org_day,
                            s1.asf_farms_point,
                            ST_Distance(s3.farm_coordinate, s1.asf_farms_point) AS distance
                        FROM step03 s3
                        CROSS JOIN step01 s1
                        -- 기준일자로부터 1달 이내 발생한 농장 집계
                        WHERE s3.standard_date BETWEEN s1.asf_org_day AND s1.asf_org_day + INTERVAL '1 month'
                        AND s3.farm_serial_no != s1.farm_serial_no -- 본인 제외
                    )
                    
                    -- 최근접 농가 선택
                    SELECT step02_farms_no as farm_serial_no, min(distance) as infection_farm_top_near_distance
                    FROM distance_calculations
                    group by step02_farms_no
                    ORDER BY infection_farm_top_near_distance
                    """)

    # 모든 농장이 역학 기간 내 농장 발생이 존재하지 않을 경우, nan 데이터프레임을 return
    if farm_nearest_asf_farm_dist.shape[0] == 0 :
        farm_nearest_asf_farm_dist = pd.DataFrame(columns=["farm_serial_no", "infection_farm_top_near_distance"])
        farm_nearest_asf_farm_dist.loc[0] = [np.nan] * len(farm_nearest_asf_farm_dist.columns)

    farm_nearest_asf_farm_dist['farm_serial_no'] = farm_nearest_asf_farm_dist['farm_serial_no'].astype(str)
    return farm_nearest_asf_farm_dist