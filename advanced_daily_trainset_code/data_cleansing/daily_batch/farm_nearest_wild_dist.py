import pandas as pd
import numpy as np
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

# +++ [이슈사항 정리] +++ ================================================================================================

# 1.
# 이슈 코드 : FROM asf.tn_diss_occ_ntcn_info
# 이슈 내용 : 원천 테이블 미 변경
# 개선 방안 : 180 서버 내 m2msys.tn_diss_occ_ntcn_info_raw로 변경 필요

# 2.
# 이슈 코드 : SELECT * FROM step00 WHERE std_dt > '{standard_date}'::date - interval '21 day'
# 이슈 내용 : 기준일자 - 21 초과로만 필터링 됨
# 개선 방안 : 기준일자-1 부터 기준일자-22까지로 변경 필요
# 농장 발생 이슈 해결 후 반영 예정

# 3.
# 이슈 코드 : 운영, 휴업 중인 양돈 농장 불러오기
# 이슈 내용 : 쿼리 내에서 운영, 휴업 중인 농장 정보를 불러오는데, 각 함수마다 운영/휴업 중인 농장을 산출하는 코드가 다름. 결과가 같은지 확인 필요
# 개선 방안 : 동일한 농장 대상을 바라보도록 모든 코드 통일 필요

# 4.
# 이슈 코드 : ON s1.std_dt BETWEEN s2.standard_date - INTERVAL '21 days' AND s2.standard_date
# 이슈 내용 : 야생 발생 데이터를 기준일자 포함 -21까지로 추출하여 거리 계산 수행함
# 개선 방안 : 기준일자-1 부터 기준일자-22까지로 변경 필요
# 농장 발생 이슈 해결 후 반영 예정

# 5.
# 이슈 코드 : select *, DATE_TRUNC('month', change_dt)::date AS std_dt from tot
# 이슈 내용 : standard_date에서 month만 짜를 필요가 없음
# 개선 방안 : standard_date 그대로 추출 필요
# 반영 완료

# 6.
# 이슈 코드 : distance_calculations 내 JOIN
# 이슈 내용 : CROSS JOIN으로 하는 것이 좀 더 표현이 명확해보임
# 개선 방안 : CROSS JOIN으로 수정
# 반영 완료


# [낙겸 이슈사항 정리] +++ ================================================================================================
# 1.
# 이슈 코드 : FROM dm.tb_farm_information_clean WHERE standard_date = '{standard_date}'
# 이슈 내용 : 농장 운영 코드 farm_operation_status_code 조건 없음
# 개선 방안 : 농장 운영 코드 farm_operation_status_code in ('1', '2') 추가 필요

# 최근접 야생발생과의 거리 생성
def create_farm_nearest_wild_dist(db, standard_date):

    farm_nearest_wild_dist = db.query(f"""WITH 
    
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
    
                                -- 야생멧돼지 야생발생 데이터 불러오기
                                step01 AS (
                                    SELECT distinct on (odr)
                                    TO_DATE(sttemnt_de, 'YYYYMMDD') AS asf_org_day,
                                    ST_Transform(ST_SetSRID(ST_Point(lo, la), 4326), 5179) AS asf_wild_point
                                    FROM m2msys.tn_diss_occ_ntcn_info_raw
                                    WHERE '{standard_date}'::date BETWEEN TO_DATE(sttemnt_de, 'YYYYMMDD') AND TO_DATE(sttemnt_de, 'YYYYMMDD') + INTERVAL '1 year'
                                    order by odr, last_change_dt desc
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

                                -- 기준일자로부터 6개월/1년 이내 발생한 야생 발생과의 거리 계산
                                distance_calculations AS (
                                    SELECT
                                        s3.farm_serial_no,
                                        CASE 
                                            WHEN s3.standard_date BETWEEN s1.asf_org_day AND s1.asf_org_day + INTERVAL '6 months'
                                            THEN ST_Distance(s3.farm_coordinate, s1.asf_wild_point)
                                            ELSE NULL
                                        END AS distance_6month,
                                        CASE 
                                            WHEN s3.standard_date BETWEEN s1.asf_org_day AND s1.asf_org_day + INTERVAL '1 year'
                                            THEN ST_Distance(s3.farm_coordinate, s1.asf_wild_point)
                                            ELSE NULL
                                        END AS distance_1year
                                    FROM step03 s3
                                    CROSS JOIN step01 s1
                                ),
                                
                                -- 농장별로 최단 거리만 남기기
                                aggregated_distances AS (
                                    SELECT
                                        farm_serial_no,
                                        MIN(distance_6month) AS specimen_picking_top_near_distance_6month,
                                        MIN(distance_1year) AS specimen_picking_top_near_distance_1year
                                    FROM distance_calculations
                                    GROUP BY farm_serial_no
                                )
                                
                                SELECT *
                                FROM aggregated_distances
                                ORDER BY specimen_picking_top_near_distance_6month NULLS LAST, specimen_picking_top_near_distance_1year NULLS LAST
                                """)

    # 모든 농장이 역학 기간 내 야생 발생이 존재하지 않을 경우, nan 데이터프레임을 return
    if farm_nearest_wild_dist.shape[0] == 0 :
        farm_nearest_wild_dist = pd.DataFrame(columns=["farm_serial_no", "specimen_picking_top_near_distance_6month", "specimen_picking_top_near_distance_1year"])
        farm_nearest_wild_dist.loc[0] = [np.nan] * len(farm_nearest_wild_dist.columns)

    farm_nearest_wild_dist['farm_serial_no'] = farm_nearest_wild_dist['farm_serial_no'].astype(str)
    return farm_nearest_wild_dist