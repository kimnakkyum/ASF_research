import pandas as pd
import geopandas as gpd
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB
from geoalchemy2 import Geometry


# +++ [이슈사항 정리] +++ ================================================================================================
# 1. *****
# 이슈 코드 : 운영, 휴업 중인 양돈 농장 불러오기
# 이슈 내용 : 쿼리 내에서 운영, 휴업 중인 농장 정보를 불러오는데, 각 함수마다 운영/휴업 중인 농장을 산출하는 코드가 다름. 결과가 같은지 확인 필요
# 개선 방안 : 동일한 농장 대상을 바라보도록 모든 코드 통일 필요

# 2. *****
# 이슈 코드 : WHERE rn = 1
# 이슈 내용 : 생성 시 rnk = 1의 조건에 의해 대표 축종에 대한 평균사육두수만 고려하고 있음.
# 개선 방안 : 평균사육두수를 sum하는 로직으로 수정 필요
# 특이사항 : 반영 했는데, 코드 한번 더 확인 필요

# [낙겸 이슈사항 정리] +++ ===================================================================================================================
# 1. *****
# 이슈 코드 : FROM dm.tb_farm_information_clean WHERE standard_date = '{standard_date}'
# 이슈 내용 : 농장 운영 코드 farm_operation_status_code 조건 없음
# 개선 방안 : 농장 운영 코드 farm_operation_status_code in ('1', '2') 추가 필요

# 운영 ,휴업 중인 양돈 농장 및 평균사육두수 생성
def create_farm_base_info(db, standard_date):

    query  = f"""with 
    
                -- 여러 축종을 사육하는 농장의 경우, 농장 운영 상태가 반영이 안되고 축종 운영 상태만 반영되는 이슈가 존재하여 해당 과정을 추가하여 해당 이슈 해결
                frmhs_info as (
                    select distinct on (frmhs_no)
                    frmhs_no, bsns_sttus_se
                    from geoai_mt.tn_mobile_frmhs_info_hist_clean
                    where std_dt <=  '{standard_date}'
                    order by frmhs_no, std_dt desc
                ),
                
                -- 기준일자 전의 농장 발생 정보 불러오기
                asf_farm AS (
                    SELECT DISTINCT ON (frmhs_no, sttemnt_de) 
                        LPAD(frmhs_no, 8, '0') AS farm_serial_no,
                        TO_DATE(sttemnt_de, 'YYYYMMDD') AS asf_org_day
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
                        AND TO_DATE(sttemnt_de, 'YYYYMMDD') < '{standard_date}'
                ),

                -- 농장&축종 테이블 중 기준일자 이전의 가장 최신 hist 정보를 해당 일자 정보로 활용
                -- 운영/휴업 중인 양돈 농가 추출 & 좌표 매칭
                step01 AS (
                    SELECT 
                        a.frmhs_no as farm_serial_no, 
                        a.brd_had_co, 
                        a.mother_pig_co, 
                        a.porker_co, 
                        a.avg_brd_had_co, 
                        a.avg_mother_pig_co,
                        a.avg_porker_co,
                        b.geometry
                    FROM (
                        SELECT DISTINCT ON (m.frmhs_no, m.lstksp_cl, m.before_lstksp_cl)
                            m.frmhs_no, m.bsns_sttus_se, m.mgr_code, m.master_sttus_se,
                            m.brd_had_co,
                            m.mother_pig_co,
                            m.porker_co,
                            m.avg_brd_had_co,
                            m.avg_mother_pig_co,
                            m.avg_porker_co
                        FROM geoai_mt.tn_mobile_frmhs_blvstck_avg_hist_clean_asf m
                        WHERE m.std_dt <=  '{standard_date}'
                          AND m.lstksp_cl LIKE '413%'
                        ORDER BY m.frmhs_no, m.lstksp_cl, m.before_lstksp_cl, m.std_dt DESC
                    ) a
                    JOIN (select * from asf.tb_farm_geometry_clean_asf where  geom_type = 'Point') b
                        ON a.frmhs_no = b.farm_serial_no
                    JOIN frmhs_info f
                        ON a.frmhs_no = f.frmhs_no
                    LEFT JOIN asf_farm af
                        ON a.frmhs_no = af.farm_serial_no
                    WHERE a.bsns_sttus_se IN ('1', '2')
                        AND f.bsns_sttus_se in ('1', '2')
                        AND a.mgr_code IN ('104001', '104002')
                        AND a.master_sttus_se IN ('1', '2', '3', 'Z')
                        AND (
                            af.asf_org_day IS NULL -- 미발생 농장은 모두 포함
                            OR 
                            (
                                af.asf_org_day IS NOT NULL
                                and a.bsns_sttus_se = '1'
                                and f.bsns_sttus_se = '1'
                                and a.mgr_code = '104001'
                                and a.master_sttus_se in ('1', '2', '3', 'Z')
                            ) -- 발생 이력이 있을 경우, 휴업 중인 농장은 분석에서 제외
                            )
                    ),
                    
                     -- 농장 단위로 groupby 하여 현재사육수수, 모돈사육수수, 비육돈사육수수, 평균사육수수 합산 산출
                    step03 AS (
                        SELECT 
                            farm_serial_no,
                             '{standard_date}' as standard_date,
                            sum(brd_had_co) as present_breeding_livestock_count, 
                            sum(COALESCE(mother_pig_co, 0)) as present_mother_pig_breeding_livestock_count,
                            sum(COALESCE(porker_co, 0)) as present_porker_breeding_livestock_count,
                            sum(COALESCE(avg_brd_had_co, 0)) as average_breeding_livestock_count,
                            sum(COALESCE(avg_mother_pig_co, 0)) as average_mother_pig_breeding_livestock_count,
                            sum(COALESCE(avg_porker_co, 0)) as average_porker_breeding_livestock_count,
                            geometry AS farm_coordinate,
                            ST_X(geometry) AS farm_longitude,
                            ST_Y(geometry) AS farm_latitude
                        FROM step01
                        GROUP BY farm_serial_no, geometry
                    )
                
                    select * from step03"""

    conn = db.engine.raw_connection()
    farm_base_info = gpd.read_postgis(query, con=conn, geom_col='farm_coordinate')
    conn.close()

    return farm_base_info