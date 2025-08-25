import pandas as pd
import geopandas as gpd
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB
from pyproj import Transformer

# 운영 ,휴업 중인 모든 축종을 사육하는 농장 생성
def create_all_farms(db, standard_date):

    all_farms = db.query(f"""with 
                    
                    -- 농장 정보 추출
                    frmhs_info as (
                        select distinct on (frmhs_no)
                        frmhs_no, bsns_sttus_se
                        from geoai_mt.tn_mobile_frmhs_info_hist_clean
                        where std_dt <= '{standard_date}'
                        order by frmhs_no, std_dt desc
                    ),
                    
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
                            a.frmhs_no as farm_serial_no
                        FROM (
                            SELECT DISTINCT ON (m.frmhs_no, m.lstksp_cl, m.before_lstksp_cl)
                                m.frmhs_no, m.bsns_sttus_se, m.mgr_code, m.master_sttus_se
                            FROM geoai_mt.tn_mobile_frmhs_blvstck_avg_hist_clean m
                            WHERE m.std_dt <= '{standard_date}'
                            ORDER BY m.frmhs_no, m.lstksp_cl, m.before_lstksp_cl, m.std_dt DESC
                        ) a
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
                            )
    
                    select distinct farm_serial_no ,'{standard_date}' as standard_date from step01
    """)

    # 농장 좌표 정보 불러오기
    location = db.query(f"""
                    SELECT distinct on (farms_no) farms_no, xmin_ as farm_longitude, ymin as farm_latitude
                    FROM m2msys.nvrqs_mobile_farms_raw
                    order by farms_no, std_dt desc
                    """)

    ## 모든 농장 정보 불러올 떄, 좌표 정보를 붙이면 시간이 많이 걸려 별도로 불러와서 파이썬에서 merge 작업 수행
    all_farms = all_farms.merge(location, left_on='farm_serial_no', right_on='farms_no')

    # 좌표 변환 (EPSG:4326 -> EPSG:5179)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
    all_farms['farm_longitude'], all_farms['farm_latitude'] = transformer.transform(all_farms['farm_longitude'].values, all_farms['farm_latitude'].values)

    return all_farms