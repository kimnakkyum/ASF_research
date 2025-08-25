from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB


######## 운영, 휴업 중인 농장 반경 이내 양돈 농장 수 생성 (산출 시간 이슈로 개선 필요)

# 농장 반경 내 양돈 농가 수 count
def create_farm_around_farm_count(db, standard_date):

    # 주변 모든 농장 수 산출
    whole_farm_with_counts = db.query(f'''with 
    
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
                        
                        farm_list AS (
                            SELECT 
                                 distinct on (a.frmhs_no)
                                 a.frmhs_no as farm_serial_no,
                                 b.geometry
                            FROM (
                                SELECT DISTINCT ON (m.frmhs_no, m.lstksp_cl, m.before_lstksp_cl)
                                    m.frmhs_no, m.bsns_sttus_se, m.mgr_code, m.master_sttus_se
                                FROM geoai_mt.tn_mobile_frmhs_blvstck_avg_hist_clean m
                                WHERE m.std_dt <= '{standard_date}'
                                ORDER BY m.frmhs_no, m.lstksp_cl, m.before_lstksp_cl, m.std_dt DESC
                            ) a
                            JOIN (select * from asf.tb_whole_farm_geometry_clean_asf) b
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
                     
                        -- 4. 반경별 거리 계산
                        radius_count AS (
                            SELECT
                                a.farm_serial_no,
                                COUNT(*) FILTER (WHERE ST_DWithin(a.geometry, b.geometry, 1000))-1 AS count_1km,
                                COUNT(*) FILTER (WHERE ST_DWithin(a.geometry, b.geometry, 3000))-1 AS count_3km,
                                COUNT(*) FILTER (WHERE ST_DWithin(a.geometry, b.geometry, 6000))-1 AS count_6km,
                                COUNT(*) FILTER (WHERE ST_DWithin(a.geometry, b.geometry, 9000))-1 AS count_9km,
                                COUNT(*) FILTER (WHERE ST_DWithin(a.geometry, b.geometry, 12000))-1 AS count_12km
                            FROM farm_list a
                            JOIN farm_list b
                              ON ST_DWithin(a.geometry, b.geometry, 12000)  -- 최대 반경 기준 join
                            GROUP BY a.farm_serial_no
                        )
                        
                        -- 최종 결과
                        SELECT 
                            farm_serial_no,
                            count_1km AS radius_1km_whole_farm_count,
                            count_3km AS radius_3km_whole_farm_count,
                            count_6km AS radius_6km_whole_farm_count,
                            count_9km AS radius_9km_whole_farm_count,
                            count_12km AS radius_12km_whole_farm_count
                        FROM radius_count
    ''')

    return whole_farm_with_counts