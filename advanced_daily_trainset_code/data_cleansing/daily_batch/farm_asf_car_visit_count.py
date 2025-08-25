from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB


# 발생 농장을 방문한 목적별 차량방문횟수 생성

def create_farm_asf_car_visit_count(db, standard_date):
    farm_asf_car_visit_count = db.query(f"""WITH 
    
                            -- standard_date 기준으로 차량 역학기간 (21일) 이내 ASF 농장 발생 정보 추출
                            asf_data AS (
                                SELECT DISTINCT ON (frmhs_no, sttemnt_de) 
                                    LPAD(frmhs_no, 8, '0') AS farm_serial_no,
                                    TO_DATE(sttemnt_de, 'YYYYMMDD') AS asf_org_day
                                FROM asf.tn_diss_occrrnc_frmhs_raw_asf_bak oc
                                JOIN asf.tb_farm_geometry_clean_asf fg
                                  ON LPAD(oc.frmhs_no, 8, '0') = fg.farm_serial_no
                                 AND fg.geom_type = 'Point'
                                WHERE diss_cl = '8' 
                                  AND delete_at = 'N' 
                                  AND SLAU_AT = 'N' 
                                  AND CNTRL_DT IS NOT NULL   
                                  AND ESTBS_NM NOT LIKE '%예살%'   
                                  AND ESTBS_NM NOT LIKE '%출하%'
                                  AND ESTBS_NM NOT LIKE '%차량%'
                                  AND ESTBS_NM LIKE '%차%'
                                  AND TO_DATE(sttemnt_de, 'YYYYMMDD') >= ('{standard_date}'::date - INTERVAL '21 days') 
                                  AND TO_DATE(sttemnt_de, 'YYYYMMDD') < ('{standard_date}'::date + INTERVAL '1 days')
                            ),
                            
                            -- ASF 발생일 전 21일 ~ 1일 사이에 방문한 차량 기록 (역학 차량) 추출
                            visit_before_asf AS (
                                SELECT 
                                v.regist_no, v.visit_de, b.farm_serial_no AS occurrence_farm_serial_no, b.asf_org_day
                                FROM asf_data b
                                JOIN m2msys.tn_visit_info_clean v
                                    ON b.farm_serial_no = v.frmhs_no
                                WHERE v.visit_de >= (b.asf_org_day  - INTERVAL '21 days')
                                AND v.visit_de < b.asf_org_day 
                            ),
                            
                            -- 역학 차량에 대해 발생일자 ~ 기준 일자 사이에 방문한 차량 기록 추출
                            visit_after_asf as (
                            SELECT DISTINCT
                                 v.frmhs_no AS farm_serial_no, 
                                 v.regist_no, v.visit_de, v.visit_ty, 
                                 vba.occurrence_farm_serial_no AS occurrence_farm_serial_no
                            FROM visit_before_asf vba
                            JOIN m2msys.tn_visit_info_clean v
                                ON v.regist_no = vba.regist_no
                                AND v.visit_de > vba.visit_de 
                                AND v.visit_de < (DATE_TRUNC('day', vba.visit_de) + INTERVAL '22 days') -- 미만 조건 위해 21일 + 1일 로 작성
                            WHERE v.frmhs_no != vba.occurrence_farm_serial_no AND v.visit_de != vba.visit_de
                            and v.visit_de < ('{standard_date}'::timestamp + INTERVAL '1 days')
                            ),
                            
                            -- 방문 유형별로 농장별 방문 횟수 계산
                            visit_counts AS (
                                    SELECT
                                        farm_serial_no,
                                        visit_ty,
                                        COUNT(*) AS visit_count
                                    FROM visit_after_asf
                                    WHERE visit_ty IS NOT NULL
                                    GROUP BY farm_serial_no, visit_ty
                            ),
                                
                            -- 피벗 테이블 생성
                            pivot_counts AS (
                                    SELECT
                                        farm_serial_no,
                                        COALESCE(SUM(CASE WHEN visit_ty = '01' THEN visit_count END), 0) AS infection_livestock_transport_car_visit_count,
                                        COALESCE(SUM(CASE WHEN visit_ty = '02' THEN visit_count END), 0) AS infection_rawmilk_transport_car_visit_count,
                                        COALESCE(SUM(CASE WHEN visit_ty = '03' THEN visit_count END), 0) AS infection_animal_medicine_transport_car_visit_count,
                                        COALESCE(SUM(CASE WHEN visit_ty = '04' THEN visit_count END), 0) AS infection_feed_transport_car_visit_count,
                                        COALESCE(SUM(CASE WHEN visit_ty = '05' THEN visit_count END), 0) AS infection_livestock_excreta_transport_car_visit_count,
                                        COALESCE(SUM(CASE WHEN visit_ty = '06' THEN visit_count END), 0) AS infection_beddingstraw_transport_car_visit_count,
                                        COALESCE(SUM(CASE WHEN visit_ty = '07' THEN visit_count END), 0) AS infection_compost_transport_car_visit_count,
                                        COALESCE(SUM(CASE WHEN visit_ty = '08' THEN visit_count END), 0) AS infection_animal_medical_car_visit_count,
                                        COALESCE(SUM(CASE WHEN visit_ty = '09' THEN visit_count END), 0) AS infection_artificial_insemination_car_visit_count,
                                        COALESCE(SUM(CASE WHEN visit_ty = '10' THEN visit_count END), 0) AS infection_consulting_car_visit_count,
                                        COALESCE(SUM(CASE WHEN visit_ty = '11' THEN visit_count END), 0) AS infection_specimen_picking_diseasecontrol_car_visit_count,
                                        COALESCE(SUM(CASE WHEN visit_ty = '12' THEN visit_count END), 0) AS infection_machine_repair_car_visit_count,
                                        COALESCE(SUM(CASE WHEN visit_ty = '13' THEN visit_count END), 0) AS infection_egg_transport_car_visit_count,
                                        COALESCE(SUM(CASE WHEN visit_ty = '14' THEN visit_count END), 0) AS infection_roughage_transport_car_visit_count,
                                        COALESCE(SUM(CASE WHEN visit_ty = '15' THEN visit_count END), 0) AS infection_eggtray_transport_car_visit_count,
                                        COALESCE(SUM(CASE WHEN visit_ty = '16' THEN visit_count END), 0) AS infection_byproducts_transport_car_visit_count,
                                        COALESCE(SUM(CASE WHEN visit_ty = '17' THEN visit_count END), 0) AS infection_poultry_shipment_manpower_transport_car_visit_count,
                                        COALESCE(SUM(CASE WHEN visit_ty = '18' THEN visit_count END), 0) AS infection_breeding_facilities_management_car_visit_count,
                                        COALESCE(SUM(CASE WHEN visit_ty = '19' THEN visit_count END), 0) AS infection_livestock_carcass_transport_car_visit_count
                                    FROM visit_counts
                                    GROUP BY farm_serial_no
                            )
                            
                            -- 합계 칼럼 추가
                            SELECT
                                farm_serial_no,
                                infection_livestock_transport_car_visit_count,
                                infection_rawmilk_transport_car_visit_count,
                                infection_animal_medicine_transport_car_visit_count,
                                infection_feed_transport_car_visit_count,
                                infection_livestock_excreta_transport_car_visit_count,
                                infection_beddingstraw_transport_car_visit_count,
                                infection_compost_transport_car_visit_count,
                                infection_animal_medical_car_visit_count,
                                infection_artificial_insemination_car_visit_count,
                                infection_consulting_car_visit_count,
                                infection_specimen_picking_diseasecontrol_car_visit_count,
                                infection_machine_repair_car_visit_count,
                                infection_egg_transport_car_visit_count,
                                infection_roughage_transport_car_visit_count,
                                infection_eggtray_transport_car_visit_count,
                                infection_byproducts_transport_car_visit_count,
                                infection_poultry_shipment_manpower_transport_car_visit_count,
                                infection_breeding_facilities_management_car_visit_count,
                                infection_livestock_carcass_transport_car_visit_count,
                                (infection_livestock_transport_car_visit_count + infection_animal_medicine_transport_car_visit_count + infection_feed_transport_car_visit_count 
                                + infection_livestock_excreta_transport_car_visit_count + infection_beddingstraw_transport_car_visit_count + infection_compost_transport_car_visit_count
                                + infection_animal_medical_car_visit_count + infection_artificial_insemination_car_visit_count + infection_consulting_car_visit_count 
                                + infection_roughage_transport_car_visit_count +  infection_breeding_facilities_management_car_visit_count 
                                + infection_livestock_carcass_transport_car_visit_count) as infection_whole_car_visit_count
                            FROM pivot_counts
                    """)
    return farm_asf_car_visit_count