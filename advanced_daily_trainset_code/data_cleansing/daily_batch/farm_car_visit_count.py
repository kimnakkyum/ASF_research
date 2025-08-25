from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB


# 목적별 차량방문횟수 생성

def create_farm_car_visit_count(db, standard_date):

    farm_car_visit_count = db.query(f"""WITH
                     
                     frmhs_info AS (
                        select distinct on (frmhs_no)
                        frmhs_no, bsns_sttus_se
                        from geoai_mt.tn_mobile_frmhs_info_hist_clean
                        where std_dt <= '{standard_date}'
                        order by frmhs_no, std_dt desc
                    ),
                     
                     -- 운영, 휴업 중인 양돈 농장 대상으로 차량방문정보 추출
                     farm AS (
                        SELECT DISTINCT ON (frmhs_no, lstksp_cl, before_lstksp_cl)
                            frmhs_no, lstksp_cl, brd_had_co, bsns_sttus_se, mgr_code, master_sttus_se
                        FROM geoai_mt.tn_mobile_frmhs_blvstck_avg_hist_clean
                        WHERE std_dt <= '{standard_date}'
                          AND lstksp_cl LIKE '413%'
                        ORDER BY frmhs_no, lstksp_cl, before_lstksp_cl, std_dt DESC
                    ),
                    
                    -- 역학 기간(21일) 내 농장을 방문한 차량 정보 집계
                    step00 AS (
                        SELECT 
                            frmhs_no as farm_serial_no,
                            regist_no as car_registration_no,
                            visit_de as visit_datetime,
                            visit_ty as car_visit_purpose_code
                        FROM m2msys.tn_visit_info_clean
                        WHERE frmhs_no != '00000000'
                          AND visit_de >= '{standard_date}'::date - INTERVAL '21 days'
                          AND visit_de < '{standard_date}'::date + INTERVAL '1 days'
                          AND regist_no IS NOT NULL
                          AND frmhs_no IN (SELECT f.frmhs_no FROM farm f
                                            JOIN frmhs_info fi
                                                ON f.frmhs_no = fi.frmhs_no
                                            WHERE f.bsns_sttus_se IN ('1','2')
                                                AND fi.bsns_sttus_se IN ('1', '2')
                                                AND f.mgr_code IN ('104001','104002') 
                                                AND f.master_sttus_se IN ('1','2','3','Z')
                                                )
                    ),
                    -- 농장별, 차량별, 목적별 방문횟수 산정
                    step01 AS (
                        SELECT
                            farm_serial_no,
                            car_registration_no,
                            car_visit_purpose_code,
                            COUNT(*) AS visit_count
                        FROM step00
                        GROUP BY
                            farm_serial_no,
                            car_registration_no,
                            car_visit_purpose_code
                    ),
                    -- 농장별, 목적별 방문횟수 산정
                    step02 AS (
                        SELECT 
                            farm_serial_no,
                            SUM(CASE WHEN visit_count > 0 AND car_visit_purpose_code = '01' THEN visit_count END) AS livestock_transport_car_visit_count,
                            SUM(CASE WHEN visit_count > 0 AND car_visit_purpose_code = '02' THEN visit_count END) AS rawmilk_transport_car_visit_count,
                            SUM(CASE WHEN visit_count > 0 AND car_visit_purpose_code = '03' THEN visit_count END) AS animal_medicine_transport_car_visit_count,
                            SUM(CASE WHEN visit_count > 0 AND car_visit_purpose_code = '04' THEN visit_count END) AS feed_transport_car_visit_count,
                            SUM(CASE WHEN visit_count > 0 AND car_visit_purpose_code = '05' THEN visit_count END) AS livestock_excreta_transport_car_visit_count,
                            SUM(CASE WHEN visit_count > 0 AND car_visit_purpose_code = '06' THEN visit_count END) AS beddingstraw_transport_car_visit_count,
                            SUM(CASE WHEN visit_count > 0 AND car_visit_purpose_code = '07' THEN visit_count END) AS compost_transport_car_visit_count,
                            SUM(CASE WHEN visit_count > 0 AND car_visit_purpose_code = '08' THEN visit_count END) AS animal_medical_car_visit_count,
                            SUM(CASE WHEN visit_count > 0 AND car_visit_purpose_code = '09' THEN visit_count END) AS artificial_insemination_car_visit_count,
                            SUM(CASE WHEN visit_count > 0 AND car_visit_purpose_code = '10' THEN visit_count END) AS consulting_car_visit_count,
                            SUM(CASE WHEN visit_count > 0 AND car_visit_purpose_code = '11' THEN visit_count END) AS specimen_picking_diseasecontrol_car_visit_count,
                            SUM(CASE WHEN visit_count > 0 AND car_visit_purpose_code = '12' THEN visit_count END) AS machine_repair_car_visit_count,
                            SUM(CASE WHEN visit_count > 0 AND car_visit_purpose_code = '13' THEN visit_count END) AS egg_transport_car_visit_count,
                            SUM(CASE WHEN visit_count > 0 AND car_visit_purpose_code = '14' THEN visit_count END) AS roughage_transport_car_visit_count,
                            SUM(CASE WHEN visit_count > 0 AND car_visit_purpose_code = '15' THEN visit_count END) AS eggtray_transport_car_visit_count,
                            SUM(CASE WHEN visit_count > 0 AND car_visit_purpose_code = '16' THEN visit_count END) AS byproducts_transport_car_visit_count,
                            SUM(CASE WHEN visit_count > 0 AND car_visit_purpose_code = '17' THEN visit_count END) AS poultry_shipment_manpower_transport_car_visit_count,
                            SUM(CASE WHEN visit_count > 0 AND car_visit_purpose_code = '18' THEN visit_count END) AS livestock_breeding_facilities_management_car_visit_count,
                            SUM(CASE WHEN visit_count > 0 AND car_visit_purpose_code = '19' THEN visit_count END) AS livestock_carcass_transport_car_visit_count
                        FROM step01
                        GROUP BY farm_serial_no
                    )
                    
                    -- 역학 기간 내 방문 목적에 해당하는 차량 방문이 없을 경우(NULL), 0으로 대체
                    SELECT 
                        farm_serial_no,
                        COALESCE(livestock_transport_car_visit_count, 0) AS livestock_transport_car_visit_count,
                        COALESCE(rawmilk_transport_car_visit_count, 0) AS rawmilk_transport_car_visit_count, 
                        COALESCE(animal_medicine_transport_car_visit_count, 0) AS animal_medicine_transport_car_visit_count,
                        COALESCE(feed_transport_car_visit_count, 0) AS feed_transport_car_visit_count,
                        COALESCE(livestock_excreta_transport_car_visit_count, 0) AS livestock_excreta_transport_car_visit_count,
                        COALESCE(beddingstraw_transport_car_visit_count, 0) AS beddingstraw_transport_car_visit_count,
                        COALESCE(compost_transport_car_visit_count, 0) AS compost_transport_car_visit_count,
                        COALESCE(animal_medical_car_visit_count, 0) AS animal_medical_car_visit_count,
                        COALESCE(artificial_insemination_car_visit_count, 0) AS artificial_insemination_car_visit_count,
                        COALESCE(consulting_car_visit_count, 0) AS consulting_car_visit_count,
                        COALESCE(specimen_picking_diseasecontrol_car_visit_count, 0) AS specimen_picking_diseasecontrol_car_visit_count,
                        COALESCE(machine_repair_car_visit_count, 0) AS machine_repair_car_visit_count,
                        COALESCE(egg_transport_car_visit_count, 0) AS egg_transport_car_visit_count,
                        COALESCE(roughage_transport_car_visit_count, 0) AS roughage_transport_car_visit_count,
                        COALESCE(eggtray_transport_car_visit_count, 0) AS eggtray_transport_car_visit_count,
                        COALESCE(byproducts_transport_car_visit_count, 0) AS byproducts_transport_car_visit_count,
                        COALESCE(poultry_shipment_manpower_transport_car_visit_count, 0) AS poultry_shipment_manpower_transport_car_visit_count,
                        COALESCE(livestock_breeding_facilities_management_car_visit_count, 0) AS livestock_breeding_facilities_management_car_visit_count,
                        COALESCE(livestock_carcass_transport_car_visit_count, 0) AS livestock_carcass_transport_car_visit_count
                    FROM step02
                    """)
    return farm_car_visit_count