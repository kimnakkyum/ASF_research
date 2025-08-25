import pandas as pd
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB
from advanced_daily_trainset_code.data_cleansing.config import config

# +++ [이슈사항 정리] +++ ================================================================================================
# 1.
# 이슈 코드 : FROM asf.tn_aph_dsnfc_manage_frmhs_info
# 이슈 내용 : 원천 테이블 미 변경
# 개선 방안 : 180 서버 내 m2msys.tn_aph_dsnfc_manage_frmhs_info_raw로 변경 필요
# 반영 완료

# 2.
# 이슈 코드 : 양돈방역관리카드 결측치 보완
# 이슈 내용 : 현재 결측치 보완을 위해 군집 분석을 활용하지만, 결측이 많은 데이터를 결측치 대체한다는 것에 한계 존재
# 결측 많은 컬럼은 제외하도록 로직 변경 고민 필요
# 개선 방안 : 일단 결측치 대체 없이 활용
# 결측치 대체 파트 제외 완료

# [낙겸 이슈사항 정리] +++ ================================================================================================
# 1.
# 이슈 코드 : SELECT DISTINCT ON (FRMHS_NO) * FROM m2msys.tn_aph_dsnfc_manage_frmhs_info_raw ORDER BY FRMHS_NO, LAST_CHANGER_DT DESC
# 이슈 내용 : 양돈방역관리카드 원천 데이터 사용 시 시점 불일치
# 개선 방안 : dm.tb_pig_diseasecontrol_status_information_clean를 std_dt 기준으로 활용 필요

# 양돈방역관리카드 변수 생성
def create_farm_manage_card_info(db):

    # 농장 별 최신 변경 날짜에 따른 방역관리카드 변수 불러오기
    farm_manage_card_info = db.query(f'''WITH 

                last_manage_card AS (
                        SELECT DISTINCT ON (frmhs_no)
                            frmhs_no AS farm_serial_no,

                            CASE WHEN brd_stle_1 = 'Y' THEN '1'
                                 WHEN brd_stle_1 = 'N' THEN '0'
                                 ELSE brd_stle_1
                            END AS windowless_breeding_yn,

                            CASE WHEN brd_stle_2 = 'Y' THEN '1'
                                 WHEN brd_stle_2 = 'N' THEN '0'
                                 ELSE brd_stle_2
                            END AS withwindows_breeding_yn,

                            CASE WHEN brd_stle_3 = 'Y' THEN '1'
                                 WHEN brd_stle_3 = 'N' THEN '0'
                                 ELSE brd_stle_3
                            END AS greenhouse_breeding_yn,

                            CASE WHEN past_occrrnc_at = 'Y' THEN '1'
                                 WHEN past_occrrnc_at = 'N' THEN '0'
                                 ELSE past_occrrnc_at
                            END AS past_occurrence_yn,

                            CASE WHEN crwd_hsmp_at = 'Y' THEN '1'
                                 WHEN crwd_hsmp_at = 'N' THEN '0'
                                 ELSE crwd_hsmp_at
                            END AS crowded_complex_yn,

                            CASE WHEN intrcp_dsnfc_fclty_01_at = 'Y' THEN '1'
                                 WHEN intrcp_dsnfc_fclty_01_at = 'N' THEN '0'
                                 ELSE intrcp_dsnfc_fclty_01_at
                            END AS fence_installation_yn,

                            CASE WHEN intrcp_dsnfc_fclty_02_at = 'Y' THEN '1'
                                 WHEN intrcp_dsnfc_fclty_02_at = 'N' THEN '0'
                                 ELSE intrcp_dsnfc_fclty_02_at
                            END AS entrance_human_disinfection_yn,

                            CASE WHEN intrcp_dsnfc_fclty_03_at = 'Y' THEN '1'
                                 WHEN intrcp_dsnfc_fclty_03_at = 'N' THEN '0'
                                 ELSE intrcp_dsnfc_fclty_03_at
                            END AS entrance_car_disinfection_yn,

                            CASE WHEN intrcp_dsnfc_fclty_04_at = 'Y' THEN '1'
                                 WHEN intrcp_dsnfc_fclty_04_at = 'N' THEN '0'
                                 ELSE intrcp_dsnfc_fclty_04_at
                            END AS compost_blocking_facility_yn,

                            CASE WHEN intrcp_dsnfc_fclty_05_at = 'Y' THEN '1'
                                 WHEN intrcp_dsnfc_fclty_05_at = 'N' THEN '0'
                                 ELSE intrcp_dsnfc_fclty_05_at
                            END AS pig_house_vent_screen_yn,

                            CASE WHEN intrcp_dsnfc_fclty_06_at = 'Y' THEN '1'
                                 WHEN intrcp_dsnfc_fclty_06_at = 'N' THEN '0'
                                 ELSE intrcp_dsnfc_fclty_06_at
                            END AS ante_room_yn,

                            CASE WHEN intrcp_dsnfc_fclty_07_at = 'Y' THEN '1'
                                 WHEN intrcp_dsnfc_fclty_07_at = 'N' THEN '0'
                                 ELSE intrcp_dsnfc_fclty_07_at
                            END AS pigpen_dedicated_boots_yn,

                            CASE WHEN intrcp_dsnfc_fclty_08_at = 'Y' THEN '1'
                                 WHEN intrcp_dsnfc_fclty_08_at = 'N' THEN '0'
                                 ELSE intrcp_dsnfc_fclty_08_at
                            END AS pigpen_sanitation_facility_yn,

                            CASE WHEN intrcp_dsnfc_fclty_09_at = 'Y' THEN '1'
                                 WHEN intrcp_dsnfc_fclty_09_at = 'N' THEN '0'
                                 ELSE intrcp_dsnfc_fclty_09_at
                            END AS pigpen_gap_blocking_net_yn,

                            CASE WHEN flfl_01_at = 'Y' THEN '1'
                                 WHEN flfl_01_at = 'N' THEN '0'
                                 ELSE flfl_01_at
                            END AS pigpen_water_puddle_removal_yn,

                            CASE WHEN flfl_02_at = 'Y' THEN '1'
                                 WHEN flfl_02_at = 'N' THEN '0'
                                 ELSE flfl_02_at
                            END AS lime_application_around_farm_yn,

                            CASE WHEN flfl_03_at = 'Y' THEN '1'
                                 WHEN flfl_03_at = 'N' THEN '0'
                                 ELSE flfl_03_at
                            END AS forbid_straw_and_crop_entry_from_wild_pig_area_yn,

                            CASE WHEN flfl_04_at = 'Y' THEN '1'
                                 WHEN flfl_04_at = 'N' THEN '0'
                                 ELSE flfl_04_at
                            END AS forbid_hunting_and_access_yn,

                            CASE WHEN flfl_05_at = 'Y' THEN '1'
                                 WHEN flfl_05_at = 'N' THEN '0'
                                 ELSE flfl_05_at
                            END AS maintain_cleanliness_and_disinfection_around_manure_feed_bin_yn,

                            CASE WHEN flfl_07_at = 'Y' THEN '1'
                                 WHEN flfl_07_at = 'N' THEN '0'
                                 ELSE flfl_07_at
                            END AS periodic_cleaning_and_disinfection_inside_pigpen_yn,

                            CASE WHEN flfl_09_at = 'Y' THEN '1'
                                 WHEN flfl_09_at = 'N' THEN '0'
                                 ELSE flfl_09_at
                            END AS handwashing_before_entering_pigpen_yn,

                            CASE WHEN flfl_10_at = 'Y' THEN '1'
                                 WHEN flfl_10_at = 'N' THEN '0'
                                 ELSE flfl_10_at
                            END AS change_boots_before_entering_pigpen_yn

                        FROM asf.tn_aph_dsnfc_manage_frmhs_info_filled_clean
                        WHERE use_at = 'Y' OR use_at IS NULL
                        ORDER BY frmhs_no, last_changer_dt DESC
                    )

                    SELECT * FROM last_manage_card;
    ''')

    return farm_manage_card_info