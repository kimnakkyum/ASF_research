import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class model_config:

    # 타겟 정보
    target_col = ['asf_occurrence_yn']

    # ++ ===============================================================================================================

    # 카테고리 별 활용 변수 정의

    # 1) 환경 카테고리
    # 1-1) 야생멧돼지 관련 정보
    wildboar_col = ['radius_500m_habiatat_possibility_median',
                    'radius_9km_wildboar_habitat_area_rate'
                    ] # 최근접 서식지 거리는 제외

    # 1-2) 주변 환경 정보
    around_env_col = ['radius_1km_farmland_dryfield_rate',
                      'radius_1km_farmland_none_dryfield_rate',
                      'radius_1km_urban_area_rate',
                      'radius_1km_forest_area_rate',
                      'radius_500m_elevation_median',
                      'radius_12km_mean_rainfall_3month',
                      'radius_3km_max_rainfall_3month'
                      ]

    # # 1-2) 주변 환경 정보
    # around_env_col = ['radius_1km_farmland_dryfield_rate',
    #                   'radius_1km_urban_area_rate',
    #                   'radius_1km_forest_area_rate',
    #                   'radius_500m_elevation_median',
    #                   'radius_12km_mean_rainfall_3month',
    #                   'radius_3km_max_rainfall_3month'
    #                   ]

    # 1-3) 주변 농장 정보
    around_farm_col = ['radius_12km_whole_farm_count',
                       'radius_12km_pig_farm_count'
                       ]

    # 1-4) 주변 농장 발생 정보
    around_farm_occ_col = ['infection_farm_top_near_distance',
                           'radius_6km_infection_farm_1month_count'
                           ]

    # 1-5) 주변 야생 발생 정보
    around_wild_occ_col = ['specimen_picking_top_near_distance_6month',
                           'radius_6km_specimen_picking_6month_count'
                           ]

    # 2. 전파 카테고리
    # 2-1) 차량 방문 횟수
    car_visit_count_col = ['livestock_transport_car_visit_count',
                           'animal_medicine_transport_car_visit_count',
                           'feed_transport_car_visit_count',
                           'livestock_excreta_transport_car_visit_count',
                           'beddingstraw_transport_car_visit_count',
                           'compost_transport_car_visit_count',
                           'animal_medical_car_visit_count',
                           'artificial_insemination_car_visit_count',
                           'consulting_car_visit_count',
                           'roughage_transport_car_visit_count',
                           'livestock_breeding_facilities_management_car_visit_count',
                           'livestock_carcass_transport_car_visit_count'
                           ]

    # 2-2) 발생 농장 차량 방문 횟수
    infection_car_visit_count_col = ['infection_whole_car_visit_count']

    # 3. 방역 카테고리
    # 3-1) 축종 사육여부
    breeding_yn_col = ['breeding_pig_breeding_yn',
                       'captive_wildboar_breeding_yn',
                       'black_pig_breeding_yn',
                       'common_pig_breeding_yn',
                       'mother_pig_breeding_yn',
                       'porker_breeding_yn'
                       ]

    # 3-2) 사육두수
    breeding_cnt_col = ['present_breeding_livestock_count',
                        'present_mother_pig_breeding_livestock_count',
                        'present_porker_breeding_livestock_count',
                        'average_breeding_livestock_count',
                        'average_mother_pig_breeding_livestock_count',
                        'average_porker_breeding_livestock_count'
                        ]

    # 3-3) 방역관리카드
    biosec_col = ['windowless_breeding_yn',
                  'withwindows_breeding_yn',
                  'greenhouse_breeding_yn',
                  'fence_installation_yn',
                  'entrance_human_disinfection_yn',
                  'entrance_car_disinfection_yn',
                  'compost_blocking_facility_yn',
                  'pig_house_vent_screen_yn',
                  'ante_room_yn',
                  'pigpen_dedicated_boots_yn',
                  'pigpen_sanitation_facility_yn',
                  'pigpen_gap_blocking_net_yn',
                  'pigpen_water_puddle_removal_yn',
                  'lime_application_around_farm_yn',
                  'forbid_straw_and_crop_entry_from_wild_pig_area_yn',
                  'forbid_hunting_and_access_yn',
                  'maintain_cleanliness_and_disinfection_around_manure_feed_bin_yn',
                  'periodic_cleaning_and_disinfection_inside_pigpen_yn',
                  'handwashing_before_entering_pigpen_yn',
                  'change_boots_before_entering_pigpen_yn'
                  ] # 과거발생여부(past_occurrence_yn), 밀집단지여부(crowded_complex_yn) 검토 후 추가 필요

    # ++ ===============================================================================================================

    # 환경 클러스터링 변수
    cluster_env_col = ['radius_9km_wildboar_habitat_area_rate',
                       'radius_1km_farmland_dryfield_rate',
                       'radius_1km_farmland_none_dryfield_rate',
                       'radius_1km_urban_area_rate',
                       'radius_1km_forest_area_rate'
                       ]

    env_col = ['center_forest', 'center_dryfield', 'center_non_dryfield', 'center_urban', 'center_habitat']

    # 방역 클러스터링 변수
    cluster_bio_col = ['average_breeding_livestock_count',
                       'average_mother_pig_breeding_livestock_count'
                       ]

    # 평균사육두수 클러스터명
    bio_col = [
        # 평균사육두수
        "None_scale_yn",  # 1~49
        "Micro_scale_yn",  # 1~49
        "Small_scale_yn",  # 49~299
        "Mid_scale_yn",  # 299~999
        "Upper_mid_scale_yn",  # 999~2999
        "Large_scale_yn",  # 2999~4999
        "Mega_scale_yn"  # 4999~
    ]

    # 모돈 평균사육두수 클러스터명
    bio_mother_pig_col = [
        "None_scale_mother_pig_yn",  # 0두
        "Micro_scale_mother_pig_yn",  # 0~ 19두
        "Small_scale_mother_pig_yn",  # 19~49두
        "Lower_mid_scale_mother_pig_yn",  # 49~99두
        "Mid_scale_mother_pig_yn",  # 99~299두
        "Upper_mid_scale_mother_pig_yn",  # 299~499두
        "Large_scale_mother_pig_yn",  # 499~999두
        "Mega_scale_mother_pig_yn"  # 999두 이상
    ]

    # 평균사육두수/모돈 평균사육두수 기준 범위
    bio_bins = [-1, 1e-6, 49, 299, 999, 2999, 4999, float('inf')]
    bio_mother_pig_bins = [-1, 1e-6, 19, 49, 99, 299, 499, 999, float('inf')]

    # 영어 칼럼명 → 한글 라벨
    general_scale_map = {
        'None_scale_yn': '미사육',
        'Micro_scale_yn': '초소형',
        'Small_scale_yn': '소형',
        'Mid_scale_yn': '중형',
        'Upper_mid_scale_yn': '중대형',
        'Large_scale_yn': '대형',
        'Mega_scale_yn': '초대형'
    }
    mother_pig_scale_map = {
        'None_scale_mother_pig_yn': '모돈_미사육',
        'Micro_scale_mother_pig_yn': '모돈_초소형',
        'Small_scale_mother_pig_yn': '모돈_소형',
        'Lower_mid_scale_mother_pig_yn': '모돈_중소형',
        'Mid_scale_mother_pig_yn': '모돈_중형',
        'Upper_mid_scale_mother_pig_yn': '모돈_중대형',
        'Large_scale_mother_pig_yn': '모돈_대형',
        'Mega_scale_mother_pig_yn': '모돈_초대형'
    }

    # 전파 클러스터링 변수
    visit_col = ['visit_cluster_0', 'visit_cluster_1', 'visit_cluster_2',
                 'visit_cluster_3', 'visit_cluster_4', 'visit_cluster_5',
                 'visit_cluster_6', 'visit_cluster_7', 'visit_cluster_8',
                 'visit_cluster_9'
                 ]

    cluster_name_map = {
        0: '저빈도·장주기형',
        1: '가축운반 단주기형',
        2: '저빈도·분뇨운반 장주기형',
        3: '퇴비운반 장주기형',
        4: '가축·사료·약품운반 다빈도형',
        5: '가축·사료·약품운반 장주기, 저빈도형',
        6: '다 차량, 단주기, 약품운반 다빈도형',
        7: '다빈도·단주기형',
        8: '단주기, 약품운반 다빈도형',
        9: '차량 미방문형'
    }

    # ++ ===============================================================================================================

    # input, output 차원 정의

    # # 1. 환경 카테고리
    # wildboar_input, wildboar_output = len(wildboar_col), 2
    # around_env_input, around_env_output = len(around_env_col), 2
    # around_farm_input, around_farm_output = len(around_farm_col), 2
    # around_farm_occ_input, around_farm_occ_output = len(around_farm_occ_col), 2
    # around_wild_occ_input, around_wild_occ_output = len(around_wild_occ_col), 2
    #
    # # 2. 전파 카테고리
    # car_visit_count_input, car_visit_count_output = len(car_visit_count_col), 2
    # infection_car_visit_count_input, infection_car_visit_count_output = len(infection_car_visit_count_col), 2
    #
    # # 3. 방역 카테고리
    # breeding_yn_input, breeding_yn_output = len(breeding_yn_col), 2
    # breeding_cnt_input, breeding_cnt_output = len(breeding_cnt_col), 2
    # biosec_input, biosec_output = len(biosec_col), 2


    # ++ ===============================================================================================================

    # FCNN input dim 정의(FCNN 연결 전 concat 결과 차원)
    # 추후 전파, 방역 output 차원도 합산 필요


    # 방역 클러스터 변수 통합
    bio_final_col = bio_col + bio_mother_pig_col

    pred_dim = len(env_col + bio_final_col + visit_col)

    pred_nn_1_multiply_num = 16
    pred_nn_2_multiply_num = 32
    pred_nn_3_multiply_num = 16
    pred_nn_4_multiply_num = 8
    pred_nn_5_multiply_num = 1

    activation_function = nn.LeakyReLU()  # nn.ReLU(), nn.Sigmoid(), nn.LeakyReLU(), nn.ELU() 중 하나 적용

    # ++ ===============================================================================================================

    # 하이퍼파라미터 정의
    batch_size = 2**10
    epoch = 5_000
    learning_rate = 1e-4
    dropout = 0.5
    patience = 300