class config :

    # 파일 경로
    PRS_PATH = r'D:\주식회사빅밸류 Dropbox\김낙겸\17 2024 업무\240101 농림축산검역본부\ASF\24년 asf 최종본\asf_research_v33\data'
    PRS_PATH_v2 = r'D:\주식회사빅밸류 Dropbox\김낙겸\17 2024 업무\240101 농림축산검역본부\ASF\24년 asf 최종본\asf_research_v33\data\고도'
    PRS_PATH_v3 = r'D:\주식회사빅밸류 Dropbox\김낙겸\17 2024 업무\240101 농림축산검역본부\ASF\24년 asf 최종본\asf_research_v33\data\농장별_서식지가능성도'
    FEATURE_PATH = r'D:\주식회사빅밸류 Dropbox\김낙겸\17 2024 업무\240101 농림축산검역본부\ASF\24년 asf 최종본\asf_research_v33\feature'
    OUTPUT_PATH = r'D:\주식회사빅밸류 Dropbox\김낙겸\17 2024 업무\240101 농림축산검역본부\ASF\24년 asf 최종본\asf_research_v33\output'
    REPORT_OUTPUT_PATH = r'D:\주식회사빅밸류 Dropbox\김낙겸\17 2024 업무\240101 농림축산검역본부\ASF\정기 리포트\RAW_REPORT'

    # DB 서버 주소
    conn_string = "host = '10.10.12.84' dbname = 'geoai' user = 'geoai' password = '1234' port = '15432'"
    # conn_string_207 = "host = '10.10.12.207' dbname = 'geoai' user = 'geoai' password = '1234' port = '15432'"

    start_date = '2020-01-01'
    end_date = '2023-12-31'


    # 사용 독립변수 데이터명 지정
    featrue_dataframes = [
        'base_df', 'avg_count', 'blvstck_at', 'card', 'visit', 'wildboar_count', 'habitat_dist',
        'pig_farm_count_500m', 'pig_farm_count_3km', 'pig_farm_count_12km', 'env_ratio',
        'asf_farms_dist', 'asf_wild_dist', 'farm_elevation', 'farm_fit', 'mother_avg_co',
        'mother_at', 'asf_org', 'mother_pig_farm_count_12km', 'mother_pig_farm_count_3km',
        'mother_pig_farm_count_500m', 'farm_habitat_possibility'
        ]

    # card 변수 제외한 독립변수 리스트
    merge_list = [
        'base_df', 'avg_count', 'blvstck_at', 'visit', 'wildboar_count', 'habitat_dist',
        'pig_farm_count_500m', 'pig_farm_count_3km', 'pig_farm_count_12km', 'env_ratio',
        'asf_farms_dist', 'asf_wild_dist', 'farm_elevation', 'farm_fit', 'mother_avg_co',
        'mother_at', 'asf_org', 'mother_pig_farm_count_12km', 'mother_pig_farm_count_3km',
        'mother_pig_farm_count_500m', 'farm_habitat_possibility'
        ]

    # 학습, 검증 데이터셋 나누는 기준일자
    split_std_dt = '2023-01-01'

    # oversampling, rf 랜덤시드
    random_seed = 42

    # 사용 변수
    x_col = ['avg_brd_had_co', 'pig_farm_12km_count',
             'common_at', 'black_at', 'breeding_at', 'boar_at',
             'BRD_STLE_1', 'BRD_STLE_2', 'BRD_STLE_3',
             'PAST_OCCRRNC_AT', 'CRWD_HSMP_AT',
             'INTRCP_DSNFC_FCLTY_01_AT', 'INTRCP_DSNFC_FCLTY_02_AT',
             'INTRCP_DSNFC_FCLTY_04_AT', 'INTRCP_DSNFC_FCLTY_05_AT',
             'INTRCP_DSNFC_FCLTY_06_AT', 'INTRCP_DSNFC_FCLTY_07_AT',
             'INTRCP_DSNFC_FCLTY_08_AT', 'INTRCP_DSNFC_FCLTY_09_AT',
             'FLFL_01_AT', 'FLFL_02_AT', 'FLFL_03_AT', 'FLFL_04_AT',
             'FLFL_05_AT', 'FLFL_07_AT', 'FLFL_09_AT', 'FLFL_10_AT',
             'livestock', 'medicines', 'feed', 'manure', 'consulting',
             'vaccination', 'compost', 'repair', 'artificial_insemination',
             'operation', 'forage', 'carcass',
             'wildboar_5km_count', 'habitat_dist', 'asf_farms_dist', 'asf_wild_dist',
             'habitat_ratio_50', 'farm_land_ratio', 'river_ratio', 'forest_ratio',
             'habitat_fit_score', 'elevation', 'avg_mother_brd_had_co', 'mother_at'
             # 'max_value',
             # 'min_value',
             # 'first_quartile',
             # 'median_value',
             # 'third_quartile'
             # 'mother_pig_farm_500m_count',
             # 'mother_pig_farm_3km_count',
             # 'mother_pig_farm_12km_count'
             # 'pig_farm_500m_count',
             # 'pig_farm_3km_count',
             # 'enfsn_co', 'frgnr_enfsn_co',
             # 'vacin_insffc_at',
             # 'nsp_detect_at',
             ]

    kor_col_name = ['평균사육두수', '12km이내 양돈농장 수',
                    '일반돼지 사육여부', '흑돼지 사육여부', '종돈 사육여부', '멧돼지 사육여부',
                    '사육형태 무창 여부', '사육형태 유창 여부', '사육형태 하우스 여부',
                    '과거 발생 여부', '밀집단지 여부',
                    '울타리 여부', '농장입구사람소독시설 여부',
                    '퇴비장방조망등차단막설치 여부', '돈사환풍기환기구방충망설치 여부',
                    '전실 여부', '돈사전용장화구입 여부',
                    '돈사전용손씻기소독시설구비 여부', '돈사구멍메우기조밀망등설치 여부',
                    '농장주변 물웅덩이·수풀제거 여부', '농장 둘레 생석회 도포 여부',
                    '멧돼지 발생지역(농경지) 생산 볏짚,작물 반입 금지 여부', '수렵 활동 및 입산 금지 여부',
                    '퇴비사·사료빈 주변 청결유지 및 소독 여부', '축사 내부 주기적 청소·소독 여부',
                    '돈사 진입 전 손씻기 여부', '돈사 진입 전 장화 갈아신기 여부',
                    '가축운반 목적 차량 방문 수', '동물 약품운반 목적 차량 방문 수',
                    '사료운반 목적차량 방문 수', '가축분뇨운반 목적차량 방문 수',
                    '컨설팅 목적차량 방문 수', '진료.예방접종 목적차량 방문 수',
                    '퇴비운반 목적차량 방문 수', '기계수리 목적차량 방문 수',
                    '인공수정 목적차량 방문 수', '가축사육시설운영관리 목적차량 방문 수',
                    '조사료운반 목적차량 방문 수', '가축사체운반 목적 차량 방문 수',
                    '5km이내 야생멧돼지 수', '최인접 야생멧돼지 서식지와의 거리',
                    '발생 농장과의 거리', '야생 발생과의 거리',
                    '야생멧돼지 서식지 비율', '농지 비율', '강 비율', '산림 비율',
                    '야생멧돼지 서식지 가능성도', '고도', '모돈 평균사육두수', '모돈 사육여부'
                    ]

    manange_col = ["BRD_STLE_1",
                    "BRD_STLE_2",
                    "BRD_STLE_3",
                    "ENFSN_CO",
                    "FRGNR_ENFSN_CO",
                    "PAST_OCCRRNC_AT",
                    "CRWD_HSMP_AT",
                    "VACIN_INSFFC_AT",
                    "NSP_DETECT_AT",
                    "INTRCP_DSNFC_FCLTY_01_AT",
                    "INTRCP_DSNFC_FCLTY_02_AT",
                    "INTRCP_DSNFC_FCLTY_04_AT",
                    "INTRCP_DSNFC_FCLTY_05_AT",
                    "INTRCP_DSNFC_FCLTY_06_AT",
                    "INTRCP_DSNFC_FCLTY_07_AT",
                    "INTRCP_DSNFC_FCLTY_08_AT",
                    "INTRCP_DSNFC_FCLTY_09_AT",
                    "FLFL_01_AT",
                    "FLFL_02_AT",
                    "FLFL_03_AT",
                    "FLFL_04_AT",
                    "FLFL_05_AT",
                    "FLFL_07_AT",
                    "FLFL_09_AT",
                    "FLFL_10_AT"]


