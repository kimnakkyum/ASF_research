def Operation(IS_SAVE = True) :
    # [ STEP 1 ] Get DataSet

    from Bulk_trainset_code.GetData.Dataloader import GetData
    from Bulk_trainset_code.src.base_farm_info import FarmFilter
    from Bulk_trainset_code.src.asf_occ_farms_info import Making_label
    from Bulk_trainset_code.src.Making_pig_breeding_yn import breeding_yn
    from Bulk_trainset_code.src.Making_manage_card import manage_card
    from Bulk_trainset_code.src.mother_breeding_yn import mother_pig_breeding_yn
    from Bulk_trainset_code.preprocessing import making_trainset


    # 데이터 불러오기
    dataloader = GetData()
    (FRAMS_INFO, ASF_OCC_FARMS_INFO, ASF_FARM_DIST,
     ASF_WILD_DIST, FARM_COUNT_12KM, VISIT_INFO, CARD_INFO,
     WILDBOAR_COUNT_5KM, HABITAT_DIST, FARMS_ENV_RATIO,
     FARM_ELEVATION, FARMS_HABITAT_POSSIBILITY, MOTHER_AVG_BRD_HAD_CO)  =  (dataloader['farms'], dataloader['asf_org'], dataloader['asf_farm_dist'],
                                                                        dataloader['asf_wild_dist'], dataloader['farm_count_12km'], dataloader['visit'],
                                                                        dataloader['manage_card'], dataloader['wildboar_count_5km'], dataloader['habitat_dist'],
                                                                        dataloader['farms_env_ratio'], dataloader['farm_elevation'], dataloader['farms_habitat_possibility'],
                                                                        dataloader['mother_avg_brd_had_co'] )

    # 운영농장정보, 평균사육수수
    farm_filter = FarmFilter()
    base_farm_info, avg_brd_had_co = farm_filter.filter(FRAMS_INFO)

    # 축종 사육 여부
    pig_breeding_yn = breeding_yn(base_farm_info)

    # asf 발생농장 여부
    making_labeling = Making_label()
    asf_org = making_labeling.labeling(ASF_OCC_FARMS_INFO, base_farm_info)

    # 최근접 발생농장과의 거리
    ASF_FARM_DIST['std_dt'] = ASF_FARM_DIST['std_dt'].astype(str)

    # 최근접 야생발생과의 거리
    ASF_WILD_DIST['std_dt'] = ASF_WILD_DIST['std_dt'].astype(str)

    # 반경 12km 양돈 농장 수
    FARM_COUNT_12KM['farms_no'] = FARM_COUNT_12KM['farms_no'].astype('str').str.zfill(8)
    FARM_COUNT_12KM['std_dt'] = FARM_COUNT_12KM['std_dt'].astype(str)
    FARM_COUNT_12KM.rename(columns={'count_within_12km' : 'pig_farm_12km_count'}, inplace=True)

    # 차량 변수
    VISIT_INFO['std_dt'] = VISIT_INFO['std_dt'].astype(str)

    # 양돈방역관리카드
    card_info = manage_card(CARD_INFO, base_farm_info, pig_breeding_yn, avg_brd_had_co)

    # 농장 반경 5km 야생멧돼지수
    WILDBOAR_COUNT_5KM['farms_no'] = WILDBOAR_COUNT_5KM['farms_no'].astype('str').str.zfill(8)

    # 최근접 서식지와의 거리
    HABITAT_DIST['farms_no'] = HABITAT_DIST['farms_no'].astype('str').str.zfill(8)

    # 농장 3km이내 환경 정보
    FARMS_ENV_RATIO

    # 농장 고도
    FARM_ELEVATION['farms_no'] = FARM_ELEVATION['farms_no'].astype('str').str.zfill(8)

    # 농장 서식지 가능성도
    FARMS_HABITAT_POSSIBILITY['farms_no'] = FARMS_HABITAT_POSSIBILITY['farms_no'].astype('str').str.zfill(8)

    # 모돈 평균사육수수
    MOTHER_AVG_BRD_HAD_CO['farms_no'] = MOTHER_AVG_BRD_HAD_CO['farms_no'].astype('str').str.zfill(8)
    MOTHER_AVG_BRD_HAD_CO['std_dt'] = MOTHER_AVG_BRD_HAD_CO['std_dt'].astype(str)

    # 모돈 사육여부
    MOTHER_PIG_BREEDING_YN = mother_pig_breeding_yn()
    MOTHER_PIG_BREEDING_YN['std_dt'] = MOTHER_PIG_BREEDING_YN['std_dt'].astype(str)


    ########## 전처리 전 데이터셋
    tot = base_farm_info[['farms_no' ,'std_dt']].merge(MOTHER_AVG_BRD_HAD_CO[['farms_no' ,'std_dt', 'avg_mother_brd_had_co']], on=['std_dt', 'farms_no'], how='left')
    tot.drop_duplicates(subset=['std_dt', 'farms_no'], inplace=True)

    tot_2 = (tot.merge(avg_brd_had_co).merge(pig_breeding_yn).
           merge(asf_org, how='left').merge(ASF_FARM_DIST, on=['farms_no', 'std_dt'], how='left').merge(ASF_WILD_DIST, on=['farms_no', 'std_dt'], how='left').
           merge(FARM_COUNT_12KM, on=['farms_no', 'std_dt'], how='left').merge(VISIT_INFO, how='left').merge(card_info, how='left').
           merge(HABITAT_DIST, how='left').merge(FARMS_ENV_RATIO, how='left').merge(FARM_ELEVATION, how='left').
           merge(FARMS_HABITAT_POSSIBILITY, how='left').merge(MOTHER_PIG_BREEDING_YN, how='left'))
    tot_2['year'] = tot_2['std_dt'].apply(lambda x: x[:4])

    tot_3 = tot_2.merge(WILDBOAR_COUNT_5KM, how='left', on=['farms_no', 'year'])


    ########## 전처리 후 데이터셋
    dataset = making_trainset(tot_3)

    # dataset.to_csv(r"D:\gitlab\asf_research\output\dataset.csv", index=False)

    return dataset
