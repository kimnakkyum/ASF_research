def making_trainset(tot_2):


    tot_2['avg_brd_had_co'].fillna(0, inplace=True)

    # 축종 사육 유무 전처리
    tot_2['asf_org'].fillna(0, inplace=True)

    # asf 발생 농장과의 거리 전처리
    tot_2['asf_farms_dist'] = tot_2['asf_farms_dist'].fillna(999999)
    tot_2['asf_farms_dist'] = tot_2['asf_farms_dist'].where(tot_2['asf_farms_dist'] < 3000, 999999)

    # asf 야생 발생과의 거리 전처리
    tot_2['asf_wild_dist'] = tot_2['asf_wild_dist'].fillna(999999)
    tot_2['asf_wild_dist'] = tot_2['asf_wild_dist'].where(tot_2['asf_wild_dist'] < 3000, 999999)

    # 차량 방문 횟수 전처리
    columns_to_fill = ['livestock', 'medicines', 'feed',
                       'manure', 'consulting', 'vaccination',
                       'compost', 'repair', 'operation',
                       'artificial_insemination', 'forage',
                       'carcass', 'sawdust', 'manpower_transportation',
                       'egg_transportation', 'egg_carrier'
                       ]
    mean_values = tot_2[columns_to_fill].mean()
    tot_2[columns_to_fill] = tot_2[columns_to_fill].fillna(mean_values)

    # 최근접 서식지와의 거리 전처리
    tot_2['habitat_dist'] = tot_2['habitat_dist'].where(tot_2['habitat_dist'] < 3000, 999999)

    # 최근접 서식지와의 거리 전처리
    tot_2['habitat_fit_score'].fillna(0, inplace=True)

    # 모돈평균사육수수 전처리
    tot_2['avg_mother_brd_had_co'].fillna(0, inplace=True)

    # 모돈 사육 여부 전처리
    tot_2['mother_at'] = tot_2['mother_at'].fillna(0)

    # 농장 반경 5km이내 야생멧돼지 수 전처리
    tot_2['wildboar_5km_count'] = tot_2['total_weighted_boar_count'].fillna(0)

    return tot_2