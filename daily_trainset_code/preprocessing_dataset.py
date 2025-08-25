def making_trainset(tot):


    tot['avg_brd_had_co'] = tot['present_breeding_livestock_count_average'].fillna(0)

    # asf 발생 농장과의 거리 전처리
    tot['asf_farms_dist'] = tot['asf_farms_dist'].fillna(999999)
    tot['asf_farms_dist'] = tot['asf_farms_dist'].where(tot['asf_farms_dist'] < 3000, 999999)

    # asf 야생 발생과의 거리 전처리
    tot['asf_wild_dist'] = tot['asf_wild_dist'].fillna(999999)
    tot['asf_wild_dist'] = tot['asf_wild_dist'].where(tot['asf_wild_dist'] < 3000, 999999)

    # 차량 방문 횟수 전처리
    columns_to_fill = ['livestock', 'medicines', 'feed',
                       'manure', 'consulting', 'vaccination',
                       'compost', 'repair', 'operation',
                       'artificial_insemination', 'forage',
                       'carcass', 'sawdust', 'manpower_transportation',
                       'egg_transportation', 'egg_carrier'
                       ]
    mean_values = tot[columns_to_fill].mean()
    tot[columns_to_fill] = tot[columns_to_fill].fillna(mean_values)

    # 최근접 서식지와의 거리 전처리
    tot['habitat_dist'] = tot['habitat_dist'].where(tot['habitat_dist'] < 3000, 999999)

    # 최근접 서식지와의 거리 전처리
    tot['habitat_fit_score'] = tot['value'].fillna(0)

    # 모돈평균사육수수 전처리
    tot['avg_mother_brd_had_co'].fillna(0, inplace=True)

    # 모돈 사육 여부 전처리
    tot['mother_at'] = tot['mother_at'].fillna(0)

    # 농장 반경 5km이내 야생멧돼지 수 전처리
    tot['wildboar_5km_count'] = tot['total_weighted_boar_count'].fillna(0)

    # 차량 데이터 전처리
    tot[['livestock', 'medicines', 'feed', 'manure',
    'sawdust', 'compost', 'vaccination', 'artificial_insemination',
    'consulting', 'disinfect', 'repair', 'egg_transportation', 'forage',
    'egg_carrier', 'byproducts', 'manpower_transportation', 'operation',
    'carcass', 'byproducts', 'disinfect']].fillna(0, inplace=True)

    tot = tot.rename(columns={'count_within_12km': 'pig_farm_12km_count'})

    return tot