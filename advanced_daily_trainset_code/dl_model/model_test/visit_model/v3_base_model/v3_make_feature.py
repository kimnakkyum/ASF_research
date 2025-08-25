import numpy as np
import pandas as pd

from advanced_daily_trainset_code.dl_model.model_test.visit_model.v3_base_model.v3_model_config import model_config

def build_propagation_features(
    visit_df: pd.DataFrame,
    trainset: pd.DataFrame,
    percentile: float = 0.95,
    main_col=None,
    other_cols=None,
    seed: int = 42,
    ref_date: pd.Timestamp = None,   # ✅ 기준일
    window_years: int = 4            # ✅ 과거 윈도우(년)
):
    """
    ASF 전파 변수 파이프라인 (이상치 클리핑 → 일평균 빈도/차량수 → 방문 주기 → 로그/스케일링 → 병합)
    - ref_date가 주어지면, ref_date - window_years ~ ref_date 구간 데이터만 사용(미래 누수 방지)
    """

    # -----------------------------
    # 0) 기본 설정 & 입력 정리
    # -----------------------------
    visit_df = visit_df.copy()
    trainset = trainset.copy()

    # 필수 날짜형 변환
    visit_df['visit_date'] = pd.to_datetime(visit_df['visit_date'])
    trainset['standard_date'] = pd.to_datetime(trainset['standard_date'])

    # ✅ 기준일/윈도우 필터: 방문 데이터 & 사육두수
    if ref_date is not None:
        ref_date = pd.to_datetime(ref_date)
        start_date = ref_date - pd.DateOffset(years=window_years)
        # 방문 데이터: ref_date 기준 과거 4년~당일
        visit_df = visit_df[(visit_df['visit_date'] >= start_date) &
                            (visit_df['visit_date'] <= ref_date)]
        # 사육두수: ref_date 이전(포함) 내 최신 1건만 농장별 선택
        trainset = (
            trainset[trainset['standard_date'] <= ref_date]
            .sort_values('standard_date', ascending=False)
            .drop_duplicates(subset='farm_serial_no', keep='first')
            .copy()
        )

    # -----------------------------
    # 1) 컬럼 정의
    # -----------------------------
    col = model_config.car_visit_count_col
    # col = [
    #     'livestock_transport_car_visit_count',
    #     'animal_medicine_transport_car_visit_count',
    #     'feed_transport_car_visit_count',
    #     'livestock_excreta_transport_car_visit_count',
    #     'beddingstraw_transport_car_visit_count',
    #     'compost_transport_car_visit_count',
    #     'animal_medical_car_visit_count',
    #     'artificial_insemination_car_visit_count',
    #     'consulting_car_visit_count',
    #     'roughage_transport_car_visit_count',
    #     'livestock_breeding_facilities_management_car_visit_count',
    #     'livestock_carcass_transport_car_visit_count'
    # ]

    if main_col is None:
        main_col = [
            'livestock_transport_car_visit_count',
            'animal_medicine_transport_car_visit_count',
            'feed_transport_car_visit_count',
            'livestock_excreta_transport_car_visit_count',
            'compost_transport_car_visit_count'
        ]
    if other_cols is None:
        other_cols = [
            'beddingstraw_transport_car_visit_count',
            'animal_medical_car_visit_count',
            'artificial_insemination_car_visit_count',
            'consulting_car_visit_count',
            'roughage_transport_car_visit_count',
            'livestock_breeding_facilities_management_car_visit_count',
            'livestock_carcass_transport_car_visit_count'
        ]

    # -----------------------------
    # 2) 방문횟수 이상치 처리 (목적*농장*차량 단위, 0 제외, 분위수 클리핑)
    # -----------------------------
    clipped_df = visit_df.copy()

    grouped = clipped_df.groupby(['frmhs_no', 'regist_no'], sort=False)
    for c in col:
        clipped_values = clipped_df[c].copy()
        for (frmhs_no, regist_no), g in grouped:
            non_zero = g.loc[g[c] > 0, c]
            if len(non_zero) == 0:
                continue
            pxx = np.percentile(non_zero, percentile * 100)
            idx = g.index[g[c] > pxx]
            clipped_values.loc[idx] = pxx
        clipped_df[c] = clipped_values

    clipped_df[col] = clipped_df[col].astype(int)

    # -----------------------------
    # 3) 목적*농장 일 단위 집계
    # -----------------------------
    daily_clipped_df = (
        clipped_df
        .groupby(['frmhs_no', 'visit_date'], as_index=False)[col]
        .sum()
    )

    # -----------------------------
    # 4) 평균 사육두수 병합 + 사육규모 카테고리
    # -----------------------------
    avg_pig_co = trainset.copy()
    avg_pig_co['farm_serial_no'] = avg_pig_co['farm_serial_no'].astype(str).str.zfill(8)

    avg_pig_co['scale_category'] = '미사육'
    non_zero_mask = avg_pig_co['average_breeding_livestock_count'] > 0
    # 분포가 빡빡할 때 대비: duplicates='drop'
    avg_pig_co.loc[non_zero_mask, 'scale_category'] = pd.qcut(
        avg_pig_co.loc[non_zero_mask, 'average_breeding_livestock_count'],
        q=4, labels=['소', '중소', '중대', '대'], duplicates='drop'
    )

    df_bio_cluster = daily_clipped_df.merge(
        avg_pig_co[['farm_serial_no', 'average_breeding_livestock_count', 'scale_category']],
        left_on='frmhs_no', right_on='farm_serial_no', how='left'
    )

    # -----------------------------
    # 5) 사육규모별(카테고리) 95% 클리핑 (0 제외)
    # -----------------------------
    def clip_outliers_per_column(df, group_col, target_cols, q=percentile):
        out = df.copy()
        p95_values = []
        for _col in target_cols:
            for cat, g in df.groupby(group_col, sort=False):
                non_zero_values = g.loc[g[_col] > 0, _col]
                if len(non_zero_values) == 0:
                    continue
                thr = non_zero_values.quantile(q)
                p95_values.append({group_col: cat, 'column': _col, 'p95': thr})
                mask = (out[group_col] == cat) & (out[_col] > thr)
                out.loc[mask, _col] = thr
        out[target_cols] = out[target_cols].astype(int)
        return out, pd.DataFrame(p95_values)

    clipped_df_bio_cluster, p95_df = clip_outliers_per_column(
        df=df_bio_cluster,
        group_col='scale_category',
        target_cols=col
    )

    # -----------------------------
    # 6) 방문 주기(diff_days) 계산 → 사육규모별 95% 클리핑 → 평균 주기
    # -----------------------------
    df_tmp = clipped_df_bio_cluster.copy()

    for purpose in col:
        diff_col = f'{purpose}_diff_days'
        df_tmp[diff_col] = pd.NA
        mask = df_tmp[purpose] > 0
        df_sub = df_tmp.loc[mask, ['frmhs_no', 'visit_date']].sort_values(['frmhs_no', 'visit_date'])
        diffs = df_sub.groupby('frmhs_no')['visit_date'].diff().dt.days
        df_tmp.loc[diffs.index, diff_col] = diffs

    col_list = [
        'livestock_transport_car_visit_count_diff_days',
        'animal_medicine_transport_car_visit_count_diff_days',
        'feed_transport_car_visit_count_diff_days',
        'livestock_excreta_transport_car_visit_count_diff_days',
        'compost_transport_car_visit_count_diff_days',
    ]

    for c_ in col_list:
        clipped = df_tmp[c_].copy()
        caps = df_tmp.groupby('scale_category')[c_].transform(lambda x: x.quantile(percentile))
        valid = caps.notna() & clipped.notna()
        clipped.loc[valid] = clipped.loc[valid].clip(upper=caps[valid])
        df_tmp[f'{c_}_clipped'] = clipped

    clipped_cols = [
        'livestock_transport_car_visit_count_diff_days_clipped',
        'animal_medicine_transport_car_visit_count_diff_days_clipped',
        'feed_transport_car_visit_count_diff_days_clipped',
        'livestock_excreta_transport_car_visit_count_diff_days_clipped',
        'compost_transport_car_visit_count_diff_days_clipped',
    ]

    mean_df = (
        df_tmp.groupby('frmhs_no', as_index=False)[clipped_cols]
              .mean()
    )
    # suffix: _diff_days_mean
    mean_df = mean_df.rename(columns={c: f'{c}_diff_days_mean' for c in clipped_cols})

    visit_freq_len = mean_df.copy()

    for c_ in visit_freq_len.columns:
        if c_ == 'frmhs_no':
            continue
        max_val = visit_freq_len[c_].max(skipna=True)
        visit_freq_len[c_] = visit_freq_len[c_].fillna(max_val)

    # -----------------------------
    # 7) 일평균 빈도/기타/차량수
    # -----------------------------
    tmp = clipped_df.copy()

    agg_dict = {'visit_date': ['min', 'max']}
    for c_ in main_col + other_cols:
        agg_dict[c_] = 'sum'

    farm_summary = (
        tmp.groupby('frmhs_no')
           .agg(agg_dict)
    )
    farm_summary.columns = ['_'.join(c) for c in farm_summary.columns]
    farm_summary = farm_summary.rename(columns={'visit_date_min':'start_date',
                                                'visit_date_max':'end_date'})
    farm_summary['period_days'] = (
        pd.to_datetime(farm_summary['end_date']) - pd.to_datetime(farm_summary['start_date'])
    ).dt.days + 1

    for c_ in main_col:
        farm_summary[f'daily_avg_{c_}'] = farm_summary[f'{c_}_sum'] / farm_summary['period_days']

    farm_summary['other_sum'] = farm_summary[[f'{c_}_sum' for c_ in other_cols]].sum(axis=1)
    farm_summary['daily_avg_other'] = farm_summary['other_sum'] / farm_summary['period_days']

    daily_avg_df = farm_summary.reset_index()[[
        'frmhs_no',
        *[f'daily_avg_{c_}' for c_ in main_col],
        'daily_avg_other'
    ]]

    car_unique_count = clipped_df.groupby('frmhs_no')['regist_no'].nunique().reset_index().rename(
        columns={'regist_no': 'car_unique_count'}
    )
    daily_avg_car_unique_count = (
        farm_summary.reset_index()
                    .merge(car_unique_count, on='frmhs_no', how='left')
    )
    daily_avg_car_unique_count['cars_per_day'] = (
        daily_avg_car_unique_count['car_unique_count'] / daily_avg_car_unique_count['period_days']
    )
    daily_avg_car_unique_count = daily_avg_car_unique_count[['frmhs_no', 'cars_per_day']]

    # -----------------------------
    # 8) 병합 + feature 지정 + 형변환
    # -----------------------------
    merged_df = (
        daily_avg_df
        .merge(daily_avg_car_unique_count, on='frmhs_no', how='left')
        .merge(visit_freq_len, on='frmhs_no', how='left')
    )

    feature = [
        'daily_avg_livestock_transport_car_visit_count',
        'daily_avg_animal_medicine_transport_car_visit_count',
        'daily_avg_feed_transport_car_visit_count',
        'daily_avg_livestock_excreta_transport_car_visit_count',
        'daily_avg_compost_transport_car_visit_count',
        'daily_avg_other',
        'cars_per_day',
        'livestock_transport_car_visit_count_diff_days_clipped_diff_days_mean',
        'animal_medicine_transport_car_visit_count_diff_days_clipped_diff_days_mean',
        'feed_transport_car_visit_count_diff_days_clipped_diff_days_mean',
        'livestock_excreta_transport_car_visit_count_diff_days_clipped_diff_days_mean',
        'compost_transport_car_visit_count_diff_days_clipped_diff_days_mean'
    ]
    merged_df[feature] = merged_df[feature].astype(float)

    # -----------------------------
    # 9) 로그 변환 + 그룹별 Min–Max 스케일링
    # -----------------------------
    df_log = merged_df.copy()
    df_log[feature] = np.log1p(df_log[feature])

    group1 = [
        'daily_avg_livestock_transport_car_visit_count',
        'daily_avg_animal_medicine_transport_car_visit_count',
        'daily_avg_feed_transport_car_visit_count',
        'daily_avg_livestock_excreta_transport_car_visit_count',
        'daily_avg_compost_transport_car_visit_count',
        'daily_avg_other'
    ]
    group2 = ['cars_per_day']
    group3 = [
        'livestock_transport_car_visit_count_diff_days_clipped_diff_days_mean',
        'animal_medicine_transport_car_visit_count_diff_days_clipped_diff_days_mean',
        'feed_transport_car_visit_count_diff_days_clipped_diff_days_mean',
        'livestock_excreta_transport_car_visit_count_diff_days_clipped_diff_days_mean',
        'compost_transport_car_visit_count_diff_days_clipped_diff_days_mean'
    ]

    df_scaled = df_log.copy()
    for grp in (group1, group2, group3):
        grp_min = df_scaled[grp].values.min()
        grp_max = df_scaled[grp].values.max()
        denom = grp_max - grp_min
        if denom == 0:
            df_scaled[grp] = 0.0
        else:
            df_scaled[grp] = (df_scaled[grp] - grp_min) / denom

    # ✅ 기준일 열 붙이기
    if ref_date is not None:
        merged_df['standard_date'] = ref_date
        df_scaled['standard_date'] = ref_date

    extras = dict(
        clipped_first=clipped_df,
        daily_clipped_df=daily_clipped_df,
        df_bio_cluster=df_bio_cluster,
        clipped_df_bio_cluster=clipped_df_bio_cluster,
        visit_freq_len=visit_freq_len,
        daily_avg_df=daily_avg_df,
        daily_avg_car_unique_count=daily_avg_car_unique_count,
        p95_df_scale_category=p95_df
    )

    return merged_df, df_scaled, feature, extras




def build_features_rolling_by_dates(
    visit_df: pd.DataFrame,
    trainset: pd.DataFrame,
    dates: list,
    window_years: int = 4,
    percentile: float = 0.95
):
    """
    여러 기준일에 대해 4년 롤링 윈도우 기반 변수를 일괄 생성.
    """

    merged_list, scaled_list = [], []
    feature_out = None

    import time

    for d in sorted(pd.to_datetime(dates)):
        start_time = time.time()  # 시작 시간 기록

        print(f"[{d.date()}] 처리 시작")

        merged_df, df_scaled, feature, _ = build_propagation_features(
            visit_df=visit_df,
            trainset=trainset,
            percentile=percentile,
            ref_date=d,
            window_years=window_years
        )

        merged_list.append(merged_df)
        scaled_list.append(df_scaled)
        # feature_out = feature  # 동일하므로 마지막 값 유지

        elapsed = time.time() - start_time  # 걸린 시간 계산
        print(f"[{d.date()}] 처리 완료 - 소요 시간: {elapsed:.2f}초")

    # merged_all = pd.concat(merged_list, ignore_index=True) if merged_list else pd.DataFrame()
    scaled_all = pd.concat(scaled_list, ignore_index=True) if scaled_list else pd.DataFrame()

    return scaled_all