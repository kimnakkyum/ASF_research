import pandas as pd
import numpy as np
import geopandas as gpd
import os
from glob import glob
import re
from advanced_daily_trainset_code.data_cleansing.config import config
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB
import sqlalchemy as sa
import warnings
warnings.filterwarnings("ignore")
from itertools import product
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
import matplotlib.ticker as mtick

db = GetDB()
path = fr'{config.dropbox_path}\19. 2025 업무\250101 농림축산검역본부\250428_ASF 분석 데이터 구축'

# ++ ===================================================================================================================

# 농장/야생 발생 시군 내 양돈 농장
asf_analysis_target_farm = db.query('''
    select farm_serial_no from asf.tb_asf_analysis_target_farm_clean_asf
    where (occurrence_yn = 1) -- or (occurrence_near_yn = 1)
''')

# ++ ===================================================================================================================

# 독립변수 데이터셋 로드
all_files = glob(os.path.join(path, '*.parquet'))
base_df = pd.concat([gpd.read_parquet(file) for file in all_files], ignore_index=True)
base_df['standard_date'] = pd.to_datetime(base_df['standard_date'])
base_df = base_df[base_df['standard_date'] >= '2020-01-01'] # 2020년 이후 데이터만 활용
base_df['standard_date'] = base_df['standard_date'].dt.strftime('%Y-%m-%d')

# 농장/야생 발생 시군 내 양돈 농장 필터링
base_df = base_df[base_df['farm_serial_no'].isin(asf_analysis_target_farm['farm_serial_no'])]

# ++ ===================================================================================================================

# 종속변수 로드
farm_occurrence = db.query('''
    SELECT DISTINCT ON (frmhs_no, sttemnt_de) 
        LPAD(frmhs_no, 8, '0') AS farm_serial_no,
        TO_DATE(sttemnt_de, 'YYYYMMDD') AS asf_org_day,
        1 AS asf_org
    FROM asf.tn_diss_occrrnc_frmhs_raw_asf oc
    join asf.tb_farm_geometry_clean_asf fg
    on LPAD(oc.frmhs_no, 8, '0') = fg.farm_serial_no
    and fg.geom_type = 'Point'
    WHERE diss_cl = '8' 
        AND delete_at = 'N' 
        AND SLAU_AT = 'N' 
        AND CNTRL_DT IS NOT NULL   
        AND ESTBS_NM NOT LIKE '%예살%'   
        AND ESTBS_NM NOT LIKE '%출하%'
        AND ESTBS_NM NOT LIKE '%차량%'   
        AND ESTBS_NM LIKE '%차%'
''')
farm_occurrence['asf_org_day'] = pd.to_datetime(farm_occurrence['asf_org_day'])
farm_occurrence = farm_occurrence[farm_occurrence['asf_org_day'] >= '2020-01-01'] # 2020년 이후 데이터만 활용
farm_occurrence['asf_org_day_yesterday'] = farm_occurrence['asf_org_day'] - pd.Timedelta(days=1)
for col in ['asf_org_day', 'asf_org_day_yesterday']:
    farm_occurrence[col] = farm_occurrence[col].dt.strftime('%Y-%m-%d')

# asf.tn_diss_occrrnc_frmhs_raw_asf 관련 히스토리
# m2msys.tn_diss_occrnc_frmhs_raw의 농장번호 80006392는 00022237 농장의 기타 시설로, 농장&축종 데이터와 매칭 시 누락됨
# 이에 수기 변경하여 80006392 -> 00022237 수정하여 asf.tn_diss_occrnc_frmhs_raw_asf에 적재함
# 추후, 농장번호 매칭 실패 건 존재 시 농장 번호를 변경하는 쿼리로 대체 예정

# ++ ===================================================================================================================

# 독립변수 + 종속변수 매칭
base_df = base_df.merge(farm_occurrence[['farm_serial_no', 'asf_org_day', 'asf_org_day_yesterday', 'asf_org']],
                                  how = 'left',
                                  left_on = ['farm_serial_no' ,'standard_date'],
                                  right_on = ['farm_serial_no', 'asf_org_day_yesterday'])
base_df['asf_org'].fillna(0, inplace=True)

# 20년 이후 발생 농장 38개 매칭 완료
print(base_df['asf_org'].sum()) # 38

# ++ ===================================================================================================================

occ_base_df = base_df[base_df['farm_serial_no'].isin(farm_occurrence['farm_serial_no'])].sort_values(by=['farm_serial_no', 'standard_date'])

# farm_serial_no별로 standard_date가 최대인 행 추출
latest_df = occ_base_df.loc[occ_base_df.groupby('farm_serial_no')['standard_date'].idxmax()]

# 조건: asf_org == 1, 6건 밖에 없음
# 발생 이후 바로 폐업되는 것은 아닌 것으로 보임
result_df = latest_df[latest_df['asf_org'] == 1].copy()

# ++ ===================================================================================================================

# 1. ASF 발생일 추출 (asf_org == 1인 행의 날짜)
asf_dates = occ_base_df[occ_base_df['asf_org'] == 1].groupby('farm_serial_no')['standard_date'].min().reset_index()
asf_dates.rename(columns={'standard_date': 'asf_date'}, inplace=True)

# 2. 최초 폐업일
first_change_dt = db.query('''
    WITH asf_org_farms AS (
        SELECT DISTINCT ON (frmhs_no, sttemnt_de) 
            LPAD(frmhs_no, 8, '0') AS farm_serial_no,
            TO_DATE(sttemnt_de, 'YYYYMMDD') AS asf_org_day,
            1 AS asf_org
        FROM asf.tn_diss_occrrnc_frmhs_raw_asf oc
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
    ),

    filtered_mngr_bsns AS (
        SELECT
            LPAD(frmhs_no, 8, '0') AS farm_serial_no,
            lstksp_cl,
            std_dt,
            mgr_code,
            bsns_sttus_se
        FROM geoai_mt.tn_mobile_frmhs_blvstck_avg_hist_clean
    ),

    joined_with_asf AS (
        SELECT f.*, a.asf_org_day
        FROM filtered_mngr_bsns f
        JOIN asf_org_farms a ON f.farm_serial_no = a.farm_serial_no
        WHERE f.std_dt > a.asf_org_day
    ),

    sequenced_status AS (
        SELECT 
            *,
            LAG(mgr_code) OVER (PARTITION BY farm_serial_no, lstksp_cl ORDER BY std_dt) AS prev_mgr_code,
            LAG(bsns_sttus_se) OVER (PARTITION BY farm_serial_no, lstksp_cl ORDER BY std_dt) AS prev_bsns_sttus_se
        FROM joined_with_asf
    ),

    mgr_rest_first_date AS (
        SELECT DISTINCT ON (f.farm_serial_no, d.lstksp_cl)
            f.farm_serial_no,
            d.lstksp_cl,
            d.std_dt AS mgr_rest_first_dt
        FROM asf_org_farms f
        JOIN filtered_mngr_bsns d
          ON f.farm_serial_no = d.farm_serial_no
         AND d.std_dt > f.asf_org_day
         AND d.mgr_code = '104002'
        ORDER BY f.farm_serial_no, d.lstksp_cl, d.std_dt
    ),

    bsns_rest_first_date AS (
        SELECT DISTINCT ON (f.farm_serial_no, d.lstksp_cl)
            f.farm_serial_no,
            d.lstksp_cl,
            d.std_dt AS bsns_rest_first_dt
        FROM asf_org_farms f
        JOIN filtered_mngr_bsns d
          ON f.farm_serial_no = d.farm_serial_no
         AND d.std_dt > f.asf_org_day
         AND d.bsns_sttus_se = '2'
        ORDER BY f.farm_serial_no, d.lstksp_cl, d.std_dt
    ),

    mgr_closure_first_date AS (
        SELECT DISTINCT ON (f.farm_serial_no, d.lstksp_cl)
            f.farm_serial_no,
            d.std_dt AS mgr_closure_first_dt
        FROM asf_org_farms f
        JOIN filtered_mngr_bsns d
          ON f.farm_serial_no = d.farm_serial_no
         AND d.std_dt > f.asf_org_day
         AND d.mgr_code = '104003'
        ORDER BY f.farm_serial_no, d.lstksp_cl, d.std_dt
    ),

    bsns_closure_first_date AS (
        SELECT DISTINCT ON (f.farm_serial_no, d.lstksp_cl)
            f.farm_serial_no,
            d.std_dt AS bsns_closure_first_dt
        FROM asf_org_farms f
        JOIN filtered_mngr_bsns d
          ON f.farm_serial_no = d.farm_serial_no
         AND d.std_dt > f.asf_org_day
         AND d.bsns_sttus_se = '3'
        ORDER BY f.farm_serial_no, d.lstksp_cl, d.std_dt
    ),

    rest_to_open_first_date_mgr AS (
        SELECT DISTINCT ON (farm_serial_no, lstksp_cl)
            farm_serial_no,
            std_dt AS rest_to_open_first_dt_mgr
        FROM sequenced_status
        WHERE prev_mgr_code in ('104002') AND mgr_code = '104001'
        ORDER BY farm_serial_no, lstksp_cl, std_dt
    ),

    rest_to_open_first_date_bsns AS (
        SELECT DISTINCT ON (farm_serial_no, lstksp_cl)
            farm_serial_no,
            std_dt AS rest_to_open_first_dt_bsns
        FROM sequenced_status
        WHERE prev_bsns_sttus_se in ('2') AND bsns_sttus_se = '1'
        ORDER BY farm_serial_no, lstksp_cl, std_dt
    ),

    closure_to_open_first_date_mgr AS (
        SELECT DISTINCT ON (farm_serial_no, lstksp_cl)
            farm_serial_no,
            std_dt AS closure_to_open_first_dt_mgr
        FROM sequenced_status
        WHERE prev_mgr_code in ('104003') AND mgr_code = '104001'
        ORDER BY farm_serial_no, lstksp_cl, std_dt
    ),

    closure_to_open_first_date_bsns AS (
        SELECT DISTINCT ON (farm_serial_no, lstksp_cl)
            farm_serial_no,
            std_dt AS closure_to_open_first_dt_bsns
        FROM sequenced_status
        WHERE prev_bsns_sttus_se in ('3') AND bsns_sttus_se = '1'
        ORDER BY farm_serial_no, lstksp_cl, std_dt
    )

    SELECT 
        a.farm_serial_no,
        a.asf_org_day,
        mr.mgr_rest_first_dt,
        br.bsns_rest_first_dt,
        mc.mgr_closure_first_dt,
        bc.bsns_closure_first_dt,
        rom.rest_to_open_first_dt_mgr,
        rob.rest_to_open_first_dt_bsns,
        com.closure_to_open_first_dt_mgr,
        cob.closure_to_open_first_dt_bsns,
        LEAST(mr.mgr_rest_first_dt, br.bsns_rest_first_dt) AS earliest_rest_dt,
        LEAST(mc.mgr_closure_first_dt, bc.bsns_closure_first_dt) AS earliest_closure_dt,
        LEAST(rom.rest_to_open_first_dt_mgr, rob.rest_to_open_first_dt_bsns) AS earliest_rest_to_open_dt,
        LEAST(com.closure_to_open_first_dt_mgr, cob.closure_to_open_first_dt_bsns) AS earliest_closure_to_open_dt
    FROM asf_org_farms a
    LEFT JOIN mgr_rest_first_date mr ON a.farm_serial_no = mr.farm_serial_no
    LEFT JOIN bsns_rest_first_date br ON a.farm_serial_no = br.farm_serial_no
    LEFT JOIN mgr_closure_first_date mc ON a.farm_serial_no = mc.farm_serial_no
    LEFT JOIN bsns_closure_first_date bc ON a.farm_serial_no = bc.farm_serial_no
    LEFT JOIN rest_to_open_first_date_mgr rom ON a.farm_serial_no = rom.farm_serial_no
    LEFT JOIN rest_to_open_first_date_bsns rob ON a.farm_serial_no = rob.farm_serial_no
    LEFT JOIN closure_to_open_first_date_mgr com ON a.farm_serial_no = com.farm_serial_no
    LEFT JOIN closure_to_open_first_date_bsns cob ON a.farm_serial_no = cob.farm_serial_no
''')

# 3. 병합
merged = pd.merge(asf_dates, first_change_dt, on='farm_serial_no', how='left')

# 4. 날짜 차이 계산
merged['rest_days_since_asf'] = (
            pd.to_datetime(merged['earliest_rest_dt']) - pd.to_datetime(merged['asf_org_day'])).dt.days
merged['closure_days_since_asf'] = (
            pd.to_datetime(merged['earliest_closure_dt']) - pd.to_datetime(merged['asf_org_day'])).dt.days
merged['rest_to_open_days'] = (
            pd.to_datetime(merged['earliest_rest_to_open_dt']) - pd.to_datetime(merged['earliest_rest_dt'])).dt.days
merged['closure_to_open_days'] = (
            pd.to_datetime(merged['earliest_closure_to_open_dt']) - pd.to_datetime(merged['earliest_closure_dt'])).dt.days

merged.sort_values(by=['farm_serial_no', 'rest_days_since_asf'], inplace=True)
merged = merged.drop_duplicates(subset=['farm_serial_no'])
merged.to_excel(fr'{path}/eda/발생 이후 최초 폐업일자.xlsx', index=False)

# 38개 농장
# case1. 운영 -> 폐업 농장 2개
case1 = merged[(merged['rest_days_since_asf'].isnull())]
print(case1['closure_days_since_asf'].median())

# case2. 운영 -> 휴업 농장 36개
case2 = merged[(merged['rest_days_since_asf'].notnull())]
print(case2['rest_days_since_asf'].median())

# case2-1. 운영 -> 계속 휴업 9개
case2_1 = merged[(merged['rest_days_since_asf'].notnull())
                 & (merged['closure_days_since_asf'].isnull())
                 & (merged['rest_to_open_days'].isnull())]

# case2-2. 운영 -> 휴업 -> 운영 11개
case2_2 = merged[(merged['rest_days_since_asf'].notnull())
                 & (merged['closure_days_since_asf'].isnull())
                 & (merged['rest_to_open_days'].notnull())]
print(case2_2['rest_to_open_days'].median())

# case2-3. 운영 -> 휴업 -> 폐업 농장 16개
case2_3 = merged[(merged['rest_days_since_asf'].notnull())
                 & (merged['closure_days_since_asf'].notnull())]
print((case2_3['closure_days_since_asf'] - case2_3['rest_days_since_asf']).median())

# case2-3-1. 운영 -> 휴업 -> 계속 폐업 농장 6개
case2_3_1 = merged[(merged['rest_days_since_asf'].notnull())
                   & (merged['closure_days_since_asf'].notnull())
                   & (merged['closure_to_open_days'].isnull())]

# case2-3-2. 운영 -> 휴업 -> 폐업 -> 운영 농장 10개
case2_3_2 = merged[(merged['rest_days_since_asf'].notnull())
                   & (merged['closure_days_since_asf'].notnull())
                   & (merged['closure_to_open_days'].notnull())]
print(case2_3_2['closure_to_open_days'].median())

# ++ ===================================================================================================================

# 분석 제외 방문 목적
car_visit_del_col = ['알운반', '원유운반', '기계수리', '가금 출하 상하차 인력운송']

# 역학 기간 내 차량 방문
car_visit_epidemic_period = db.query('''
    WITH asf_org AS (
        SELECT DISTINCT ON (frmhs_no, sttemnt_de) 
            LPAD(frmhs_no, 8, '0') AS farm_serial_no,
            TO_DATE(sttemnt_de, 'YYYYMMDD') - INTERVAL '1 day' AS asf_org_day, 
            1 AS asf_org
        FROM asf.tn_diss_occrrnc_frmhs_raw_asf oc
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
          AND TO_DATE(sttemnt_de, 'YYYYMMDD') - INTERVAL '1 day' >= '2020-01-01'
    )
    
    SELECT 
        a.asf_org_day,
        v.std_dt,
        v.frmhs_no,
        v.visit_de,
        v.visit_ty,
        v.visit_purps_cn,
        v.visit_nm,
        v.regist_no
    FROM m2msys.tn_visit_info_clean v
    JOIN asf_org a
      ON v.frmhs_no = a.farm_serial_no
    WHERE v.visit_de >= a.asf_org_day - INTERVAL '21 days'
      AND v.visit_de <  a.asf_org_day + INTERVAL '1 day'
    ORDER BY a.asf_org_day, v.frmhs_no, v.visit_de
''')
car_visit_epidemic_period = car_visit_epidemic_period[~car_visit_epidemic_period['visit_purps_cn'].isin(car_visit_del_col)]
car_visit_epidemic_period2 = car_visit_epidemic_period[~car_visit_epidemic_period['visit_purps_cn'].isin(['시료채취,방역'])]

# 발생 농장들은 역학 기간 내 어떤 차량 방문이 많았는지
car_visit_epidemic_period_ty_counts = car_visit_epidemic_period2.groupby(['visit_purps_cn']).size().sort_values(ascending=False).reset_index(name = '역학기간 내 방문 수')
car_visit_epidemic_period_ty_counts['ratio'] = car_visit_epidemic_period_ty_counts['역학기간 내 방문 수'] / car_visit_epidemic_period2.shape[0]

# 농장 단위 분석
frmhs_car_visit_epidemic_period_ty_counts = car_visit_epidemic_period2.groupby(['frmhs_no', 'visit_purps_cn']).size().sort_values(ascending=False).reset_index(name = '역학기간 내 방문 수')
frmhs_car_visit_epidemic_period_ty_counts['역학기간 내 일평균 방문 수'] = frmhs_car_visit_epidemic_period_ty_counts['역학기간 내 방문 수'] / 21
frmhs_car_visit_epidemic_period_ty_counts = pd.pivot(frmhs_car_visit_epidemic_period_ty_counts, index = ['frmhs_no'], columns = 'visit_purps_cn', values = '역학기간 내 일평균 방문 수').fillna(0)

# ++ ===================================================================================================================

# 발생 농장 대상으로 역학 기간 이전의 방문 비율과 크게 차이나는 것은 무엇인지?
car_visit_non_epidemic_period = db.query('''
    WITH asf_org AS (
        SELECT DISTINCT ON (frmhs_no, sttemnt_de) 
            LPAD(frmhs_no, 8, '0') AS farm_serial_no,
            TO_DATE(sttemnt_de, 'YYYYMMDD') - INTERVAL '1 day' AS asf_org_day, 
            1 AS asf_org
        FROM asf.tn_diss_occrrnc_frmhs_raw_asf oc
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
          AND TO_DATE(sttemnt_de, 'YYYYMMDD') - INTERVAL '1 day' >= '2020-01-01'
    )

    SELECT 
        a.asf_org_day,
        v.std_dt,
        v.frmhs_no,
        v.visit_de,
        v.visit_ty,
        v.visit_purps_cn,
        v.visit_nm,
        v.regist_no
    FROM m2msys.tn_visit_info_clean v
    JOIN asf_org a
      ON v.frmhs_no = a.farm_serial_no
    WHERE v.visit_de >= a.asf_org_day - INTERVAL '21 days' - INTERVAL '365 days' 
    AND v.visit_de < a.asf_org_day - INTERVAL '21 days'
    ORDER BY a.asf_org_day, v.frmhs_no, v.visit_de
''')
car_visit_non_epidemic_period = car_visit_non_epidemic_period[~car_visit_non_epidemic_period['visit_purps_cn'].isin(car_visit_del_col)]
car_visit_non_epidemic_period2 = car_visit_non_epidemic_period[~car_visit_non_epidemic_period['visit_purps_cn'].isin(['시료채취,방역'])]

# 발생 농장들은 역학 기간 이전에 어떤 차량 방문이 많았는지
car_visit_non_epidemic_period_ty_counts = car_visit_non_epidemic_period2.groupby(['visit_purps_cn']).size().sort_values(ascending=False).reset_index(name = '역학기간 이전 방문 수')
car_visit_non_epidemic_period_ty_counts['ratio'] = car_visit_non_epidemic_period_ty_counts['역학기간 이전 방문 수'] / car_visit_non_epidemic_period2.shape[0]

# 농장 단위 분석
frmhs_car_visit_non_epidemic_period_ty_counts = car_visit_non_epidemic_period2.groupby(['frmhs_no', 'visit_purps_cn']).size().sort_values(ascending=False).reset_index(name = '역학기간 이전 방문 수')
frmhs_car_visit_non_epidemic_period_ty_counts['역학기간 이전 일평균 방문 수'] = frmhs_car_visit_non_epidemic_period_ty_counts['역학기간 이전 방문 수'] / 365
frmhs_car_visit_non_epidemic_period_ty_counts = pd.pivot(frmhs_car_visit_non_epidemic_period_ty_counts, index = ['frmhs_no'], columns = 'visit_purps_cn', values = '역학기간 이전 일평균 방문 수').fillna(0)

# ++ ===================================================================================================================

# 두 데이터프레임을 'visit_purps_cn' 기준으로 병합
df_merged = pd.merge(
    car_visit_epidemic_period_ty_counts,
    car_visit_non_epidemic_period_ty_counts,
    on='visit_purps_cn',
    how='outer'
).fillna(0)

# 전체 방문 수 대비 비율 계산
df_merged['역학기간 내 비율'] = df_merged['역학기간 내 방문 수'] / df_merged['역학기간 내 방문 수'].sum()
df_merged['역학기간 이전 비율'] = df_merged['역학기간 이전 방문 수'] / df_merged['역학기간 이전 방문 수'].sum()

# 막대그래프용 데이터 준비
labels = df_merged['visit_purps_cn']
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(x - width/2, df_merged['역학기간 이전 비율'], width, label='역학기간 이전')
bar2 = ax.bar(x + width/2, df_merged['역학기간 내 비율'], width, label='역학기간 내')

# y축을 퍼센트로 표시
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))  # 1.0 기준 → 0.2 → 20%

# 라벨 및 제목 설정
ax.set_ylabel('전체 방문 수 대비 비율')
ax.set_title('방문 목적별 비율 비교 (역학기간 이전 vs 내)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(fr'{path}/eda/차량 방문/방문 목적별 비율 비교 (역학기간 이전 vs 내).png')
plt.show()

# 역학기간 내, 이전에 대해 목적별 평균 계산 차이 plot
mean_non_epi = frmhs_car_visit_non_epidemic_period_ty_counts.mean().rename('역학기간 이전')
mean_epi = frmhs_car_visit_epidemic_period_ty_counts.mean().rename('역학기간 내')

# 하나로 합치기
mean_df = pd.concat([mean_non_epi, mean_epi], axis=1)
mean_df['dif'] = mean_df['역학기간 내'] - mean_df['역학기간 이전']

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(mean_df.index))
width = 0.35

ax.bar(x - width/2, mean_df['역학기간 이전'], width, label='역학기간 이전')
ax.bar(x + width/2, mean_df['역학기간 내'], width, label='역학기간 내')

ax.set_xticks(x)
ax.set_xticklabels(mean_df.index, rotation=45, ha='right')
ax.set_ylabel('일평균 방문 수 (발생 농장 평균)')
ax.set_title('방문 목적별 일평균 방문 수 비교 (역학기간 이전 vs 내)')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(fr'{path}/eda/차량 방문/방문 목적별 일평균 방문 수 비교 (역학기간 이전 vs 내).png')
plt.show()

# ++ ===================================================================================================================

# 발생 농장에 대해 역학기간 전 1년과 역학기간 내 평균 방문 빈도 비교 match
match = pd.merge(frmhs_car_visit_epidemic_period_ty_counts.mean().reset_index(name = '일평균 방문 수 평균'),
                 frmhs_car_visit_non_epidemic_period_ty_counts.mean().reset_index(name = '일평균 방문 수 평균'),
                 on = ['visit_purps_cn'])

# frmhs_car_visit_epidemic_period_ty_counts['합계'] = np.sum(frmhs_car_visit_epidemic_period_ty_counts, axis=1)
# frmhs_car_visit_non_epidemic_period_ty_counts['합계'] = np.sum(frmhs_car_visit_non_epidemic_period_ty_counts, axis=1)

joined_df = frmhs_car_visit_non_epidemic_period_ty_counts.join(
    frmhs_car_visit_epidemic_period_ty_counts,
    lsuffix="_before", rsuffix="_during", how="inner"
)
joined_df = joined_df[sorted(joined_df.columns)]

# wilcoxon 순위 검정. 대응표본에 대해 전, 후 차이가 존재하는지
from scipy.stats import wilcoxon
import numpy as np

def calc_rank_biserial(before, during):
    diff = during - before
    non_zero_diff = diff[diff != 0]

    if len(non_zero_diff) == 0:
        return 0.0  # 모든 값이 동일할 경우 효과 없음

    positive_ranks = np.sum(non_zero_diff > 0)
    negative_ranks = np.sum(non_zero_diff < 0)

    r = (positive_ranks - negative_ranks) / len(non_zero_diff)
    return r

# 결과 저장 list
results = []

# 방문 목적 추출
visit_purposes = [col.replace('_before', '') for col in joined_df.columns if col.endswith('_before')]

for purp in visit_purposes:
    before = joined_df[f"{purp}_before"]
    during = joined_df[f"{purp}_during"]

    # 0 초과하는 행이 10개 이상인 경우에만 수행
    if ((before > 0) | (during > 0)).sum() >= 10:
        try:
            stat, p = wilcoxon(before, during)
            r = calc_rank_biserial(before, during)
            results.append({
                "visit_purps_cn": purp,
                "before_mean": before.mean(),
                "during_mean": during.mean(),
                "mean_diff": during.mean() - before.mean(),
                "p_value": p,
                "effect_size": r,
                "significant": p < 0.05
            })
        except Exception as e:
            print(f"{purp} 실패: {e}")

stats_df = pd.DataFrame(results)
stats_df = stats_df.sort_values("p_value")

# 검증 결과, 차이가 통계적으로 유의미한 방문은 없음
# 결론 : 모든 발생 농장에 대해 통계적으로 유의미한 방문 수 변화가 일어난 방문 목적은 없음
stats_df["significant"] = stats_df["p_value"] < 0.05  # 유의성 표시
stats_df.to_excel(fr'{path}/eda/차량 방문/방문목적별 역학기간 이전, 내 일평균 방문 수 차이 통계적 검정.xlsx', index=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=stats_df,
    x='visit_purps_cn',
    y='p_value',
    hue='significant',
    palette={True: 'orange', False: 'skyblue'}
)
plt.axhline(0.05, color='red', linestyle='--', label='p = 0.05 기준선')
plt.xticks(rotation=45, ha='right')
plt.xlabel('')
plt.ylabel('p-value')
plt.title('방문 목적별 일평균 방문 수 차이 통계적 검정 p-value')
plt.legend(title='유의성 여부')
plt.tight_layout()
plt.savefig(fr'{path}/eda/차량 방문/방문 목적별 일평균 방문 수 차이 통계적 검정 p-value.png')
plt.show()

# ++ ===================================================================================================================

# 각 방문의 발생일자와의 차이가 어느정도 되는지. 발생과 너무 뚜렷한 관계를 갖는 변수는 아닐지 파악 목적
car_visit_epidemic_period['dif'] = (car_visit_epidemic_period['asf_org_day'] + pd.Timedelta(days = 1) - car_visit_epidemic_period['visit_de']).dt.days
car_visit_epidemic_period['within_7_days'] = car_visit_epidemic_period['dif'] <= 7

# 시료채취.방역, 진료.예방접종 방문 목적
disinfection_vaccine = car_visit_epidemic_period[car_visit_epidemic_period['visit_purps_cn'].isin(['시료채취,방역', '진료.예방접종'])]
disinfection_vaccine.sort_values(by=['frmhs_no', 'visit_de'], inplace=True)

# 요약 집계
summary_df = car_visit_epidemic_period.groupby('visit_purps_cn').agg(
    dif_median=('dif', 'median'),
    within_7_days_ratio=('within_7_days', 'mean')
).reset_index().sort_values('dif_median')

# 시각화
plt.figure(figsize=(10, 6))
bars = sns.barplot(
    data=summary_df,
    x='visit_purps_cn',
    y='dif_median',
    palette=sns.color_palette("OrRd", n_colors=len(summary_df))
)

# # 막대 위에 7일 이내 비율 텍스트 추가
# for bar, ratio in zip(bars.patches, summary_df['within_7_days_ratio']):
#     height = bar.get_height()
#     plt.text(
#         bar.get_x() + bar.get_width() / 2,
#         height + 0.3,
#         f"{ratio:.0%}",
#         ha='center',
#         va='bottom',
#         fontsize=10
#     )

plt.xticks(rotation=45, ha='right')
plt.xlabel('')
plt.ylabel('ASF 발생일과의 방문 시차 중위값 (일)')
plt.title('역학기간 내 방문 목적별 농장 발생일과 차량 방문 간 시차 중위값 (일)') # \n(막대 위 숫자: 7일 이내 방문 비율)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(fr'{path}/eda/차량 방문/역학기간 내 방문 목적별 농장 발생일과 차량 방문 간 시차 중위값 (일).png')
plt.show()

# ++ ===================================================================================================================

# 일상적으로 방문하는 차량 방문 목적인지, 아니면 특수한 상황에서만 방문하는 차량 방문 목적인지
# 이를 파악하기 위해, 역학기간 이전 차량 방문에 대해 방문 주기를 분석

# 날짜 정렬
filtered_df = car_visit_non_epidemic_period.copy()
filtered_df = filtered_df.sort_values(['frmhs_no', 'visit_purps_cn', 'visit_de'])

# 각 농장-방문목적 단위로 방문 간격 계산
for (farm, purpose), group in filtered_df.groupby(['frmhs_no', 'visit_purps_cn']):
    dates = group['visit_de'].sort_values().drop_duplicates()

    if len(dates) >= 2:
        intervals = dates.diff().dt.days.dropna()
        avg_interval = intervals.mean()
        std_interval = intervals.std()
        visit_cnt = len(dates)
        results.append({
            'frmhs_no': farm,
            'visit_purps_cn': purpose,
            'visit_cnt': visit_cnt,
            'avg_interval_days': avg_interval,
            'std_interval_days': std_interval
        })

# 데이터프레임으로 정리
visit_pattern_df = pd.DataFrame(results)
gap_summary = visit_pattern_df.groupby('visit_purps_cn')['avg_interval_days'].describe().reset_index()

summary_df = pd.merge(summary_df, gap_summary[['visit_purps_cn', '50%']], on = 'visit_purps_cn')
summary_df['dif'] = summary_df['50%'] - summary_df['dif_median']

plt.figure(figsize = (10, 6))
sns.barplot(data = summary_df.sort_values(by='50%'),
            x = 'visit_purps_cn',
            y = '50%',
            palette=sns.color_palette("OrRd", n_colors=len(summary_df))
            )
plt.title('방문 목적별 방문 주기 중위값 비교(역학기간 이전)')
plt.xlabel('')
plt.ylabel('방문 간 시차 (일)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(fr'{path}/eda/차량 방문/방문 목적별 방문 주기 중위값 비교.png')
plt.show()

# fig, ax = plt.subplots(figsize=(10, 6))
# x = np.arange(len(summary_df.index))
# width = 0.35
#
# ax.bar(x - width/2, summary_df['dif_median'], width, label='발생일 기준 방문 시차 (역학기간 내)')
# ax.bar(x + width/2, summary_df['50%'], width, label='방문 주기 중위값 (역학기간 이전)')
#
# ax.set_xticks(x)
# ax.set_xticklabels(summary_df['visit_purps_cn'], rotation=45, ha='right')
#
# ax.set_ylabel('방문 간 시차 (일)')
# ax.set_title('방문 목적별 ASF 발생일 기준 방문 시차 및 방문 주기 중위값 비교')
# ax.legend()
# ax.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.savefig(fr'{path}/eda/차량 방문/방문 목적별 ASF 발생일 기준 방문 시차 및 방문 주기 중위값 비교.png')
# plt.show()

# ++ ===================================================================================================================

# 발생농장
occ_frmhs = gpd.read_postgis(sql = sa.text(f'''
        SELECT DISTINCT ON (frmhs_no, sttemnt_de) 
            LPAD(frmhs_no, 8, '0') AS farm_serial_no,
            TO_DATE(sttemnt_de, 'YYYYMMDD') AS asf_org_day,
            estbs_nm, 
            1 AS asf_org,
            fg.geometry
        FROM asf.tn_diss_occrrnc_frmhs_raw_asf oc
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
          AND TO_DATE(sttemnt_de, 'YYYYMMDD') >= '2020-01-01'
          '''), con=db.engine, geom_col = 'geometry')
occ_frmhs['odr'] = occ_frmhs['estbs_nm'].apply(lambda x : re.findall(r'\d+차', x)[0])
occ_frmhs.to_parquet(fr'{path}/eda/발생농장.parquet')

# 야생발생
query = """
    SELECT distinct on (odr)
    TO_DATE(sttemnt_de, 'YYYYMMDD') AS wild_asf_org_day,
    ST_Transform(ST_SetSRID(ST_Point(lo, la), 4326), 5179) AS geometry
    FROM m2msys.tn_diss_occ_ntcn_info_raw
    WHERE STTEMNT_DE::date >= '2020-01-01' 
    order by odr, last_change_dt desc
"""
wild_occ = gpd.read_postgis(query, db.engine, geom_col='geometry')
wild_occ['wild_asf_org_day'] = pd.to_datetime(wild_occ['wild_asf_org_day'])
wild_occ['wild_asf_org_day'] = wild_occ['wild_asf_org_day'].dt.strftime('%Y-%m-%d')
wild_occ.to_parquet(fr'{path}/eda/야생발생.parquet')

# 야생멧돼지 수
query = """
    SELECT *
    FROM geoai_polygon.tb_forest_wildboar_count_clean
    where year = '2023'
"""
wildboar = gpd.read_postgis(query, db.engine, geom_col='geometry')
wildboar.to_parquet(fr'{path}/eda/야생멧돼지수.parquet')

# 역학기간 내 방문 차량을 집계했을 때, 여러 농장에 공통적으로 방문한 차량
car_frmhs_counts = car_visit_epidemic_period.groupby(['regist_no'])['frmhs_no'].nunique().sort_values(ascending=False).reset_index(name='frmhs_count')
car_frmhs_counts_over1 = car_frmhs_counts[car_frmhs_counts['frmhs_count'] > 1]
print(car_frmhs_counts_over1.shape[0] / len(car_frmhs_counts))

frmhs_car_counts = car_visit_epidemic_period.groupby(['frmhs_no'])['regist_no'].size().reset_index(name='차량 방문 수')
frmhs_car_unique_counts = car_visit_epidemic_period.groupby(['frmhs_no'])['regist_no'].nunique().reset_index(name='방문 차량 unique 개수')
frmhs_car_unique = car_visit_epidemic_period.groupby(['frmhs_no'])['regist_no'].unique().reset_index(name='unique')

for i in frmhs_car_unique.index :
    car_visit_epidemic_period_frmhs = car_visit_epidemic_period[car_visit_epidemic_period['frmhs_no'] == frmhs_car_unique.loc[i, 'frmhs_no']]
    another_frmhs_visit_unique = set(car_frmhs_counts_over1['regist_no']) & set(frmhs_car_unique.loc[i, 'unique'])
    car_visit_epidemic_period_frmhs = car_visit_epidemic_period_frmhs[car_visit_epidemic_period_frmhs['regist_no'].isin(another_frmhs_visit_unique)]
    frmhs_car_counts.loc[i, '타 발생 농장 방문이력 존재 차량 방문 수'] = len(car_visit_epidemic_period_frmhs)
    frmhs_car_unique_counts.loc[i, '타 발생 농장 방문이력 존재 차량 unique 개수'] = len(another_frmhs_visit_unique)

match = pd.merge(frmhs_car_counts, frmhs_car_unique_counts, on = 'frmhs_no')

# ++ ===================================================================================================================

farm_point = gpd.read_postgis(sql=sa.text(f'''
    with
    -- 농장 좌표 데이터
    farm_point AS (
        SELECT distinct on (farms_no) farms_no, xmin_, ymin
        FROM m2msys.nvrqs_mobile_farms_raw
        order by farms_no, std_dt desc
    )

    select 
        farms_no,
        st_transform(('SRID=4326;POINT(' || xmin_ || ' ' || ymin || ')')::geometry, 5179) as geometry
    from farm_point
'''), con=db.engine, geom_col='geometry')

# ++ ===================================================================================================================

# 정리하면, 28 ~ 30차 발생은 서로 엮여있음
# 28차 발생(20243602)이 22년 11월 9일 가장 먼저 발생. 이미 농장 주변은 오염되어 있음
# 29차 발생(01720048)은 23년 1월 5일 발생. 28차 발생 농장 근처 농장(30차 발생 농장)으로 자돈 분양 및 비육돈 출하과정에서 오염
# 30차 발생(20399236)은 23년 1월 11일 발생. 29차 -> 30차 농장으로 자돈 입식 과정에서 소독 미실시로 인해 오염
# -> 그럼 여기서 찾아야 하는건? 29차 발생 농장과 30차 발생 농장을 동시에 방문한 차량이 있는지 찾는 것

target_farm_list = ['20243602', '01720048', '20399236']
occ_frmhs_28_30 = occ_frmhs[occ_frmhs['farm_serial_no'].isin(target_farm_list)]
occ_frmhs_28_30['odr_std_dt'] = occ_frmhs_28_30['odr'] + '\n' + '(' + occ_frmhs_28_30['asf_org_day'].astype(str) + ')'
occ_frmhs_28_30.to_parquet(fr'{path}/eda/차량 방문/수평 전파/28~30차/28~30차 발생 농장.parquet')

occ_frmhs_28_30_buffer = occ_frmhs_28_30.copy()
occ_frmhs_28_30_buffer['geometry'] = occ_frmhs_28_30_buffer.geometry.buffer(10000) # 10km buffer

match = gpd.sjoin(wild_occ[['wild_asf_org_day', 'geometry']], occ_frmhs_28_30_buffer, predicate = 'within').drop(['index_right'], axis=1)
match['wild_asf_org_day'] = pd.to_datetime(match['wild_asf_org_day'])
match['asf_org_day'] = pd.to_datetime(match['asf_org_day'])
match['dif'] = (match['asf_org_day'] - match['wild_asf_org_day']).dt.days
print(((match['dif'] <= 365) & (match['dif'] > 0)).sum())

tmp = car_visit_epidemic_period2[car_visit_epidemic_period2['frmhs_no'].isin(target_farm_list)]
tmp2 = tmp.groupby(['regist_no'])['frmhs_no'].nunique().reset_index(name = 'nunique')
tmp2 = tmp2[tmp2['nunique']>=2]

for regist_no in tmp2['regist_no'] :
    tmp3 = car_visit_epidemic_period2[car_visit_epidemic_period2['regist_no'] == regist_no].sort_values(by='visit_de')
    tmp3 = tmp3[['regist_no', 'frmhs_no', 'visit_de', 'visit_purps_cn']]
    tmp3.to_excel(fr'{path}/eda/차량 방문/수평 전파/28~30차/{regist_no}.xlsx', index=False)

target_farm_str = ",".join([f"'{i}'" for i in target_farm_list])
target_car_str = ",".join([f"'{i}'" for i in tmp2['regist_no'].to_list()])
around_car_visit_epidemic_period = db.query(f'''
    WITH asf_org AS (
        SELECT DISTINCT ON (frmhs_no, sttemnt_de) 
            LPAD(frmhs_no, 8, '0') AS farm_serial_no,
            TO_DATE(sttemnt_de, 'YYYYMMDD') - INTERVAL '1 day' AS asf_org_day
        FROM asf.tn_diss_occrrnc_frmhs_raw_asf oc
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
          AND TO_DATE(sttemnt_de, 'YYYYMMDD') - INTERVAL '1 day' >= '2020-01-01'
          AND LPAD(frmhs_no, 8, '0') IN ({target_farm_str})
    ),
    
    -- 차량 방문 데이터
    car_visit AS (
        SELECT 
            std_dt,
            frmhs_no,
            visit_de,
            visit_ty,
            visit_purps_cn,
            visit_nm,
            regist_no
        FROM m2msys.tn_visit_info_clean
        WHERE regist_no in ({target_car_str})
        AND visit_de >= (select min(asf_org_day) from asf_org) - INTERVAL '21 days'
        AND visit_de < (select max(asf_org_day) from asf_org) + INTERVAL '1 day'
    )
    
    SELECT 
    distinct on (v.regist_no, v.frmhs_no, v.visit_de)
        v.regist_no,
        v.frmhs_no,
        v.std_dt,
        v.visit_de,
        v.visit_ty,
        v.visit_purps_cn,
        v.visit_nm
    FROM car_visit v
    CROSS JOIN asf_org a
    WHERE v.visit_de >= a.asf_org_day - INTERVAL '21 days'
      AND v.visit_de <  a.asf_org_day + INTERVAL '1 day'
    ORDER BY v.regist_no, v.visit_de
''')

# 줄임 처리 함수 정의
def summarize_dates(dates):
    dates = list(dates)
    n = len(dates)
    if n <= 10:
        return '\n'.join(dates)
    else:
        top = dates[:5]
        bottom = dates[-5:]
        return '\n'.join(top + ['...'] + bottom)

for regist_no in tmp2['regist_no'] :
    around_car_visit_epidemic_period2 = around_car_visit_epidemic_period[around_car_visit_epidemic_period['regist_no'] == regist_no]
    around_car_visit_epidemic_period2['std_dt'] = pd.to_datetime(around_car_visit_epidemic_period2['std_dt'])
    around_car_visit_epidemic_period2 = around_car_visit_epidemic_period2[(around_car_visit_epidemic_period2['std_dt'] >= '2022-11-09')
                                                            & (around_car_visit_epidemic_period2['std_dt'] < '2023-01-05')]
    around_car_visit_epidemic_period2['std_dt'] = around_car_visit_epidemic_period2['std_dt'].dt.strftime('%Y-%m-%d')

    around_car_visit_epidemic_period2 = around_car_visit_epidemic_period2.sort_values(by = ['regist_no', 'frmhs_no', 'std_dt'])
    around_car_visit_epidemic_period2['rank'] = around_car_visit_epidemic_period2.groupby(['regist_no', 'frmhs_no'])['std_dt'].rank(method = 'first').astype(int).astype(str)
    around_car_visit_epidemic_period2['std_dt'] = around_car_visit_epidemic_period2['rank'] + '. ' + around_car_visit_epidemic_period2['std_dt']

    # 그룹 후 요약
    around_car_visit_epidemic_period2 = around_car_visit_epidemic_period2.groupby(['regist_no', 'frmhs_no'])['std_dt'].apply(summarize_dates).reset_index()
    around_car_visit_epidemic_period2 = pd.merge(farm_point, around_car_visit_epidemic_period2, left_on='farms_no', right_on='frmhs_no')

    around_car_visit_epidemic_period2_occ = around_car_visit_epidemic_period2[
        around_car_visit_epidemic_period2['frmhs_no'].isin(target_farm_list)]
    around_car_visit_epidemic_period2_occ.to_parquet(fr'{path}/eda/차량 방문/수평 전파/28~30차/{regist_no} 차량 발생농장 방문 이력.parquet')

    around_car_visit_epidemic_period2_nonocc = around_car_visit_epidemic_period2[
        ~around_car_visit_epidemic_period2['frmhs_no'].isin(target_farm_list)]
    around_car_visit_epidemic_period2_nonocc.to_parquet(fr'{path}/eda/차량 방문/수평 전파/28~30차/{regist_no} 차량 발생농장 외 방문 이력.parquet')

# ++ ===================================================================================================================

# 33~35차 농장 발생
target_farm_list = ['00052626', '00500425', '01106596']
occ_frmhs_33_35 = occ_frmhs[occ_frmhs['farm_serial_no'].isin(target_farm_list)]
occ_frmhs_33_35['odr_std_dt'] = occ_frmhs_33_35['odr'] + '\n' + '(' + occ_frmhs_33_35['asf_org_day'].astype(str) + ')'
occ_frmhs_33_35.to_parquet(fr'{path}/eda/차량 방문/수평 전파/33~35차/33~35차 발생 농장.parquet')

occ_frmhs_33_35_buffer = occ_frmhs_33_35.copy()
occ_frmhs_33_35_buffer['geometry'] = occ_frmhs_33_35_buffer.geometry.buffer(10000) # 10km buffer

match = gpd.sjoin(wild_occ[['wild_asf_org_day', 'geometry']], occ_frmhs_33_35_buffer, predicate = 'within').drop(['index_right'], axis=1)
match['wild_asf_org_day'] = pd.to_datetime(match['wild_asf_org_day'])
match['asf_org_day'] = pd.to_datetime(match['asf_org_day'])
match['dif'] = (match['asf_org_day'] - match['wild_asf_org_day']).dt.days
print(((match['dif'] <= 365) & (match['dif'] > 0)).sum())

tmp = car_visit_epidemic_period2[car_visit_epidemic_period2['frmhs_no'].isin(target_farm_list)]
tmp2 = tmp.groupby(['regist_no'])['frmhs_no'].nunique().reset_index(name = 'nunique')
tmp2 = tmp2[tmp2['nunique']>=3]

for regist_no in tmp2['regist_no'] :
    tmp3 = car_visit_epidemic_period2[car_visit_epidemic_period2['regist_no'] == regist_no].sort_values(by='visit_de')
    tmp3 = tmp3[['regist_no', 'frmhs_no', 'visit_de', 'visit_purps_cn']]
    print(regist_no, tmp3.shape[0])
    tmp3.to_excel(fr'{path}/eda/차량 방문/수평 전파/33~35차/{regist_no}.xlsx', index=False)

target_farm_str = ",".join([f"'{i}'" for i in target_farm_list])
target_car_str = ",".join([f"'{i}'" for i in tmp2['regist_no'].to_list()])
around_car_visit_epidemic_period = db.query(f'''
    WITH asf_org AS (
        SELECT DISTINCT ON (frmhs_no, sttemnt_de) 
            LPAD(frmhs_no, 8, '0') AS farm_serial_no,
            TO_DATE(sttemnt_de, 'YYYYMMDD') - INTERVAL '1 day' AS asf_org_day
        FROM asf.tn_diss_occrrnc_frmhs_raw_asf oc
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
          AND TO_DATE(sttemnt_de, 'YYYYMMDD') - INTERVAL '1 day' >= '2020-01-01'
          AND LPAD(frmhs_no, 8, '0') IN ({target_farm_str})
    ),

    -- 차량 방문 데이터
    car_visit AS (
        SELECT 
            std_dt,
            frmhs_no,
            visit_de,
            visit_ty,
            visit_purps_cn,
            visit_nm,
            regist_no
        FROM m2msys.tn_visit_info_clean
        WHERE regist_no in ({target_car_str})
        AND visit_de >= (select min(asf_org_day) from asf_org) - INTERVAL '21 days'
        AND visit_de < (select max(asf_org_day) from asf_org) + INTERVAL '1 day'
    )

    SELECT 
    distinct on (v.regist_no, v.frmhs_no, v.visit_de)
        v.regist_no,
        v.frmhs_no,
        v.std_dt,
        v.visit_de,
        v.visit_ty,
        v.visit_purps_cn,
        v.visit_nm
    FROM car_visit v
    CROSS JOIN asf_org a
    WHERE v.visit_de >= a.asf_org_day - INTERVAL '21 days'
      AND v.visit_de <  a.asf_org_day + INTERVAL '1 day'
    ORDER BY v.regist_no, v.visit_de
''')

for regist_no in tmp2['regist_no']:

    around_car_visit_epidemic_period2 = around_car_visit_epidemic_period[
        around_car_visit_epidemic_period['regist_no'] == regist_no]
    around_car_visit_epidemic_period2['std_dt'] = pd.to_datetime(around_car_visit_epidemic_period2['std_dt'])
    around_car_visit_epidemic_period2 = around_car_visit_epidemic_period2[(around_car_visit_epidemic_period2['std_dt'] < '2023-03-31')]
    around_car_visit_epidemic_period2['std_dt'] = around_car_visit_epidemic_period2['std_dt'].dt.strftime('%Y-%m-%d')

    around_car_visit_epidemic_period2 = around_car_visit_epidemic_period2.sort_values(
        by=['regist_no', 'frmhs_no', 'std_dt'])
    around_car_visit_epidemic_period2['rank'] = around_car_visit_epidemic_period2.groupby(['regist_no', 'frmhs_no'])[
        'std_dt'].rank(method='first').astype(int).astype(str)
    around_car_visit_epidemic_period2['std_dt'] = around_car_visit_epidemic_period2['rank'] + '. ' + \
                                                  around_car_visit_epidemic_period2['std_dt']

    around_car_visit_epidemic_period2 = around_car_visit_epidemic_period2.groupby(['regist_no', 'frmhs_no'])['std_dt'].apply(summarize_dates).reset_index()
    around_car_visit_epidemic_period2 = pd.merge(farm_point, around_car_visit_epidemic_period2, left_on='farms_no',
                                                 right_on='frmhs_no')

    around_car_visit_epidemic_period2_occ = around_car_visit_epidemic_period2[
        around_car_visit_epidemic_period2['frmhs_no'].isin(target_farm_list)]
    around_car_visit_epidemic_period2_occ.to_parquet(
        fr'{path}/eda/차량 방문/수평 전파/33~35차/{regist_no} 차량 발생농장 방문 이력.parquet')

    around_car_visit_epidemic_period2_nonocc = around_car_visit_epidemic_period2[
        ~around_car_visit_epidemic_period2['frmhs_no'].isin(target_farm_list)]
    around_car_visit_epidemic_period2_nonocc.to_parquet(
        fr'{path}/eda/차량 방문/수평 전파/33~35차/{regist_no} 차량 발생농장 외 방문 이력.parquet')

# 추가로, 33~35차 발생 농장(00052626, 00500425, 01106596) 모두 방문한 경기-포천-15-0010 차량이
# 29차 발생 농장(01720048)을 방문했던 이력이 존재하는 것을 확인
# 하지만, 방문 시차가 약 2달 정도로, 역학기간 21일을 벗어나 전파의 가능성은 낮을 것으로 사료됨

# ++ ===================================================================================================================

# 49~52차 농장 발생
target_farm_list = ['20432278', '00035758', '00971362', '00035629']
occ_frmhs_49_52 = occ_frmhs[occ_frmhs['farm_serial_no'].isin(target_farm_list)]
occ_frmhs_49_52['odr_std_dt'] = occ_frmhs_49_52['odr'] + '\n' + '(' + occ_frmhs_49_52['asf_org_day'].astype(str) + ')'
occ_frmhs_49_52.to_parquet(fr'{path}/eda/차량 방문/수평 전파/49~52차/49~52차 발생 농장.parquet')

occ_frmhs_49_52_buffer = occ_frmhs_49_52.copy()
occ_frmhs_49_52_buffer['geometry'] = occ_frmhs_49_52_buffer.geometry.buffer(10000) # 10km buffer

match = gpd.sjoin(wild_occ[['wild_asf_org_day', 'geometry']], occ_frmhs_49_52_buffer, predicate = 'within').drop(['index_right'], axis=1)
match['wild_asf_org_day'] = pd.to_datetime(match['wild_asf_org_day'])
match['asf_org_day'] = pd.to_datetime(match['asf_org_day'])
match['dif'] = (match['asf_org_day'] - match['wild_asf_org_day']).dt.days
print(((match['dif'] <= 365) & (match['dif'] > 0)).sum())

tmp = car_visit_epidemic_period2[car_visit_epidemic_period2['frmhs_no'].isin(target_farm_list)]
tmp2 = tmp.groupby(['regist_no'])['frmhs_no'].nunique().reset_index(name = 'nunique')
tmp2 = tmp2[tmp2['nunique']>=2]

car_visit_dict = {}
for regist_no in tmp2['regist_no'] :
    tmp3 = car_visit_epidemic_period2[car_visit_epidemic_period2['regist_no'] == regist_no].sort_values(by='visit_de')
    tmp3 = tmp3[['regist_no', 'frmhs_no', 'visit_de', 'visit_purps_cn']]
    print(regist_no, tmp3.shape[0])
    car_visit_dict[regist_no] = tmp3
    tmp3.to_excel(fr'{path}/eda/차량 방문/수평 전파/49~52차/{regist_no}.xlsx', index=False)

target_farm_str = ",".join([f"'{i}'" for i in target_farm_list])
target_car_str = ",".join([f"'{i}'" for i in tmp2['regist_no'].to_list()])
around_car_visit_epidemic_period = db.query(f'''
    WITH asf_org AS (
        SELECT DISTINCT ON (frmhs_no, sttemnt_de) 
            LPAD(frmhs_no, 8, '0') AS farm_serial_no,
            TO_DATE(sttemnt_de, 'YYYYMMDD') - INTERVAL '1 day' AS asf_org_day
        FROM asf.tn_diss_occrrnc_frmhs_raw_asf oc
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
          AND TO_DATE(sttemnt_de, 'YYYYMMDD') - INTERVAL '1 day' >= '2020-01-01'
          AND LPAD(frmhs_no, 8, '0') IN ({target_farm_str})
    ),

    -- 차량 방문 데이터
    car_visit AS (
        SELECT 
            std_dt,
            frmhs_no,
            visit_de,
            visit_ty,
            visit_purps_cn,
            visit_nm,
            regist_no
        FROM m2msys.tn_visit_info_clean
        WHERE regist_no in ({target_car_str})
        AND visit_de >= (select min(asf_org_day) from asf_org) - INTERVAL '21 days'
        AND visit_de < (select max(asf_org_day) from asf_org) + INTERVAL '1 day'
    )

    SELECT 
    distinct on (v.regist_no, v.frmhs_no, v.visit_de)
        v.regist_no,
        v.frmhs_no,
        v.std_dt,
        v.visit_de,
        v.visit_ty,
        v.visit_purps_cn,
        v.visit_nm
    FROM car_visit v
    CROSS JOIN asf_org a
    WHERE v.visit_de >= a.asf_org_day - INTERVAL '21 days'
      AND v.visit_de <  a.asf_org_day + INTERVAL '1 day'
    ORDER BY v.regist_no, v.visit_de
''')

for regist_no in tmp2['regist_no']:

    around_car_visit_epidemic_period2 = around_car_visit_epidemic_period[
        around_car_visit_epidemic_period['regist_no'] == regist_no]
    around_car_visit_epidemic_period2['std_dt'] = pd.to_datetime(around_car_visit_epidemic_period2['std_dt'])
    around_car_visit_epidemic_period2 = around_car_visit_epidemic_period2[(around_car_visit_epidemic_period2['std_dt'] < '2025-03-16')]
    around_car_visit_epidemic_period2['std_dt'] = around_car_visit_epidemic_period2['std_dt'].dt.strftime('%Y-%m-%d')

    around_car_visit_epidemic_period2 = around_car_visit_epidemic_period2.sort_values(
        by=['regist_no', 'frmhs_no', 'std_dt'])
    around_car_visit_epidemic_period2['rank'] = around_car_visit_epidemic_period2.groupby(['regist_no', 'frmhs_no'])[
        'std_dt'].rank(method='first').astype(int).astype(str)
    around_car_visit_epidemic_period2['std_dt'] = around_car_visit_epidemic_period2['rank'] + '. ' + \
                                                  around_car_visit_epidemic_period2['std_dt']

    around_car_visit_epidemic_period2 = around_car_visit_epidemic_period2.groupby(['regist_no', 'frmhs_no'])['std_dt'].apply(summarize_dates).reset_index()
    around_car_visit_epidemic_period2 = pd.merge(farm_point, around_car_visit_epidemic_period2, left_on='farms_no',
                                                 right_on='frmhs_no')

    around_car_visit_epidemic_period2_occ = around_car_visit_epidemic_period2[
        around_car_visit_epidemic_period2['frmhs_no'].isin(target_farm_list)]
    around_car_visit_epidemic_period2_occ.to_parquet(
        fr'{path}/eda/차량 방문/수평 전파/49~52차/{regist_no} 차량 발생농장 방문 이력.parquet')

    around_car_visit_epidemic_period2_nonocc = around_car_visit_epidemic_period2[
        ~around_car_visit_epidemic_period2['frmhs_no'].isin(target_farm_list)]
    around_car_visit_epidemic_period2_nonocc.to_parquet(
        fr'{path}/eda/차량 방문/수평 전파/49~52차/{regist_no} 차량 발생농장 외 방문 이력.parquet')

# 49차 농장 발생(20432278) 24.12.16
# 50차 농장 발생(00035758) 25.01.20
# 51차 농장 발생(00971362) 25.01.28
# 52차 농장 발생(00035629) 25.03.16

# '경기-양주-13-0061' : 20432278(25.11.27) -> 00035758(25.01.02) 전파 가능성. 실제 거리도 2km가 안됨
# '경기-양주-12-0260' : 00035758(25.01.07) -> 00971362(25.01.16) 전파 가능성
# 경기-양주-12-0255 : 00035758(25.01.13) -> 00971362(25.01.16) 전파 가능성
# '인천-서-14-0021' : 00035758(25.01.15) -> 00971362(25.01.18) 전파 가능성
# '경기-양주-15-0032' : 방문 목적이 시료채취,방역으로 발생 1~2주 전 쯤과 얼추 맞아떨어짐
# '경기-파주-12-0147' : 방문 목적이 시료채취,방역으로 발생 1~3주 전 쯤과 얼추 맞아떨어짐

# ++ ===================================================================================================================

target_frmhs_no = '00035629'

# 역학 기간 내 간접 차량 방문
car_indirect_visit_epidemic_period = db.query(f'''
    WITH asf_org AS (
        SELECT DISTINCT ON (frmhs_no, sttemnt_de)
            LPAD(frmhs_no, 8, '0') AS farm_serial_no,
            TO_DATE(sttemnt_de, 'YYYYMMDD') - INTERVAL '1 day' AS asf_org_day
        FROM asf.tn_diss_occrrnc_frmhs_raw_asf
        WHERE diss_cl = '8' 
          AND delete_at = 'N' 
          AND SLAU_AT = 'N' 
          AND CNTRL_DT IS NOT NULL   
          AND ESTBS_NM NOT LIKE '%예살%'   
          AND ESTBS_NM NOT LIKE '%출하%'
          AND ESTBS_NM NOT LIKE '%차량%'   
          AND ESTBS_NM LIKE '%차%'
          AND TO_DATE(sttemnt_de, 'YYYYMMDD') - INTERVAL '1 day' >= '2020-01-01'
    ),
    
    -- 1. A 농장 방문 차량 ①
    vehicle1_from_asf_farm_a AS (
        SELECT DISTINCT v.regist_no AS vehicle1,
               a.farm_serial_no AS asf_farm_a,
               a.asf_org_day AS asf_org_day_a,
               v.visit_de AS visit_de_a
        FROM m2msys.tn_visit_info_clean v
        JOIN asf_org a ON v.frmhs_no = a.farm_serial_no
        WHERE a.farm_serial_no = '{target_frmhs_no}' 
          AND v.visit_de >= a.asf_org_day - INTERVAL '21 days' 
          AND v.visit_de <  a.asf_org_day + INTERVAL '1 days'
    ),
    
    -- 2. 차량①이 방문한 미발생 농장 B (A 농장 방문 시점 기준 21일 내)
    farm_b_visited_by_vehicle1 AS (
        SELECT DISTINCT v.regist_no AS vehicle1,
               v.frmhs_no AS farm_b,
               v.visit_de AS visit_de_b,
               vf.asf_farm_a,
               vf.asf_org_day_a,
               vf.visit_de_a
        FROM m2msys.tn_visit_info_clean v
        JOIN vehicle1_from_asf_farm_a vf ON v.regist_no = vf.vehicle1
        WHERE v.frmhs_no != vf.asf_farm_a
          AND v.frmhs_no NOT IN (SELECT farm_serial_no FROM asf_org)
          AND v.visit_de >= vf.visit_de_a - INTERVAL '21 days' 
          AND v.visit_de <  vf.visit_de_a + INTERVAL '1 days'
    ),
    
    -- 3. 농장 B를 차량②가 1일 이내로 방문
    vehicle2_from_farm_b AS (
        SELECT DISTINCT v.regist_no AS vehicle2,
               b.vehicle1,
               b.farm_b,
               v.visit_de AS visit_de_b_by_vehicle2,
               b.asf_farm_a,
               b.asf_org_day_a,
               b.visit_de_a,
               b.visit_de_b
        FROM m2msys.tn_visit_info_clean v
        JOIN farm_b_visited_by_vehicle1 b ON v.frmhs_no = b.farm_b
        WHERE v.regist_no != b.vehicle1
            AND DATE_PART('day', b.visit_de_b - v.visit_de) >= 0
            AND DATE_PART('day', b.visit_de_b - v.visit_de) <= 1
    ),
    
    -- 4. 차량②가 발생 농장 C 방문
    vehicle2_to_asf_farm_c_raw AS (
        SELECT v2.regist_no AS vehicle2,
               vb.vehicle1,
               v2.frmhs_no AS asf_farm_c,
               v2.visit_de AS visit_de_c,
               vb.farm_b,
               vb.asf_farm_a,
               vb.asf_org_day_a,
               vb.visit_de_a,
               vb.visit_de_b,
               vb.visit_de_b_by_vehicle2
        FROM m2msys.tn_visit_info_clean v2
        JOIN vehicle2_from_farm_b vb ON v2.regist_no = vb.vehicle2
        WHERE v2.frmhs_no IN (SELECT farm_serial_no FROM asf_org)
          AND v2.frmhs_no != vb.asf_farm_a
    ),
    
    -- 4-1. C 농장의 역학기간 내 방문인지 확인
    vehicle2_to_asf_farm_c AS (
        SELECT v.*,
               ao.asf_org_day AS asf_org_day_c
        FROM vehicle2_to_asf_farm_c_raw v
        JOIN asf_org ao ON v.asf_farm_c = ao.farm_serial_no
        WHERE v.visit_de_c >= ao.asf_org_day - INTERVAL '21 days'
          AND v.visit_de_c <  ao.asf_org_day + INTERVAL '1 days'
    )
    
    -- 최종 결과
    SELECT *
    FROM vehicle2_to_asf_farm_c;
''')

car_indirect_visit_epidemic_period = car_indirect_visit_epidemic_period[
    ~car_indirect_visit_epidemic_period['visit_purps_cn'].isin(car_visit_del_col)]

car_indirect_visit_epidemic_period.to_parquet(fr'{path}/eda/차량 방문/수평 전파/car_indirect_visit_epidemic_period_{target_frmhs_no}.parquet')

# 발생농장(A) <-> 미발생농장(B) <-> 발생농장(C)
# 일단 A를 방문한 차량이 A를 방문한 시점과 B를 방문한 시점의 차이가 적아야 함
# 그리고 B를 방문한 시점과 C를 방문한 시점의 차이도 적어야 함

# ++ ===================================================================================================================

# 고도 분석
base_df_dropdup = base_df[base_df['standard_date'] == '2025-03-15']
base_df_dropdup = base_df_dropdup[['farm_serial_no', 'elevation_nearest', 'elevation_median_500m', 'elevation_median_1km', 'asf_org', 'geometry']]
base_df_dropdup.to_parquet(fr'{path}/eda/고도/고도.parquet')

# ++ ===================================================================================================================

# 야생발생과 가장 가까운 고도, 500m 중위 고도, 1km 중위 고도를 가져옴
# 야생발생 경향을 봤을 때, 특정 고도에서 가장 야생 발생이 많이 발생하더라
# 그래서 그 고도 인근에 있는 농장들은 실제 발생은 안했지만 인근에서 야생발생 위험이 높으므로 발생 가능성이 높다

# 야생발생
query = """
    SELECT distinct on (odr)
    TO_DATE(sttemnt_de, 'YYYYMMDD') AS wild_asf_org_day,
    ST_Transform(ST_SetSRID(ST_Point(lo, la), 4326), 5179) AS geometry
    FROM m2msys.tn_diss_occ_ntcn_info_raw
    WHERE STTEMNT_DE::date >= '2020-01-01' 
    order by odr, last_change_dt desc
"""
wild_occ = gpd.read_postgis(query, db.engine, geom_col='geometry')
wild_occ['wild_asf_org_day'] = pd.to_datetime(wild_occ['wild_asf_org_day'])
wild_occ['wild_asf_org_day'] = wild_occ['wild_asf_org_day'].dt.strftime('%Y-%m-%d')

asf_near_sigun = gpd.read_postgis(sql = 'select * from asf.tb_asf_near_sigun_clean_asf', con = db.engine, geom_col = 'geometry')
wild_occ = gpd.sjoin(wild_occ, asf_near_sigun, how='inner', predicate = 'within').drop(['index_right'], axis=1)

# 농장 Point x, y
wild_occ['x'], wild_occ['y'] = wild_occ.geometry.x, wild_occ.geometry.y
wild_occ_coords = wild_occ[['x', 'y']].values

# ++ ===================================================================================================================

# 고도 포인트
elevation_data = pd.read_sql(
    '''
    SELECT elevation, ST_X(geom) as x, ST_Y(geom) as y
    FROM geoai_polygon.tb_elevation_geom_raw
    ''', con=db.engine
)

# 고도 포인트 KDTree 구축
elevation_coords = elevation_data[['x', 'y']].values
elevation_tree = cKDTree(elevation_coords)

# ++ ===================================================================================================================

# 농장별 거리 검색
# 1. 가장 가까운 elevation
dist_nearest, idx_nearest = elevation_tree.query(wild_occ_coords, k=1)

# 2. 500m 이내 elevation
idxs_500m = elevation_tree.query_ball_point(wild_occ_coords, r=500)

# 3. 1km 이내 elevation
idxs_1km = elevation_tree.query_ball_point(wild_occ_coords, r=1000)

# ++ ===================================================================================================================

# 결과 저장용
nearest_elevation = []
median_500m = []
median_1km = []

for nearest_d, nearest_idx, idxs_500, idxs_1k in zip(dist_nearest, idx_nearest, idxs_500m, idxs_1km):

    # (1) 가장 가까운 elevation. 1km를 벗어나면 None 부여
    if nearest_d <= 1000:
        nearest_elevation.append(elevation_data.iloc[nearest_idx]['elevation'])
    else:
        nearest_elevation.append(None)

    # (2) 500m 이내 중위값
    if idxs_500:
        median_500m.append(np.median(elevation_data.iloc[idxs_500]['elevation']))
    else:
        median_500m.append(None)

    # (3) 1km 이내 중위값
    if idxs_1k:
        median_1km.append(np.median(elevation_data.iloc[idxs_1k]['elevation']))
    else:
        median_1km.append(None)

wild_occ['elevation_nearest'] = nearest_elevation
wild_occ['elevation_median_500m'] = median_500m
wild_occ['elevation_median_1km'] = median_1km

wild_occ = wild_occ[['wild_asf_org_day', 'elevation_nearest', 'elevation_median_500m', 'elevation_median_1km', 'geometry']]
wild_occ.to_parquet(fr'{path}/eda/고도/야생발생 고도.parquet')

print(wild_occ['elevation_nearest'].describe())

plt.figure(figsize=(10, 6))
sns.kdeplot(wild_occ['elevation_nearest'], label='야생발생 지점 고도')
sns.kdeplot(wild_occ['elevation_median_500m'], label='야생발생 반경 500m 내 고도 중위값')
sns.kdeplot(wild_occ['elevation_median_1km'], label='야생발생 반경 1km 내 고도 중위값')
plt.title('ASF 야생발생 지점 고도 분포')
plt.xlabel('고도 (m)')
plt.ylabel('상대적 밀집도')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(fr'{path}/eda/고도/ASF 야생발생 지점 고도 kdeplot')
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=wild_occ[['elevation_nearest', 'elevation_median_500m', 'elevation_median_1km']])
plt.xticks([0, 1, 2], ['야생발생 지점 고도', '야생발생 반경 500m 내\n고도 중위값', '야생발생 반경 1km 내\n고도 중위값'])
plt.title('ASF 야생발생 지점 고도 비교')
plt.ylabel('고도 (m)')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(fr'{path}/eda/고도/ASF 야생발생 지점 고도 boxplot')
plt.show()

base_df_dropdup_200_400 = base_df_dropdup[(base_df_dropdup['elevation_nearest'] >= 200) & (base_df_dropdup['elevation_nearest'] <= 400)]
base_df_dropdup_200_400_1 = base_df_dropdup_200_400[base_df_dropdup_200_400['farm_serial_no'].isin(base_df[base_df['asf_org'] == 1]['farm_serial_no'].unique())]
print(base_df_dropdup_200_400_1['elevation_nearest'].describe())
plt.figure(figsize=(10,6))
sns.boxplot(data=base_df_dropdup_200_400_1.drop(['asf_org'], axis=1))
plt.xticks([0, 1, 2], ['농장 고도', '농장 반경 500m 내\n고도 중위값', '농장 반경 1km 내\n고도 중위값'])
plt.title('ASF 발생 농장 고도 비교')
plt.ylabel('고도 (m)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
base_df_dropdup_200_400_1.to_parquet(fr'{path}/eda/고도/base_df_dropdup_200_400_1.parquet')

base_df_dropdup_200_400_0 = base_df_dropdup_200_400[~base_df_dropdup_200_400['farm_serial_no'].isin(base_df[base_df['asf_org'] == 1]['farm_serial_no'].unique())]
plt.figure(figsize=(10,6))
sns.boxplot(data=base_df_dropdup_200_400_0.drop(['asf_org'], axis=1))
plt.xticks([0, 1, 2], ['농장 고도', '농장 반경 500m 내\n고도 중위값', '농장 반경 1km 내\n고도 중위값'])
plt.title('ASF 미발생 농장 고도 비교')
plt.ylabel('고도 (m)')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
base_df_dropdup_200_400_0.to_parquet(fr'{path}/eda/고도/base_df_dropdup_200_400_0.parquet')

# ++ ===================================================================================================================

# 서식지가능성도 분석
base_df_dropdup = base_df.sort_values(by=['asf_org'], ascending=False).drop_duplicates(subset='farm_serial_no')
base_df_dropdup = base_df_dropdup[['farm_serial_no', 'habitat_possibility_nearest', 'habitat_possibility_median_500m', 'habitat_possibility_median_1km', 'asf_org', 'geometry']]
base_df_dropdup.to_parquet(fr'{path}/eda/서식지가능성도/농장 서식지가능성도.parquet')
base_df_dropdup[base_df_dropdup['habitat_possibility_median_500m'].isnull()].to_parquet(fr'{path}/eda/서식지가능성도/농장 서식지가능성도_결측.parquet')