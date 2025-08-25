import re
import pandas as pd
import numpy as np
import geopandas as gpd
import os
from glob import glob
from advanced_daily_trainset_code.data_cleansing.config import config
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB
import sqlalchemy as sa
import warnings
warnings.filterwarnings("ignore")
from itertools import product
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu

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

# eda를 위해 원본 데이터는 따로 관리
eda_df = base_df.select_dtypes(exclude = ['object']).copy() # object 타입 컬럼 제외
eda_df.drop(['x', 'y', 'geometry'], axis=1, inplace=True) # 좌표 관련 정보 제외

# ++ ===================================================================================================================

# 축종별 사육 여부
lstksp_breeding_at_col = ['breeding_at', 'boar_at', 'black_at', 'common_at']
lstksp_breeding_at_col_map = {'breeding_at' : '종돈 사육',
                              'boar_at' : '멧돼지 사육',
                              'black_at' : '흑돼지 사육',
                              'common_at' : '일반 사육',
                              }

# 대부분 돼지-일반을 키우는 농장에 해당. 합산이 1이 아닌 것은, 현재사육수수가 0인 농장 때문
lstksp_breeding_at_ratio = eda_df[lstksp_breeding_at_col][eda_df['asf_org'] == 0].sum().reset_index(name = '개수')
lstksp_breeding_at_ratio['비율'] = lstksp_breeding_at_ratio['개수'] / (eda_df['asf_org'] == 0).sum()
lstksp_breeding_at_ratio['index'] = lstksp_breeding_at_ratio['index'].map(lstksp_breeding_at_col_map)
lstksp_breeding_at_ratio.rename(columns = {'index' : '축종 사육 여부'}, inplace=True)
lstksp_breeding_at_ratio.loc[4] = ['미 사육',
                                   (eda_df['asf_org'] == 0).sum() - lstksp_breeding_at_ratio['개수'].sum(),
                                   1 - lstksp_breeding_at_ratio['비율'].sum()
                                   ]
lstksp_breeding_at_ratio.to_excel(fr'{path}/eda/축종별 사육 여부/미발생 농장 축종별 사육 여부.xlsx', index=False)
print(lstksp_breeding_at_ratio)

# 발생 농장 또한 돼지-일반을 키우는 농장에서 주로 감염됨
asf_org_farm_lstksp_breeding_at_ratio = eda_df[lstksp_breeding_at_col][eda_df['asf_org'] == 1].sum().reset_index(name = '개수')
asf_org_farm_lstksp_breeding_at_ratio['비율'] = asf_org_farm_lstksp_breeding_at_ratio['개수'] / (eda_df['asf_org'] == 1).sum()
asf_org_farm_lstksp_breeding_at_ratio['index'] = asf_org_farm_lstksp_breeding_at_ratio['index'].map(lstksp_breeding_at_col_map)
asf_org_farm_lstksp_breeding_at_ratio.rename(columns = {'index' : '축종 사육 여부'}, inplace=True)
asf_org_farm_lstksp_breeding_at_ratio.loc[4] = ['미 사육',
                                   (eda_df['asf_org'] == 1).sum() - asf_org_farm_lstksp_breeding_at_ratio['개수'].sum(),
                                   1 - asf_org_farm_lstksp_breeding_at_ratio['비율'].sum()
                                   ]
asf_org_farm_lstksp_breeding_at_ratio.to_excel(fr'{path}/eda/축종별 사육 여부/발생 농장 축종별 사육 여부.xlsx', index=False)
print(asf_org_farm_lstksp_breeding_at_ratio)

# 시각화
lstksp_breeding_at_ratio['구분'] = '미발생'
asf_org_farm_lstksp_breeding_at_ratio['구분'] = '발생'

concat = pd.concat([lstksp_breeding_at_ratio, asf_org_farm_lstksp_breeding_at_ratio], ignore_index=True)
concat['비율'] = concat['비율'] * 100

plt.figure(figsize = (8, 6))
sns.barplot(x = '축종 사육 여부', y = '비율', data = concat, hue = '구분')
plt.title('ASF 발생 여부에 따른 축종 별 사육 농장 비율')
plt.ylabel('비율(%)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(fr'{path}/eda/축종별 사육 여부/ASF 발생 여부에 따른 축종 별 사육 농장 비율.png')
plt.show()

# 결론 : 축종별 사육 여부는 큰 의미를 확인할 수 없을 것으로 판단됨

# ++ ===================================================================================================================

# 기초통계량 함수
def get_describe(df, variable, label):
    desc = df[variable].describe()

    return {
        '변수명' : variable,
        '발생구분': label,
        '데이터 총 개수': len(df),
        '결측 개수': df[variable].isnull().sum(),
        '결측 비율': df[variable].isnull().sum() / len(df) * 100,
        '유효값 개수' : len(df) - df[variable].isnull().sum(),
        '유효값 비율' : (len(df) - df[variable].isnull().sum()) / len(df) * 100,
        '0개수': (df[variable] == 0).sum(),
        '0개수 비율': (df[variable] == 0).sum() / len(df) * 100,
        '1분위수': desc.get('25%', None),
        '2분위수(중앙값)': desc.get('50%', None),
        '3분위수': desc.get('75%', None),
        'min': desc.get('min', None),
        'max': desc.get('max', None),
        'mean': desc.get('mean', None),
        'std': desc.get('std', None)
    }

# 방역관리카드 결측률 확인
describe_list = []
for col in config.manange_card_info_col : # 방역관리카드 컬럼

    # 전체, asf_org == 0, asf_org == 1 통계 계산
    col_describe_concat_list = [
        get_describe(eda_df, col, '전체')
    ]
    col_describe = pd.DataFrame(col_describe_concat_list)
    describe_list.append(col_describe)

manage_card_null_ratio = pd.concat(describe_list)
manage_card_null_ratio.drop(['발생구분'] + list(manage_card_null_ratio.columns[9:]), axis=1, inplace=True)
manage_card_null_ratio.to_excel(fr'{path}/eda/방역관리카드/방역관리카드 결측률.xlsx', index=False)

# 결론 : 방역관리카드는 모든 변수의 결측률이 20%를 넘어, 기초통계량 분석에서 제외
# Action Item : 방역관리카드변수 중 주기적으로 변화하는 변수, 변화하지 않는 변수 구분 필요
eda_df.drop(config.manange_card_info_col, axis=1, inplace=True)

# ++ ===================================================================================================================

# 차량방문 관련 변수 중 원유운반, 기계수리, 알운반, 난좌운반, 가금부산물운반, 가금 출하 상하차 인력운송, 시료채취.방역은 양돈농가 대상 방문이 아니므로 제외
car_visit_remove_col = ['milk', 'repair', 'egg_transportation', 'egg_carrier', 'byproducts', 'manpower_transportation', 'disinfect']
eda_df.drop(car_visit_remove_col, axis=1, inplace=True)

# ++ ===================================================================================================================

# 차량 방문 수
car_visit_col = ['livestock', 'medicines', 'feed', 'manure', 'sawdust', 'compost', 'vaccination', 'artificial_insemination', 'consulting', 'forage', 'operation', 'carcass']
wildboar_count_col = [col for col in eda_df.columns if 'wildboar_count' in col]
env_ratio_col = [col for col in eda_df.columns if ('산림' in col) | ('하천' in col) | ('농지' in col) | ('시가화' in col)]
asf_farm_wild_count_col = [col for col in eda_df.columns if 'cnt_within' in col]

# 0으로 결측 보완
eda_df[car_visit_col + wildboar_count_col + env_ratio_col + asf_farm_wild_count_col] = eda_df[car_visit_col + wildboar_count_col + env_ratio_col + asf_farm_wild_count_col].fillna(0)

# 최근접 농장/야생 발생과의 거리는 최대값을 6km로 제한하고, 결측값도 6km로 대체함
eda_df['asf_farms_dist'] = np.where(eda_df['asf_farms_dist'] > 6000,
                                        6000,
                                        eda_df['asf_farms_dist'].fillna(6000))
eda_df['asf_wild_dist_6month'] = np.where(eda_df['asf_wild_dist_6month'] > 6000,
                                        6000,
                                        eda_df['asf_wild_dist_6month'].fillna(6000))
eda_df['asf_wild_dist_1year'] = np.where(eda_df['asf_wild_dist_1year'] > 6000,
                                        6000,
                                        eda_df['asf_wild_dist_1year'].fillna(6000))

# ++ ===================================================================================================================

# 기초통계량 분석 대상으로 전체/발생/미발생에 대한 기초 통계값 산출
describe_list = []
for col in eda_df.drop(['asf_org'], axis=1).columns : # asf_org를 제외한 모든 변수의 기초통계량 산출

    # 전체, asf_org == 0, asf_org == 1 통계 계산
    col_describe_concat_list = [
        get_describe(eda_df, col, '전체'),
        get_describe(eda_df[eda_df['asf_org'] == 1], col, '발생'),
        get_describe(eda_df[eda_df['asf_org'] == 0], col, '미발생')
    ]
    col_describe = pd.DataFrame(col_describe_concat_list)
    describe_list.append(col_describe)

describe = pd.concat(describe_list)
describe.to_excel(fr'{path}/eda/기초통계량.xlsx', index=False)

# ++ ===================================================================================================================

# 발생/미발생에 대한 KDE, BOX PLOT 시각화

# 색상 설정
colors = {'미발생': 'green', '발생': 'red'}

# PLOT 편의를 위해 영문 컬럼명을 한글 컬럼명으로 변경
for col in eda_df.columns[:-1] :
    eda_df.rename(columns = {col : config.column_translation.get(col, None)}, inplace=True)
eda_df['asf_org_kor'] = eda_df['asf_org'].map({0 : '미발생', 1 : '발생'})

# 통계적 검정 결과 dict
statistics_dict = {'col' : [],
                   'shapiro_stat_0' : [],
                   'shapiro_pvalue_0' : [],
                   'shapiro_stat_1': [],
                   'shapiro_pvalue_1': [],
                   'mannwhitneyu_stat' : [],
                   'mannwhitneyu_pvalue' : [],
                   "cliff's delta" : []
                   }

# Cliff's Delta(효과 크기)를 측정하는 함수
# 각 반경 별 어떤 반경을 선택했을 때, 가장 발생/미발생 간 차이에 미치는 영향이 큰지 산출
def calculate_cliffs_delta(x, y):
    """ 계산 함수"""
    x, y = np.array(x), np.array(y)
    n = len(x) * len(y)
    diff = np.array([i - j for i, j in product(x, y)])
    delta = np.sum(diff > 0) - np.sum(diff < 0)
    return delta / n

# PLOT 대상 변수
for col in eda_df.columns[:-2] : # asf_org 관련 변수 제외

    statistics_dict['col'].append(col)

    # 그룹별 데이터
    group_0 = eda_df[eda_df['asf_org'] == 0][col]
    group_1 = eda_df[eda_df['asf_org'] == 1][col]

    # Figure 및 GridSpec 설정
    plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)

    # 1. KDE Plot
    ax0 = plt.subplot(gs[0])
    sns.kdeplot(group_0, color=colors['미발생'], fill=True, alpha=0.5, label='미발생', linewidth=2, clip=(0, eda_df[col].max()))
    sns.kdeplot(group_1, color=colors['발생'], fill=True, alpha=0.5, label='발생', linewidth=2, clip=(0, eda_df[col].max()))
    ax0.set_xlabel('')
    ax0.set_ylabel('상대적 밀집도')
    ax0.legend()
    ax0.set_title(f'ASF 발생 여부에 따른 {col} 분포 시각화')
    ax0.grid(axis='y', linestyle='--', alpha=0.3)

    # 2. boxplot
    ax1 = plt.subplot(gs[1], sharex=ax0)
    sns.boxplot(data=eda_df, x=col, y='asf_org_kor', palette=colors, orient='h', ax=ax1)
    # ax1.set_xlim(x_min, x_max)
    ax1.set_ylabel('')
    ax1.set_xlabel(col)

    plt.tight_layout()
    plt.savefig(fr'{path}/eda/plot/{col}.png')
    plt.show()

    # ++ ===============================================================================================================

    # 발생/미발생에 대한 정규성 검정 (Shapiro-Wilk Test)
    group_0 = eda_df[eda_df['asf_org'] == 0][col].dropna()
    group_1 = eda_df[eda_df['asf_org'] == 1][col].dropna()

    # Shapiro-Wilk Test 수행
    stat_0, p_0 = shapiro(group_0)
    statistics_dict['shapiro_stat_0'].append(stat_0)
    statistics_dict['shapiro_pvalue_0'].append(p_0)
    print(f"Group 0: Shapiro-Wilk Test Stat: {stat_0:.3f}, p-value: {p_0:.5f}")

    stat_1, p_1 = shapiro(group_1)
    statistics_dict['shapiro_stat_1'].append(stat_1)
    statistics_dict['shapiro_pvalue_1'].append(p_1)
    print(f"Group 1: Shapiro-Wilk Test Stat: {stat_1:.3f}, p-value: {p_1:.5f}")

    # ++ ===============================================================================================================

    # 발생/미발생에 대한 비모수 검정 (Mann-Whitney U Test)

    # Mann-Whitney U Test 수행
    stat, p_value = mannwhitneyu(group_1, group_0)
    statistics_dict['mannwhitneyu_stat'].append(stat)
    statistics_dict['mannwhitneyu_pvalue'].append(p_value)
    print(f'Mann-Whitney U Test Stat: {stat:.3f}, p-value: {p_value:.5f}')

    # ++ ===============================================================================================================

    # 각 변수 별 발생/미발생에 대해 효과크기 측정
    cliffs_delta = calculate_cliffs_delta(group_1, group_0)
    statistics_dict["cliff's delta"].append(cliffs_delta)

statistics_df = pd.DataFrame(statistics_dict)
statistics_df.to_excel(fr'{path}/eda/statistics_df.xlsx', index=False)

# ++ ===================================================================================================================

# 반경 내 농장 발생 횟수 cliff's delta 시각화
around_frmhs_occ = statistics_df[statistics_df['col'].str.contains('최근 1개월 간 농장발생')]
around_frmhs_occ['radius'] = around_frmhs_occ['col'].apply(lambda x : re.findall('\d+', x)[0])

# 그래프 그리기
plt.figure(figsize=(8, 6))
plt.plot(around_frmhs_occ['radius'], around_frmhs_occ["cliff's delta"], marker='o')
plt.title("반경에 따른 발생/미발생 그룹 간 차이의 통계적 크기", fontsize=16)
plt.xlabel("반경 (km)", fontsize=14)
plt.ylabel("통계적 크기", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(fr"{path}/eda/plot/반경 내 농장 발생 횟수 cliff's delta.png")
plt.show()

# ++ ===================================================================================================================

# 반경 내 야생 발생 횟수 cliff's delta 시각화
around_wild_occ = statistics_df[statistics_df['col'].str.contains('최근 6개월 간 야생발생')]
around_wild_occ['radius'] = around_wild_occ['col'].apply(lambda x : re.findall('\d+', x)[0])

# 그래프 그리기
plt.figure(figsize=(8, 6))
plt.plot(around_wild_occ['radius'], around_wild_occ["cliff's delta"], marker='o')
plt.title("반경에 따른 발생/미발생 그룹 간 차이의 통계적 크기", fontsize=16)
plt.xlabel("반경 (km)", fontsize=14)
plt.ylabel("통계적 크기", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig(fr"{path}/eda/plot/반경 내 야생 발생 횟수 cliff's delta.png")
plt.show()