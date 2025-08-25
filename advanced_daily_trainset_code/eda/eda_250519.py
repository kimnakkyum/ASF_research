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
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
import matplotlib.ticker as mtick

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

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

# 유형분류

# 유형분류를 위한 분석 대상 컬럼 선정
analysis_col = ['brd_had_co_sum', 'avg_brd_had_co',
                'asf_farms_dist', 'asf_farm_1month_cnt_within_6km',
                'asf_wild_dist_6month', 'asf_wild_6month_cnt_within_6km',
                'farm_count_within_12km',
                '농지지역_1000',
                '시가화지역_1000',
                'elevation_median_500m',
                'habitat_possibility_median_500m',
                '산림지역_1000'
                ]
cluster_df = base_df[['standard_date', 'farm_serial_no', 'asf_org', 'geometry'] + analysis_col]

# 결측 보완 및 clip
# 농장/야생 발생 거리는 6km 초과하는 것은 6km로 clip, 결측 값도 6km로 대체
cluster_df['asf_farms_dist'] = np.where(cluster_df['asf_farms_dist'] > 6000,
                                        6000,
                                        cluster_df['asf_farms_dist'].fillna(6000))
cluster_df['asf_wild_dist_6month'] = np.where(cluster_df['asf_wild_dist_6month'] > 6000,
                                        6000,
                                        cluster_df['asf_wild_dist_6month'].fillna(6000))
cluster_df[['asf_farm_1month_cnt_within_6km', 'asf_wild_6month_cnt_within_6km', '농지지역_1000', '시가화지역_1000', '산림지역_1000']] = cluster_df[['asf_farm_1month_cnt_within_6km', 'asf_wild_6month_cnt_within_6km', '농지지역_1000', '시가화지역_1000', '산림지역_1000']].fillna(0)

# ++ ===================================================================================================================

# 스케일러
scaler = MinMaxScaler()
cluster_df_scaled = scaler.fit_transform(cluster_df.iloc[:, 4:])

random_state = 42

# 군집 개수 별 inertia 및 실루엣 계수 도출
clusters_inertia_silhouette_dict = {'n_clusters': [], 'inertia': [], 'silhouette': []}

for n_clusters in range(2, 10 + 1):
    clusters_inertia_silhouette_dict['n_clusters'].append(n_clusters)

    # K-means를 통한 유형분류 수행
    model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state)
    cluster_df['cluster'] = model.fit_predict(cluster_df_scaled)
    clusters_inertia_silhouette_dict['inertia'].append(model.inertia_)

    kmeans_silhouette = silhouette_score(cluster_df_scaled, cluster_df['cluster'])
    print(n_clusters, kmeans_silhouette)
    print(cluster_df['cluster'].value_counts())
    clusters_inertia_silhouette_dict['silhouette'].append(kmeans_silhouette)

    # PCA를 통해 차원 축소 후, 군집 분석 결과 시각화
    pca = PCA(n_components=2)
    cluster_df[['pca_1', 'pca_2']] = pca.fit_transform(cluster_df_scaled)

    palette = sns.color_palette(n_colors=n_clusters)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=cluster_df['pca_1'], y=cluster_df['pca_2'], hue=cluster_df['cluster'], alpha=0.7,
                    legend=False, palette=palette)
    plt.title(f'KMeans 군집 개수 {n_clusters}개 분석 결과 시각화')
    plt.xlabel('차원 1')
    plt.ylabel('차원 2')
    plt.tight_layout()
    plt.savefig(
        f'{path}/eda/군집분석/clustering_model_n_cluster_{n_clusters}_pca.png')
    plt.show()

    # ++ ===============================================================================================================

clusters_inertia_silhouette = pd.DataFrame(clusters_inertia_silhouette_dict)
clusters_inertia_silhouette.to_csv(
    f'{path}/eda/군집분석/clusters_inertia_silhouette.txt',
    encoding='utf-8', sep='|', index=False)

# 엘보우 기법 시각화
plt.figure(figsize=(8, 6))
plt.plot(range(2, 10 + 1), clusters_inertia_silhouette['inertia'], 'bo-')
plt.xlabel('군집 수')
plt.ylabel('군집 중심과의 거리 합계')
plt.title('최적 군집 수 선정을 위한 군집 중심과의 거리 합계 시각화')
plt.xticks(range(2, 10 + 1))
plt.tight_layout()
plt.savefig(
    f'{path}/eda/군집분석/clustering_model_inertia.png')
plt.show()

# ++ ===================================================================================================================

# 덴드로그램 시각화
merge_matrix = linkage(cluster_df_scaled, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(merge_matrix, orientation='top', distance_sort='descending',
           show_leaf_counts=False)

plt.title('계층적 군집화 덴드로그램')
plt.xticks([])
plt.ylabel('거리')
plt.tight_layout()
plt.savefig(f'{path}/eda/군집분석/dendrogram.png')
plt.show()

# ++ ===================================================================================================================

# K-means를 통한 유형분류 수행
n_clusters = 4
model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state)
cluster_df['cluster'] = model.fit_predict(cluster_df_scaled)

# 군집 별 기초통계 저장
clustering_model_result_describe_list = []
for cluster in range(4):
    clustering_model_result_describe = cluster_df[cluster_df['cluster'] == cluster][
        cluster_df.columns[3:-3]].describe().reset_index()
    clustering_model_result_describe_list.append(clustering_model_result_describe)

clustering_model_result_describe = pd.concat(clustering_model_result_describe_list, ignore_index=True)
clustering_model_result_describe.to_excel(f'{path}/eda/군집분석/clustering_model_result_describe.xlsx', index=False)

# ++ ===================================================================================================================

# 가장 최근 시점 데이터
latest_df = cluster_df[cluster_df['standard_date'] == '2025-03-15']
print(latest_df['cluster'].value_counts())
latest_df.to_parquet(fr'{path}/eda/군집분석/군집분석_결과_2025_03_15.parquet')

# ++ ===================================================================================================================

wild_occ = gpd.read_parquet(fr'{path}/eda/야생발생.parquet')
wild_occ['wild_asf_org_day'] = pd.to_datetime(wild_occ['wild_asf_org_day'])
wild_occ[(wild_occ['wild_asf_org_day'] >= '2024-09-15') &
         (wild_occ['wild_asf_org_day'] <= '2025-03-15')].to_parquet(fr'{path}/eda/군집분석/야생발생_2024_09_15_이후.parquet')

import pandas as pd
import geopandas as gpd
farm = gpd.read_parquet(fr'{path}/eda/농장.parquet')
occ_farm = gpd.read_parquet(fr'{path}/eda/발생농장.parquet')

# 농장/야생 발생 시군 내 양돈 농장
asf_analysis_target_farm = db.query('''
    select farm_serial_no from asf.tb_asf_analysis_target_farm_clean_asf
    where (occurrence_yn = 1) -- or (occurrence_near_yn = 1)
''')

farm = farm[farm['farm_serial_no'].isin(asf_analysis_target_farm['farm_serial_no'])]
farm.to_parquet(fr'{path}/eda/분석시군 내 농장.parquet')

farm[~farm['farm_serial_no'].isin(occ_farm['farm_serial_no'])].to_parquet(fr'{path}/eda/분석시군 내 미발생농장.parquet')