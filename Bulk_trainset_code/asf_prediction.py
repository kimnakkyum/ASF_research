import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier
import os
import shap
import joblib
import matplotlib.pyplot as plt

from config import config
from Bulk_trainset_code.DataBuildOperator import Operation


# 데이터셋 불러오기
tr_train = pd.read_csv(r"D:\gitlab\asf_research\output\dataset.csv", encoding='cp949')
# tr_train = Operation()

# train, test split
df_train = tr_train[tr_train['std_dt'] < config.split_std_dt]
df_target = tr_train[tr_train['std_dt'] >= config.split_std_dt]

rf_list = []
x_original_list = []

# x,y 축 지정
x = df_train[config.x_col].values
y = np.array(df_train['asf_org'].values)

x_original = df_target[config.x_col].values
y_original = np.array(df_target['asf_org'].values)

x_original_list.append(x_original)

# # 오버샘플링, 랜덤지정에 따라 변경
X_resampled, y_resampled = ADASYN(random_state=config.random_seed).fit_resample(x, y)  # 오버샘플링 + 랜덤성 고정할 때
# X_resampled, y_resampled = x, y  #오버샘플링 안할 때

# 모델 학습
rf = RandomForestClassifier(max_depth=3, n_estimators=800, random_state=config.random_seed, n_jobs=-2)
rf.fit(X_resampled, y_resampled)

# 해당월 데이터 예측
y_original_pred = rf.predict(x_original)
y_original_pred_prob = rf.predict_proba(x_original)[:, 1]
df_target['pred_prob'] = y_original_pred_prob.tolist()

df_target["ASF_PERCENTRANK"] = df_target['pred_prob'].rank(pct=True)

df_target["pred_30"] = 0
df_target.loc[df_target['ASF_PERCENTRANK'] >= 0.70, "pred_30"] = 1

df_target["pred_20"] = 0
df_target.loc[df_target['ASF_PERCENTRANK'] >= 0.80, "pred_20"] = 1

df_target["pred_10"] = 0
df_target.loc[df_target['ASF_PERCENTRANK'] >= 0.90, "pred_10"] = 1

# 성능 확인
print("발생 예측 성공률\n", df_target[(df_target['asf_org'] == 1)][['farms_no' ,'pred_prob', 'ASF_PERCENTRANK', 'pred_10', 'pred_20', 'pred_30']])
print("=================================================================")
print("상위 30%이내 예측 성곻률\n",  df_target[(df_target['asf_org'] == 1)]['pred_30'].sum() /len(df_target[(df_target['asf_org'] == 1)]) * 100)
print("상위 20%이내 예측 성곻률\n",  df_target[(df_target['asf_org'] == 1)]['pred_20'].sum() /len(df_target[(df_target['asf_org'] == 1)]) * 100)
print("상위 10%이내 예측 성곻률\n",  df_target[(df_target['asf_org'] == 1)]['pred_10'].sum() /len(df_target[(df_target['asf_org'] == 1)]) * 100)


# # 결과 저장
# df_target_path  = os.path.join(config.OUTPUT_PATH, "df_target.csv")
# df_target.to_csv(df_target_path, encoding='cp949', index=False)


# 위험도 랭크 히스토그램
df_target = df_target[df_target['asf_org'] == 1]
df = df_target["ASF_PERCENTRANK"]

ai_percentrank_data = df * 100

plt.hist(ai_percentrank_data, bins=10, range=(0, 100))

# 그래프 출력
plt.show()



# 모델 저장
# joblib.dump(rf, r"D:\gitlab\asf_research\output\model.pkl")


tot = df_target[['avg_brd_had_co', 'pig_farm_12km_count',
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
             'habitat_fit_score', 'elevation', 'avg_mother_brd_had_co', 'mother_at', 'asf_org', 'farms_no']]


tot2 = tot[tot['asf_org'] == 1]

tot3 = tot2.drop(columns=['asf_org', 'farms_no'])

# # shapely 분석
# df_tmp = pd.DataFrame(x_original, columns=config.x_col)
explainer = shap.TreeExplainer(rf)
shap_values = explainer(tot3)
shap_values_res = shap.Explanation(shap_values[:, :, 1], data=tot3, feature_names=config.x_col)
new_shap_df = pd.DataFrame(shap_values_res.values, columns = config.x_col)
new_shap_df.index = tot2.farms_no
new_shap_df = new_shap_df.reset_index()
df_shap = (new_shap_df.melt(id_vars='farms_no', var_name="variable_name", value_name='variable_importance_level'))



# 관심 있는 변수 목록
selected_columns = [
    'pig_farm_12km_count', 'elevation', 'medicines',
    'avg_mother_brd_had_co', 'feed', 'mother_at',
    'INTRCP_DSNFC_FCLTY_01_AT', 'forest_ratio'
]

# df_shap에서 'variable_name'이 선택한 변수들인 행들만 필터링
df_selected_shap = df_shap[df_shap['variable_name'].isin(selected_columns)]

# 농장별로 데이터 피벗
df_farmwise_shap = df_selected_shap.pivot_table(
    index='farms_no',  # 농장번호를 인덱스로
    columns='variable_name',  # 변수 이름을 컬럼으로
    values='variable_importance_level',  # 변수 중요도 값을 채움
    aggfunc='mean'  # 여러 값이 있을 경우 평균값을 사용
)

# 기초 통계량 계산 (평균, 표준편차, 최소값, 최대값, 25% 분위수, 50% 분위수, 75% 분위수)
df_farmwise_shap_stats = df_farmwise_shap.describe()

df_farmwise_shap_stats.to_csv(r"C:\Users\BV-KIMNAKKYUM\Desktop\df_farmwise_shap_stats.csv")




# 관심 있는 변수 목록
selected_columns = [
    'pig_farm_12km_count', 'elevation', 'medicines',
    'avg_mother_brd_had_co', 'feed', 'mother_at',
    'INTRCP_DSNFC_FCLTY_01_AT', 'forest_ratio'
]

# df_shap에서 'variable_name'이 선택한 변수들인 행들만 필터링
df_selected_shap = df_shap[df_shap['variable_name'].isin(selected_columns)]

# 이제 농장별로 데이터를 추출하고 싶다면, 'farms_no' 기준으로 피벗
df_farmwise_shap = df_selected_shap.pivot_table(
    index='farms_no',  # 농장번호를 인덱스로
    columns='variable_name',  # 변수 이름을 컬럼으로
    values='variable_importance_level',  # 변수 중요도 값을 채움
    aggfunc='mean'  # 여러 값이 있을 경우 평균값을 사용
)

# 농장별로 특정 변수들에 대한 SHAP 값이 추출된 df_farmwise_shap 출력
print(df_farmwise_shap)


df_farmwise_shap.to_csv(r"C:\Users\BV-KIMNAKKYUM\Desktop\df_farmwise_shapdf_farmwise_shapdf_farmwise_shapdf_farmwise_shapdf_farmwise_shap.csv")






variable_importance = df_shap.groupby("variable_name")["variable_importance_level"].mean().abs()

# 2. 중요도 순으로 정렬
variable_importance = variable_importance.sort_values(ascending=False)

# 3. 결과 출력
print(variable_importance)
#
#
#
# df_shap.groupby("variable_name")["variable_importance_level"].mean()



# 1. ASF 발생(1) 농장 목록 가져오기
# asf_farms = df_target[df_target["asf_org"] == 1]["farms_no"]
#
# # 2. 해당 농장의 SHAP 값만 필터링
# df_shap_asf = df_shap[df_shap["farms_no"].isin(asf_farms)]


df_target[df_target["asf_org"] == 1]


df_shap

# 3. 변수별 평균 기여도 계산 (절대값 & 원본)
shap_importance_abs = df_shap_asf.groupby("variable_name")["variable_importance_level"].mean().abs().sort_values(ascending=False)
shap_importance_raw = df_shap_asf.groupby("variable_name")["variable_importance_level"].mean().sort_values(ascending=False)
