import pandas as pd
import joblib
import psycopg2
import shap
from Bulk_trainset_code.config import config
from daily_trainset_code.Making_pred_dataset import Operation
import numpy as np

from daily_trainset_code.Get_ctgr_XAI import get_ctgr_XAI



# 분석 날짜 지정하기
target_date = '2025-01-27'

# 데이터셋 불러오기
df_target = pd.read_csv(fr"D:\주식회사빅밸류 Dropbox\김낙겸\17 2024 업무\240101 농림축산검역본부\ASF\정기 리포트\RAW_REPORT\{target_date}_final_dataset.csv", encoding='cp949')
# df_target = Operation(target_date)

########### 추론
rf = joblib.load(r"D:\gitlab\asf_research\output\model.pkl")

x_original = df_target[config.x_col].values
y_original_pred = rf.predict(x_original)
y_original_pred_prob = rf.predict_proba(x_original)[:, 1]
df_target['pred_prob'] = y_original_pred_prob.tolist()

df_target["ASF_PERCENTRANK"] = df_target['pred_prob'].rank(pct=True)

tot = df_target[['farm_serial_no', 'pred_prob', 'ASF_PERCENTRANK']].sort_values(by=['ASF_PERCENTRANK'], ascending=False)


########### 엑셀 출력
tot['farm_serial_no'] = tot['farm_serial_no'].astype('str').str.zfill(8)

conn = psycopg2.connect(config.conn_string)
cur = conn.cursor()
cur.execute(f""" with step01 as (
                                select a.farm_serial_no, a.farm_name, a.farm_owner_name, a.statutorydong_name,
                                b.livestock_species_class_code, b.present_breeding_livestock_count, b.present_breeding_livestock_count_average
                                from (select * from monthly_report_partition.tb_farm_information where standard_date = '{target_date}') a
                                left join (SELECT *
                                            FROM (
                                                SELECT *,
                                                       ROW_NUMBER() OVER (
                                                           PARTITION BY standard_date, farm_serial_no
                                                           ORDER BY livestock_species_representative_no ASC
                                                       ) AS rn
                                                FROM monthly_report_partition.tb_livestock_species_information
                                                WHERE standard_date = '{target_date}'
                                                  AND livestock_species_class_code LIKE '413%'
                                            ) sub
                                            WHERE rn = 1) b
                                on a.farm_serial_no = b.farm_serial_no
                                and a.standard_date = b.standard_date
                                ),

                                step02 as (
                                select * from (select * from step01) a
                                left join (select * from geoai_mt.tb_code) b
                                on a.livestock_species_class_code = b.code_no
                                )

                                select * from step02 where livestock_species_class_code is not null """)

farm_info = cur.fetchall()
farm_info = pd.DataFrame(farm_info)
farm_info.columns = [desc[0] for desc in cur.description]
conn.close()

tot2 = tot.merge(farm_info, on=['farm_serial_no'], how='left')



######### ASF 농장 위험도 (전체)
# 광역시도, 시군구, 읍면동 칼럼 생성
tot2['광역시도'] = tot2['statutorydong_name'].str.split().str[0]
tot2['시군구'] = tot2['statutorydong_name'].str.split().str[1]
tot2['읍면동'] = tot2['statutorydong_name'].str.split().str[2]

# 농장번호, 농장명, 농장주, 현재사육수수, 평균사육수수, 대표품종 칼럼 생성
tot2.rename(columns={'farm_serial_no' : '농장번호', 'farm_name' : '농장명', 'farm_owner_name' : '농장주', 'code_nm' : '대표품종',
                     'present_breeding_livestock_count' : '현재사육두수'}, inplace=True)

tot2['농장주'] = tot2['농장주'].apply(
    lambda x: x[0] + 'O' * (len(x) - 2) + x[-1] if isinstance(x, str) and len(x) > 1 else (x if isinstance(x, str) else np.nan))

tot2['대표품종'] = tot2['대표품종'].astype(str).str[3:]
tot2['대표품종'] = tot2['대표품종'].replace('일반', '일반돼지')


# 연번 추가
tot2['연번'] = range(1, len(tot2) + 1)

# 위험도 등급 (임의로 등급 작성)
tot2['위험도 등급'] = ''
tot2.loc[(tot2['ASF_PERCENTRANK'] > 0.90) & (tot2['ASF_PERCENTRANK'] <= 1.00), "위험도 등급"] = '매우 위험'
tot2.loc[(tot2['ASF_PERCENTRANK'] > 0.80) & (tot2['ASF_PERCENTRANK'] <= 0.90), "위험도 등급"] = '위험'
tot2.loc[(tot2['ASF_PERCENTRANK'] > 0.70) & (tot2['ASF_PERCENTRANK'] <= 0.80), "위험도 등급"] = '경고'
tot2.loc[(tot2['ASF_PERCENTRANK'] > 0.50) & (tot2['ASF_PERCENTRANK'] <= 0.70), "위험도 등급"] = '주의'
tot2.loc[(tot2['ASF_PERCENTRANK'] <= 0.50) , "위험도 등급"] = '관심'


tot2.to_csv(r"C:\Users\BV-KIMNAKKYUM\Desktop\tot2.csv", encoding='cp949', index=False)


result = tot2[['연번', '광역시도', '시군구', '읍면동', '농장번호', '농장명', '농장주', '대표품종', '현재사육두수', '위험도 등급']]

# 대표 위험 범주
conn = psycopg2.connect(config.conn_string)
cur = conn.cursor()
cur.execute(f"""select name_eng, name_kor, category_name, variable_code from asf.tb_variable_category_v2""")
ctgr = cur.fetchall()
ctgr = pd.DataFrame(ctgr)
ctgr.columns = [desc[0] for desc in cur.description]
variable_code = ctgr.copy()

conn = psycopg2.connect(config.conn_string)
cur = conn.cursor()
cur.execute(f""" SELECT 
    farm_serial_no as farms_no,
    farm_pnu,
    SUBSTRING(farm_pnu FROM 1 FOR 2) AS sido_code,      -- 앞 2자리
    SUBSTRING(farm_pnu FROM 1 FOR 5) AS sigungu_code,   -- 앞 5자리
    SUBSTRING(farm_pnu FROM 1 FOR 8) AS eupmyundong_code -- 앞 8자리
FROM 
    monthly_report_partition.tb_farm_information
WHERE 
    standard_date = '{target_date}' """)
boundary  = cur.fetchall()
boundary  = pd.DataFrame(boundary )
boundary .columns = [desc[0] for desc in cur.description]
conn.close()


df_target.fillna(0, inplace=True)

위험범주 = get_ctgr_XAI(df_target, ctgr, boundary)

max_indices = 위험범주.groupby('farm_serial_no')['nationwide_standard_rank'].idxmax()
대표위험범주 = 위험범주.loc[max_indices]
대표위험범주 = 대표위험범주[['farm_serial_no', 'category_name']]

final_result = result.merge(대표위험범주, left_on='농장번호', right_on = 'farm_serial_no', how='left')


# shapely 분석
df_tmp = pd.DataFrame(x_original, columns=config.x_col)
explainer = shap.TreeExplainer(rf)
shap_values = explainer(df_tmp)
shap_values_res = shap.Explanation(shap_values[:, :, 1], data=df_tmp, feature_names=config.x_col)
new_shap_df = pd.DataFrame(shap_values_res.values, columns = config.x_col)
new_shap_df.index = df_target.farm_serial_no
new_shap_df = new_shap_df.reset_index()
df_shap = (new_shap_df.melt(id_vars='farm_serial_no', var_name="variable_name", value_name='variable_importance_level'))


# variable_importance = df_shap.groupby("variable_name")["variable_importance_level"].mean().abs()
#
# # 2. 중요도 순으로 정렬
# variable_importance = variable_importance.sort_values(ascending=False)
#
# # 3. 결과 출력
# print(variable_importance)


res_df = df_shap[['farm_serial_no', 'variable_name', 'variable_importance_level']]

# 각 카테고리별 상위 3개 선택
ctgr['name_eng'] = ctgr['name_eng'].str.lower()

res_df = res_df.merge(ctgr, left_on=['variable_name'], right_on = 'name_eng' ,how='left')

top3_df = res_df[res_df['variable_importance_level'] >= 0]

top3_df = top3_df.groupby(['farm_serial_no', 'category_name']).apply(lambda x: x.nlargest(3, 'variable_importance_level')).reset_index(drop=True)

top3_df['rank'] = top3_df.groupby(['farm_serial_no', 'category_name']).cumcount() + 1

top3_feature = top3_df.pivot_table(index='farm_serial_no',
                              columns=['category_name', 'rank'],
                              values='name_kor',
                              aggfunc='first')

top3_feature.columns = [f"{category}_top{rank}" for category, rank in top3_feature.columns]

top3_feature = top3_feature.reset_index()
top3_feature['farm_serial_no'] = top3_feature['farm_serial_no'].astype(str).str.zfill(8)

final_result = final_result.merge(top3_feature, on='farm_serial_no', how='left')


# 변수별 영향도
res_df = (res_df[['farm_serial_no', 'variable_name', 'variable_importance_level']].
          merge(variable_code[['name_eng', 'variable_code']], how='left', left_on='variable_name', right_on = 'name_eng'))

res_df_pivot = res_df.pivot_table(
    index='farm_serial_no',
    columns='variable_code',
    values='variable_importance_level',
    aggfunc='first').reset_index()

res_df_pivot.columns.name = None

res_df_pivot['farm_serial_no'] = res_df_pivot['farm_serial_no'].astype(str).str.zfill(8)
final_result = final_result.merge(res_df_pivot, on = 'farm_serial_no', how='left')
final_result.drop(columns='farm_serial_no', inplace=True)

output_path = f"{config.REPORT_OUTPUT_PATH}/{target_date}_ASF_농장위험도_결과.csv"
final_result.to_csv(path_or_buf=output_path, encoding='cp949', index=False)


# 지역별 위험도
emd_farm_result = tot2.loc[tot2.groupby(['광역시도', '시군구', '읍면동'])['pred_prob'].idxmax()]

emd_farm_result["asf_region_percentrank"] = emd_farm_result['pred_prob'].rank(pct=True)

emd_farm_result.sort_values(by=['asf_region_percentrank'], ascending=False, inplace=True)

emd_farm_result['위험도 등급'] = ''
emd_farm_result.loc[(emd_farm_result['asf_region_percentrank'] > 0.90) & (emd_farm_result['asf_region_percentrank'] <= 1.00), "위험도 등급"] = '매우 위험'
emd_farm_result.loc[(emd_farm_result['asf_region_percentrank'] > 0.80) & (emd_farm_result['asf_region_percentrank'] <= 0.90), "위험도 등급"] = '위험'
emd_farm_result.loc[(emd_farm_result['asf_region_percentrank'] > 0.70) & (emd_farm_result['asf_region_percentrank'] <= 0.80), "위험도 등급"] = '경고'
emd_farm_result.loc[(emd_farm_result['asf_region_percentrank'] > 0.50) & (emd_farm_result['asf_region_percentrank'] <= 0.70), "위험도 등급"] = '주의'
emd_farm_result.loc[(emd_farm_result['asf_region_percentrank'] <= 0.50), "위험도 등급"] = '관심'

# 연번 추가
emd_farm_result['연번'] = range(1, len(emd_farm_result) + 1)

emd_farm_result = emd_farm_result[['연번' ,'광역시도', '시군구', '읍면동', '위험도 등급']]

output_path_2 = f"{config.REPORT_OUTPUT_PATH}/{target_date}_ASF_지역별위험도_결과.csv"
emd_farm_result.to_csv(path_or_buf=output_path_2, encoding='cp949', index=False)

print("########### 끝 ################")



# 시도 필터링
df = pd.read_excel(r"C:\Users\BV-KIMNAKKYUM\Desktop\(25.02.31기준) 빅데이터기반 아프리카돼지열병 위험도 예측 보고서_농장,지역별_v2.xlsx", sheet_name='농장 위험도', skiprows=3)
result = pd.read_csv(r"C:\Users\BV-KIMNAKKYUM\Desktop\tot2.csv", encoding='cp949')

dff = df.merge(result[['농장번호', 'pred_prob']], on =['농장번호'])



# 분석 제외 시도 제외
tot2 = dff[~dff['Unnamed: 1'].isin(['경상남도', '전라남도', '전라북도', '충청남도'])]
tot2["ASF_PERCENTRANK"] = tot2['pred_prob'].rank(pct=True)
tot3 = tot2[['농장번호', 'pred_prob', 'ASF_PERCENTRANK']].sort_values(by=['ASF_PERCENTRANK'], ascending=False)


# 위험도 등급 (임의로 등급 작성)
tot3['위험도 등급'] = ''
tot3.loc[(tot3['ASF_PERCENTRANK'] > 0.90) & (tot3['ASF_PERCENTRANK'] <= 1.00), "위험도 등급"] = '매우 위험'
tot3.loc[(tot3['ASF_PERCENTRANK'] > 0.80) & (tot3['ASF_PERCENTRANK'] <= 0.90), "위험도 등급"] = '위험'
tot3.loc[(tot3['ASF_PERCENTRANK'] > 0.70) & (tot3['ASF_PERCENTRANK'] <= 0.80), "위험도 등급"] = '경고'
tot3.loc[(tot3['ASF_PERCENTRANK'] > 0.50) & (tot3['ASF_PERCENTRANK'] <= 0.70), "위험도 등급"] = '주의'
tot3.loc[(tot3['ASF_PERCENTRANK'] <= 0.50) , "위험도 등급"] = '관심'


final = df.merge(tot3, on =['농장번호'], how='inner')

tot2['연번'] = range(1, len(tot2) + 1)

final.to_csv(r"C:\Users\BV-KIMNAKKYUM\Desktop\finalfinalfinalfinal.csv", encoding='cp949', index=False)


# 지역별 위험도
emd_farm_result = tot2.loc[tot2.groupby(['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3'])['pred_prob'].idxmax()]

emd_farm_result["asf_region_percentrank"] = emd_farm_result['pred_prob'].rank(pct=True)

emd_farm_result.sort_values(by=['asf_region_percentrank'], ascending=False, inplace=True)

emd_farm_result['위험도 등급'] = ''
emd_farm_result.loc[(emd_farm_result['asf_region_percentrank'] > 0.90) & (emd_farm_result['asf_region_percentrank'] <= 1.00), "위험도 등급"] = '매우 위험'
emd_farm_result.loc[(emd_farm_result['asf_region_percentrank'] > 0.80) & (emd_farm_result['asf_region_percentrank'] <= 0.90), "위험도 등급"] = '위험'
emd_farm_result.loc[(emd_farm_result['asf_region_percentrank'] > 0.70) & (emd_farm_result['asf_region_percentrank'] <= 0.80), "위험도 등급"] = '경고'
emd_farm_result.loc[(emd_farm_result['asf_region_percentrank'] > 0.50) & (emd_farm_result['asf_region_percentrank'] <= 0.70), "위험도 등급"] = '주의'
emd_farm_result.loc[(emd_farm_result['asf_region_percentrank'] <= 0.50), "위험도 등급"] = '관심'