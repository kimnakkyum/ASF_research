from sklearn.linear_model import LinearRegression
import numpy as np

def get_ctgr_XAI(df_target, ctgr, boundary):
    category_lst = ['환경', '전파', '방역']
    # df_base = df_target[["farm_serial_no", "std_dt"]]
    df_base = df_target[["farm_serial_no"]]
    df_base['farm_serial_no'] = df_base['farm_serial_no'].astype(str).str.zfill(8)

    df_target.columns = df_target.columns.str.lower()

    for category_nm in category_lst:
        var_list = ctgr[ctgr["category_name"] == f'{category_nm}'].name_eng
        var_list = var_list.str.lower()
        X = df_target[var_list]
        y = df_target['asf_percentrank']
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        df_base[f'{category_nm}'] = predictions

    df = df_base.merge(boundary, how='left', left_on='farm_serial_no', right_on = 'farms_no')

    df = df[['farm_serial_no', '환경', '전파', '방역', 'sido_code', 'sigungu_code', 'eupmyundong_code']]
    df2 = df.melt(id_vars=["farm_serial_no",  "sido_code", "sigungu_code", "eupmyundong_code"], value_name='importance',
                  var_name='category_name')

    df2["nationwide_standard_rank"] = df2.groupby("category_name")["importance"].rank(pct=True)
    df2["sido_standard_rank"] = df2.groupby(["category_name", "sido_code"])["importance"].rank(pct=True)
    df2["sigungu_standard_rank"] = df2.groupby(["category_name", "sigungu_code"])["importance"].rank(pct=True)

    df2["nationwide_standard_rank"] = np.floor(df2["nationwide_standard_rank"] * 100 ) / 100
    df2["sido_standard_rank"] = np.floor(df2["sido_standard_rank"] * 100 ) / 100
    df2["sigungu_standard_rank"] = np.floor(df2["sigungu_standard_rank"] * 100 ) / 100

    df2 = df2[["farm_serial_no",  "category_name", "nationwide_standard_rank", "sido_standard_rank", "sigungu_standard_rank"]]
    return df2