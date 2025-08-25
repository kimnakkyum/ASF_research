import pandas as pd
import psycopg2

from Bulk_trainset_code.config import config

from Bulk_trainset_code.GetData.Dataloader import GetData
from Bulk_trainset_code.src.base_farm_info import FarmFilter

def breeding_yn(base_df):
    # 데이터베이스 연결 및 쿼리 실행
    conn = psycopg2.connect(config.conn_string)
    cur = conn.cursor()
    cur.execute("""
        SELECT frmhs_no, lstksp_cl,
            TO_DATE(TO_CHAR(change_dt, 'YYYY-MM') || '-01', 'YYYY-MM-DD') AS std_dt
        FROM m2msys.tn_mobile_blvstck_hist_2017 
            --WHERE DATE_TRUNC('day', change_dt) BETWEEN '2019-01-01' AND '2023-12-31'
            WHERE DATE_TRUNC('day', change_dt) BETWEEN '{config.start_date}' AND '{config.end_date}'
            AND frmhs_no != '        ' 
            AND (lstksp_cl LIKE '413%' OR lstksp_cl LIKE '419%')
            AND mgr_code IN ('104001', '104002')
            AND master_sttus_se IN ('1', '2', '3', 'Z')
    """)
    df = cur.fetchall()
    df = pd.DataFrame(df)
    df.columns = [desc[0] for desc in cur.description]
    conn.close()

    # DataFrame 전처리
    base_df['farms_no'] = base_df['farms_no'].astype('str').str.zfill(8)
    base_df['std_dt'] = base_df['std_dt'].str[:10]
    df['farms_no'] = df['frmhs_no'].astype('str').str.zfill(8)

    df = df[['lstksp_cl', 'std_dt', 'farms_no']]
    grouped_df = df.groupby(['std_dt', 'farms_no'])['lstksp_cl'].unique().reset_index()

    grouped_df['lstksp_cl'] = grouped_df['lstksp_cl'].apply(lambda x: ', '.join(x))
    grouped_df['lstksp_cl'] = grouped_df['lstksp_cl'].apply(lambda x: x.split(', '))
    exploded_df = grouped_df.explode('lstksp_cl').reset_index(drop=True)

    pivot_df = exploded_df.pivot_table(index=['std_dt', 'farms_no'], columns='lstksp_cl', aggfunc='size', fill_value=0).reset_index()
    pivot_df.columns.name = None
    pivot_df = pivot_df.rename(columns={
        '413016': 'common_at', '413015': 'black_at',
        '413013': 'breeding_at', '413014': 'boar_at'
    })

    for column in ['common_at', 'black_at', 'breeding_at', 'boar_at']:
        if column not in pivot_df:
            pivot_df[column] = 0

    pivot_df[['common_at', 'black_at', 'breeding_at', 'boar_at']] = pivot_df[
        ['common_at', 'black_at', 'breeding_at', 'boar_at']
    ].applymap(lambda x: 1 if x > 0 else 0)

    pivot_df['std_dt'] = pivot_df['std_dt'].astype(str)
    base_df['std_dt'] = base_df['std_dt'].astype(str)

    base_df_blvtsck_at = base_df.merge(pivot_df, how='outer', on=['std_dt', 'farms_no'])

    def fill_na_mode(series):
        filtered = series[series != -1]
        if not filtered.empty:
            mode = filtered.mode()
            if not mode.empty:
                return series.fillna(mode.iloc[0])
        return series.fillna(-1)

    for col in ['common_at', 'black_at', 'breeding_at', 'boar_at']:
        base_df_blvtsck_at[col] = base_df_blvtsck_at.groupby('farms_no')[col].transform(fill_na_mode)

    return base_df_blvtsck_at