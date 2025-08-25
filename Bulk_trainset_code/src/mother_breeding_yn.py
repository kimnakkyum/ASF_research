import psycopg2
import pandas as pd

from Bulk_trainset_code.config import config

def mother_pig_breeding_yn():

    conn = psycopg2.connect(config.conn_string)
    cur = conn.cursor()
    cur.execute("""SELECT *, TO_DATE(TO_CHAR(change_dt, 'YYYY-MM') || '-01', 'YYYY-MM-DD') AS std_dt
              FROM m2msys.tn_mobile_blvstck_hist_2017
             WHERE 1=1
               AND DATE_TRUNC('day', change_dt) BETWEEN '2019-01-01' AND '2023-12-31'
               AND MASTER_STTUS_SE IN ('1','2','3','Z')
               AND MGR_CODE IN ('104001','104002')
               AND SEAR_SE = 'Y'
               AND lstksp_cl LIKE '413%'
               AND SUBSTR(BRD_PURPS_CODE,7) = '0003'
               AND frmhs_no != '        '
            UNION ALL
            SELECT *, TO_DATE(TO_CHAR(change_dt, 'YYYY-MM') || '-01', 'YYYY-MM-DD') AS std_dt
              FROM m2msys.tn_mobile_blvstck_hist_2017
             WHERE 1=1
               AND DATE_TRUNC('day', change_dt) BETWEEN '2019-01-01' AND '2023-12-31'
               AND MASTER_STTUS_SE IN ('1','2','3','Z')
               AND MGR_CODE IN ('104001','104002')
               AND SEAR_SE = 'Y'
               AND lstksp_cl LIKE '413%'
               AND SUBSTR(BRD_PURPS_CODE,7) = '0004'
               AND MOTHER_PIG_CO > 0
               AND frmhs_no != '        '
               """)

    df = cur.fetchall()
    df = pd.DataFrame(df)
    df.columns = [desc[0] for desc in cur.description]
    conn.close()

    df['farms_no'] = df['frmhs_no'].astype('str').str.zfill(8)

    df.drop_duplicates(subset=['std_dt', 'farms_no'], inplace=True)

    df = df[['std_dt', 'farms_no']]

    df['mother_at'] = 1

    return df