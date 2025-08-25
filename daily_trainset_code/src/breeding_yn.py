from Bulk_trainset_code.config import config
import psycopg2
import pandas as pd
import numpy as np

# 운영 ,휴업 중인 양돈 농장 및 축종 별 사육 여부 생성

def making_breeding_yn(target_date):
    conn = psycopg2.connect(config.conn_string)
    cur = conn.cursor()
    cur.execute(f""" with step01 as (
                                    select a.farm_serial_no, a.farm_name, a.farm_owner_name, a.statutorydong_name, 
                                    b.livestock_species_class_code, b.present_breeding_livestock_count, b.present_breeding_livestock_count_average
                                    from (select * from monthly_report_partition.tb_farm_information where standard_date = '{target_date}') a
                                    left join (SELECT *
                                                    FROM monthly_report_partition.tb_livestock_species_information
                                                    WHERE standard_date = '{target_date}'
                                                      AND livestock_species_class_code LIKE '413%') b
                                    on a.farm_serial_no = b.farm_serial_no 
                                    and a.standard_date = b.standard_date
                                    )
                                    
                                    select farm_serial_no, livestock_species_class_code, present_breeding_livestock_count from step01 where livestock_species_class_code is not null """)

    df = cur.fetchall()
    df = pd.DataFrame(df)
    df.columns = [desc[0] for desc in cur.description]
    conn.close()

    # 현재사육두수가 1보다 크면 여(1) 1보다 작으면 부(0)
    df['present_breeding_livestock_count'] = np.where(df['present_breeding_livestock_count'] >= 1, 1, 0)
    df = df.pivot_table(
        index='farm_serial_no',
        columns='livestock_species_class_code',
        values='present_breeding_livestock_count',
        fill_value=0
    ).reset_index()
    df.columns.name = None
    df = df.rename(columns={
        '413016': 'common_at', '413015': 'black_at',
        '413013': 'breeding_at', '413014': 'boar_at'
    })


    return df


