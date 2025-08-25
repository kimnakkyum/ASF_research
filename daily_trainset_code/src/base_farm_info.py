from Bulk_trainset_code.config import config
import psycopg2
import pandas as pd

# 운영 ,휴업 중인 양돈 농장 및 평균사육두수 생성

def base_farm_info(target_date):

    conn = psycopg2.connect(config.conn_string)
    cur = conn.cursor()
    cur.execute(f""" with step01 as (
                                    select a.farm_serial_no, a.farm_name, a.farm_owner_name, a.statutorydong_name, a.farm_latitude, a.farm_longitude,
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
                                    )
                                    
                                    select * from step01 where livestock_species_class_code is not null """)

    farm_info = cur.fetchall()
    farm_info = pd.DataFrame(farm_info)
    farm_info.columns = [desc[0] for desc in cur.description]
    conn.close()

    avg_brd_had_co = farm_info[['farm_serial_no', 'farm_latitude', 'farm_longitude','present_breeding_livestock_count_average']]

    return avg_brd_had_co