from advanced_daily_trainset_code.data_cleansing.config import config
import pandas as pd
import geopandas as gpd
from shapely import wkt

# 학습 데이터셋 upload
def Upload_trainingset(db, full_dataset):

    # 종속변수 load
    farm_occurrence = db.query('''
        SELECT DISTINCT ON (frmhs_no, sttemnt_de) 
            LPAD(frmhs_no, 8, '0') AS farm_serial_no,
            TO_DATE(sttemnt_de, 'YYYYMMDD') AS asf_org_day,
            1 AS asf_occurrence_yn
        FROM asf.tn_diss_occrrnc_frmhs_raw_asf_bak oc
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
    farm_occurrence['asf_org_day_yesterday'] = farm_occurrence['asf_org_day'] - pd.Timedelta(days=1)
    for col in ['asf_org_day', 'asf_org_day_yesterday']:
        farm_occurrence[col] = farm_occurrence[col].dt.strftime('%Y-%m-%d')

    # 발생 여부 칼럼 추가
    full_dataset = full_dataset.merge(farm_occurrence[['farm_serial_no', 'asf_org_day', 'asf_org_day_yesterday', 'asf_occurrence_yn']],
                            how='left',
                            left_on=['farm_serial_no', 'standard_date'],
                            right_on=['farm_serial_no', 'asf_org_day_yesterday'])

    full_dataset['asf_occurrence_yn'].fillna(0, inplace=True)

    # 환경 비율 관련 칼럼명 변경 필요
    full_dataset.rename(columns=config.full_dataset_column_translation, inplace=True)

    # 학습셋 칼럼 추출
    training_df = full_dataset[config.training_set_col]

    # 문자열을 shapely Point 객체로 변환
    training_df['farm_coordinate'] = training_df['farm_coordinate'].apply(wkt.loads)
    training_df = gpd.GeoDataFrame(training_df, geometry='farm_coordinate', crs='EPSG:5179')

    # 학습셋 upload
    training_df.to_postgis(
        'tb_trainingset_raw_asf',
        con=db.engine,
        if_exists='append',
        chunksize=1000,
        schema='asf',
        index=False
    )