import pandas as pd

class FarmFilter:
    def __init__(self, excluded_farms=None):

        if excluded_farms is None:
            excluded_farms = [11865, 199454, 200320, 580241, 646223, 20001297]
        self.excluded_farms = excluded_farms

    def filter(self, df):

        if 'farms_no' not in df.columns:
            raise ValueError("The DataFrame must contain a 'farms_no' column.")

        # base_farm_info
        # df['std_dt'] = pd.to_datetime(df['std_dt'], format='%Y%m%d')
        df = df[~df['farms_no'].isin(self.excluded_farms)]

        base_farm_info = df.drop_duplicates(subset=['farms_no', 'std_dt'])[['std_dt', 'farms_no', 'xmin_', 'ymin']]
        base_farm_info['std_dt'] = base_farm_info['std_dt'].str[:10]

        # 평균사육수수
        avg_brd_had_co = df.groupby(['std_dt', 'farms_no'])['avg_brd_had_co'].sum().reset_index()
        avg_brd_had_co['farms_no'] = avg_brd_had_co['farms_no'].astype('str').str.zfill(8)
        avg_brd_had_co['std_dt'] = avg_brd_had_co['std_dt'].str[:10]

        return base_farm_info, avg_brd_had_co

