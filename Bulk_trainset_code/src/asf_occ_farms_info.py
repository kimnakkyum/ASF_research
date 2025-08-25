import pandas as pd

class Making_label:

    def labeling(self, asf_farm, base_df):

        asf_farm['sttemnt_de'] = pd.to_datetime(asf_farm['sttemnt_de'], format='%Y%m%d')

        asf_farm = asf_farm[(asf_farm['sttemnt_de'] >= '2020-01-01') & (asf_farm['sttemnt_de'] <= '2023-12-31')]

        asf_farm['sttemnt_de'] = asf_farm['sttemnt_de'].apply(lambda x: x.replace(day=1))

        asf_farm.rename(columns={'frmhs_no' : 'farms_no' ,'sttemnt_de' : 'std_dt'}, inplace=True)

        asf_farm['asf_org'] = 1

        asf_farm['farms_no'] = asf_farm['farms_no'].astype('str').str.zfill(8)
        # asf_farm['std_dt'] = asf_farm['std_dt'].astype(str)
        # base_df['std_dt'] = pd.to_datetime(base_df['std_dt'])
        # asf_farm.drop_duplicates(subset=['std_dt' ,'farms_no'])

        asf_farm = asf_farm[['farms_no', 'std_dt', 'asf_org']]
        asf_farm['std_dt'] = asf_farm['std_dt'].astype(str)
        # base_df_asf_org = base_df.merge(asf_farm[['farms_no', 'std_dt', 'asf_org']], on=['std_dt', 'farms_no'],how= 'left')

        # base_df_asf_org['asf_org'].fillna(0, inplace=True)

        return asf_farm