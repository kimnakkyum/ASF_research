import pandas as pd
from functools import reduce
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB
from advanced_daily_trainset_code.data_cleansing.config import config

from advanced_daily_trainset_code.data_cleansing.daily_batch.farm_base_info import create_farm_base_info
from advanced_daily_trainset_code.data_cleansing.daily_batch.all_farms import create_all_farms
from advanced_daily_trainset_code.data_cleansing.daily_batch.farm_breeding_yn import create_farm_breeding_yn
from advanced_daily_trainset_code.data_cleansing.daily_batch.farm_manage_card_info import create_farm_manage_card_info
from advanced_daily_trainset_code.data_cleansing.daily_batch.farm_nearest_asf_farm_dist import create_farm_nearest_asf_farm_dist
from advanced_daily_trainset_code.data_cleansing.daily_batch.farm_around_asf_farm_count import create_farm_around_asf_farm_count
from advanced_daily_trainset_code.data_cleansing.daily_batch.farm_nearest_wild_dist import create_farm_nearest_wild_dist
from advanced_daily_trainset_code.data_cleansing.daily_batch.farm_around_asf_wild_count import create_farm_around_asf_wild_count
from advanced_daily_trainset_code.data_cleansing.daily_batch.farm_around_all_farm_count import create_farm_around_farm_count
from advanced_daily_trainset_code.data_cleansing.daily_batch.farm_around_pig_farm_count import create_farm_around_pig_farm_count
from advanced_daily_trainset_code.data_cleansing.daily_batch.farm_car_visit_count import create_farm_car_visit_count
from advanced_daily_trainset_code.data_cleansing.daily_batch.farm_asf_car_visit_count import create_farm_asf_car_visit_count
from advanced_daily_trainset_code.data_cleansing.daily_batch.farm_cum_rainfall import create_farm_cum_rainfall

# 일 데이터 생성
class DailyBatchOperator :
    def __init__(self, db, standard_date) :
        self.standard_date = standard_date
        self.db = db

    # ++ ===============================================================================================================

    def _create_farm_base_info(self) :
        farm_base_info = create_farm_base_info(self.db, self.standard_date)
        return farm_base_info

    def _create_all_farms(self) :
        all_farms = create_all_farms(self.db, self.standard_date)
        return all_farms

    def _create_farm_breeding_yn(self) :
        farm_breeding_yn = create_farm_breeding_yn(self.db, self.standard_date)
        return farm_breeding_yn

    def _create_farm_manage_card_info(self) :
        farm_manage_card_info = create_farm_manage_card_info(self.db)
        return farm_manage_card_info

    def _create_farm_nearest_asf_farm_dist(self) :
        farm_nearest_asf_farm_dist = create_farm_nearest_asf_farm_dist(self.db, self.standard_date)
        return farm_nearest_asf_farm_dist

    def _create_farm_around_asf_farm_count(self) :
        farm_around_asf_farm_count = create_farm_around_asf_farm_count(self.db, self.standard_date)
        return farm_around_asf_farm_count

    def _create_farm_nearest_wild_dist(self) :
        farm_nearest_wild_dist = create_farm_nearest_wild_dist(self.db, self.standard_date)
        return farm_nearest_wild_dist

    def _create_farm_around_asf_wild_count(self) :
        farm_around_asf_wild_count = create_farm_around_asf_wild_count(self.db, self.standard_date)
        return farm_around_asf_wild_count

    def _create_farm_around_farm_count(self, farm_base_info, all_farms, farm_around_radius_list) :
        farm_around_farm_count = create_farm_around_farm_count(farm_base_info, all_farms, farm_around_radius_list)
        return farm_around_farm_count

    def _create_farm_around_pig_farm_count(self, farm_base_info, farm_around_radius_list) :
        farm_around_pig_farm_count = create_farm_around_pig_farm_count(farm_base_info, farm_around_radius_list)
        return farm_around_pig_farm_count

    def _create_farm_car_visit_count(self) :
        farm_car_visit_count = create_farm_car_visit_count(self.db, self.standard_date)
        return farm_car_visit_count

    def _create_farm_asf_car_visit_count(self) :
        farm_asf_car_visit_count = create_farm_asf_car_visit_count(self.db, self.standard_date)
        return farm_asf_car_visit_count

    def _create_farm_cum_rainfall(self) :
        farm_cum_rainfall = create_farm_cum_rainfall(self.db, self.standard_date)
        return farm_cum_rainfall

    # ++ ===============================================================================================================

    def load_daily_dataset(self, farm_around_radius_list) : # 여러 반경에 대해 테스트 할 경우, input을 어떻게 받을지에 대한 고민 필요
        farm_base_info = self._create_farm_base_info()
        all_farms = self._create_all_farms()
        farm_breeding_yn = self._create_farm_breeding_yn()
        farm_manage_card_info = self._create_farm_manage_card_info()
        farm_nearest_asf_farm_dist = self._create_farm_nearest_asf_farm_dist()
        farm_around_asf_farm_count = self._create_farm_around_asf_farm_count()
        farm_nearest_wild_dist = self._create_farm_nearest_wild_dist()
        farm_around_asf_wild_count = self._create_farm_around_asf_wild_count()
        farm_around_farm_count = self._create_farm_around_farm_count(farm_base_info, all_farms, farm_around_radius_list)
        farm_around_pig_farm_count = self._create_farm_around_pig_farm_count(farm_base_info, farm_around_radius_list)
        farm_car_visit_count = self._create_farm_car_visit_count()
        farm_asf_car_visit_count = self._create_farm_asf_car_visit_count()
        farm_cum_rainfall = self._create_farm_cum_rainfall()

        daily_dataset = reduce(lambda x, y: pd.merge(x, y, on='farm_serial_no', how="left"),
                               [farm_base_info,
                                farm_breeding_yn,
                                farm_manage_card_info,
                                farm_nearest_asf_farm_dist,
                                farm_around_asf_farm_count,
                                farm_nearest_wild_dist,
                                farm_around_asf_wild_count,
                                farm_around_farm_count,
                                farm_around_pig_farm_count,
                                farm_car_visit_count,
                                farm_asf_car_visit_count,
                                farm_cum_rainfall
                                ]
                               )

        # ++ ===============================================================================================================

        # 방역관리카드 결측치 디폴트 1로 대체 (과거 발생 여부, 밀집 단지 여부 칼럼은 제외하고 진행)
        exclude_cols = {"past_occurrence_yn", "crowded_complex_yn"}
        target_cols = [col for col in config.manange_card_info_col if col not in exclude_cols]
        daily_dataset[target_cols] = daily_dataset[target_cols].fillna('1')
        daily_dataset[target_cols] = daily_dataset[target_cols].astype(int)

        # ++ ===============================================================================================================

        # 발생 여부 칼럼 생성
        db = GetDB()
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
        daily_dataset = daily_dataset.merge(
            farm_occurrence[['farm_serial_no', 'asf_org_day', 'asf_org_day_yesterday', 'asf_occurrence_yn']],
            how='left',
            left_on=['farm_serial_no', 'standard_date'],
            right_on=['farm_serial_no', 'asf_org_day_yesterday'])

        daily_dataset['asf_occurrence_yn'].fillna(0, inplace=True)

        return daily_dataset