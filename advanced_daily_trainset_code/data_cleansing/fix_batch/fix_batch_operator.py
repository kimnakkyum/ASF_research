import pandas as pd
from functools import reduce
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

from advanced_daily_trainset_code.data_cleansing.fix_batch.farm_elevation_python import create_farm_elevation
from advanced_daily_trainset_code.data_cleansing.fix_batch.farm_habitat_possibility_python import create_farm_habitat_possibility
from advanced_daily_trainset_code.data_cleansing.fix_batch.tb_farm_nearest_habitat_dist_clean_asf_python import create_farm_nearest_habitat_dist
from advanced_daily_trainset_code.data_cleansing.fix_batch.farm_environment import create_farm_environment_fix_batch
from advanced_daily_trainset_code.data_cleansing.fix_batch.farm_habitat_ratio_python import create_farm_habitat_ratio_fix_batch

# 시점 고정 데이터 생성
class FixBatchOperator :
    def __init__(self, db) :
        self.db = db

    # ++ ===============================================================================================================

    def _create_farm_elevation(self):
        create_farm_elevation(self.db)

    def _create_farm_habitat_possibility(self):
        create_farm_habitat_possibility(self.db)

    def _create_farm_nearest_habitat_dist(self):
        create_farm_nearest_habitat_dist(self.db)

    def _create_farm_environment_fix_batch(self, radius_list):
        for radius in radius_list:
            create_farm_environment_fix_batch(self.db, radius)

    def _create_farm_habitat_ratio_fix_batch(self, radius_list):
        for radius in radius_list:
            create_farm_habitat_ratio_fix_batch(self.db, radius)

    def create_fix_dataset(self, radius_list) :
        create_farm_elevation(self.db)
        create_farm_habitat_possibility(self.db)
        create_farm_nearest_habitat_dist(self.db)

        for radius in radius_list :
            create_farm_environment_fix_batch(self.db, radius)
            create_farm_habitat_ratio_fix_batch(self.db, radius)

    # ++ ===============================================================================================================

    def load_farm_elevation(self) :
        farm_elevation = self.db.query(f'''
        SELECT * FROM asf.tb_farm_elevation_clean_asf
        ''')
        return farm_elevation

    def load_farm_habitat_possibility(self) :
        farm_habitat_possibility = self.db.query('''
        SELECT * FROM asf.tb_farm_habitat_possibility_clean_asf
        ''')
        return farm_habitat_possibility

    def load_farm_nearest_habitat_dist(self) :
        farm_nearest_habitat_dist = self.db.query('''
        SELECT * FROM asf.tb_farm_nearest_habitat_dist_clean_asf
        ''')
        return farm_nearest_habitat_dist


    def load_farm_environment_fix_batch(self, radius_list) :

        joined_radius = ", ".join(f"'{r}'" for r in radius_list)
        radius_list_str = f"({joined_radius})"

        farm_environment_fix_batch = self.db.query(f'''
        SELECT * FROM asf.tb_farm_environment_clean_asf
        WHERE year = '9999' 
        AND radius in {radius_list_str}
        AND code = '산림지역_10000_고도'
        ''')
        farm_environment_fix_batch = farm_environment_fix_batch.sort_values(
            by=['farm_serial_no', 'code', 'radius']
        )
        farm_environment_fix_batch = pd.pivot(farm_environment_fix_batch,
                                                    index = 'farm_serial_no',
                                                    columns = ['code', 'radius'],
                                                    values = 'total_intersection_area_m2_ratio'
                                              )
        farm_environment_fix_batch.columns = [
            f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
            for col in farm_environment_fix_batch.columns
        ]
        farm_environment_fix_batch = farm_environment_fix_batch.reset_index()
        return farm_environment_fix_batch

    def load_farm_habitat_ratio_fix_batch(self, radius_list) :

        joined_radius = ", ".join(f"'{r}'" for r in radius_list)
        radius_list_str = f"({joined_radius})"

        farm_habitat_ratio_fix_batch = self.db.query(f'''
        SELECT * FROM asf.tb_farm_environment_clean_asf
        WHERE year = '9999' 
        AND code = '서식지'
        AND radius in {radius_list_str}
        ''')
        farm_habitat_ratio_fix_batch = farm_habitat_ratio_fix_batch.sort_values(
            by=['farm_serial_no', 'code', 'radius']
        )
        farm_habitat_ratio_fix_batch = pd.pivot(farm_habitat_ratio_fix_batch,
                                                    index = 'farm_serial_no',
                                                    columns = ['code', 'radius'],
                                                    values = 'total_intersection_area_m2_ratio'
                                              )
        farm_habitat_ratio_fix_batch.columns = [
            f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
            for col in farm_habitat_ratio_fix_batch.columns
        ]
        farm_habitat_ratio_fix_batch = farm_habitat_ratio_fix_batch.reset_index()
        return farm_habitat_ratio_fix_batch

    def load_fix_dataset(self, radius_list):
        farm_elevation = self.load_farm_elevation()
        farm_habitat_possibility = self.load_farm_habitat_possibility()
        farm_nearest_habitat_dist = self.load_farm_nearest_habitat_dist()
        farm_environment_fix_batch = self.load_farm_environment_fix_batch(radius_list)
        farm_habitat_ratio_fix_batch = self.load_farm_habitat_ratio_fix_batch(radius_list)

        fix_dataset = reduce(lambda x, y: pd.merge(x, y, on='farm_serial_no', how="left"),
                               [
                                farm_elevation,
                                farm_habitat_possibility,
                                farm_nearest_habitat_dist,
                                farm_environment_fix_batch,
                                farm_habitat_ratio_fix_batch
                                ]
                               )
        return fix_dataset