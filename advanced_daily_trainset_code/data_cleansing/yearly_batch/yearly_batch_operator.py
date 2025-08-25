import pandas as pd
from functools import reduce
from tqdm import tqdm
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

from advanced_daily_trainset_code.data_cleansing.yearly_batch.farm_environment import create_farm_environment
from advanced_daily_trainset_code.data_cleansing.yearly_batch.farm_wildboar_count import create_farm_wildboar_count
from advanced_daily_trainset_code.data_cleansing.yearly_batch.forest_wildboar_count import create_forest_wildboar_count

class YearlyBatchOperator:
    def __init__(self, db) :
        self.db = db

    # ++ ===============================================================================================================

    def _create_farm_environment(self, year_start, year_end, radius_list) :
        for radius in radius_list :
            for year in tqdm(list(map(str, range(int(year_start), int(year_end) + 1)))):
                create_farm_environment(self.db, year, radius)

    def _create_forest_wildboar_count(self, year_start, year_end):
        for year in tqdm(list(map(str, range(int(year_start), int(year_end) + 1)))):
            create_forest_wildboar_count(self.db, year)

    def _create_farm_wildboar_count(self, year_start, year_end, radius_list) :
        for radius in radius_list :
            for year in tqdm(list(map(str, range(int(year_start), int(year_end) + 1)))):
                create_farm_wildboar_count(self.db, year, radius)

    # ++ ===============================================================================================================

    def load_farm_environment(self, year, radius_list) :

        joined_radius = ", ".join(f"'{r}'" for r in radius_list)
        radius_list_str = f"({joined_radius})"

        farm_environment = self.db.query(f'''
        SELECT * FROM asf.tb_farm_environment_clean_asf
        WHERE year = '{year}'
        AND radius in {radius_list_str}
        ''')

        farm_environment = farm_environment.sort_values(
            by=['farm_serial_no', 'code', 'radius']
        )
        farm_environment = pd.pivot(farm_environment,
                                                    index = 'farm_serial_no',
                                                    columns = ['code', 'radius'],
                                                    values = 'total_intersection_area_m2_ratio'
                                              )
        farm_environment.columns = [
            f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col
            for col in farm_environment.columns
        ]
        farm_environment = farm_environment.reset_index()
        return farm_environment

    def load_farm_wildboar_count(self, year, radius_list) :

        joined_radius = ", ".join(f"'{r}'" for r in radius_list)
        radius_list_str = f"({joined_radius})"

        farm_wildboar_count = self.db.query(f'''
        SELECT * FROM asf.tb_farm_wildboar_count_clean_asf
        WHERE year = '{year}'
        AND radius in {radius_list_str}
        ''')

        farm_wildboar_count = farm_wildboar_count.sort_values(
            by=['farm_serial_no', 'radius']
        )
        farm_wildboar_count = pd.pivot(farm_wildboar_count,
                                                    index = 'farm_serial_no',
                                                    columns = 'radius',
                                                    values = 'wildboar_count'
                                              )

        # 컬럼 이름 변경
        farm_wildboar_count.columns = [
            f"wildboar_count_{col}"
            for col in farm_wildboar_count.columns
        ]
        farm_wildboar_count = farm_wildboar_count.reset_index()
        return farm_wildboar_count

    def load_yearly_dataset(self, year, radius_list):
        farm_environment = self.load_farm_environment(year, radius_list)
        farm_wildboar_count = self.load_farm_wildboar_count(year, radius_list)

        # 위 두 산출물 모두 농장과 매칭 대상 Polygon과 Inner Join이었기 때문에, 각각 누락될 가능성이 존재. 그러므로 outer 조인
        yearly_dataset = reduce(lambda x, y: pd.merge(x, y, on='farm_serial_no', how="outer"),
                               [farm_environment,
                                farm_wildboar_count
                                ]
                               )
        return yearly_dataset