import sqlalchemy as sa
import pandas as pd
from sqlalchemy import create_engine
from advanced_daily_trainset_code.DBConnect.DB_config import config as DB_config

class GetDB :
    def __init__(self):
        self.engine = create_engine(
            f"postgresql+psycopg2://{DB_config.USER}:{DB_config.PASSWORD}@{DB_config.HOST}:{DB_config.PORT}/{DB_config.DB}")

    def query(self, SQL) :
        with self.engine.connect() as conn:
            df = pd.read_sql_query(sa.text(SQL), con = conn)
        return df