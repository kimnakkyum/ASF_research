import os
import psycopg2
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point
from tqdm import tqdm

from Bulk_trainset_code.config import config


# 야생멧돼지서식지 (야생멧돼지수가 50마리 이상인 산림) 불러오기
conn = psycopg2.connect(config.conn_string)
cur = conn.cursor()
cur.execute("""select * from asf.wildboar_habitat""")
wildboar_habiat = cur.fetchall()
wildboar_habiat = pd.DataFrame(wildboar_habiat)
wildboar_habiat.columns = [desc[0] for desc in cur.description]
conn.close()

file_path  = os.path.join(config.FEATURE_PATH, "base_df_재산출.csv")
base_df = pd.read_csv(file_path, encoding='cp949')
farm = base_df.drop_duplicates(subset=['farms_no'])[['farms_no', 'xmin_', 'ymin']]

geometry = [Point(xy) for xy in zip(farm['xmin_'], farm['ymin'])]
farm = gpd.GeoDataFrame(farm, geometry=geometry, crs="EPSG:4326")
farm = farm.to_crs("EPSG:5179")

wildboar_habiat['geometry'] = wildboar_habiat['geom'].apply(wkt.loads)
wildboar_habiat = gpd.GeoDataFrame(wildboar_habiat, geometry='geometry', crs='EPSG:5179')

distances = []
for farm_geom in tqdm(farm.geometry, desc="Calculating distances"):
    # wildboar_habiat 각 geometry와의 거리를 계산
    boar_distances = wildboar_habiat.geometry.distance(farm_geom)
    # 농장이 wildboar_habiat geometry에 포함되면 거리를 0으로 설정
    min_distance = 0 if wildboar_habiat.geometry.contains(farm_geom).any() else boar_distances.min()
    distances.append(min_distance)

# 농장 GeoDataFrame에 최단 거리 추가
farm['habitat_dist'] = distances

habitat_dist_path  = os.path.join(config.FEATURE_PATH, "야생멧돼지서식지_거리_50.csv")
farm[['farms_no' ,'habitat_dist']].to_csv(habitat_dist_path, encoding='cp949', index=False)