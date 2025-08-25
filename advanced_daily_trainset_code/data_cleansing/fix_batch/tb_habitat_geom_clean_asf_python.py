import geopandas as gpd
from shapely import wkb
import pandas as pd


# ++ ===================================================================================================================

#서식지가능성도 중위값 기반 서식지 생성
def create_habitat(db):

    # 재정의된 산림 정보
    forest_df = pd.read_sql("SELECT * FROM asf.tb_forest_geom_clean_asf;", con=db.engine)

    # 서식지가능성도 정보
    habitat_df = pd.read_sql("SELECT * FROM geoai_polygon.tb_wildboar_habitat_possibility_raw", con=db.engine)

    # Geometry 컬럼 변환
    forest_df["geometry"] = forest_df["geom"].apply(wkb.loads)
    habitat_df["geometry"] = habitat_df["geom"].apply(wkb.loads)

    # GeoDataFrame으로 변환
    forest_gdf = gpd.GeoDataFrame(forest_df, geometry="geometry", crs="EPSG:5179")
    habitat_gdf = gpd.GeoDataFrame(habitat_df, geometry="geometry", crs="EPSG:5179")

    # 공간조인: habitat이 forest 내부에 있는 것
    joined = gpd.sjoin(habitat_gdf, forest_gdf, predicate='within', how='inner')

    # forest별 중위값 계산
    median_df = joined.groupby('index_right')['habitat_possibility'].median().reset_index()
    median_df.columns = ['index_right', 'median_habitat_possibility']

    # forest_gdf에 중위값 병합
    forest_with_median = forest_gdf.reset_index().merge(median_df, left_index=True, right_on='index_right')

    # 서식지 정의에 따른 임계치 변경 가능성이 존재하여 0.5로 지정
    habitat = forest_with_median[forest_with_median['median_habitat_possibility'] >= 0.50]

    # 최종 서식지
    habitat_final = habitat[['id', 'geometry', 'median_habitat_possibility']].copy()

    habitat_final.columns = ['id', 'geom', 'median_habitat_possibility']
    habitat_final = habitat_final.set_geometry("geom")


    # PostGIS용 테이블에 저장할 때는 to_postgis 사용
    habitat_final.to_postgis(
        'tb_habitat_geom_clean_asf',
        con=db.engine,
        if_exists='append',
        chunksize=1000,
        schema='asf',
        index=False
    )