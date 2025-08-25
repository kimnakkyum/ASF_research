import geopandas as gpd
import pandas as pd
import psycopg2
from shapely import wkt

from Bulk_trainset_code.config import config




def manage_card(base_farm_info):

    farm_df = base_farm_info.drop_duplicates(subset=['farms_no'])


    # GeoDataFrame 생성 (lon, lat을 Point 객체로 변환)
    gdf_farms = gpd.GeoDataFrame(
        farm_df,
        geometry=gpd.points_from_xy(farm_df['xmin_'], farm_df['ymin']),
        crs="EPSG:4326"  # WGS84 좌표계 사용
    )

    # 고도 데이터 불러오기
    conn = psycopg2.connect(config.conn_string)
    query = "SELECT *, ST_AsText(geom) AS geom_wkt FROM 검역본부2.고도"
    gdf_elevation = pd.read_sql(query, conn)
    geometry = [wkt.loads(wkt_str) for wkt_str in gdf_elevation['geom_wkt']]
    gdf_elevation = gpd.GeoDataFrame(gdf_elevation, geometry=geometry, crs = 'EPSG:5179')
    conn.close()


    # 고도 데이터의 좌표계 확인 및 변환 (필요시)
    if gdf_elevation.crs != gdf_farms.crs:
        gdf_elevation = gdf_elevation.to_crs(gdf_farms.crs)

    # 공간 조인: 농장 포인트에 가장 가까운 고도 값 할당
    gdf_farms = gpd.sjoin_nearest(gdf_farms, gdf_elevation, how='left', distance_col='distance')

    gdf_farms['elevation'] = gdf_farms['고도']

    gdf_farms = gdf_farms[['farms_no', 'elevation']]

    return gdf_farms
