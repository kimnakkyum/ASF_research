import geopandas as gpd

# ++ ===================================================================================================================

# 최근접 서식지와의 거리 생성
def create_farm_nearest_habitat_dist(db):

    # 농장 폴리곤
    farm_data = gpd.read_postgis(
        '''
        SELECT farm_serial_no, geometry
        FROM asf.tb_farm_geometry_clean_asf
        WHERE geom_type = 'Point'
        ''',
        con=db.engine,
        geom_col='geometry'
    )

    # ++ ===================================================================================================================

    # 서식지 폴리곤
    habitat_data = gpd.read_postgis(
        '''
        SELECT id, geom
        FROM asf.tb_habitat_geom_clean_asf
        WHERE median_habitat_possibility >= 0.75
        ''',
        con=db.engine,
        geom_col='geom'
    )

    # ++ ===================================================================================================================

    # 최근접 서식지와의 거리 공간 연산 수행
    distances = []
    for farm_geom in farm_data.geometry:
        # wildboar_habiat 각 geometry와의 거리를 계산
        boar_distances = habitat_data.geometry.distance(farm_geom)
        # 농장이 wildboar_habiat geometry에 포함되면 거리를 0으로 설정
        min_distance = 0 if habitat_data.geometry.contains(farm_geom).any() else boar_distances.min()
        distances.append(min_distance)

    # 농장 GeoDataFrame에 최단 거리 추가
    farm_data['nearest_habitat_dist'] = distances

    nearest_habitat_dist = farm_data[['farm_serial_no', 'nearest_habitat_dist']].copy()

    # ++ ===================================================================================================================

    # DB 저장
    nearest_habitat_dist.to_sql(
                        'tb_farm_nearest_habitat_dist_clean_asf',
                        con=db.engine,
                        if_exists='append',
                        chunksize=1000,
                        schema='asf',
                        index=False
                    )