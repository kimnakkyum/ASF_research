import pandas as pd
import sqlalchemy as sa
from advanced_daily_trainset_code.DBConnect.DB_conn import GetDB

# +++ [이슈사항 정리] +++ ================================================================================================
# 1. *****
# 이슈 코드 : 운영, 휴업 중인 양돈 농장 불러오기
# 이슈 내용 : 쿼리 내에서 운영, 휴업 중인 농장 정보를 불러오는데, 각 함수마다 운영/휴업 중인 농장을 산출하는 코드가 다름. 결과가 같은지 확인 필요
# 개선 방안 : 동일한 농장 대상을 바라보도록 모든 코드 통일 필요

# 2.
# 이슈 코드 : FROM asf.서식지가능성도
# 이슈 내용 : 원천 테이블 미 변경
# 개선 방안 : 180 서버 내 geoai_polygon.tb_wildboar_habitat_possibility_raw로 변경 필요
# 반영 완료

# 3.
# 이슈 코드 : ON ST_DWithin(f.farm_coordinate, h.geom_habitat, 1000) -- 1km 이내만 필터링
# 이슈 내용 : 왜 1km 이내로 필터링 하는지? 서식지 가능성도 데이터를 확인했을 때 point 데이터로 확인되는데,
# 농장 기준 1km로 필터링 했을 때 서식지 가능성도 point가 포함이 안되는 경우는 없었는지
# 개선 방안 : 확인 필요
# 확인 완료

# 4.
# 이슈 코드 : habitat_possibility.drop_duplicates(subset=['farm_serial_no'], inplace=True)
# 이슈 내용 : 농장 별로 가장 가까운 서식지가능도를 가져왔는데, 왜 중복제거가 필요한지? 쿼리의 결과가 농장 번호가 중복된 적이 있는지?
# 개선 방안 : farm_point_5179에서 축종이 여러개 있어서 중복 행 발생. 여기서 농장 별로 1개만 남겨야함
# 반영 완료

# 5.
# 이슈 코드 : FROM monthly_report_partition
# 이슈 내용 : 원천 테이블 미 변경
# 개선 방안 : 180 서버 내 이관된 테이블로 변경 필요
# 반영 완료

# ++ ===================================================================================================================

# 농장 야생멧돼지 서식지가능성도 생성
def create_farm_habitat_possibility(db):

    query = sa.text(f'''
    
        INSERT INTO asf.tb_farm_habitat_possibility_clean_asf (
            farm_serial_no,
            habitat_possibility_nearest,
            habitat_possibility_median_500m,
            habitat_possibility_median_1km
        )
        
        WITH 
        farm_point_5179 AS (
            SELECT 
                farm_serial_no,
                geometry AS farm_coordinate
            FROM asf.tb_farm_geometry_clean_asf
            WHERE geom_type = 'Point'
        ),
        
        habitat_data AS (
            SELECT 
                habitat_possibility, 
                geom AS geom_habitat
            FROM geoai_polygon.tb_wildboar_habitat_possibility_raw
        ),
        
        nearest_habitat AS (
            -- 농장별 가장 가까운 서식지 가능성도
            SELECT DISTINCT ON (f.farm_serial_no)
                f.farm_serial_no,
                h.habitat_possibility AS nearest_habitat
            FROM farm_point_5179 f
            LEFT JOIN habitat_data h 
                ON ST_DWithin(f.farm_coordinate, h.geom_habitat, 1000)
            ORDER BY f.farm_serial_no, ST_Distance(f.farm_coordinate, h.geom_habitat)
        ),
        
        habitat_500m AS (
            -- 500m 이내 서식지 가능성도 중위값
            SELECT 
                f.farm_serial_no,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY h.habitat_possibility) AS median_500m
            FROM farm_point_5179 f
            JOIN habitat_data h 
                ON ST_DWithin(f.farm_coordinate, h.geom_habitat, 500)
            GROUP BY f.farm_serial_no
        ),
        
        habitat_1km AS (
            -- 1km 이내 서식지 가능성도 중위값
            SELECT 
                f.farm_serial_no,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY h.habitat_possibility) AS median_1km
            FROM farm_point_5179 f
            JOIN habitat_data h 
                ON ST_DWithin(f.farm_coordinate, h.geom_habitat, 1000)
            GROUP BY f.farm_serial_no
        )
        
        SELECT 
            f.farm_serial_no,
            nh.nearest_habitat as habitat_possibility_nearest,
            h500.median_500m as habitat_possibility_median_500m,
            h1k.median_1km as habitat_possibility_median_1km
        FROM farm_point_5179 f
        LEFT JOIN nearest_habitat nh ON f.farm_serial_no = nh.farm_serial_no
        LEFT JOIN habitat_500m h500 ON f.farm_serial_no = h500.farm_serial_no
        LEFT JOIN habitat_1km h1k ON f.farm_serial_no = h1k.farm_serial_no
    ''')

    # 쿼리 실행
    with db.engine.begin() as conn:
        conn.execute(query)
