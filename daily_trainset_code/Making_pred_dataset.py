from datetime import datetime, timedelta

from daily_trainset_code.src.base_farm_info import base_farm_info
from daily_trainset_code.src.breeding_yn import making_breeding_yn
from daily_trainset_code.src.nearest_asf_farm_dist import making_asf_farm_dist
from daily_trainset_code.src.nearest_asf_wild_dist import making_asf_wild_dist
from daily_trainset_code.src.pig_farm_count_12km import farm_count
from daily_trainset_code.src.visit_count import making_visit_info
from daily_trainset_code.src.card_info import manage_card
from daily_trainset_code.src.wildboar_count_5km import wildboar_count
from daily_trainset_code.src.wildboar_habitat_dist import nearest_wildboar_habitat_dist
from daily_trainset_code.src.env_info_3km import making_env_ratio
from daily_trainset_code.src.farm_elevation import farm_elevation
from daily_trainset_code.src.habitat_possibility import farm_habitat_possibility
from daily_trainset_code.src.avg_mother_brd_had_co import avg_mother_brd_had_co
from daily_trainset_code.src.mother_pig_breeding_yn import making_mother_pig_breeding_yn
from daily_trainset_code.preprocessing_dataset import making_trainset

from Bulk_trainset_code.config import config


def Operation(target_date):

        # 평균사육수수
        avg_brd_had_co = base_farm_info(target_date)

        # 축종 사육 여부
        breeding_yn = making_breeding_yn(target_date)

        # 최근접 발생농장과의 거리
        asf_farm_dist = making_asf_farm_dist(target_date)

        # 최근접 야생발생과의 거리
        asf_wild_dist = making_asf_wild_dist(target_date)

        # 농장 반경 12km이내 양돈 농장 수
        base_farm = avg_brd_had_co[['farm_serial_no', 'farm_latitude', 'farm_longitude']]
        farm_count_12km = farm_count(base_farm)

        # 차량 변수 (1분 내외)
        visit_info = making_visit_info(target_date[:7])

        # 양돈방역관리카드
        card_info = manage_card(breeding_yn, avg_brd_had_co)

        # 농장 반경 5km이내 야생멧돼지 수 (3분)
        target_year = int(target_date[:4]) - 2
        wildboar_count_5km = wildboar_count(target_date ,str(target_year))

        # 최근접서식지와의 거리 (8분)
        nearest_habitat_dist = nearest_wildboar_habitat_dist(avg_brd_had_co)

        # 3km이내 환경정보 (5분)
        env_ratio = making_env_ratio(target_date)

        # 농장 고도 (8분)
        elevation = farm_elevation(base_farm)

        # 농장 서식지 가능성도 (1분)
        habitat_possibility = farm_habitat_possibility(target_date)

        # 모돈평균사육두수 (13분)
        target_date_dt = datetime.strptime(target_date, '%Y-%m-%d')
        one_year_before = target_date_dt - timedelta(days=365)
        one_year_before_date = one_year_before.strftime('%Y-%m-%d')
        avg_motehr_brd_had_co = avg_mother_brd_had_co(one_year_before_date, target_date)

        # 모돈 사육 여부
        mother_pig_breeding_yn = making_mother_pig_breeding_yn(target_date)

        # 전처리 전 예측 데이터셋
        tot = (avg_brd_had_co.merge(breeding_yn, on='farm_serial_no', how='left').merge(asf_farm_dist, on='farm_serial_no', how='left').merge(asf_wild_dist, on='farm_serial_no', how='left').merge(
                farm_count_12km, on='farm_serial_no', how='left').merge( visit_info, on='farm_serial_no', how='left').merge(card_info, on='farm_serial_no', how='left').
                merge(wildboar_count_5km, on='farm_serial_no', how='left').merge(nearest_habitat_dist, on='farm_serial_no', how='left').merge(env_ratio, on='farm_serial_no', how='left').
                merge(elevation, on='farm_serial_no', how='left').merge(habitat_possibility, on='farm_serial_no', how='left').merge(avg_motehr_brd_had_co, on='farm_serial_no', how='left').
                merge(mother_pig_breeding_yn, on='farm_serial_no', how='left'))

        output_path = f"{config.REPORT_OUTPUT_PATH}/{target_date}_tempt_dataset.csv"
        tot.to_csv(path_or_buf=output_path, encoding='cp949', index=False)

        # # 전처리 후 예측 데이터셋
        dataset = making_trainset(tot)

        output_path_2 = f"{config.REPORT_OUTPUT_PATH}/{target_date}_final_dataset.csv"
        dataset.to_csv(path_or_buf=output_path_2, encoding='cp949', index=False)

        return dataset