import torch
import torch.nn as nn

import pandas as pd
from advanced_daily_trainset_code.dl_model.model_test.visit_model.v3_base_model.v3_model_config import model_config

# 모델 구조
class HDNN(nn.Module):
    def __init__(self):
        super(HDNN, self).__init__()

        # model config
        self.mc = model_config()

        # 독립변수 리스트(전파, 방역 카테고리 변수 추가 필요)
        self.col_list = pd.DataFrame({'col': list(
            self.mc.visit_col
        )})

        # ++ ===========================================================================================================

        # 변수군 별 축약 레이어 정의(BatchNorm 추가 여부 고민 필요)

        # 1) 환경 카테고리
        
        # # 야생멧돼지 관련 정보
        # self.wildboar_nn = nn.Sequential(
        #     nn.Linear(self.mc.wildboar_input, self.mc.wildboar_input * 2),
        #     nn.BatchNorm1d(self.mc.wildboar_input * 2),
        #     nn.LeakyReLU(),
        #
        #     nn.Linear(self.mc.wildboar_input * 2, self.mc.wildboar_input),
        #     nn.BatchNorm1d(self.mc.wildboar_input),
        #     nn.LeakyReLU(),
        #
        #     nn.Linear(self.mc.wildboar_input, self.mc.wildboar_output),
        #     nn.BatchNorm1d(self.mc.wildboar_output),
        #     nn.LeakyReLU()
        # )
        #
        # # 주변 환경 정보
        # self.around_env_nn = nn.Sequential(
        #     nn.Linear(self.mc.around_env_input, self.mc.around_env_input * 2),
        #     nn.BatchNorm1d(self.mc.around_env_input * 2),
        #     nn.LeakyReLU(),
        #
        #     nn.Linear(self.mc.around_env_input * 2, self.mc.around_env_input),
        #     nn.BatchNorm1d(self.mc.around_env_input),
        #     nn.LeakyReLU(),
        #
        #     nn.Linear(self.mc.around_env_input, self.mc.around_env_output),
        #     nn.BatchNorm1d(self.mc.around_env_output),
        #     nn.LeakyReLU()
        # )
        #
        # # 주변 농장 정보
        # self.around_farm_nn = nn.Sequential(
        #     nn.Linear(self.mc.around_farm_input, self.mc.around_farm_input * 2),
        #     nn.BatchNorm1d(self.mc.around_farm_input * 2),
        #     nn.LeakyReLU(),
        #
        #     nn.Linear(self.mc.around_farm_input * 2, self.mc.around_farm_input),
        #     nn.BatchNorm1d(self.mc.around_farm_input),
        #     nn.LeakyReLU(),
        #
        #     nn.Linear(self.mc.around_farm_input, self.mc.around_farm_output),
        #     nn.BatchNorm1d(self.mc.around_farm_output),
        #     nn.LeakyReLU()
        # )
        #
        # # 주변 농장 발생 정보
        # self.around_farm_occ_nn = nn.Sequential(
        #     nn.Linear(self.mc.around_farm_occ_input, self.mc.around_farm_occ_input * 2),
        #     nn.BatchNorm1d(self.mc.around_farm_occ_input * 2),
        #     nn.LeakyReLU(),
        #
        #     nn.Linear(self.mc.around_farm_occ_input * 2, self.mc.around_farm_occ_input),
        #     nn.BatchNorm1d(self.mc.around_farm_occ_input),
        #     nn.LeakyReLU(),
        #
        #     nn.Linear(self.mc.around_farm_occ_input, self.mc.around_farm_occ_output),
        #     nn.BatchNorm1d(self.mc.around_farm_occ_output),
        #     nn.LeakyReLU()
        # )
        #
        # # 주변 야생 발생 정보
        # self.around_wild_occ_nn = nn.Sequential(
        #     nn.Linear(self.mc.around_wild_occ_input, self.mc.around_wild_occ_input * 2),
        #     nn.BatchNorm1d(self.mc.around_wild_occ_input * 2),
        #     nn.LeakyReLU(),
        #
        #     nn.Linear(self.mc.around_wild_occ_input * 2, self.mc.around_wild_occ_input),
        #     nn.BatchNorm1d(self.mc.around_wild_occ_input),
        #     nn.LeakyReLU(),
        #
        #     nn.Linear(self.mc.around_wild_occ_input, self.mc.around_wild_occ_output),
        #     nn.BatchNorm1d(self.mc.around_wild_occ_output),
        #     nn.LeakyReLU()
        # )

        # 2. 전파 카테고리(추후 추가 예정)

        # 3. 방역 카테고리(추후 추가 예정)

        # 환경변수 클러스터링
        # self.cluster_env_nn = nn.Sequential(
        #     nn.Linear(self.mc.cluster_env_input, self.mc.cluster_env_output),
        #     nn.BatchNorm1d(self.mc.cluster_env_input),
        #     nn.LeakyReLU()
        # )

        # ++ ===========================================================================================================

        # 축약 Vector Concat 후 FCNN 레이어 정의
        self.pred_nn_1 = nn.Linear(self.mc.pred_dim,
                                   self.mc.pred_dim * self.mc.pred_nn_1_multiply_num)
        self.pred_nn_1_batchnorm = nn.BatchNorm1d(self.mc.pred_dim * self.mc.pred_nn_1_multiply_num)

        self.pred_nn_2 = nn.Linear(self.mc.pred_dim * self.mc.pred_nn_1_multiply_num,
                                   self.mc.pred_dim * self.mc.pred_nn_2_multiply_num)
        self.pred_nn_2_batchnorm = nn.BatchNorm1d(self.mc.pred_dim * self.mc.pred_nn_2_multiply_num)

        self.pred_nn_3 = nn.Linear(self.mc.pred_dim * self.mc.pred_nn_2_multiply_num,
                                   self.mc.pred_dim * self.mc.pred_nn_3_multiply_num)
        self.pred_nn_3_batchnorm = nn.BatchNorm1d(self.mc.pred_dim * self.mc.pred_nn_3_multiply_num)

        self.pred_nn_4 = nn.Linear(self.mc.pred_dim * self.mc.pred_nn_3_multiply_num,
                                   self.mc.pred_dim * self.mc.pred_nn_4_multiply_num)
        self.pred_nn_4_batchnorm = nn.BatchNorm1d(self.mc.pred_dim * self.mc.pred_nn_4_multiply_num)

        self.pred_nn_5 = nn.Linear(self.mc.pred_dim * self.mc.pred_nn_4_multiply_num,
                                   1)

        # ++ ===========================================================================================================

        # 활성화 함수 및 드롭아웃 비율 정의
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(self.mc.dropout)

    def forward(self, x):

        # 1) 환경 카테고리
        # 1-1) 야생멧돼지 관련 정보
        # wildboar_col = list(self.col_list[self.col_list.col.isin(self.mc.wildboar_col)].index)
        # wildboar_vec = self.wildboar_nn(x[:, wildboar_col])
        #
        # # 1-2) 주변 환경 정보
        # around_env_col = list(self.col_list[self.col_list.col.isin(self.mc.around_env_col)].index)
        # around_env_vec = self.around_env_nn(x[:, around_env_col])
        #
        # # 1-3) 주변 농장 정보
        # around_farm_col = list(self.col_list[self.col_list.col.isin(self.mc.around_farm_col)].index)
        # around_farm_vec = self.around_farm_nn(x[:, around_farm_col])
        #
        # # 1-4) 주변 농장 발생 정보
        # around_farm_occ_col = list(self.col_list[self.col_list.col.isin(self.mc.around_farm_occ_col)].index)
        # around_farm_occ_vec = self.around_farm_occ_nn(x[:, around_farm_occ_col])
        #
        # # 1-5) 주변 야생 발생 정보
        # around_wild_occ_col = list(self.col_list[self.col_list.col.isin(self.mc.around_wild_occ_col)].index)
        # around_wild_occ_vec = self.around_wild_occ_nn(x[:, around_wild_occ_col])

        # 2. 전파 카테고리
        # 2-1) 차량 방문 횟수
        # 2-2) 발생 농장 차량 방문 횟수

        # 3. 방역 카테고리

        # ++ ===========================================================================================================

        # 1) 환경 클러스터 카테고리
        cluster_visit_col = list(self.col_list[self.col_list.col.isin(self.mc.visit_col)].index)
        cluster_visit_vec = x[:, cluster_visit_col]

        # ++ ===========================================================================================================

        # 축약된 벡터 concat
        vec_concat = torch.cat([
            cluster_visit_vec
        ], axis=1)

        # ++ ===========================================================================================================

        # FCNN 레이어 연결
        # baseline에서는 dropout 모두 제외하여 진행
        # 첫번째, 마지막 전 완전연결층은 dropout x -> 정보 보존 목적
        out = self.pred_nn_1(vec_concat)
        # out = self.pred_nn_1_batchnorm(out)
        out = self.relu(out)
        # out = self.dropout(out)

        out = self.pred_nn_2(out)
        # out = self.pred_nn_2_batchnorm(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.pred_nn_3(out)
        # out = self.pred_nn_3_batchnorm(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.pred_nn_4(out)
        # out = self.pred_nn_4_batchnorm(out)
        out = self.relu(out)
        # out = self.dropout(out)

        out = self.pred_nn_5(out)
        out = self.sigmoid(out)
        return out